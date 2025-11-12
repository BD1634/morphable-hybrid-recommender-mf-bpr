import numpy as np


class SyntheticCatalog:
    """
    Synthetic item catalog with:
    - an interpretable category keyword (so we can "filter by intent")
    - lightweight "text" (title tokens)
    - an item embedding computed from token embeddings (no external deps)
    """

    def __init__(
        self,
        num_items: int = 500,
        num_categories: int = 12,
        words_per_item: int = 4,
        embed_dim: int = 64,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        self.num_items = num_items
        self.num_categories = num_categories
        self.words_per_item = words_per_item
        self.embed_dim = embed_dim

        # Category keywords double as high-signal text tokens for the demo.
        self.category_keywords = [
            "modern",
            "classic",
            "minimal",
            "premium",
            "cozy",
            "sport",
            "eco",
            "smart",
            "compact",
            "studio",
            "outdoor",
            "quiet",
        ][:num_categories]

        extra_words = ["bold", "eco", "relaxed", "studio", "fast", "warm", "clean", "light", "quiet", "durable"]
        vocab = list(dict.fromkeys(self.category_keywords + extra_words))
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # Learnable-ish token embeddings (randomly initialized).
        W = rng.normal(0, 1.0, size=(len(self.vocab), embed_dim)).astype(np.float64)
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        self.word_embeddings = W

        self.categories = rng.integers(0, num_categories, size=num_items).astype(np.int64)
        self.prices = rng.uniform(10, 120, size=num_items).astype(np.float64).round(2)

        self.item_word_lists: list[list[str]] = []
        self.titles: list[str] = []
        item_vecs = np.zeros((num_items, embed_dim), dtype=np.float64)

        for iid in range(num_items):
            cat_word = self.category_keywords[self.categories[iid]]
            other_words = rng.choice(self.vocab, size=words_per_item - 1, replace=True).tolist()
            # Ensure the first token is always the category keyword for easier parsing.
            words = [cat_word] + other_words
            self.item_word_lists.append(words)
            self.titles.append(" ".join(words))

            vec = np.sum([self.word_embeddings[self.word_to_idx[w]] for w in words], axis=0)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            item_vecs[iid] = vec

        self.item_vectors = item_vecs

    def get_item(self, item_id: int) -> dict:
        return {
            "item_id": int(item_id),
            "category": int(self.categories[item_id]),
            "category_keyword": self.category_keywords[int(self.categories[item_id])],
            "price": float(self.prices[item_id]),
            "title": self.titles[item_id],
        }

    def get_items(self, item_ids: list[int]) -> list[dict]:
        return [self.get_item(iid) for iid in item_ids]

    def get_item_vectors(self, item_ids: np.ndarray) -> np.ndarray:
        return self.item_vectors[item_ids]

    def get_user_vector_from_keywords(self, keywords: list[str]) -> np.ndarray:
        """
        Build an embedding vector from a small set of preference keywords.
        """
        idxs = [self.word_to_idx[k] for k in keywords if k in self.word_to_idx]
        if not idxs:
            # Fallback: near-zero vector.
            return np.zeros(self.embed_dim, dtype=np.float64)
        vec = np.sum(self.word_embeddings[idxs], axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec


class SyntheticInteractions:
    """
    Generates implicit positives using a *mixture* of:
    - latent collaborative factors (MF/BPR learns this)
    - "text" preference similarity between user keywords and item vectors (content learns this)
    """

    def __init__(
        self,
        catalog: SyntheticCatalog,
        num_users: int = 200,
        factors_k: int = 32,
        positives_per_user: int = 20,
        seed: int = 7,
        w_collab: float = 1.0,
        w_content: float = 1.0,
    ):
        rng = np.random.default_rng(seed)

        self.catalog = catalog
        self.num_users = num_users
        self.num_items = catalog.num_items
        self.factors_k = factors_k
        self.positives_per_user = positives_per_user

        # Hidden "true" factors (collaborative side).
        self.user_true = rng.normal(0, 1.0, size=(num_users, factors_k))
        self.item_true = rng.normal(0, 1.0, size=(self.num_items, factors_k))
        self.item_bias = rng.normal(0, 0.25, size=(self.num_items,))

        # Generate user personas as keyword sets.
        self.user_keywords: list[list[str]] = []
        user_content_vecs = np.zeros((num_users, catalog.embed_dim), dtype=np.float64)
        for u in range(num_users):
            # Pick 2-3 category keywords and mix in one extra word sometimes.
            cat_kw = rng.choice(catalog.category_keywords, size=2, replace=False).tolist()
            maybe_extra = rng.choice(catalog.vocab, size=1, replace=True).tolist()
            keywords = cat_kw + maybe_extra
            # Keep list unique but order-stable enough for readability.
            seen = set()
            filtered = []
            for k in keywords:
                if k not in seen:
                    filtered.append(k)
                    seen.add(k)
            self.user_keywords.append(filtered)
            user_content_vecs[u] = catalog.get_user_vector_from_keywords(filtered)

        self.user_content_vecs = user_content_vecs

        # Score and pick top positives for each user.
        self.positives: list[list[int]] = []
        for u in range(num_users):
            collab_scores = self.user_true[u] @ self.item_true.T + self.item_bias
            content_scores = self.user_content_vecs[u] @ catalog.item_vectors.T
            scores = w_collab * collab_scores + w_content * content_scores
            scores = scores + rng.normal(0, 0.2, size=self.num_items)  # noise

            top_idx = np.argpartition(scores, -positives_per_user)[-positives_per_user:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            self.positives.append([int(x) for x in top_idx])

        # Precompute user "content centroids" from positives for the content model.
        self.user_pos_content_vecs = np.zeros((num_users, catalog.embed_dim), dtype=np.float64)
        for u in range(num_users):
            pos_ids = np.array(self.positives[u], dtype=np.int64)
            vec = np.mean(catalog.get_item_vectors(pos_ids), axis=0)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            self.user_pos_content_vecs[u] = vec

        # Item popularity from interaction logs.
        counts = np.zeros(self.num_items, dtype=np.int32)
        for u in range(num_users):
            for iid in self.positives[u]:
                counts[iid] += 1
        self.item_popularity = counts / max(1, counts.max())

    def interacted_set(self, user_id: int) -> set[int]:
        return set(self.positives[user_id])

    def get_user_keywords(self, user_id: int) -> list[str]:
        return self.user_keywords[user_id]


