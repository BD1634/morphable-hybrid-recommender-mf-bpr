import numpy as np


class MFRecommender:
    """
    Simple matrix factorization recommender trained with BPR pairwise ranking.

    This is intentionally lightweight (numpy-only) to keep the demo runnable
    without heavyweight dependencies.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        k: int = 32,
        seed: int = 0,
        lr: float = 1e-2,
        reg: float = 1e-3,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.lr = lr
        self.reg = reg

        rng = np.random.default_rng(seed)
        self.P = 0.1 * rng.normal(size=(num_users, k))
        self.Q = 0.1 * rng.normal(size=(num_items, k))
        self.bu = np.zeros(num_users, dtype=np.float64)
        self.bi = np.zeros(num_items, dtype=np.float64)
        self.global_bias = 0.0

    def predict_scores(self, user_id: int) -> np.ndarray:
        """
        Return a score for every item for the given user.
        """
        # scores[u] = bu + bi + dot(P[u], Q[i])
        return (
            self.global_bias
            + self.bu[user_id]
            + self.bi
            + (self.P[user_id] @ self.Q.T)
        )

    def predict_scores_normalized(self, user_id: int) -> np.ndarray:
        """
        Z-score normalize scores so they can be combined with content scores.
        """
        scores = self.predict_scores(user_id).astype(np.float64)
        mean = float(np.mean(scores))
        std = float(np.std(scores)) + 1e-9
        return (scores - mean) / std

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit_bpr(
        self,
        positives: list[list[int]],
        num_epochs: int = 12,
        samples_per_epoch: int = 2500,
        seed: int = 123,
        neg_samples: int = 1,
    ) -> None:
        rng = np.random.default_rng(seed)

        num_users = len(positives)
        for epoch in range(num_epochs):
            # SGD-like training with sampled triplets (u, i+, i-).
            for _ in range(samples_per_epoch):
                u = int(rng.integers(0, num_users))
                pos_list = positives[u]
                if not pos_list:
                    continue
                i_pos = int(pos_list[rng.integers(0, len(pos_list))])

                # Sample negatives uniformly (skipping positives).
                # (For better negatives you'd sample from item popularity / hard negatives.)
                j = int(rng.integers(0, self.num_items))
                while j in pos_list:
                    j = int(rng.integers(0, self.num_items))

                # BPR loss uses: -log(sigmoid(s_ui - s_uj))
                s_ui = self.global_bias + self.bu[u] + self.bi[i_pos] + self.P[u] @ self.Q[i_pos]
                s_uj = self.global_bias + self.bu[u] + self.bi[j] + self.P[u] @ self.Q[j]
                x = s_ui - s_uj
                sig = self._sigmoid(x)

                # Gradients for maximizing log(sigmoid(x)):
                # d/dx log(sigmoid(x)) = 1 - sigmoid(x)
                grad = 1.0 - sig

                # Update biases
                self.bu[u] += self.lr * (grad * (1.0 - 0.0) - self.reg * self.bu[u])
                self.bi[i_pos] += self.lr * (grad * 1.0 - self.reg * self.bi[i_pos])
                self.bi[j] += self.lr * (-grad * 1.0 - self.reg * self.bi[j])

                # Update factors
                P_u = self.P[u].copy()
                Q_i = self.Q[i_pos].copy()
                Q_j = self.Q[j].copy()

                self.P[u] += self.lr * (grad * (Q_i - Q_j) - self.reg * P_u)
                self.Q[i_pos] += self.lr * (grad * P_u - self.reg * Q_i)
                self.Q[j] += self.lr * (-grad * P_u - self.reg * Q_j)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_interacted=None,
        item_popularity=None,
        diversity: bool = False,
    ) -> list[int]:
        scores = self.predict_scores(user_id)
        if exclude_interacted:
            # Hard-mask interacted items.
            for iid in exclude_interacted:
                if 0 <= iid < self.num_items:
                    scores[iid] = -np.inf

        # Basic top-k from collaborative signal.
        ranked = np.argsort(-scores)[: max(k * 3, k)]

        if not diversity:
            return [int(iid) for iid in ranked[:k]]

        # Diversity re-rank: pick items to cover multiple categories by
        # using popularity-penalized scoring. (Category info should be external;
        # for our demo we implement diversity as "less-popular items".)
        if item_popularity is None:
            item_popularity = np.zeros(self.num_items, dtype=np.float64)

        selected: list[int] = []
        used = set()
        for iid in ranked:
            if len(selected) >= k:
                break
            if iid in used:
                continue
            # Prefer items that are not just top-popularity.
            pop_penalty = 0.5 * float(item_popularity[iid])
            if scores[iid] - pop_penalty == scores[iid] or scores[iid] != -np.inf:
                selected.append(int(iid))
                used.add(int(iid))
        return selected


class HybridRecommender:
    """
    Hybrid recommender that combines:
    - collaborative signal from MF/BPR
    - content signal from synthetic "text" token embeddings
    - novelty/diversity re-ranking
    """

    def __init__(
        self,
        mf: MFRecommender,
        item_vectors: np.ndarray,
        item_categories: np.ndarray,
        item_prices: np.ndarray,
        item_popularity: np.ndarray,
        user_content_vecs: np.ndarray,
    ):
        self.mf = mf
        self.item_vectors = item_vectors.astype(np.float64)  # already L2-normalized in catalog
        self.item_categories = item_categories.astype(np.int64)
        self.item_prices = item_prices.astype(np.float64)
        self.item_popularity = item_popularity.astype(np.float64)
        self.user_content_vecs = user_content_vecs.astype(np.float64)

        if self.item_vectors.ndim != 2:
            raise ValueError("item_vectors must be 2D: [num_items, embed_dim]")
        if self.user_content_vecs.ndim != 2:
            raise ValueError("user_content_vecs must be 2D: [num_users, embed_dim]")

    @staticmethod
    def _soft_clip_mask(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply a boolean mask. Mask=False entries get -inf.
        """
        out = scores.copy()
        out[~mask] = -np.inf
        return out

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_interacted=None,
        allowed_categories=None,
        price_max=None,
        weights=None,
        diversity: bool = False,
        category_diversity_strength: float = 0.35,
        similarity_diversity_strength: float = 0.5,
    ) -> list[int]:
        if weights is None:
            weights = {"collab": 0.7, "content": 0.3, "novelty": 0.15}

        w_collab = float(weights.get("collab", 0.7))
        w_content = float(weights.get("content", 0.3))
        w_novelty = float(weights.get("novelty", 0.15))

        num_items = self.item_vectors.shape[0]
        if user_id < 0 or user_id >= self.user_content_vecs.shape[0]:
            raise ValueError("Invalid user_id")

        # 1) Score components
        collab_scores = self.mf.predict_scores_normalized(user_id)  # [num_items]
        user_vec = self.user_content_vecs[user_id]
        content_scores = (user_vec @ self.item_vectors.T).astype(np.float64)
        content_scores = (content_scores - float(np.mean(content_scores))) / (float(np.std(content_scores)) + 1e-9)

        novelty_scores = 1.0 - self.item_popularity  # higher for less popular

        final_scores = w_collab * collab_scores + w_content * content_scores + w_novelty * novelty_scores

        # 2) Hard filters
        allowed_mask = np.ones(num_items, dtype=bool)
        if allowed_categories is not None and len(allowed_categories) > 0:
            allowed_mask &= np.isin(self.item_categories, np.array(list(allowed_categories), dtype=np.int64))

        if price_max is not None:
            allowed_mask &= self.item_prices <= float(price_max)

        if exclude_interacted:
            allowed_mask[list(exclude_interacted)] = False

        final_scores = self._soft_clip_mask(final_scores, allowed_mask)

        # Candidate list (top-N, then diversity re-rank).
        top_n = max(50, k * 10)
        ranked = np.argsort(-final_scores)[:top_n]
        ranked = [int(i) for i in ranked if np.isfinite(final_scores[i])]

        if not diversity:
            return ranked[:k]

        # 3) Diversity re-ranking (greedy)
        selected: list[int] = []
        selected_categories: set[int] = set()
        selected_vecs = []

        for _ in range(k):
            best_i = None
            best_score = -np.inf
            for iid in ranked:
                if iid in selected:
                    continue
                score = final_scores[iid]
                # Penalize picking the same category repeatedly.
                if self.item_categories[iid] in selected_categories:
                    score -= category_diversity_strength
                # Penalize high semantic similarity to already selected items.
                if selected_vecs:
                    sims = [float(self.item_vectors[iid] @ v) for v in selected_vecs]
                    score -= similarity_diversity_strength * max(sims)
                if score > best_score:
                    best_score = score
                    best_i = iid

            if best_i is None:
                break
            selected.append(int(best_i))
            selected_categories.add(int(self.item_categories[best_i]))
            selected_vecs.append(self.item_vectors[best_i])

        return selected[:k]

