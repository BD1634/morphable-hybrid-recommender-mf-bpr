import os
from dataclasses import dataclass

import numpy as np
from flask import Flask, render_template, request

from nlp_parser import derive_weights, detect_diversity, extract_category_keywords, parse_price_max
from recommender import HybridRecommender, MFRecommender
from synthetic_data import SyntheticCatalog, SyntheticInteractions


@dataclass
class ModelBundle:
    recommender: HybridRecommender
    catalog: SyntheticCatalog
    interactions: SyntheticInteractions


def build_bundle(seed: int = 42) -> ModelBundle:
    catalog = SyntheticCatalog(num_items=500, num_categories=12, seed=seed + 1)
    interactions = SyntheticInteractions(
        catalog=catalog,
        num_users=220,
        factors_k=32,
        positives_per_user=22,
        seed=seed + 2,
        w_collab=1.0,
        w_content=1.0,
    )

    model = MFRecommender(
        num_users=interactions.num_users,
        num_items=interactions.num_items,
        k=32,
        seed=seed + 2,
        lr=2e-2,
        reg=1e-3,
    )

    # Keep training time bounded for a demo.
    # Keep demo startup fast enough for local runs.
    epochs = int(os.environ.get("TRAIN_EPOCHS", "5"))
    samples_per_epoch = int(os.environ.get("TRAIN_SAMPLES_PER_EPOCH", "1500"))
    model.fit_bpr(
        positives=interactions.positives,
        num_epochs=epochs,
        samples_per_epoch=samples_per_epoch,
        seed=seed + 3,
    )

    hybrid = HybridRecommender(
        mf=model,
        item_vectors=catalog.item_vectors,
        item_categories=catalog.categories,
        item_prices=catalog.prices,
        item_popularity=interactions.item_popularity,
        user_content_vecs=interactions.user_pos_content_vecs,
    )

    return ModelBundle(recommender=hybrid, catalog=catalog, interactions=interactions)


bundle = build_bundle()

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        max_user_id=bundle.interactions.num_users - 1,
        max_k=30,
        default_user_id=0,
    )


@app.post("/recommend")
def recommend():
    user_id = int(request.form["user_id"])
    k = int(request.form.get("k", 10))
    mode = request.form.get("mode", "collab").strip().lower()
    request_text = request.form.get("request_text", "").strip()

    k = max(1, min(k, 30))
    if user_id < 0 or user_id >= bundle.interactions.num_users:
        return render_template("error.html", message="Invalid user_id")

    exclude = bundle.interactions.interacted_set(user_id) if mode != "allow_repeats" else set()

    allowed_categories = None
    if request_text:
        allowed_categories = extract_category_keywords(request_text, bundle.catalog.category_keywords)
        if not allowed_categories:
            allowed_categories = None

    price_max = parse_price_max(request_text) if request_text else None

    diverse_from_text = detect_diversity(request_text)
    diversity = (mode == "diverse") or diverse_from_text

    # Base weights from either the dropdown or the natural language request.
    weights = None
    if mode == "collab":
        weights = {"collab": 0.85, "content": 0.15, "novelty": 0.15}
    elif mode == "content":
        weights = {"collab": 0.15, "content": 0.85, "novelty": 0.15}
    elif mode == "diverse":
        weights = {"collab": 0.35, "content": 0.45, "novelty": 0.40}
    else:
        weights = derive_weights(request_text)
        if not request_text:
            weights = {"collab": 0.75, "content": 0.25, "novelty": 0.15}

    if request_text:
        # If user wrote intent words, let them "morph" the weights.
        weights = derive_weights(request_text)

    rec_item_ids = bundle.recommender.recommend(
        user_id=user_id,
        k=k,
        exclude_interacted=exclude,
        allowed_categories=allowed_categories,
        price_max=price_max,
        weights=weights,
        diversity=diversity,
    )

    return render_template(
        "recommendations.html",
        user_id=user_id,
        mode=mode,
        request_text=request_text,
        items=bundle.catalog.get_items(rec_item_ids),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Debug only for local demo.
    app.run(host="0.0.0.0", port=port, debug=True)

