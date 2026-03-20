# Morphable Hybrid Recommender (MF + BPR) Demo

Build a small, resume-friendly recommender that feels interactive: users can type a natural-language request, and the system “morphs” recommendations by shifting weights across:
1) collaborative relevance (MF/BPR),
2) content relevance (synthetic text embeddings),
3) novelty/diversity (popularity-aware re-ranking).

This repo ships with a lightweight Flask UI so you can demo the product behavior immediately.

## What you can demo
- Personalize recommendations per user id using an MF/BPR model trained on synthetic implicit feedback
- Take a free-form request like `modern eco under 50 and diverse` and translate it into ranking constraints and weight adjustments
- Apply hard filters like category intent (via keyword matches) and price caps (via simple numeric parsing)
- Offer a “diverse” mode that re-ranks results to reduce repetitiveness across categories

## How “morphing” works (high level)
- The `/recommend` handler reads `request_text` and extracts simple signals (category keywords, `under $X`-style limits, and diversity intent)
- A heuristic weight mapper converts the request into mixture weights for collaborative vs. content vs. novelty
- The hybrid ranker combines three score components:
- `collab`: MF/BPR predicted relevance
- `content`: similarity between user “preference keywords” and item token-based vectors
- `novelty`: higher scores for less-popular items
- Optional re-ranking encourages category and embedding diversity when “diverse” is requested

## Endpoints / UI
- `GET /` shows a simple form (user id, top-K, mode, and an optional natural-language request)
- `POST /recommend` returns a table of recommended items (Item ID, category, price, title)

## Quickstart
```bash
cd recsys_flask
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Faster startup (optional)
Training runs on startup, but it is bounded for demo speed.

Environment variables:
- `TRAIN_EPOCHS` (default: `5`)
- `TRAIN_SAMPLES_PER_EPOCH` (default: `1500`)

Example:
```bash
TRAIN_EPOCHS=2 TRAIN_SAMPLES_PER_EPOCH=500 python app.py
```

## Example requests to try
- `modern eco under 50`
- `quiet minimal premium and diverse`
- `sport smart under 100, surprise me`

## Notes
- The demo uses synthetic users/items and synthetic interactions, so it runs quickly without large datasets.
- The goal is to showcase a product-style interactive recommender pipeline (request understanding + hybrid ranking + constraints/re-ranking) rather than production-grade data infrastructure.

