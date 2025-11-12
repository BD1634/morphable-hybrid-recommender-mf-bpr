# Recsys Flask Demo (Morphable Hybrid MF+BPR)

This project is a **portfolio-friendly** recommender systems demo with a small amount of "morphing":

- Synthetic item catalog with interpretable category keywords + token-based text embeddings
- Synthetic implicit interactions generated from a mixture of collaborative (MF/BPR) + content similarity
- Hybrid recommender: collaborative + content + novelty/diversity re-ranking
- Simple Flask UI: pick a user id, enter a natural language request, get top-K recommendations

## Setup

Create a virtual environment (optional) and install requirements:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open: `http://127.0.0.1:5000`

## Demo controls

Training is bounded so it runs quickly.

Environment variables:

- `TRAIN_EPOCHS` (default: `8`)
- `TRAIN_SAMPLES_PER_EPOCH` (default: `2500`)

For faster startup:

```bash
TRAIN_EPOCHS=3 python app.py
```

