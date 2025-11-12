import re


def parse_price_max(text: str):
    """
    Extract a numeric upper bound from phrases like:
      - "under $50"
      - "less than 30"
      - "$40 or less"
    Returns float or None.
    """
    if not text:
        return None
    t = text.lower()

    # "$<num>" / "under $<num>"
    m = re.search(r"(?:under|below|less than|<=|\$)\s*\$?\s*(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))

    # "<num> or less"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:or\s*less|max)", t)
    if m:
        return float(m.group(1))

    return None


def extract_category_keywords(text: str, category_keywords: list[str]):
    """
    Extract which category keywords appear in the user's request.
    """
    if not text:
        return set()
    t = text.lower()
    found = set()
    for kw in category_keywords:
        if kw.lower() in t:
            found.add(int(category_keywords.index(kw)))
    return found


def detect_diversity(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    triggers = ["diverse", "variety", "different", "surprise", "novel", "mix", "wide"]
    return any(x in t for x in triggers)


def derive_weights(text: str):
    """
    Heuristic weighting:
    - If the user asks for "similar" / "more like my history" => collab weight up
    - If the user mentions intent keywords (we treat them as content cues) => content weight up
    - If the user asks for "diverse"/"surprise" => novelty weight up
    """
    if not text:
        return {"collab": 0.75, "content": 0.25, "novelty": 0.15}

    t = text.lower()
    diverse = detect_diversity(t)

    collab = 0.7
    content = 0.3
    novelty = 0.15

    if any(x in t for x in ["similar", "more like", "history", "what i liked", "again"]):
        collab = 0.85
        content = 0.15
    if any(x in t for x in ["cozy", "eco", "modern", "classic", "minimal", "premium", "sport", "outdoor", "quiet", "smart", "compact", "studio", "bold"]):
        content = 0.65
        collab = 0.35

    if diverse:
        novelty = 0.45
        # Keep content slightly higher for "tasteful diversity" in this demo.
        content = max(content, 0.45)
        collab = max(0.15, 1.0 - content)

    return {"collab": collab, "content": content, "novelty": novelty}

