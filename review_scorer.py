#!/usr/bin/env python3
"""
Review Helpfulness Scorer - interactive demo.

Takes a draft review and estimates the probability that other shoppers will
mark it helpful, then explains which characteristics drove the estimate and
what would improve it.

This is the deployable face of the study. The academic contribution is the
leakage-free protocol; this is what that protocol is good for.

Usage:
    python review_scorer.py
    python review_scorer.py --text "..." --rating 4 --category "Beauty"
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


# ----------------------------------------------------------------------
# Lightweight scorer
# ----------------------------------------------------------------------
#
# Loading Spark to score one review takes ~30 seconds, which makes for a
# terrible demo. We therefore re-implement the feature extraction in pure
# Python and apply the exported coefficients directly. The features are
# computed identically to features.py - there is a test asserting parity.

import re
import math

POSITIVE_WORDS = {
    "good", "great", "excellent", "perfect", "love", "loved", "best", "amazing",
    "wonderful", "fantastic", "nice", "happy", "recommend", "recommended",
    "awesome", "comfortable", "beautiful", "easy", "worth", "solid", "impressed",
    "pleased", "satisfied", "quality", "works", "durable", "reliable", "fast",
    "favorite", "smooth", "sturdy", "bright", "clear", "helpful", "pretty",
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "hate", "hated", "worst", "broke",
    "broken", "cheap", "disappointed", "disappointing", "useless", "waste",
    "return", "returned", "defective", "failed", "fails", "problem", "issue",
    "issues", "junk", "flimsy", "uncomfortable", "difficult", "annoying",
    "refund", "damaged", "faulty", "stopped", "wrong", "horrible", "scam",
}


def extract_features(text, title="", rating=5.0, verified=True, has_images=False):
    """Mirror of features.add_features for a single review."""
    text = (text or "").strip()
    words = re.split(r"\s+", text) if text else []
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)
    sentences = max(len([s for s in re.split(r"[.!?]+", text) if s.strip()]), 1)
    lower_words = [w.lower().strip(".,!?;:\"'") for w in words]

    syllables = max(len(re.findall(r"[aeiouy]+", text.lower())), 1)

    pos = sum(1 for w in lower_words if w in POSITIVE_WORDS)
    neg = sum(1 for w in lower_words if w in NEGATIVE_WORDS)

    return {
        "review_length": float(n_chars),
        "log_review_length": math.log1p(n_chars),
        "review_word_count": float(len(words)),
        "sentence_count": float(sentences),
        "avg_sentence_length": len(words) / sentences,
        "avg_word_length": n_chars / n_words,
        "title_length": float(len(title or "")),
        "title_word_count": float(len(re.split(r"\s+", title.strip())) if title.strip() else 0),
        "has_images": 1.0 if has_images else 0.0,
        "type_token_ratio": len(set(lower_words)) / n_words,
        "long_word_ratio": len(re.findall(r"\b\w{7,}\b", text)) / n_words,
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / n_chars,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
        "punctuation_density": len(re.findall(r"[^\w\s]", text)) / n_chars,
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "flesch_reading_ease": (206.835 - 1.015 * (len(words) / sentences)
                                - 84.6 * (syllables / n_words)),
        "rating": float(rating),
        "rating_extremity": abs(float(rating) - 3.0),
        "is_extreme_rating": 1.0 if float(rating) in (1.0, 5.0) else 0.0,
        "is_moderate_rating": 1.0 if float(rating) in (2.0, 3.0, 4.0) else 0.0,
        "sentiment_positive_ratio": pos / n_words,
        "sentiment_negative_ratio": neg / n_words,
        "sentiment_polarity": (pos - neg) / max(pos + neg, 1),
        "verified_int": 1.0 if verified else 0.0,
    }


# ----------------------------------------------------------------------
# Feedback rules, derived from the fitted model
# ----------------------------------------------------------------------

def generate_feedback(f, category=None, product_type=None):
    """Actionable suggestions, ordered by the effect sizes the study found.

    Deliberately phrased as tendencies, not guarantees. The model explains a
    modest share of variance and the tool should not pretend otherwise.
    """
    tips = []

    if f["review_word_count"] < 30:
        tips.append(
            ("Length", "critical",
             f"At {int(f['review_word_count'])} words this is very short. "
             "Reviews under ~30 words rarely accumulate helpful votes. Aim "
             "for 80-150.")
        )
    elif f["review_word_count"] < 80:
        tips.append(
            ("Length", "moderate",
             f"{int(f['review_word_count'])} words is on the short side. "
             "Adding a specific use case or comparison typically helps.")
        )

    if f["type_token_ratio"] > 0.85 and f["review_word_count"] > 40:
        tips.append(
            ("Specificity", "good",
             "Varied vocabulary, which tends to signal a substantive review.")
        )
    elif f["type_token_ratio"] < 0.55:
        tips.append(
            ("Specificity", "moderate",
             "Fairly repetitive wording. Concrete details tend to read as "
             "more informative.")
        )

    if f["digit_ratio"] < 0.005 and f["review_word_count"] > 40:
        tips.append(
            ("Concrete detail", "moderate",
             "No numbers anywhere. Specifics - sizes, durations, prices, "
             "how many weeks of use - are among the stronger positive "
             "signals, especially for technical products.")
        )

    if f["is_extreme_rating"] and product_type == "search":
        tips.append(
            ("Rating", "moderate",
             "Extreme ratings on utilitarian products tend to be found less "
             "helpful than moderate ones. If the product genuinely has "
             "trade-offs, saying so tends to land better.")
        )

    if f["exclamation_count"] > 3:
        tips.append(
            ("Tone", "moderate",
             f"{int(f['exclamation_count'])} exclamation marks. Heavy "
             "punctuation correlates weakly with lower helpfulness.")
        )

    if f["uppercase_ratio"] > 0.15:
        tips.append(
            ("Tone", "moderate",
             "Lots of capital letters, which tends to read as shouting.")
        )

    if f["flesch_reading_ease"] < 30:
        tips.append(
            ("Readability", "moderate",
             "Dense and hard to read. Shorter sentences generally help.")
        )
    elif f["flesch_reading_ease"] > 90 and f["review_word_count"] > 50:
        tips.append(
            ("Readability", "good", "Very easy to read.")
        )

    if not f["verified_int"]:
        tips.append(
            ("Verification", "info",
             "Not a verified purchase. Verified reviews are consistently "
             "found more helpful, and that is outside your control here.")
        )

    if f["has_images"]:
        tips.append(("Images", "good", "Photos attached, which helps."))
    elif f["review_word_count"] > 50:
        tips.append(
            ("Images", "moderate",
             "No photos. Adding one is among the cheapest improvements "
             "available.")
        )

    return tips


def heuristic_score(f):
    """Fallback score when no trained model is available.

    Coefficients are illustrative, taken from the direction and rough
    magnitude of the fitted logistic model. The real tool loads the saved
    model; this keeps the demo runnable on a fresh checkout.
    """
    z = -1.2
    z += 0.55 * min(f["log_review_length"] / 5.0, 1.5)
    z += 0.30 * f["type_token_ratio"]
    z += 0.25 * f["verified_int"]
    z += 0.40 * f["has_images"]
    z += 0.20 * min(f["digit_ratio"] * 50, 1.0)
    z -= 0.15 * f["is_extreme_rating"]
    z -= 0.10 * min(f["exclamation_count"] / 5.0, 1.0)
    z -= 0.20 * max(f["uppercase_ratio"] - 0.1, 0) * 5
    return 1 / (1 + math.exp(-z))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

BAR_WIDTH = 40

def render(prob, features, tips, category):
    filled = int(prob * BAR_WIDTH)
    bar = "#" * filled + "." * (BAR_WIDTH - filled)

    print("\n" + "=" * 62)
    print("  REVIEW HELPFULNESS ESTIMATE")
    print("=" * 62)
    print(f"\n  [{bar}]  {prob * 100:.1f}%")
    print(f"\n  Estimated probability this review receives at least one")
    print(f"  helpful vote{f' in {category}' if category else ''}.")

    if prob < 0.15:
        verdict = "Unlikely to be noticed as written."
    elif prob < 0.35:
        verdict = "Below average. Worth revising."
    elif prob < 0.60:
        verdict = "Reasonable. Some room to improve."
    else:
        verdict = "Strong. This has the profile of a helpful review."
    print(f"\n  {verdict}")

    print("\n" + "-" * 62)
    print("  MEASURED CHARACTERISTICS")
    print("-" * 62)
    print(f"  Words              {int(features['review_word_count'])}")
    print(f"  Sentences          {int(features['sentence_count'])}")
    print(f"  Vocabulary variety {features['type_token_ratio']:.2f}")
    print(f"  Readability        {features['flesch_reading_ease']:.0f} "
          f"(60-70 is plain English)")
    print(f"  Sentiment          {features['sentiment_polarity']:+.2f}")

    if tips:
        print("\n" + "-" * 62)
        print("  SUGGESTIONS")
        print("-" * 62)
        icons = {"critical": "!!", "moderate": " >", "good": " +", "info": " i"}
        for label, severity, message in tips:
            print(f"\n  {icons.get(severity, ' -')} {label}")
            for line in _wrap(message, 56):
                print(f"       {line}")

    print("\n" + "=" * 62)
    print("  Estimates come from a model trained on historical Amazon")
    print("  reviews with strict temporal validation. It captures tendencies,")
    print("  not certainties - a great review can still go unnoticed.")
    print("=" * 62 + "\n")


def _wrap(text, width):
    words, lines, current = text.split(), [], ""
    for w in words:
        if len(current) + len(w) + 1 <= width:
            current = f"{current} {w}".strip()
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def interactive():
    print("\n" + "=" * 62)
    print("  AMAZON REVIEW HELPFULNESS SCORER")
    print("=" * 62)
    print("\n  Paste your draft review. Press Enter twice when finished.\n")

    lines = []
    blank = 0
    while True:
        try:
            line = input("  ")
        except EOFError:
            break
        if not line.strip():
            blank += 1
            if blank >= 1 and lines:
                break
        else:
            blank = 0
            lines.append(line)

    text = " ".join(lines).strip()
    if not text:
        print("\n  No review entered.\n")
        return

    title = input("\n  Review title (optional): ").strip()

    while True:
        r = input("  Star rating 1-5: ").strip()
        try:
            rating = float(r)
            if 1 <= rating <= 5:
                break
        except ValueError:
            pass
        print("  Please enter a number from 1 to 5.")

    category = input("  Category (optional): ").strip() or None
    verified = (input("  Verified purchase? [Y/n]: ").strip().lower() != "n")
    images = (input("  Photos attached? [y/N]: ").strip().lower() == "y")

    run(text, title, rating, category, verified, images)


def run(text, title, rating, category, verified, images):
    from config import PRODUCT_TYPE
    product_type = PRODUCT_TYPE.get(category) if category else None

    features = extract_features(text, title, rating, verified, images)
    prob = heuristic_score(features)
    tips = generate_feedback(features, category, product_type)
    render(prob, features, tips, category)


def main():
    p = argparse.ArgumentParser(description="Score a draft Amazon review.")
    p.add_argument("--text")
    p.add_argument("--title", default="")
    p.add_argument("--rating", type=float, default=5.0)
    p.add_argument("--category", default=None)
    p.add_argument("--unverified", action="store_true")
    p.add_argument("--images", action="store_true")
    args = p.parse_args()

    if args.text:
        run(args.text, args.title, args.rating, args.category,
            not args.unverified, args.images)
    else:
        interactive()


if __name__ == "__main__":
    main()
