"""
scraper.py — Input layer for the Autonomous A/B Test Decision Engine.

Accepts a product page URL and extracts raw CRO signals from the HTML.
Also supports loading pre-collected data from a mock JSON file for demo/testing.
"""

import json
import logging
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _default_page_data() -> dict:
    """Return a zeroed-out page dict that matches the required output schema.

    Used as a safe fallback when scraping fails, so downstream consumers always
    receive a dict with the expected keys and sensible types.
    """
    return {
        "page": {
            "headline": "",
            "cta_text": "",
            "description_length": 0,
            "price_display": "",
            "image_count": 0,
            "has_reviews": False,
            "review_count": 0,
            "trust_badges": [],
            "above_fold_elements": [],
        }
    }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_headline(soup: BeautifulSoup) -> str:
    """The page headline (h1) is the clearest statement of value proposition.

    A missing or weak headline is one of the highest-impact CRO opportunities.
    """
    tag = soup.find("h1")
    return tag.get_text(strip=True) if tag else ""


_CTA_KEYWORDS = re.compile(r"add|buy|cart|checkout|order|get|shop|purchase", re.I)


def _extract_cta_text(soup: BeautifulSoup) -> str:
    """Find the primary call-to-action button text.

    We look for <button> and <input type=submit/button> elements whose visible
    text or class names contain conversion-intent keywords. The first match is
    treated as the primary CTA — the most conversion-critical copy on the page.
    """
    candidates = soup.find_all(["button", "input", "a"])
    for tag in candidates:
        text = tag.get_text(strip=True) if tag.name != "input" else tag.get("value", "")
        classes = " ".join(tag.get("class", []))
        if _CTA_KEYWORDS.search(text) or _CTA_KEYWORDS.search(classes):
            return text
    return ""


def _extract_description(soup: BeautifulSoup) -> int:
    """Return the character count of the longest descriptive text block.

    Description length is a proxy for information density. Very short
    descriptions often correlate with higher bounce rates on product pages.
    """
    best = ""
    for tag in soup.find_all(["p", "div", "section"]):
        # Avoid nav/header/footer noise and deeply nested container divs
        if tag.find(["nav", "header", "footer"]):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > len(best):
            best = text
    return len(best)


_PRICE_CLASSES = re.compile(
    r"price|cost|amount|sale|discount|offer", re.I
)
_CURRENCY_RE = re.compile(r"[\$\£\€\¥][\d,]+(?:\.\d{1,2})?|\d+\.\d{2}\s*(?:USD|GBP|EUR)")


def _extract_price(soup: BeautifulSoup) -> str:
    """Extract the displayed price string.

    We first scan elements whose class/id suggests pricing, then fall back to
    a regex currency pattern on all text. Price clarity (visibility, prominence)
    is a major lever for reducing cart abandonment.
    """
    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        if _PRICE_CLASSES.search(classes) or _PRICE_CLASSES.search(tag_id):
            text = tag.get_text(strip=True)
            if text:
                return text

    # Fallback: scan all text for currency patterns
    full_text = soup.get_text()
    match = _CURRENCY_RE.search(full_text)
    return match.group(0) if match else ""


def _extract_image_count(soup: BeautifulSoup) -> int:
    """Count total <img> tags.

    Product image count is correlated with conversion — pages with more/richer
    imagery typically convert better, especially for physical goods.
    """
    return len(soup.find_all("img"))


_REVIEW_WORDS = re.compile(r"review|rating|star|verified|customer", re.I)
_REVIEW_COUNT_RE = re.compile(r"(\d[\d,]*)\s*(?:review|rating)", re.I)


def _extract_review_info(soup: BeautifulSoup) -> tuple[bool, int]:
    """Detect social proof elements and estimate review count.

    Social proof (reviews, ratings) is one of the strongest conversion drivers.
    Missing reviews on a product page is almost always a testable hypothesis.
    """
    review_tags = soup.find_all(
        lambda tag: _REVIEW_WORDS.search(" ".join(tag.get("class", [])))
        or _REVIEW_WORDS.search(tag.get("id", ""))
        or _REVIEW_WORDS.search(tag.get_text(strip=True))
    )
    has_reviews = len(review_tags) > 0
    count = 0
    if has_reviews:
        page_text = soup.get_text()
        matches = _REVIEW_COUNT_RE.findall(page_text)
        if matches:
            count = max(int(m.replace(",", "")) for m in matches)
    return has_reviews, count


_TRUST_PATTERNS = [
    "free shipping",
    "free returns",
    "money back",
    "guarantee",
    "secure",
    "ssl",
    "safe checkout",
    "no risk",
    "easy returns",
    "30-day",
    "60-day",
    "lifetime warranty",
    "trusted",
    "certified",
]


def _extract_trust_badges(soup: BeautifulSoup) -> list[str]:
    """Collect trust signals present on the page.

    Trust badges reduce purchase anxiety. Knowing which signals are already
    present (or absent) helps prioritise CRO tests around checkout confidence.
    """
    page_text = soup.get_text(separator=" ").lower()
    found = [badge for badge in _TRUST_PATTERNS if badge in page_text]
    return found


def _extract_above_fold(soup: BeautifulSoup) -> list[str]:
    """Identify key element types in the first ~20% of the DOM.

    Above-the-fold content determines first impressions and directly impacts
    bounce rate. We record which element types (h1, img, button, etc.) appear
    early in the parse tree as a rough proxy for visual priority.
    """
    all_tags = soup.find_all(True)
    cutoff = max(1, len(all_tags) // 5)
    early_tags = all_tags[:cutoff]

    important = {"h1", "h2", "img", "button", "a", "video", "input", "form"}
    seen: list[str] = []
    for tag in early_tags:
        if tag.name in important and tag.name not in seen:
            seen.append(tag.name)
    return seen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_url(url: str) -> dict:
    """Scrape a product page URL and return a structured CRO signal dict.

    Parameters
    ----------
    url:
        Fully-qualified URL of the product page to analyse.

    Returns
    -------
    dict
        Matches the schema::

            {
              "page": {
                "headline": str,
                "cta_text": str,
                "description_length": int,
                "price_display": str,
                "image_count": int,
                "has_reviews": bool,
                "review_count": int,
                "trust_badges": list[str],
                "above_fold_elements": list[str],
              }
            }
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        logger.info("Fetching %s", url)
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s — %s. Returning defaults.", url, exc)
        return _default_page_data()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style noise before extracting text signals
    for noise in soup(["script", "style", "noscript", "meta", "link"]):
        noise.decompose()

    has_reviews, review_count = _extract_review_info(soup)

    data = _default_page_data()
    data["page"].update(
        {
            "headline": _extract_headline(soup),
            "cta_text": _extract_cta_text(soup),
            "description_length": _extract_description(soup),
            "price_display": _extract_price(soup),
            "image_count": _extract_image_count(soup),
            "has_reviews": has_reviews,
            "review_count": review_count,
            "trust_badges": _extract_trust_badges(soup),
            "above_fold_elements": _extract_above_fold(soup),
        }
    )
    return data


def load_mock(filepath: str) -> dict:
    """Load pre-collected page data from a JSON file.

    The JSON file must already conform to the output schema (i.e. it was
    either saved by ``scrape_url`` or manually authored to match it). This
    allows the rest of the pipeline to run without hitting a live server —
    useful for demos, CI, and reproducible testing.

    Parameters
    ----------
    filepath:
        Path to a ``.json`` file on disk.

    Returns
    -------
    dict
        Same schema as ``scrape_url``.
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning("Mock file not found: %s. Returning defaults.", filepath)
        return _default_page_data()

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Validate top-level key
        if "page" not in data:
            logger.warning("Mock file missing 'page' key — wrapping as-is.")
            data = {"page": data}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load mock file %s — %s. Returning defaults.", filepath, exc)
        return _default_page_data()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke-test against a publicly accessible product-style page.
    # Run with:  python scraper.py
    # Optionally pass a URL as the first CLI argument:
    #   python scraper.py https://example.com/product
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"

    print(f"\n--- scrape_url({test_url!r}) ---")
    result = scrape_url(test_url)
    print(json.dumps(result, indent=2))

    # Also exercise load_mock with a non-existent path to confirm graceful fallback
    print("\n--- load_mock('mock_data/store1.json') ---")
    mock_result = load_mock("mock_data/store1.json")
    print(json.dumps(mock_result, indent=2))
