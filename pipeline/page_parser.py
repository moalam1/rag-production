"""
pipeline/page_parser.py — Fetch and parse Equinix AEM resource pages.

Content strategy (pages are JS-rendered, SSR has ~150-200 words only):
  - Whitepaper / report / data-sheet  → page teaser + PDF via LlamaParse
  - Video / webinar                   → page teaser + YouTube transcript
  - Case study / article              → page teaser + PDF if present

Page teaser is always indexed as a lightweight summary chunk.
PDF and transcript are the primary deep-content sources.
"""
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
}
FETCH_TIMEOUT   = 30
MAX_CONTENT_LEN = 5_000_000

# Resource types that have PDFs as primary content
PDF_TYPES = {"whitepaper", "analyst-report", "data-sheet", "blueprint", "playbook", "solution-brief"}

# Resource types that have YouTube transcripts as primary content
VIDEO_TYPES = {"multimedia", "webinar"}


@dataclass
class ParsedPage:
    """Structured output from page_parser."""
    url:             str
    title:           str
    teaser:          str        # short SSR text — always present
    resource_type:   str
    template:        str
    tags:            list[str] = field(default_factory=list)
    published_date:  str = ""
    og_updated_time: str = ""
    pdf_url:         str = ""   # present for PDF_TYPES
    youtube_id:      str = ""   # present for VIDEO_TYPES
    transcript:      str = ""   # populated if youtube_id found
    content_hash:    str = ""
    document_family: str = ""
    page_url:        str = ""
    word_count:      int = 0    # teaser word count
    has_pdf:         bool = False
    has_transcript:  bool = False


def parse_page(url: str, fetch_transcript: bool = True) -> Optional[ParsedPage]:
    """
    Fetch and parse an Equinix resource page.

    Args:
        url:              Full resource URL
        fetch_transcript: If True, fetch YouTube transcript for video pages

    Returns:
        ParsedPage on success, None on failure.
    """
    url = url.strip().rstrip("/")
    log.info("Parsing page: %s", url)

    # ── Fetch HTML ────────────────────────────────────────────────────────────
    raw_html, content_hash = _fetch(url)
    if not raw_html:
        return None

    soup = BeautifulSoup(raw_html, "html.parser")

    # ── Extract metadata ──────────────────────────────────────────────────────
    template       = _meta(soup, "meta-template")
    title          = (_meta(soup, "og:title") or _meta(soup, "title")
                      or _tag_text(soup, "h1") or url.split("/")[-1])
    og_updated     = _meta(soup, "og:updated_time")
    published_date = _parse_date(og_updated)
    tags           = _extract_tags(soup)
    pdf_url        = _extract_pdf_url(soup)
    youtube_id     = _extract_youtube_id(soup)
    resource_type  = _derive_resource_type(url, template)
    family         = _derive_family(url)

    # ── Extract teaser text (SSR content only) ────────────────────────────────
    teaser = _extract_teaser(soup)
    if not teaser:
        log.warning("No extractable teaser content: %s", url)
        return None

    # ── Fetch YouTube transcript if video ─────────────────────────────────────
    transcript = ""
    if youtube_id and fetch_transcript:
        transcript = _fetch_transcript(youtube_id)
        if transcript:
            log.info("Fetched transcript: %d words for video %s", len(transcript.split()), youtube_id)

    result = ParsedPage(
        url             = url,
        title           = title,
        teaser          = teaser,
        resource_type   = resource_type,
        template        = template,
        tags            = tags,
        published_date  = published_date,
        og_updated_time = og_updated,
        pdf_url         = pdf_url,
        youtube_id      = youtube_id,
        transcript      = transcript,
        content_hash    = content_hash,
        document_family = family,
        page_url        = url,
        word_count      = len(teaser.split()),
        has_pdf         = bool(pdf_url),
        has_transcript  = bool(transcript),
    )

    log.info(
        "✅ Parsed '%s' | type=%s | teaser=%d words | pdf=%s | transcript=%s",
        title[:50], resource_type, result.word_count,
        "yes" if pdf_url else "no",
        f"{len(transcript.split())} words" if transcript else "no",
    )
    return result


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch(url: str) -> tuple[str, str]:
    try:
        with httpx.Client(headers=HEADERS, timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_bytes = resp.content
            if len(content_bytes) > MAX_CONTENT_LEN:
                log.warning("Page too large, skipping: %s", url)
                return "", ""
            return resp.text, hashlib.sha256(content_bytes).hexdigest()[:32]
    except httpx.HTTPStatusError as e:
        log.warning("HTTP %d for %s", e.response.status_code, url)
        return "", ""
    except httpx.RequestError as e:
        log.warning("Request error for %s: %s", url, e)
        return "", ""


# ── Teaser extraction ─────────────────────────────────────────────────────────

def _extract_teaser(soup: BeautifulSoup) -> str:
    """
    Extract all meaningful SSR text from the page.
    Combines paragraphs and list items — typically 100-250 words on Equinix pages.
    """
    for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
        tag.decompose()

    parts = []

    # Paragraphs
    for p in soup.find_all("p"):
        t = p.get_text(separator=" ", strip=True)
        if t and len(t) > 20:
            parts.append(t)

    # List items
    for li in soup.find_all("li"):
        t = li.get_text(separator=" ", strip=True)
        if t and len(t) > 10:
            parts.append(f"- {t}")

    # H1/H2 headings for context
    for h in soup.find_all(["h1", "h2"]):
        t = h.get_text(strip=True)
        if t and len(t) > 5:
            parts.insert(0, f"# {t}")

    text = "\n".join(parts)
    return _clean_text(text)


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l for l in text.splitlines() if l.strip() and len(l.strip()) >= 3]
    return "\n".join(lines).strip()


# ── YouTube transcript ────────────────────────────────────────────────────────

def _fetch_transcript(youtube_id: str) -> str:
    """
    Fetch YouTube transcript via youtube-transcript-api.
    Free, no API key needed. Returns empty string on failure.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # Note: AWS IPs are blocked by YouTube — transcript fetch will fail
        # on EC2. Transcripts must be fetched from a non-cloud IP and passed
        # in directly. For now we return empty and rely on page teaser.
        ytt     = YouTubeTranscriptApi()
        fetched = ytt.fetch(youtube_id)
        text    = " ".join(s.text.strip() for s in fetched if s.text)
        text    = re.sub(r"\[.*?\]", "", text)
        text    = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        log.debug("Transcript unavailable (likely AWS IP block) for %s: %s", youtube_id, e)
        return ""


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _meta(soup: BeautifulSoup, name: str) -> str:
    tag = (
        soup.find("meta", attrs={"name": name})
        or soup.find("meta", attrs={"property": name})
    )
    return (tag.get("content", "") if tag else "").strip()


def _tag_text(soup: BeautifulSoup, tag: str) -> str:
    el = soup.find(tag)
    return el.get_text(strip=True) if el else ""


def _extract_tags(soup: BeautifulSoup) -> list[str]:
    tags = set()
    for selector in [".tags a", ".resource-tags a", ".tag-list a"]:
        for el in soup.select(selector):
            t = el.get_text(strip=True)
            if t:
                tags.add(t)
    if not tags:
        keywords = _meta(soup, "keywords")
        if keywords:
            tags.update(k.strip() for k in keywords.split(",") if k.strip())
    return sorted(tags)


def _extract_pdf_url(soup: BeautifulSoup) -> str:
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/content/dam/" in href and href.lower().endswith(".pdf"):
            if href.startswith("http"):
                return href
            return f"https://www.equinix.com{href}"
    return ""


def _extract_youtube_id(soup: BeautifulSoup) -> str:
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"]
        if "youtube.com/embed/" in src:
            match = re.search(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})", src)
            if match:
                return match.group(1)
    return ""


def _derive_resource_type(url: str, template: str) -> str:
    path = urlparse(url).path.lower()
    mapping = {
        "whitepapers":       "whitepaper",
        "analyst-reports":   "analyst-report",
        "data-sheets":       "data-sheet",
        "case-studies":      "case-study",
        "solution-briefs":   "solution-brief",
        "blueprints":        "blueprint",
        "playbooks":         "playbook",
        "articles":          "article",
        "videos":            "multimedia",
        "webinars":          "webinar",
        "media":             "media",
        "infopapers":        "infopaper",
        "product-documents": "product-document",
        "infographics":      "infographic",
        "success-stories":   "success-story",
    }
    for segment, rtype in mapping.items():
        if f"/{segment}/" in path or path.endswith(f"/{segment}"):
            return rtype
    if "multimedia" in template.lower():
        return "multimedia"
    return "article"


def _derive_family(url: str) -> str:
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9_]", "_", slug.lower().replace("-", "_"))[:50]


def _parse_date(og_updated_time: str) -> str:
    if not og_updated_time:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(og_updated_time.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""
