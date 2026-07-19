"""Web content extraction using trafilatura."""

from dataclasses import dataclass
from datetime import datetime, timezone

import trafilatura


@dataclass
class ScrapedContent:
    url: str
    title: str
    text: str
    scraped_at: str

    @property
    def is_valid(self) -> bool:
        return len(self.text.strip()) > 50


def scrape_url(url: str) -> ScrapedContent:
    """Download and extract main content from a URL."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to download: {url}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )

    metadata = trafilatura.extract(
        downloaded,
        output_format="json",
        include_comments=False,
    )

    title = url  # fallback
    if metadata:
        import json

        meta = json.loads(metadata)
        title = meta.get("title", url) or url

    return ScrapedContent(
        url=url,
        title=title,
        text=text or "",
        scraped_at=datetime.now(timezone.utc).isoformat(),
    )
