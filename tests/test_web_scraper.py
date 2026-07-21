from knowledge_bot.web_scraper import ScrapedContent


def _content(text):
    return ScrapedContent(url="http://example.com", title="Test", text=text, scraped_at="now")


def test_valid_content():
    assert _content("a" * 100).is_valid is True


def test_short_content_is_invalid():
    assert _content("hi").is_valid is False


def test_empty_content_is_invalid():
    assert _content("").is_valid is False


def test_whitespace_content_is_invalid():
    assert _content("   ").is_valid is False
