"""Check the built public documentation before it is released."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


SITE_ROOT = Path("site").resolve()
SITE_PREFIX = "/aiython-docs/"
PRIVATE_REPOSITORY_PATH = "/sunmodza/aiython"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        attribute = "src" if tag in {"img", "script"} else "href"
        if tag in {"a", "img", "link", "script"} and values.get(attribute):
            self.links.append(values[attribute])

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def local_target(page: Path, link: str) -> Path | None:
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    if parsed.path.startswith(SITE_PREFIX):
        target = SITE_ROOT / unquote(parsed.path.removeprefix(SITE_PREFIX))
    elif parsed.path.startswith("/"):
        return SITE_ROOT / unquote(parsed.path.lstrip("/"))
    else:
        target = page.parent / unquote(parsed.path)
    target = target.resolve()
    if not target.is_relative_to(SITE_ROOT):
        raise ValueError(f"Link escapes the site: {link}")
    return target / "index.html" if target.is_dir() else target


def main() -> None:
    required = [SITE_ROOT / "index.html", SITE_ROOT / "examples" / "index.html"]
    errors = [f"Missing page: {page}" for page in required if not page.is_file()]
    examples_text = ""
    pages = sorted(SITE_ROOT.rglob("*.html"))

    for page in pages:
        html = page.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(html)
        if page == SITE_ROOT / "examples" / "index.html":
            examples_text = "".join(parser.text)
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.netloc == "github.com" and (
                parsed.path == PRIVATE_REPOSITORY_PATH
                or parsed.path.startswith(f"{PRIVATE_REPOSITORY_PATH}/")
            ):
                errors.append(f"Private repository URL in {page}: {link}")
            try:
                target = local_target(page, link)
            except ValueError as exc:
                errors.append(f"{page}: {exc}")
                continue
            if target is not None and not target.is_file():
                errors.append(f"Broken link in {page}: {link}")

    if "--8<--" in examples_text:
        errors.append("An unexpanded source snippet is visible on the examples page")

    snippets = sorted(Path("examples/recipes").glob("*.py"))
    snippets.extend(
        Path("examples/capabilities") / name
        for name in (
            "documents.py",
            "media.py",
            "policy.md",
            "products.py",
            "video_generation.py",
        )
    )
    for source in snippets:
        if source.read_text(encoding="utf-8").strip() not in examples_text:
            errors.append(f"Source snippet was not rendered: {source}")

    for name in (
        "blue-boot.jpg",
        "clip.mp4",
        "meeting.mp3",
        "red-shoe.jpg",
        "shoe.jpg",
    ):
        source = Path("examples/capabilities") / name
        published = SITE_ROOT / "assets" / "examples" / name
        if not published.is_file() or source.read_bytes() != published.read_bytes():
            errors.append(f"Published example asset differs from {source}")

    transcript = Path("examples/capabilities/meeting-transcript.md")
    public_transcript = Path("docs/assets/examples/meeting-transcript.md")
    if transcript.read_bytes() != public_transcript.read_bytes():
        errors.append("Published meeting transcript differs from the example source")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(pages)} public documentation pages and their local links.")


if __name__ == "__main__":
    main()
