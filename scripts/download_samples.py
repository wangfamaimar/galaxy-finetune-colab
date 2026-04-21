"""Download a small set of public-domain galaxy thumbnails from Wikipedia.

Uses the MediaWiki ``pageimages`` API to fetch the lead image of each galaxy's
Wikipedia article, then downloads the thumbnail. Images are saved under
``sample_data/{E,S,SB}/`` and recorded in ``sample_data/SOURCES.md``.

Most galaxy lead images are NASA/ESA/ESO imagery — individually public domain or
CC-BY. Attribution is preserved in SOURCES.md.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sample_data"
THUMB_WIDTH = 512
UA = "galaxy-finetune-colab/0.1 (https://github.com/wangfamaimar/galaxy-finetune-colab)"

# (class, slug, Wikipedia article title, source note)
IMAGES: list[tuple[str, str, str, str]] = [
    # Ellipticals (E)
    ("E",  "m87",     "Messier 87",  "NASA / ESA / HST (public domain)"),
    ("E",  "m49",     "Messier 49",  "NASA / ESA / HST (public domain)"),
    ("E",  "m60",     "Messier 60",  "NASA / ESA / HST (public domain)"),
    ("E",  "m105",    "Messier 105", "NASA / ESA / HST (public domain)"),
    ("E",  "ngc1316", "NGC 1316",    "NASA / ESA / HST (public domain)"),

    # Spirals (S)
    ("S",  "m101",    "Messier 101", "NASA / ESA / HST (public domain)"),
    ("S",  "m51",     "Whirlpool Galaxy", "NASA / ESA / HST (public domain)"),
    ("S",  "m81",     "Messier 81",  "NASA / ESA / HST (public domain)"),
    ("S",  "m33",     "Triangulum Galaxy", "ESO / VST (CC-BY 4.0)"),
    ("S",  "ngc3370", "NGC 3370",    "NASA / ESA / HST (public domain)"),

    # Barred spirals (SB)
    ("SB", "ngc1300", "NGC 1300",    "NASA / ESA / HST (public domain)"),
    ("SB", "m109",    "Messier 109", "NASA / ESA / HST (public domain)"),
    ("SB", "ngc1365", "NGC 1365",    "ESO (CC-BY 4.0)"),
    ("SB", "ngc2903", "NGC 2903",    "NASA / ESA / HST (public domain)"),
    ("SB", "ngc6217", "NGC 6217",    "NASA / ESA / HST (public domain)"),
]


def _http_get(url: str, binary: bool = False, retries: int = 5) -> bytes | str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return data if binary else data.decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                print(f"  retry after {delay:.0f}s (HTTP {e.code})")
                time.sleep(delay)
                delay *= 2
                continue
            raise


def fetch_lead_image(title: str) -> tuple[str, str]:
    """Return (image_filename, thumb_url) for the lead image of the given Wikipedia article."""
    api = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&titles={urllib.parse.quote(title)}"
        "&prop=pageimages&pithumbsize=" + str(THUMB_WIDTH) + "&redirects=1"
    )
    payload = json.loads(_http_get(api))
    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    if "thumbnail" not in page:
        raise RuntimeError(f"No lead image for article: {title}")
    return page.get("pageimage", "unknown"), page["thumbnail"]["source"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows: list[str] = [
        "# Sample galaxy images",
        "",
        "Small sample set bundled with the repo so you can smoke-test fine-tuning",
        "without uploading anything to Drive. Thumbnails (~512 px) sourced from each",
        "galaxy's Wikipedia lead image via the MediaWiki `pageimages` API.",
        "",
        "| Class | File | Galaxy | Wikipedia | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    total_bytes = 0
    for cls, slug, title, src in IMAGES:
        dst = OUT / cls / f"{slug}.jpg"
        if dst.exists() and dst.stat().st_size > 1024:
            size = dst.stat().st_size
            print(f"-> {cls}/{slug}  (cached, {size // 1024} KB)")
            rows.append(
                f"| {cls} | `{cls}/{slug}.jpg` | {title} | "
                f"[article](https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}) | {src} |"
            )
            total_bytes += size
            continue
        print(f"-> {cls}/{slug}  ({title})")
        fname, url = fetch_lead_image(title)
        data = _http_get(url, binary=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        total_bytes += len(data)
        rows.append(
            f"| {cls} | `{cls}/{slug}.jpg` | {title} | "
            f"[article](https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}) | "
            f"[{fname}](https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(fname)}) — {src} |"
        )
        time.sleep(1.0)

    rows += [
        "",
        f"_Total: {len(IMAGES)} images, ~{total_bytes // 1024} KB._",
        "",
        "All included imagery is NASA/ESA (public domain) or ESO (CC-BY 4.0).",
        "Attribution is preserved above. To refresh or swap images, re-run",
        "`python scripts/download_samples.py`.",
    ]
    (OUT / "SOURCES.md").write_text("\n".join(rows) + "\n")
    print(f"\nWrote {len(IMAGES)} images, total {total_bytes // 1024} KB")


if __name__ == "__main__":
    main()
