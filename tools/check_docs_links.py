"""Check generated local links/fragments and current external Markdown links.

Run after mkdocs build. Core-site URLs are checked against the local build so
new documentation does not depend on deployment. External TLS stays verified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import ssl
import subprocess
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlparse


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for field in ("href", "src"):
            if attrs.get(field):
                self.targets.append(attrs[field])


def markdown_links(root):
    for path in [root / "README.md", *sorted((root / "docs").rglob("*.md"))]:
        if "marketing" not in path.parts and not path.name.startswith("_audit"):
            yield path, re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8"))


def local_links(root, site):
    parsed = {}
    for path in site.rglob("*.html"):
        page = Links()
        page.feed(path.read_text(encoding="utf-8"))
        parsed[path.resolve()] = page
    failures, count = [], 0
    for path, page in parsed.items():
        for link in page.targets:
            target, fragment = urldefrag(link)
            url = urlparse(target)
            if url.netloc == "core.lazybridge.com":
                dest = site / url.path.lstrip("/")
            elif url.scheme or url.netloc:
                continue
            else:
                dest = (
                    site / unquote(url.path).lstrip("/")
                    if url.path.startswith("/")
                    else path.parent / unquote(url.path)
                )
                if not url.path:
                    dest = path
            if dest.is_dir():
                dest = dest / "index.html"
            count += 1
            dest = dest.resolve()
            if not dest.exists():
                failures.append(f"{path.relative_to(site)} -> {link}: missing file")
            elif fragment and dest in parsed and unquote(fragment) not in parsed[dest].ids:
                failures.append(f"{path.relative_to(site)} -> {link}: missing fragment")
    # README is not a MkDocs page, so its core-site and relative file links
    # need a separate pass. Core-site links in Markdown also bypass redirects
    # and validate against the documentation we are about to publish.
    for path, links in markdown_links(root):
        for link in links:
            target, fragment = urldefrag(link)
            url = urlparse(target)
            if url.netloc == "core.lazybridge.com":
                dest = site / unquote(url.path).lstrip("/")
                if dest.is_dir():
                    dest /= "index.html"
            elif path == root / "README.md" and not url.scheme and not url.netloc and url.path:
                dest = path.parent / unquote(url.path)
            else:
                continue
            count += 1
            dest = dest.resolve()
            if not dest.exists():
                failures.append(f"{path.relative_to(root)} -> {link}: missing file")
            elif fragment and dest in parsed and unquote(fragment) not in parsed[dest].ids:
                failures.append(f"{path.relative_to(root)} -> {link}: missing fragment")
    return count, failures


def external_links(root, ref=None):
    links = {}
    for path in [root / "README.md", *sorted((root / "docs").rglob("*.md"))]:
        if "marketing" in path.parts or path.name.startswith("_audit"):
            continue
        content = (
            subprocess.run(
                ["git", "show", ref + ":" + path.relative_to(root).as_posix()],
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
            if ref
            else path.read_text(encoding="utf-8")
        )
        urls = re.findall(r"\]\((https?://[^\s)]+)\)|<(https?://[^\s>]+)>", content)
        for values in urls:
            url = next(value for value in values if value)
            if urlparse(url).netloc == "core.lazybridge.com":
                continue
            links.setdefault(url, []).append(str(path.relative_to(root)))
    return links


def fetch(url):
    import certifi

    target, fragment = urldefrag(url)
    request = urllib.request.Request(target, headers={"User-Agent": "LazyBridge-docs-link-check/1.0"})
    try:
        with urllib.request.urlopen(
            request, timeout=25, context=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            result = {"url": url, "status": response.status, "final": response.url}
            if "html" in response.headers.get("Content-Type", ""):
                body = response.read() if fragment else response.read(4096)
                body = body.decode("utf-8", errors="replace")
                if re.search(r"<title>\s*(?:404\b|Page not found\b)", body, re.I):
                    result["error"] = "HTML reports a missing page despite successful HTTP status"
                if fragment:
                    page = Links()
                    page.feed(body)
                    if unquote(fragment) not in page.ids and "user-content-" + unquote(fragment) not in page.ids:
                        result["error"] = (
                            "Missing HTML fragment; inspect client-rendered pages before declaring it dead"
                        )
            return result
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "error": str(exc)}
    except Exception as exc:
        return {"url": url, "status": None, "error": str(exc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ref", help="Read external Markdown destinations from a Git revision, e.g. e475be8")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    count, failures = local_links(root, (root / "site").resolve())
    report = {"local_checked": count, "local_failures": failures}
    print(f"Local links/assets/fragments: {count} checked, {len(failures)} failures")
    for failure in failures:
        print(failure)
    if args.external:
        links = external_links(root, args.ref)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(fetch, links))
        for result in results:
            result["pages"] = links[result["url"]]
        report["external"] = results
        uncertain = [item for item in results if "error" in item or item["status"] is None or item["status"] >= 400]
        print(f"External URLs: {len(results)} checked, {len(uncertain)} need inspection")
        for result in uncertain:
            print(json.dumps(result))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(bool(failures or (args.external and uncertain)))
