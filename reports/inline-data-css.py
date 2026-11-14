#!/usr/bin/env python3
"""Turn Quarto's `data:text/css,...` stylesheet links into inline <style> blocks.

`embed-resources: true` inlines most things, but a few stylesheets come back as
`<link rel="stylesheet" href="data:text/css,...">`. A browser opening the file
directly honours those. Anywhere a Content-Security-Policy governs `style-src`
-- a published artifact, most hosts -- a `data:` URL is not in the allow-list,
the fetch is refused, and the deck renders as unstyled serif text with black
SVG shapes. `'unsafe-inline'` is almost always allowed, so a <style> block
survives where the link does not.

Idempotent: run it as often as you like after `quarto render`.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

# href may be single- or double-quoted; the payload runs to the closing quote.
LINK = re.compile(
    r"""<link\b[^>]*?href=(?P<q>["'])data:text/css,(?P<css>.*?)(?P=q)[^>]*>""",
    re.DOTALL | re.IGNORECASE,
)


def inline(html: str) -> tuple[str, int]:
    count = 0

    def swap(m: re.Match[str]) -> str:
        nonlocal count
        css = urllib.parse.unquote(m.group("css"))
        # A literal `</style` inside the CSS would close the block early. It has
        # no business being in a stylesheet, but escape the slash if it appears.
        css = re.sub(r"</(?=\s*style)", r"<\\/", css, flags=re.IGNORECASE)
        count += 1
        return f"<style>\n{css}\n</style>"

    return LINK.sub(swap, html), count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <rendered.html>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    html = path.read_text(encoding="utf-8")
    out, count = inline(html)

    if count:
        path.write_text(out, encoding="utf-8")

    left = out.count("data:text/css")
    print(f"{path.name}: inlined {count} data: stylesheet(s); {left} remaining")
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
