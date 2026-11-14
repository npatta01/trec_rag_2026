#!/usr/bin/env bash
# Render the talk deck the only way that produces a shippable file.
#
# `quarto render` on its own leaves three stylesheets as `data:text/css,...`
# links. Those load when the file is opened locally and are refused anywhere a
# Content-Security-Policy governs `style-src`, where the deck then renders as
# unstyled serif text with black SVG shapes. inline-data-css.py folds them into
# <style> blocks; the guards fail if it has not run.
set -euo pipefail
cd "$(dirname "$0")"

quarto render 2026-trec-rag-talk.qmd
python3 inline-data-css.py 2026-trec-rag-talk.html
node 2026-trec-rag-talk.test.js
