// The two hub pages share one stylesheet so "consistent" is a fact the repo
// can check, not a thing somebody remembered to do twice. This test owns both
// halves of that: the shared tokens clear AA in light and dark, and neither
// page is allowed to grow a private theme beside them.
const fs = require("node:fs");
const path = require("node:path");

const repo = path.join(__dirname, "..");
const css = fs.readFileSync(path.join(__dirname, "site.css"), "utf8");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function tokensOf(block) {
  const out = {};
  for (const m of block.matchAll(/--([a-z-]+):\s*(#[0-9a-fA-F]{3,8})/g)) out[m[1]] = m[2];
  return out;
}

const rootBlock = /:root\s*{([\s\S]*?)}/.exec(css);
const darkBlock = /prefers-color-scheme:\s*dark\)\s*{\s*:root\s*{([\s\S]*?)}/.exec(css);
assert(rootBlock, "site.css should declare a :root token block");
assert(darkBlock, "site.css should redefine the tokens for dark mode");

const light = tokensOf(rootBlock[1]);
const dark = { ...light, ...tokensOf(darkBlock[1]) };

function luminance(hex) {
  let h = hex.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  const channels = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrast(a, b) {
  const x = luminance(a);
  const y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

// Every foreground/background pair the pages actually put on screen. The label
// pills are small bold text, so they owe the full 4.5:1, not the large-text 3:1.
const pairs = [
  ["ink", "bg"], ["ink", "panel"],
  ["muted", "bg"], ["muted", "panel"],
  ["ret", "bg"], ["ret", "panel"],
  ["ok", "bg"], ["ok", "panel"],
  ["ret", "ret-soft"], ["gen", "gen-soft"], ["ok", "ok-soft"], ["adv", "adv-soft"],
  ["ink", "adv-soft"],
  // the primary button paints the page colour on the retrieval accent
  ["bg", "ret"],
];

for (const [name, tokens] of [["light", light], ["dark", dark]]) {
  for (const [fg, bg] of pairs) {
    assert(tokens[fg] && tokens[bg], `${name}: site.css should define --${fg} and --${bg}`);
    const ratio = contrast(tokens[fg], tokens[bg]);
    assert(
      ratio >= 4.5,
      `${name}: --${fg} on --${bg} is ${ratio.toFixed(2)}:1, under AA`,
    );
  }
}

for (const behavior of [
  "focus-visible",
  "prefers-reduced-motion",
  "@media (max-width: 620px)",
  "@media print",
]) {
  assert(css.includes(behavior), `site.css should carry ${behavior}`);
}

// Both hubs, one sheet, one bar.
const pages = [
  ["index.html", "assets/site.css"],
  ["reports/index.html", "../assets/site.css"],
];

const navLabels = [];
for (const [page, href] of pages) {
  const html = fs.readFileSync(path.join(repo, page), "utf8");
  assert(html.includes(`<link rel="stylesheet" href="${href}">`), `${page} should load ${href}`);
  assert(!/<style[\s>]/.test(html), `${page} should not carry a private theme block`);
  assert(html.includes('class="topbar"'), `${page} should carry the shared top bar`);
  assert(html.includes('<span class="brand-mark" aria-hidden="true">NP</span>'), `${page} should carry the shared brand mark`);
  assert(html.includes('aria-current="page"'), `${page} should mark its own nav entry`);

  const nav = /<nav class="nav"[^>]*>([\s\S]*?)<\/nav>/.exec(html);
  assert(nav, `${page} should carry the shared project nav`);
  navLabels.push([...nav[1].matchAll(/>([^<>]+)<\/a>/g)].map((m) => m[1].trim()).join(" | "));

  for (const forbidden of ["PYSERINI_API_TOKEN", "OPENROUTER_API_KEY", "Authorization", "Bearer ", "outputs/", "cache/"]) {
    assert(!html.includes(forbidden), `${page} should not expose ${forbidden}`);
  }
}

assert(
  navLabels[0] === navLabels[1],
  `both hubs should offer the same navigation, found:\n  ${navLabels[0]}\n  ${navLabels[1]}`,
);
assert(
  navLabels[0] === "Hub | Paper | Talk | Architecture | Scores | Reports | GitHub",
  `unexpected shared navigation: ${navLabels[0]}`,
);

console.log("shared site stylesheet and hub consistency passed");
