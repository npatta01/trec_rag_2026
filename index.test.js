const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const htmlPath = path.join(root, "index.html");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(fs.existsSync(htmlPath), "root index.html should exist");

const html = fs.readFileSync(htmlPath, "utf8");
const hrefs = [...html.matchAll(/href="([^"]+)"/g)].map((match) => match[1]);
const internalHrefs = hrefs.filter(
  (href) => !href.startsWith("#") && !/^[a-z][a-z0-9+.-]*:/i.test(href),
);

assert(internalHrefs.length >= 18, "artifact hub should expose the full project link map");
for (const href of internalHrefs) {
  const target = href.split("#", 1)[0].split("?", 1)[0];
  assert(target && !target.startsWith("/"), `internal link must be repository-relative: ${href}`);
  assert(fs.existsSync(path.resolve(root, target)), `internal link should resolve: ${href}`);
}

assert((html.match(/<article class="link-card /g) || []).length === 6, "artifact hub should render six cards");
for (const landmark of ["<main", "<header", "<section", "<footer", "aria-label="]) {
  assert(html.includes(landmark), `artifact hub should expose ${landmark}`);
}
for (const behavior of ["focus-visible", "prefers-reduced-motion", "@media (max-width: 620px)", "@media print"]) {
  assert(html.includes(behavior), `artifact hub should preserve ${behavior}`);
}
for (const forbidden of [
  "outputs/",
  "cache/",
  "PYSERINI_API_TOKEN",
  "OPENROUTER_API_KEY",
  "Authorization",
  "Bearer ",
]) {
  assert(!html.includes(forbidden), `artifact hub should not expose ${forbidden}`);
}

console.log("root artifact hub smoke test passed");
