-- Inline each SVG figure instead of linking it.
--
-- An <img src="figure.svg"> is a separate document: the deck's CSS custom
-- properties cannot cross that boundary, so a palette switch on the page could
-- never repaint the figures. Inlining puts the markup in the same tree, where
-- var(--ret) and friends resolve against the page.
--
-- On the way in, each file's own :root colour block is dropped so the page is
-- the single source of colour. The files keep those blocks on disk, which is
-- why they still render correctly when opened on their own.
--
-- Figure CSS classes are namespaced per file (ap-, ln-, tw-, pl-, ev-, rt-)
-- so six style blocks in one document cannot collide.

local function read_file(path)
  local handle = io.open(path, "rb")
  if not handle then
    return nil
  end
  local contents = handle:read("a")
  handle:close()
  return contents
end

-- Remove `:root { ... }` and the dark-scheme override that only contains it.
local function strip_palette(svg)
  svg = svg:gsub("@media%s*%(prefers%-color%-scheme:%s*dark%)%s*%b{}", "")
  svg = svg:gsub(":root%b{}", "")
  return svg
end

function Image(el)
  local src = el.src
  if not src or not src:match("%.svg$") then
    return nil
  end

  local svg = read_file(src)
  if not svg then
    io.stderr:write("inline-svg: could not read " .. src .. ", leaving the <img> in place\n")
    return nil
  end

  if not svg:match("<desc") then
    io.stderr:write("inline-svg: " .. src .. " has no <desc>; screen readers will get nothing\n")
  end

  svg = svg:gsub("<%?xml.-%?>", "")
  svg = strip_palette(svg)

  return pandoc.RawInline("html", svg)
end
