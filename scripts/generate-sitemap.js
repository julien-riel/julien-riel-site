// Post-build sitemap + robots.txt. The Eleventy-Vite plugin only keeps HTML and
// bundled assets in _site, so non-HTML template output (.xml, .txt) never ships —
// same reason the RSS feed is generated here rather than as a template.
import { readdirSync, readFileSync, writeFileSync } from "fs";
import path from "path";

const siteUrl = "https://julien-riel.com";
const outDir = "_site";

// Every index.html in _site is a page; redirect stubs opt out via their noindex meta.
const collectUrls = (dir) => {
  const urls = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "assets") continue;
      urls.push(...collectUrls(full));
    } else if (entry.name === "index.html") {
      const html = readFileSync(full, "utf8");
      if (html.includes('name="robots" content="noindex"')) continue;
      const rel = path.relative(outDir, path.dirname(full));
      urls.push(rel === "" ? "/" : `/${rel.split(path.sep).join("/")}/`);
    }
  }
  return urls;
};

const urls = collectUrls(outDir).sort();

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url><loc>${siteUrl}${u}</loc></url>`).join("\n")}
</urlset>
`;

const robots = `# robots.txt for ${siteUrl}
User-agent: *
Allow: /

Sitemap: ${siteUrl}/sitemap.xml
`;

writeFileSync(path.join(outDir, "sitemap.xml"), sitemap);
writeFileSync(path.join(outDir, "robots.txt"), robots);

console.log(`Wrote _site/sitemap.xml (${urls.length} URLs) and _site/robots.txt`);
