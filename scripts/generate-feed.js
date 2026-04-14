import { existsSync, readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs";
import path from "path";

const siteUrl = "https://julien-riel.com";
const siteTitle = "Julien Riel";
const authorName = "Julien Riel";

const parseFrontmatter = (raw) => {
  const m = raw.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return null;
  const data = {};
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^(\w+):\s*"?(.*?)"?$/);
    if (kv) data[kv[1]] = kv[2];
  }
  return data;
};

const collectEntries = (sourceDirs) => {
  const items = [];
  for (const [dir, section] of sourceDirs) {
    if (!existsSync(dir)) continue;
    for (const file of readdirSync(dir)) {
      if (!file.endsWith(".md")) continue;
      const fm = parseFrontmatter(readFileSync(path.join(dir, file), "utf8"));
      if (!fm || !fm.title) continue;
      const slug = file.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/\.md$/, "");
      items.push({
        title: fm.title,
        date: fm.date || "2026-01-01",
        description: fm.description || "",
        section,
        slug,
      });
    }
  }
  return items.sort((a, b) => (a.date < b.date ? 1 : -1));
};

const esc = (s) =>
  String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const renderFeed = (lang, entries, subtitle) => {
  const feedPath = lang === "en" ? "/en/feed.xml" : "/feed.xml";
  const homePath = lang === "en" ? "/en/" : "/";
  const urlPrefix = lang === "en" ? "/en" : "";
  const latest = entries[0]?.date || "2026-04-12";
  const entryXml = entries
    .slice(0, 40)
    .map((e) => {
      const url = `${urlPrefix}/${e.section}/${e.slug}/`;
      return `  <entry>
    <title>${esc(e.title)}</title>
    <link href="${siteUrl}${url}"/>
    <updated>${e.date}T00:00:00Z</updated>
    <id>${siteUrl}${url}</id>
    <summary>${esc(e.description)}</summary>
  </entry>`;
    })
    .join("\n");
  return `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>${esc(siteTitle)}</title>
  <subtitle>${esc(subtitle)}</subtitle>
  <link href="${siteUrl}${feedPath}" rel="self"/>
  <link href="${siteUrl}${homePath}"/>
  <updated>${latest}T00:00:00Z</updated>
  <id>${siteUrl}${homePath}</id>
  <author>
    <name>${authorName}</name>
  </author>
${entryXml}
</feed>
`;
};

const frEntries = collectEntries([
  ["src/posts", "posts"],
  ["src/articles", "articles"],
  ["src/case-studies", "case-studies"],
]);
const enEntries = collectEntries([
  ["src/en/posts", "posts"],
  ["src/en/articles", "articles"],
  ["src/en/case-studies", "case-studies"],
]);

const frFeed = renderFeed(
  "fr",
  frEntries,
  "Essais, études de cas et guides techniques sur la programmation agentique"
);
const enFeed = renderFeed(
  "en",
  enEntries,
  "Essays, case studies, and technical guides on agentic programming"
);

if (!existsSync("_site")) mkdirSync("_site", { recursive: true });
if (!existsSync("_site/en")) mkdirSync("_site/en", { recursive: true });
writeFileSync("_site/feed.xml", frFeed);
writeFileSync("_site/en/feed.xml", enFeed);

console.log(`Wrote _site/feed.xml (${frEntries.length} entries) and _site/en/feed.xml (${enEntries.length} entries)`);
