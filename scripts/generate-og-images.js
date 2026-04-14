import { existsSync, readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs";
import path from "path";
import satori from "satori";
import { Resvg } from "@resvg/resvg-js";

const SITE_TITLE = "JULIEN RIEL";
const SITE_DOMAIN = "julien-riel.com";
const WIDTH = 1200;
const HEIGHT = 630;

const COLOR_BG = "#FAF7F2";
const COLOR_COPPER = "#C45D35";
const COLOR_TEAL = "#1A535C";
const COLOR_TEXT = "#1C1917";
const COLOR_MUTED = "#57534E";
const COLOR_FADED = "#78716C";

const FONT_DIR = "node_modules/@fontsource";
const loadFont = (p) => (existsSync(p) ? readFileSync(p) : null);

const fonts = [
  {
    name: "DM Serif Display",
    data: loadFont(`${FONT_DIR}/dm-serif-display/files/dm-serif-display-latin-400-normal.woff`),
    weight: 400,
    style: "normal",
  },
  {
    name: "Outfit",
    data: loadFont(`${FONT_DIR}/outfit/files/outfit-latin-300-normal.woff`),
    weight: 300,
    style: "normal",
  },
  {
    name: "Outfit",
    data: loadFont(`${FONT_DIR}/outfit/files/outfit-latin-400-normal.woff`),
    weight: 400,
    style: "normal",
  },
  {
    name: "Outfit",
    data: loadFont(`${FONT_DIR}/outfit/files/outfit-latin-600-normal.woff`),
    weight: 600,
    style: "normal",
  },
].filter((f) => f.data);

const SECTION_LABELS = {
  posts: { fr: "ESSAI", en: "ESSAY" },
  articles: { fr: "ARTICLE", en: "ARTICLE" },
  "case-studies": { fr: "ÉTUDE DE CAS", en: "CASE STUDY" },
};

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

// Title size picked on char count. Short titles get huge display; long ones shrink.
const titleFontSize = (title) => {
  const len = title.length;
  if (len <= 10) return 160;
  if (len <= 20) return 120;
  if (len <= 40) return 84;
  return 64;
};

// Tiny element factory — avoids needing JSX / React.
const el = (type, props = {}, children) => ({
  type,
  props: { ...props, children },
});

const buildTemplate = ({ title, description, section, lang }) => {
  const label = SECTION_LABELS[section]?.[lang] || section.toUpperCase();
  const fontSize = titleFontSize(title);

  return el(
    "div",
    {
      style: {
        width: WIDTH,
        height: HEIGHT,
        display: "flex",
        flexDirection: "column",
        backgroundColor: COLOR_BG,
        fontFamily: "Outfit",
        padding: "80px",
        position: "relative",
      },
    },
    [
      // Decorative copper circle top-right
      el("div", {
        style: {
          position: "absolute",
          top: -160,
          right: -160,
          width: 440,
          height: 440,
          borderRadius: "50%",
          backgroundColor: COLOR_COPPER,
          opacity: 0.06,
        },
      }),
      // Decorative teal circle top-right (smaller, offset)
      el("div", {
        style: {
          position: "absolute",
          top: -20,
          right: -40,
          width: 280,
          height: 280,
          borderRadius: "50%",
          backgroundColor: COLOR_TEAL,
          opacity: 0.04,
        },
      }),
      // Giant faded "97" watermark bottom-right
      el(
        "div",
        {
          style: {
            position: "absolute",
            right: 40,
            bottom: -120,
            fontFamily: "DM Serif Display",
            fontSize: 540,
            color: COLOR_COPPER,
            opacity: 0.06,
            lineHeight: 1,
          },
        },
        "97"
      ),
      // Top bar: site mark
      el(
        "div",
        {
          style: {
            display: "flex",
            alignItems: "center",
            gap: 20,
          },
        },
        [
          el(
            "div",
            {
              style: {
                fontSize: 22,
                fontWeight: 600,
                color: COLOR_TEAL,
                letterSpacing: 3,
              },
            },
            SITE_TITLE
          ),
          el("div", {
            style: {
              width: 80,
              height: 2,
              backgroundColor: COLOR_COPPER,
            },
          }),
        ]
      ),
      // Spacer
      el("div", { style: { flexGrow: 1, display: "flex" } }),
      // Section label
      el(
        "div",
        {
          style: {
            fontSize: 20,
            fontWeight: 600,
            color: COLOR_COPPER,
            letterSpacing: 4,
            marginBottom: 32,
          },
        },
        label
      ),
      // Title
      el(
        "div",
        {
          style: {
            fontFamily: "DM Serif Display",
            fontSize,
            color: COLOR_TEXT,
            lineHeight: 1.05,
            marginBottom: 28,
            maxWidth: 1040,
          },
        },
        title
      ),
      // Description (optional)
      description
        ? el(
            "div",
            {
              style: {
                fontSize: 26,
                fontWeight: 300,
                color: COLOR_MUTED,
                lineHeight: 1.4,
                maxWidth: 980,
              },
            },
            description
          )
        : null,
      // Spacer
      el("div", { style: { flexGrow: 1, display: "flex" } }),
      // Bottom bar
      el(
        "div",
        {
          style: {
            display: "flex",
            flexDirection: "column",
            gap: 12,
          },
        },
        [
          el("div", {
            style: {
              width: 80,
              height: 3,
              backgroundColor: COLOR_COPPER,
            },
          }),
          el(
            "div",
            {
              style: {
                fontSize: 18,
                color: COLOR_FADED,
              },
            },
            SITE_DOMAIN
          ),
        ]
      ),
    ].filter(Boolean)
  );
};

const renderCard = async (page) => {
  const element = buildTemplate(page);
  const svg = await satori(element, {
    width: WIDTH,
    height: HEIGHT,
    fonts,
  });
  const resvg = new Resvg(svg, { fitTo: { mode: "width", value: WIDTH } });
  return resvg.render().asPng();
};

const collectPages = () => {
  const pages = [];
  const sources = [
    { dir: "src/posts", section: "posts", lang: "fr" },
    { dir: "src/articles", section: "articles", lang: "fr" },
    { dir: "src/case-studies", section: "case-studies", lang: "fr" },
    { dir: "src/en/posts", section: "posts", lang: "en" },
    { dir: "src/en/articles", section: "articles", lang: "en" },
    { dir: "src/en/case-studies", section: "case-studies", lang: "en" },
  ];
  for (const { dir, section, lang } of sources) {
    if (!existsSync(dir)) continue;
    for (const file of readdirSync(dir)) {
      if (!file.endsWith(".md")) continue;
      const fm = parseFrontmatter(readFileSync(path.join(dir, file), "utf8"));
      if (!fm || !fm.title) continue;
      const slug = file.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/\.md$/, "");
      pages.push({
        title: fm.title,
        description: fm.description || "",
        section,
        lang,
        slug,
      });
    }
  }
  return pages;
};

const STATIC_CARDS = [
  {
    title: "97 choses que tout programmeur agentique devrait savoir",
    description: "Courts essais sur la construction, l'utilisation et la compréhension des agents IA.",
    section: "posts",
    lang: "fr",
    slug: "_home",
  },
  {
    title: "97 Things Every Agentic Programmer Should Know",
    description: "Short essays on building, using, and thinking clearly about AI agents.",
    section: "posts",
    lang: "en",
    slug: "_home",
  },
];

const main = async () => {
  if (fonts.length === 0) {
    throw new Error("No fonts loaded — check FONT_DIR paths");
  }
  const outDir = "_site/assets/og";
  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

  const pages = [...collectPages(), ...STATIC_CARDS];
  let count = 0;
  for (const page of pages) {
    const png = await renderCard(page);
    const outPath = path.join(outDir, `${page.lang}-${page.section}-${page.slug}.png`);
    writeFileSync(outPath, png);
    count++;
  }
  console.log(`Generated ${count} OG images → ${outDir}/`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
