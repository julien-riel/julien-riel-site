import EleventyVitePlugin from "@11ty/eleventy-plugin-vite";
import syntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";
import markdownIt from "markdown-it";
import markdownItAnchor from "markdown-it-anchor";
import Image from "@11ty/eleventy-img";
import path from "path";
import { existsSync } from "fs";

/**
 * Generates a table of contents from markdown content
 * @param {string} content - HTML content
 * @returns {Array} - TOC array with nested structure
 */
function generateTOC(content) {
  const headingRegex = /<h([2-3])[^>]*id="([^"]*)"[^>]*>(.*?)<\/h[2-3]>/gi;
  const toc = [];
  let match;

  while ((match = headingRegex.exec(content)) !== null) {
    const level = parseInt(match[1], 10);
    const id = match[2];
    // Strip HTML tags and anchor link text
    const text = match[3]
      .replace(/<[^>]*>/g, "")
      .replace(/#$/, "")
      .trim();

    if (level === 2) {
      toc.push({ level, id, text, children: [] });
    } else if (level === 3 && toc.length > 0) {
      toc[toc.length - 1].children.push({ level, id, text });
    }
  }

  return toc;
}

/**
 * Gets related posts based on shared tags
 * @param {string} currentUrl - Current post URL
 * @param {Array} currentTags - Current post tags
 * @param {Array} allPosts - All posts collection
 * @param {number} limit - Max number of related posts
 * @returns {Array} - Related posts
 */
function getRelatedPosts(currentUrl, currentTags, allPosts, limit = 3) {
  const postTags = currentTags || [];

  return allPosts
    .filter((p) => {
      // Exclude current post
      if (p.url === currentUrl) return false;
      // Exclude drafts
      if (p.data.draft) return false;
      return true;
    })
    .map((p) => {
      const pTags = p.data.tags || [];
      const sharedTags = postTags.filter((tag) =>
        tag !== "posts" && pTags.includes(tag)
      );
      return { post: p, score: sharedTags.length };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => item.post);
}

/**
 * Gets related content across essays, articles, and case studies.
 * Ranks by shared tags; ties broken by section diversity so the "See also"
 * block tends to span multiple sections rather than pile up in one.
 */
const STRUCTURAL_TAGS = new Set(["posts", "articles", "case-studies"]);

function getRelatedContent(currentUrl, currentTags, currentLang, collections, limit = 4) {
  const postTags = (currentTags || []).filter((t) => !STRUCTURAL_TAGS.has(t));
  if (postTags.length === 0) return [];

  const pools = [
    { section: "posts", items: collections.posts || [] },
    { section: "articles", items: collections.articles || [] },
    { section: "case-studies", items: collections["case-studies"] || [] },
  ];

  const candidates = [];
  for (const { section, items } of pools) {
    for (const p of items) {
      if (p.url === currentUrl) continue;
      if (p.data.draft) continue;
      if (currentLang && p.data.lang !== currentLang) continue;
      const pTags = (p.data.tags || []).filter((t) => !STRUCTURAL_TAGS.has(t));
      const score = postTags.filter((t) => pTags.includes(t)).length;
      if (score > 0) candidates.push({ post: p, score, section });
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  // Section-aware selection: walk the sorted list and prefer items from sections
  // not yet represented, until each section has one, then fill remaining slots.
  const seenSections = new Set();
  const picked = [];
  for (const c of candidates) {
    if (picked.length >= limit) break;
    if (!seenSections.has(c.section)) {
      picked.push(c);
      seenSections.add(c.section);
    }
  }
  for (const c of candidates) {
    if (picked.length >= limit) break;
    if (!picked.includes(c)) picked.push(c);
  }

  return picked.map((c) => c.post);
}

export default function(eleventyConfig) {
  // Configure markdown-it with anchor plugin
  const mdOptions = {
    html: true,
    breaks: false,
    linkify: true,
  };

  const mdAnchorOptions = {
    permalink: markdownItAnchor.permalink.linkAfterHeader({
      placement: "after",
      class: "header-anchor",
      symbol: "#",
      style: "visually-hidden",
      assistiveText: (title) => `Lien vers la section ${title}`,
      visuallyHiddenClass: "visually-hidden",
    }),
    level: [2, 3, 4],
    slugify: (s) => {
      return s
        .toString()
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/(^-|-$)/g, "");
    },
  };

  const md = markdownIt(mdOptions).use(markdownItAnchor, mdAnchorOptions);
  eleventyConfig.setLibrary("md", md);

  // Syntax highlighting plugin
  eleventyConfig.addPlugin(syntaxHighlight, {
    preAttributes: {
      tabindex: 0,
      "data-language": function({ language }) {
        return language;
      },
    },
  });

  // Image shortcode: optimizes images, converts to WebP, adds width/height
  eleventyConfig.addAsyncShortcode("image", async function(src, alt, sizes = "100vw") {
    const inputPath = this.page.inputPath;
    const inputDir = path.dirname(inputPath);

    // Resolve image path relative to the current file or from src/assets
    let imagePath;
    if (src.startsWith("/")) {
      imagePath = path.join("src", src);
    } else if (src.startsWith("http")) {
      imagePath = src;
    } else {
      imagePath = path.join(inputDir, src);
    }

    // Check if file exists for local images
    if (!src.startsWith("http") && !existsSync(imagePath)) {
      console.warn(`[eleventy-img] Image not found: ${imagePath}`);
      return `<img src="${src}" alt="${alt || ""}" loading="lazy" decoding="async">`;
    }

    const metadata = await Image(imagePath, {
      widths: [400, 800, 1200],
      formats: ["webp", "jpeg"],
      outputDir: "./_site/assets/images/",
      urlPath: "/assets/images/",
      filenameFormat: function(id, src, width, format) {
        const name = path.basename(src, path.extname(src));
        return `${name}-${width}w.${format}`;
      },
    });

    const imageAttributes = {
      alt: alt || "",
      sizes,
      loading: "lazy",
      decoding: "async",
    };

    return Image.generateHTML(metadata, imageAttributes);
  });

  // Computed data: make translation strings available everywhere
  eleventyConfig.addGlobalData("eleventyComputed", {
    currentLang: (data) => data.lang || "fr",
    t: (data) => {
      const lang = data.lang || "fr";
      return data.i18n ? data.i18n[lang] : {};
    },
    langPrefix: (data) => (data.lang || "fr") === "en" ? "/en" : "",
  });

  // Collection: all posts sorted by date (newest first), excluding drafts
  eleventyConfig.addCollection("posts", function(collectionApi) {
    return collectionApi
      .getFilteredByGlob(["src/posts/**/*.md", "src/en/posts/**/*.md"])
      .filter((post) => !post.data.draft)
      .sort((a, b) => a.date - b.date);
  });

  // Collection: list of all unique tags (excluding structural collection tags)
  const collectionTags = new Set(["posts", "articles", "case-studies"]);

  eleventyConfig.addCollection("tagList", function(collectionApi) {
    const tagSet = new Set();

    collectionApi.getAll().forEach((item) => {
      if (item.data.tags) {
        item.data.tags.forEach((tag) => {
          if (!collectionTags.has(tag)) {
            tagSet.add(tag);
          }
        });
      }
    });

    return [...tagSet].sort((a, b) => a.localeCompare(b, "fr"));
  });

  // Filter: filter collection by language
  eleventyConfig.addFilter("filterByLang", (collection, lang) => {
    if (!collection || !lang) return collection || [];
    return collection.filter((item) => item.data.lang === lang);
  });

  // Filter for formatting dates (language-aware)
  eleventyConfig.addFilter("dateFormat", (date, format, lang) => {
    const d = date === "now" ? new Date() : new Date(date);
    const locale = lang === "fr" ? "fr-FR" : "en-US";

    if (format === "iso") {
      return d.toISOString().split("T")[0];
    }

    if (format === "year") {
      return d.getFullYear().toString();
    }

    if (format === "short") {
      return d.toLocaleDateString(locale, {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    }

    if (format === "long") {
      return d.toLocaleDateString(locale, {
        day: "numeric",
        month: "long",
        year: "numeric",
      });
    }

    return d.toLocaleDateString(locale);
  });

  // Filter: limit array length
  eleventyConfig.addFilter("head", (array, n) => {
    if (!Array.isArray(array)) return [];
    return array.slice(0, n);
  });

  // Filter: slugify for URLs
  eleventyConfig.addFilter("slugify", (str) => {
    if (!str) return "";
    return str
      .toString()
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "");
  });

  // Filter: generate TOC from content
  eleventyConfig.addFilter("toc", (content) => {
    return generateTOC(content);
  });

  // Filter: get related posts
  eleventyConfig.addFilter("relatedPosts", function(url, tags, collections, limit = 3) {
    const allPosts = collections.posts || [];
    return getRelatedPosts(url, tags, allPosts, limit);
  });

  // Filter: get related content across essays, articles, and case studies
  eleventyConfig.addFilter("relatedContent", function(url, tags, lang, collections, limit = 4) {
    return getRelatedContent(url, tags, lang, collections, limit);
  });

  // Filter: filter posts by tag
  eleventyConfig.addFilter("filterByTag", (posts, tag) => {
    if (!posts || !tag) return [];
    return posts.filter((post) => {
      const tags = post.data.tags || [];
      return tags.includes(tag);
    });
  });

  // Filter: count posts by tag
  eleventyConfig.addFilter("postCountByTag", (posts, tag) => {
    if (!posts || !tag) return 0;
    return posts.filter((post) => {
      const tags = post.data.tags || [];
      return tags.includes(tag);
    }).length;
  });

  // Filter: estimate reading time in minutes
  eleventyConfig.addFilter("readingTime", (content) => {
    if (!content) return "1 min";
    const words = content.replace(/<[^>]*>/g, " ").split(/\s+/).filter(Boolean).length;
    const minutes = Math.max(1, Math.round(words / 200));
    return `${minutes} min`;
  });

  // Filter: strip HTML tags from content
  eleventyConfig.addFilter("striptags", (content) => {
    if (!content) return "";
    return content
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  });

  // Vite plugin for asset bundling
  eleventyConfig.addPlugin(EleventyVitePlugin, {
    viteOptions: {
      clearScreen: false,
      server: {
        mode: "development",
        middlewareMode: true,
      },
      build: {
        mode: "production",
        emptyOutDir: false,
        copyPublicDir: false,
        rollupOptions: {
          output: {
            assetFileNames: "assets/[name]-[hash][extname]",
            chunkFileNames: "assets/[name]-[hash].js",
            entryFileNames: "assets/[name]-[hash].js",
          },
        },
      },
    },
  });

  // Transform: Add lazy loading, accessibility, and dimensions to images
  eleventyConfig.addTransform("optimizeImages", async function(content) {
    if (!this.page.outputPath || !this.page.outputPath.endsWith(".html")) {
      return content;
    }

    // Add loading="lazy" and decoding="async" to images without these attributes
    content = content.replace(
      /<img(?![^>]*loading=)([^>]*)>/gi,
      '<img loading="lazy"$1>'
    );
    content = content.replace(
      /<img(?![^>]*decoding=)([^>]*)>/gi,
      '<img decoding="async"$1>'
    );
    // Ensure images have alt text (add empty alt if missing for decorative images)
    content = content.replace(
      /<img(?![^>]*alt=)([^>]*)>/gi,
      '<img alt=""$1>'
    );

    // Add width and height to images that don't have them
    const imgRegex = /<img([^>]*)src="([^"]+)"([^>]*)>/gi;
    const matches = [...content.matchAll(imgRegex)];

    for (const match of matches) {
      const fullMatch = match[0];
      const beforeSrc = match[1];
      const src = match[2];
      const afterSrc = match[3];

      // Skip if already has width and height
      if (fullMatch.includes('width=') && fullMatch.includes('height=')) {
        continue;
      }

      // Skip external images and data URIs
      if (src.startsWith('http') || src.startsWith('data:')) {
        continue;
      }

      try {
        // Resolve image path
        let imagePath;
        if (src.startsWith('/')) {
          imagePath = path.join('src', src);
        } else {
          imagePath = path.join('src', src);
        }

        if (!existsSync(imagePath)) {
          // Try _site directory for processed images
          imagePath = path.join('_site', src);
        }

        if (existsSync(imagePath)) {
          const metadata = await Image(imagePath, {
            widths: [null], // Keep original width
            formats: [null], // Keep original format
            dryRun: true, // Don't output files, just get metadata
          });

          const format = Object.keys(metadata)[0];
          const { width, height } = metadata[format][0];

          // Build new img tag with dimensions
          let newTag = `<img${beforeSrc}src="${src}"${afterSrc}>`;
          if (!fullMatch.includes('width=')) {
            newTag = newTag.replace('<img', `<img width="${width}"`);
          }
          if (!fullMatch.includes('height=')) {
            newTag = newTag.replace('<img', `<img height="${height}"`);
          }

          content = content.replace(fullMatch, newTag);
        }
      } catch (error) {
        // Silently skip images that can't be processed
      }
    }

    return content;
  });

  // Copy static assets
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

  // Self-hosted fonts: copy specific woff2 files from @fontsource packages
  const fontFiles = {
    "node_modules/@fontsource/dm-serif-display/files/dm-serif-display-latin-400-normal.woff2": "assets/fonts/dm-serif-display-400.woff2",
    "node_modules/@fontsource/dm-serif-display/files/dm-serif-display-latin-400-italic.woff2": "assets/fonts/dm-serif-display-400-italic.woff2",
    "node_modules/@fontsource/outfit/files/outfit-latin-300-normal.woff2": "assets/fonts/outfit-300.woff2",
    "node_modules/@fontsource/outfit/files/outfit-latin-400-normal.woff2": "assets/fonts/outfit-400.woff2",
    "node_modules/@fontsource/outfit/files/outfit-latin-500-normal.woff2": "assets/fonts/outfit-500.woff2",
    "node_modules/@fontsource/outfit/files/outfit-latin-600-normal.woff2": "assets/fonts/outfit-600.woff2",
    "node_modules/@fontsource/outfit/files/outfit-latin-700-normal.woff2": "assets/fonts/outfit-700.woff2",
    "node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2": "assets/fonts/jetbrains-mono-400.woff2",
    "node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2": "assets/fonts/jetbrains-mono-500.woff2",
    "node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-600-normal.woff2": "assets/fonts/jetbrains-mono-600.woff2",
  };
  eleventyConfig.addPassthroughCopy(fontFiles);

  // Watch CSS and JS files
  eleventyConfig.addWatchTarget("src/assets/");

  // Directory configuration
  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data",
      layouts: "_includes/layouts",
    },
    templateFormats: ["njk", "md", "html"],
    htmlTemplateEngine: "njk",
    markdownTemplateEngine: "njk",
  };
}
