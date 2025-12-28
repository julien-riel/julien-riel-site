import EleventyVitePlugin from "@11ty/eleventy-plugin-vite";
import syntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";
import markdownIt from "markdown-it";
import markdownItAnchor from "markdown-it-anchor";

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

  // Collection: all posts sorted by date (newest first), excluding drafts
  eleventyConfig.addCollection("posts", function(collectionApi) {
    return collectionApi
      .getFilteredByGlob("src/posts/**/*.md")
      .filter((post) => !post.data.draft)
      .sort((a, b) => b.date - a.date);
  });

  // Collection: list of all unique tags
  eleventyConfig.addCollection("tagList", function(collectionApi) {
    const tagSet = new Set();

    collectionApi.getAll().forEach((item) => {
      if (item.data.tags) {
        item.data.tags.forEach((tag) => {
          if (tag !== "posts") {
            tagSet.add(tag);
          }
        });
      }
    });

    return [...tagSet].sort((a, b) => a.localeCompare(b, "fr"));
  });

  // Filter for formatting dates
  eleventyConfig.addFilter("dateFormat", (date, format) => {
    const d = date === "now" ? new Date() : new Date(date);

    if (format === "iso") {
      return d.toISOString().split("T")[0];
    }

    if (format === "year") {
      return d.getFullYear().toString();
    }

    if (format === "fr") {
      return d.toLocaleDateString("fr-FR", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      });
    }

    if (format === "long") {
      return d.toLocaleDateString("fr-FR", {
        day: "numeric",
        month: "long",
        year: "numeric",
      });
    }

    return d.toLocaleDateString("fr-FR");
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

  // Transform: Add lazy loading and accessibility to images
  eleventyConfig.addTransform("lazyImages", function(content) {
    if (this.page.outputPath && this.page.outputPath.endsWith(".html")) {
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
    }
    return content;
  });

  // Copy static assets
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

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
