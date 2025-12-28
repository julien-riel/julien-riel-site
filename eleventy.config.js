import EleventyVitePlugin from "@11ty/eleventy-plugin-vite";

export default function(eleventyConfig) {
  // Filtre pour formater les dates
  eleventyConfig.addFilter("dateFormat", (date, format) => {
    const d = new Date(date);

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
        year: "numeric"
      });
    }

    return d.toLocaleDateString("fr-FR");
  });

  // Filtre head pour limiter les tableaux
  eleventyConfig.addFilter("head", (array, n) => {
    if (!Array.isArray(array)) return [];
    return array.slice(0, n);
  });

  // Filtre slugify pour les URLs
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

  // Plugin Vite pour le bundling des assets
  eleventyConfig.addPlugin(EleventyVitePlugin, {
    viteOptions: {
      clearScreen: false,
      server: {
        mode: "development",
        middlewareMode: true,
      },
      build: {
        mode: "production",
      },
    },
  });

  // Copier les assets statiques
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

  // Watch pour les fichiers CSS et JS
  eleventyConfig.addWatchTarget("src/assets/");

  // Configuration des dossiers
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
