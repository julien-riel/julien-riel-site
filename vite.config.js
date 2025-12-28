import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: "src",
  build: {
    outDir: "../_site",
    emptyOutDir: false,
    // Enable minification in production
    minify: "terser",
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
    // CSS minification
    cssMinify: true,
    // Code splitting
    cssCodeSplit: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "src/assets/js/main.js"),
        styles: resolve(__dirname, "src/assets/css/main.css"),
      },
      output: {
        // Optimize chunk names for caching
        chunkFileNames: "assets/js/[name]-[hash].js",
        entryFileNames: "assets/js/[name]-[hash].js",
        assetFileNames: (assetInfo) => {
          const extType = assetInfo.name.split(".").pop();
          if (/css/i.test(extType)) {
            return "assets/css/[name]-[hash][extname]";
          }
          if (/png|jpe?g|svg|gif|tiff|bmp|ico|webp|avif/i.test(extType)) {
            return "assets/images/[name]-[hash][extname]";
          }
          if (/woff2?|ttf|eot|otf/i.test(extType)) {
            return "assets/fonts/[name]-[hash][extname]";
          }
          return "assets/[name]-[hash][extname]";
        },
      },
    },
    // Generate source maps for debugging
    sourcemap: false,
    // Target modern browsers
    target: "es2020",
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  // Optimize dependencies
  optimizeDeps: {
    include: ["lunr", "mermaid"],
  },
});
