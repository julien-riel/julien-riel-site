import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: "src",
  build: {
    outDir: "../_site",
    emptyOutDir: false,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "src/assets/js/main.js"),
        styles: resolve(__dirname, "src/assets/css/main.css"),
      },
    },
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
});
