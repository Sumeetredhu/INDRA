import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * `base` must match the GitHub Pages sub-path (https://<user>.github.io/INDRA/) or every asset
 * 404s. Override with BASE_PATH=/ when deploying to a root domain (Netlify, Vercel, S3, Docker).
 */
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH ?? "/INDRA/",
  server: { port: 5173 },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1200 },
});
