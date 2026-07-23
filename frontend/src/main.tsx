import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

// The service worker must be registered under the deployment base path — a hardcoded "/sw.js"
// 404s on GitHub Pages, where the app is served from /INDRA/.
if ("serviceWorker" in navigator) {
  const base = (import.meta as { env?: Record<string, string | undefined> }).env?.BASE_URL ?? "/";
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register(`${base}sw.js`).catch(() => {
      /* offline caching is a progressive enhancement; never block boot on it */
    });
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode><App /></StrictMode>,
);
