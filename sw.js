/**
 * INDRA console service worker.
 *
 * Strategy is deliberately split, because the previous cache-first-everything version pinned every
 * visitor to the first build they ever loaded — a redeploy simply never reached them. It also
 * cached "/" rather than the deployment scope, so it missed entirely under /INDRA/.
 *
 *   - Navigations and the app shell: NETWORK FIRST, cache as fallback.
 *     Fresh code whenever the network is up; the last good shell when it is not.
 *   - Hashed build assets (/assets/index-<hash>.js): CACHE FIRST.
 *     Safe because the filename changes whenever the content does.
 *   - demo-snapshot.json / backends.json: STALE-WHILE-REVALIDATE.
 *     Instant paint from cache, refreshed in the background for the next load.
 *   - API calls and anything cross-origin: never touched. Caching a live backend response would
 *     make the console show stale plant data, which for this product is worse than showing nothing.
 */

const VERSION = "v3";
const CACHE = `indra-console-${VERSION}`;

// Scope-relative, so this behaves identically at "/" and at "/INDRA/".
const SCOPE = new URL(self.registration.scope);
const SHELL = [SCOPE.pathname, `${SCOPE.pathname}manifest.webmanifest`];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL).catch(() => undefined)),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

function isHashedAsset(url) {
  // Vite emits /assets/index-<hash>.js — content-addressed, so it can be cached indefinitely.
  return /\/assets\/[^/]+-[A-Za-z0-9_-]{8,}\.(js|css|woff2?|png|svg)$/.test(url.pathname);
}

function isData(url) {
  return url.pathname.endsWith("/demo-snapshot.json") || url.pathname.endsWith("/backends.json");
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response && response.ok) {
      const cache = await caches.open(CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) return cached;
    // A navigation with no network and no cached copy still needs the shell.
    const shell = await caches.match(SCOPE.pathname);
    if (shell) return shell;
    throw error;
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response && response.ok) {
    const cache = await caches.open(CACHE);
    cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const network = fetch(request)
    .then(async (response) => {
      if (response && response.ok) {
        const cache = await caches.open(CACHE);
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => undefined);
  return cached || (await network) || Response.error();
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Only ever handle our own origin and scope. API traffic must always reach the network.
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(SCOPE.pathname)) return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request));
    return;
  }
  if (isHashedAsset(url)) {
    event.respondWith(cacheFirst(request));
    return;
  }
  if (isData(url)) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }
  event.respondWith(networkFirst(request));
});

// Lets a running page force an update without a hard reload.
self.addEventListener("message", (event) => {
  if (event.data === "skip-waiting") self.skipWaiting();
});
