/**
 * Backend discovery.
 *
 * The hosted console must work for a stranger with no instructions. So instead of asking anyone to
 * paste a URL, it resolves one in priority order:
 *
 *   1. `?api=<url>` in the query string  — makes a shareable link that pins a specific backend,
 *                                          and is remembered afterwards
 *   2. `localStorage`                    — whatever worked last time
 *   3. `VITE_API_BASE` at build time     — for forks that bake in their own host
 *   4. `backends.json` candidates        — probed in parallel; first healthy one wins
 *   5. nothing                           — the recorded session answers instead
 *
 * Discovery runs in the background. The page renders from the recording immediately and *upgrades*
 * to the live backend when one answers, because a free-tier instance that has gone to sleep takes
 * ~50 seconds to wake and blocking the first paint on that would be far worse than showing a
 * recording that is already correct.
 */

const STORAGE_KEY = "indra.api.base";

/** Fast probe for a backend that is already awake. */
const WARM_TIMEOUT_MS = 4_000;

/** Patient probe, sized for a free-tier cold start (Render sleeps after ~15 min idle). */
const COLD_TIMEOUT_MS = 75_000;

function normalise(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

function fromQuery(): string | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const api = params.get("api");
    return api ? normalise(api) : null;
  } catch {
    return null;
  }
}

function fromStorage(): string | null {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? normalise(stored) : null;
  } catch {
    return null; // private browsing / storage disabled
  }
}

export function rememberBackend(url: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, normalise(url));
  } catch {
    /* non-fatal: discovery just runs again next load */
  }
}

export function forgetBackend(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

/** Ask one candidate whether it is alive. Never throws. */
async function probe(base: string, timeoutMs: number): Promise<boolean> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${base}/health`, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

/** Candidate hosts shipped with the build. Editing this file and redeploying is the whole config. */
async function candidates(): Promise<string[]> {
  try {
    const base = (import.meta as { env?: Record<string, string | undefined> }).env?.BASE_URL ?? "/";
    const response = await fetch(`${base}backends.json`);
    if (!response.ok) return [];
    const data = (await response.json()) as { backends?: string[] };
    return (data.backends ?? []).map(normalise).filter(Boolean);
  } catch {
    return [];
  }
}

export interface Discovery {
  /** A backend to try first, synchronously available (may still be unreachable). */
  initial: string;
  /** Resolves to a *verified* live backend, or null if none answered. */
  live: Promise<string | null>;
  /** True when the user pinned a backend explicitly, so discovery should not override it. */
  pinned: boolean;
}

const ENV_BASE = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE;

/** Whether this build is running from a local dev server rather than a hosted origin. */
function isLocalOrigin(): boolean {
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "";
}

export function discoverBackend(): Discovery {
  const pinnedUrl = fromQuery();
  if (pinnedUrl) {
    rememberBackend(pinnedUrl);
    return {
      initial: pinnedUrl,
      pinned: true,
      live: probe(pinnedUrl, COLD_TIMEOUT_MS).then((ok) => (ok ? pinnedUrl : null)),
    };
  }

  const stored = fromStorage();
  const localDefault = isLocalOrigin() ? "http://localhost:8000" : null;
  const initial = stored ?? ENV_BASE ?? localDefault ?? "";

  const live = (async (): Promise<string | null> => {
    // Anything already known gets first refusal, with a patient timeout.
    for (const url of [stored, ENV_BASE, localDefault].filter((u): u is string => Boolean(u))) {
      if (await probe(url, url === localDefault ? WARM_TIMEOUT_MS : COLD_TIMEOUT_MS)) {
        rememberBackend(url);
        return url;
      }
    }

    const list = await candidates();
    if (!list.length) return null;

    // Warm pass first: if any candidate is already awake we want it in seconds, not after a
    // cold-start wait on an earlier entry in the list.
    const warm = await Promise.all(list.map(async (url) => ((await probe(url, WARM_TIMEOUT_MS)) ? url : null)));
    const awake = warm.find((url): url is string => Boolean(url));
    if (awake) {
      rememberBackend(awake);
      return awake;
    }

    // Nothing awake — one of them may be a sleeping free instance. Waking them in parallel means
    // the wait is one cold start, not N of them.
    const cold = await Promise.all(list.map(async (url) => ((await probe(url, COLD_TIMEOUT_MS)) ? url : null)));
    const woken = cold.find((url): url is string => Boolean(url));
    if (woken) {
      rememberBackend(woken);
      return woken;
    }
    return null;
  })();

  return { initial, live, pinned: false };
}
