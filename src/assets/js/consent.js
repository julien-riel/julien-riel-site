/**
 * Consent banner + GA4 event tracking.
 *
 * GDPR model: analytics_storage is denied by default (set in analytics.njk).
 * We only call gtag('consent', 'update', ...) once the user explicitly accepts.
 * Choice is persisted in localStorage under `consent.analytics` with value
 * 'granted' | 'denied'. Banner only shows when no choice has been recorded.
 */

const STORAGE_KEY = "consent.analytics";
const SCROLL_THRESHOLDS = [25, 50, 75, 100];

function gtagSafe() {
  if (typeof window.gtag === "function") {
    window.gtag.apply(null, arguments);
  }
}

function getConsent() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch (e) {
    return null;
  }
}

function setConsent(value) {
  try {
    localStorage.setItem(STORAGE_KEY, value);
  } catch (e) {}
}

function hideBanner(banner) {
  banner.hidden = true;
  banner.classList.remove("is-visible");
}

function showBanner(banner) {
  banner.hidden = false;
  requestAnimationFrame(() => banner.classList.add("is-visible"));
}

export function initConsent() {
  const banner = document.getElementById("consent-banner");
  if (!banner) return;

  const existing = getConsent();
  if (!existing) {
    showBanner(banner);
  }

  banner.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-consent]");
    if (!btn) return;
    const choice = btn.getAttribute("data-consent");
    if (choice === "accept") {
      setConsent("granted");
      gtagSafe("consent", "update", { analytics_storage: "granted" });
    } else {
      setConsent("denied");
      gtagSafe("consent", "update", { analytics_storage: "denied" });
    }
    hideBanner(banner);
  });
}

/**
 * Track custom events. gtag() buffers events until consent is granted,
 * so we can call this unconditionally — no data leaves the browser until then.
 */
export function initAnalyticsEvents() {
  if (typeof window.gtag !== "function") return;

  initScrollDepth();
  initOutboundLinks();
  initLangSwitch();
  initRssClicks();
}

function initScrollDepth() {
  const fired = new Set();
  let ticking = false;

  function check() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (docHeight <= 0) return;
    const percent = Math.round((scrollTop / docHeight) * 100);
    SCROLL_THRESHOLDS.forEach((threshold) => {
      if (percent >= threshold && !fired.has(threshold)) {
        fired.add(threshold);
        gtagSafe("event", "scroll_depth", {
          percent_scrolled: threshold,
          page_path: location.pathname,
        });
      }
    });
    ticking = false;
  }

  window.addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        requestAnimationFrame(check);
        ticking = true;
      }
    },
    { passive: true }
  );
}

function initOutboundLinks() {
  const host = location.hostname;
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[href]");
    if (!link) return;
    const href = link.getAttribute("href");
    if (!href) return;
    let url;
    try {
      url = new URL(href, location.href);
    } catch (err) {
      return;
    }
    if (url.hostname && url.hostname !== host) {
      gtagSafe("event", "click_outbound", {
        link_url: url.href,
        link_domain: url.hostname,
        link_text: (link.textContent || "").trim().slice(0, 100),
      });
    }
  });
}

function initLangSwitch() {
  const switcher = document.querySelector("[data-lang-switch], .lang-switch, a[hreflang]");
  if (!switcher) return;
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[hreflang]");
    if (!link) return;
    gtagSafe("event", "language_switch", {
      target_language: link.getAttribute("hreflang"),
      from_path: location.pathname,
    });
  });
}

function initRssClicks() {
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[href$='feed.xml'], a[href*='/feed.xml']");
    if (!link) return;
    gtagSafe("event", "rss_click", {
      link_url: link.href,
    });
  });
}
