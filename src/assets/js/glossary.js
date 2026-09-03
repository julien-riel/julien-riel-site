/**
 * Glossary filtering
 * Progressive enhancement: the full list is server-rendered, this only narrows it.
 */

/**
 * Lowercase and strip accents so "evaluation" matches "évaluation".
 * @param {string} value
 * @returns {string}
 */
function normalize(value) {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

export function initGlossary() {
  const toolbar = document.querySelector("[data-glossary-toolbar]");
  const input = document.querySelector("[data-glossary-input]");
  const status = document.querySelector("[data-glossary-status]");
  const empty = document.querySelector("[data-glossary-empty]");

  if (!toolbar || !input) return;

  const root = document.querySelector("[data-glossary]");
  const sections = Array.from(document.querySelectorAll("[data-glossary-section]"));
  const entries = Array.from(document.querySelectorAll("[data-glossary-entry]")).map((el) => ({
    el,
    text: normalize(el.textContent),
  }));

  if (entries.length === 0) return;

  // Only reveal the filter once we know it will work.
  toolbar.hidden = false;

  const countLabel = status?.dataset.label || "";

  const render = (query) => {
    const needle = normalize(query.trim());
    let visible = 0;

    entries.forEach(({ el, text }) => {
      const match = needle === "" || text.includes(needle);
      el.hidden = !match;
      if (match) visible += 1;
    });

    sections.forEach((section) => {
      const hasVisible = section.querySelector("[data-glossary-entry]:not([hidden])");
      section.hidden = !hasVisible;
    });

    // Per-section counts describe the whole glossary, so hide them while filtering.
    root?.classList.toggle("is-filtering", needle !== "");

    if (empty) empty.hidden = visible > 0;
    if (status) {
      status.textContent = needle === "" ? "" : `${visible} ${countLabel}`.trim();
    }
  };

  let debounce = null;
  input.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => render(input.value), 120);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      input.value = "";
      render("");
    }
  });
}
