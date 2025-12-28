/**
 * Main JavaScript for julien-riel.com
 */

import { initSearch } from "./search.js";

/**
 * Initialize Mermaid diagrams
 * Loads Mermaid from CDN and renders all .language-mermaid code blocks
 */
async function initMermaid() {
  const mermaidBlocks = document.querySelectorAll("pre.language-mermaid");
  if (mermaidBlocks.length === 0) return;

  try {
    const { default: mermaid } = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");

    mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      securityLevel: "loose",
    });

    const renderPromises = [];

    mermaidBlocks.forEach((pre, index) => {
      const code = pre.querySelector("code");
      if (!code) return;

      const diagramCode = code.textContent.trim();
      const container = document.createElement("div");
      container.className = "mermaid-diagram";
      container.id = `mermaid-diagram-${index}`;

      pre.replaceWith(container);

      const promise = mermaid.render(`mermaid-svg-${index}`, diagramCode)
        .then(({ svg }) => {
          container.innerHTML = svg;
        })
        .catch((error) => {
          console.error("Mermaid render error:", error);
          container.innerHTML = `<pre class="mermaid-error">Erreur de rendu Mermaid: ${error.message}</pre>`;
        });

      renderPromises.push(promise);
    });

    await Promise.all(renderPromises);
  } catch (error) {
    console.error("Failed to load Mermaid:", error);
  }
}

/**
 * Initialize PlantUML diagrams
 * Converts .language-plantuml code blocks to images using PlantUML server
 */
function initPlantUML() {
  const plantumlBlocks = document.querySelectorAll("pre.language-plantuml");
  if (plantumlBlocks.length === 0) return;

  plantumlBlocks.forEach((pre) => {
    const code = pre.querySelector("code");
    if (!code) return;

    const diagramCode = code.textContent.trim();

    // Encode PlantUML diagram using hex encoding (simpler and reliable)
    const encoded = plantumlHexEncode(diagramCode);
    const imgUrl = `https://www.plantuml.com/plantuml/svg/~h${encoded}`;

    const container = document.createElement("div");
    container.className = "plantuml-diagram";

    const img = document.createElement("img");
    img.src = imgUrl;
    img.alt = "PlantUML diagram";
    img.loading = "lazy";
    img.onerror = () => {
      container.innerHTML = `<pre class="plantuml-error">Erreur de chargement du diagramme PlantUML</pre>`;
    };

    container.appendChild(img);
    pre.replaceWith(container);
  });
}

/**
 * Encode PlantUML text as hexadecimal
 * PlantUML server accepts ~h prefix followed by hex-encoded text
 * @param {string} text - PlantUML source code
 * @returns {string} - Hex encoded string
 */
function plantumlHexEncode(text) {
  return Array.from(text)
    .map((char) => char.charCodeAt(0).toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Initialize Table of Contents scroll tracking
 * Highlights the current section in the TOC based on scroll position
 */
function initTOCScrollTracking() {
  const toc = document.querySelector(".toc");
  if (!toc) return;

  const tocLinks = toc.querySelectorAll("a");
  const headings = [];

  tocLinks.forEach((link) => {
    const id = link.getAttribute("href")?.slice(1);
    if (id) {
      const heading = document.getElementById(id);
      if (heading) headings.push({ id, element: heading, link });
    }
  });

  if (headings.length === 0) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const { id } = headings.find((h) => h.element === entry.target) || {};
        if (!id) return;

        const link = toc.querySelector(`a[href="#${id}"]`);
        if (entry.isIntersecting) {
          tocLinks.forEach((l) => l.classList.remove("active"));
          link?.classList.add("active");
        }
      });
    },
    {
      rootMargin: "-20% 0px -60% 0px",
      threshold: 0,
    }
  );

  headings.forEach(({ element }) => observer.observe(element));
}

/**
 * Initialize mobile menu toggle
 */
function initMobileMenu() {
  const menuButton = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".site-nav");

  if (!menuButton || !nav) return;

  menuButton.addEventListener("click", () => {
    const isExpanded = menuButton.getAttribute("aria-expanded") === "true";
    menuButton.setAttribute("aria-expanded", !isExpanded);
    nav.classList.toggle("is-open");
  });
}

/**
 * Initialize copy button for code blocks
 */
function initCodeCopyButtons() {
  const codeBlocks = document.querySelectorAll("pre[class*='language-']");

  codeBlocks.forEach((pre) => {
    // Skip if already has a copy button or is mermaid/plantuml
    if (pre.querySelector(".copy-button")) return;
    if (pre.classList.contains("language-mermaid") ||
        pre.classList.contains("language-plantuml")) return;

    const code = pre.querySelector("code");
    if (!code) return;

    const button = document.createElement("button");
    button.className = "copy-button";
    button.textContent = "Copier";
    button.type = "button";
    button.setAttribute("aria-label", "Copier le code");

    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copie!";
        setTimeout(() => {
          button.textContent = "Copier";
        }, 2000);
      } catch (err) {
        console.error("Failed to copy:", err);
      }
    });

    pre.style.position = "relative";
    pre.appendChild(button);
  });
}

// Initialize all features when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initMobileMenu();
  initMermaid();
  initPlantUML();
  initTOCScrollTracking();
  initCodeCopyButtons();
  initSearch();
});
