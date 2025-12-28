/**
 * Search functionality using Lunr.js
 * Provides instant search results with keyboard navigation
 */

let searchIndex = null;
let searchData = null;
let lunrIndex = null;

/**
 * Initialize the search functionality
 * Loads the search index and sets up event listeners
 */
export async function initSearch() {
  const searchInput = document.getElementById("search-input");
  const searchResults = document.querySelector(".search-results");

  if (!searchInput || !searchResults) return;

  // Load Lunr.js from CDN
  await loadLunr();

  // Load the search index
  await loadSearchIndex();

  if (!lunrIndex) {
    console.error("Failed to initialize search index");
    return;
  }

  // Set up event listeners
  let debounceTimer = null;

  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      performSearch(e.target.value.trim(), searchResults);
    }, 150);
  });

  // Handle form submission
  searchInput.closest("form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    performSearch(searchInput.value.trim(), searchResults);
  });

  // Close results when clicking outside
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-container")) {
      searchResults.innerHTML = "";
      searchResults.classList.remove("is-open");
    }
  });

  // Keyboard navigation
  searchInput.addEventListener("keydown", (e) => {
    handleKeyboardNavigation(e, searchResults);
  });

  // Check for query parameter on page load
  const urlParams = new URLSearchParams(window.location.search);
  const query = urlParams.get("q");
  if (query) {
    searchInput.value = query;
    performSearch(query, searchResults);
  }
}

/**
 * Load Lunr.js from CDN
 */
async function loadLunr() {
  if (window.lunr) return;

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

/**
 * Load and build the search index
 */
async function loadSearchIndex() {
  try {
    // Get search data from inline script tag
    const searchDataElement = document.getElementById("search-data");
    if (!searchDataElement) {
      return;
    }

    const textContent = searchDataElement.textContent.trim();
    if (!textContent || textContent === "[]") {
      return;
    }

    searchData = JSON.parse(textContent);

    if (!searchData || searchData.length === 0) {
      return;
    }

    // Build the Lunr index
    lunrIndex = window.lunr(function() {
      this.ref("id");
      this.field("title", { boost: 10 });
      this.field("description", { boost: 5 });
      this.field("tags", { boost: 3 });
      this.field("content");

      // French language support (basic stemming)
      this.pipeline.remove(window.lunr.stemmer);
      this.pipeline.remove(window.lunr.stopWordFilter);

      searchData.forEach((doc) => {
        this.add({
          id: doc.id,
          title: doc.title,
          description: doc.description,
          tags: doc.tags.join(" "),
          content: doc.content,
        });
      });
    });

    // Create a lookup map for quick access to document data
    searchIndex = {};
    searchData.forEach((doc) => {
      searchIndex[doc.id] = doc;
    });
  } catch (error) {
    console.error("Error loading search index:", error);
  }
}

/**
 * Perform search and display results
 * @param {string} query - Search query
 * @param {HTMLElement} resultsContainer - Container for results
 */
function performSearch(query, resultsContainer) {
  if (!query || query.length < 2) {
    resultsContainer.innerHTML = "";
    resultsContainer.classList.remove("is-open");
    return;
  }

  if (!lunrIndex) {
    resultsContainer.innerHTML = '<div class="search-loading">Chargement...</div>';
    resultsContainer.classList.add("is-open");
    return;
  }

  try {
    // Perform the search with wildcard support
    const searchQuery = query
      .split(/\s+/)
      .filter(term => term.length > 0)
      .map(term => `${term}*`)
      .join(" ");

    const results = lunrIndex.search(searchQuery);

    if (results.length === 0) {
      resultsContainer.innerHTML = `
        <div class="search-no-results">
          Aucun resultat pour « ${escapeHtml(query)} »
        </div>
      `;
      resultsContainer.classList.add("is-open");
      return;
    }

    // Limit to top 10 results
    const topResults = results.slice(0, 10);

    const html = topResults
      .map((result, index) => {
        const doc = searchIndex[result.ref];
        if (!doc) return "";

        const excerpt = createExcerpt(doc.content, query);
        const highlightedTitle = highlightMatch(doc.title, query);

        return `
          <a href="${escapeHtml(doc.url)}" class="search-result-item" data-index="${index}">
            <div class="search-result-title">${highlightedTitle}</div>
            ${doc.description ? `<div class="search-result-description">${escapeHtml(doc.description)}</div>` : ""}
            <div class="search-result-excerpt">${excerpt}</div>
            <div class="search-result-meta">
              <time>${formatDate(doc.date)}</time>
              ${doc.tags.length > 0 ? `<span class="search-result-tags">${doc.tags.slice(0, 3).map(t => escapeHtml(t)).join(", ")}</span>` : ""}
            </div>
          </a>
        `;
      })
      .filter(Boolean)
      .join("");

    resultsContainer.innerHTML = `
      <div class="search-results-header">
        ${results.length} resultat${results.length > 1 ? "s" : ""} pour « ${escapeHtml(query)} »
      </div>
      <div class="search-results-list">${html}</div>
    `;
    resultsContainer.classList.add("is-open");
  } catch (error) {
    console.error("Search error:", error);
    resultsContainer.innerHTML = `
      <div class="search-error">Erreur de recherche</div>
    `;
    resultsContainer.classList.add("is-open");
  }
}

/**
 * Handle keyboard navigation in search results
 * @param {KeyboardEvent} e - Keyboard event
 * @param {HTMLElement} resultsContainer - Container for results
 */
function handleKeyboardNavigation(e, resultsContainer) {
  const items = resultsContainer.querySelectorAll(".search-result-item");
  if (items.length === 0) return;

  const currentIndex = [...items].findIndex((item) =>
    item.classList.contains("is-focused")
  );

  let newIndex = -1;

  switch (e.key) {
    case "ArrowDown":
      e.preventDefault();
      newIndex = currentIndex < items.length - 1 ? currentIndex + 1 : 0;
      break;
    case "ArrowUp":
      e.preventDefault();
      newIndex = currentIndex > 0 ? currentIndex - 1 : items.length - 1;
      break;
    case "Enter":
      if (currentIndex >= 0) {
        e.preventDefault();
        items[currentIndex].click();
      }
      return;
    case "Escape":
      resultsContainer.innerHTML = "";
      resultsContainer.classList.remove("is-open");
      return;
    default:
      return;
  }

  items.forEach((item) => item.classList.remove("is-focused"));
  if (newIndex >= 0) {
    items[newIndex].classList.add("is-focused");
    items[newIndex].scrollIntoView({ block: "nearest" });
  }
}

/**
 * Create an excerpt from content with query highlighting
 * @param {string} content - Full content
 * @param {string} query - Search query
 * @returns {string} - HTML excerpt
 */
function createExcerpt(content, query) {
  const maxLength = 150;
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);

  if (terms.length === 0) {
    return escapeHtml(content.slice(0, maxLength)) + "...";
  }

  // Find the first occurrence of any search term
  const lowerContent = content.toLowerCase();
  let firstMatch = content.length;

  for (const term of terms) {
    const index = lowerContent.indexOf(term);
    if (index !== -1 && index < firstMatch) {
      firstMatch = index;
    }
  }

  // Create excerpt around the match
  let start = Math.max(0, firstMatch - 30);
  let end = Math.min(content.length, start + maxLength);

  if (start > 0) start = content.indexOf(" ", start) + 1 || start;
  if (end < content.length) end = content.lastIndexOf(" ", end) || end;

  let excerpt = content.slice(start, end);
  if (start > 0) excerpt = "..." + excerpt;
  if (end < content.length) excerpt = excerpt + "...";

  return highlightMatch(excerpt, query);
}

/**
 * Highlight search terms in text
 * @param {string} text - Text to highlight
 * @param {string} query - Search query
 * @returns {string} - HTML with highlights
 */
function highlightMatch(text, query) {
  const terms = query.split(/\s+/).filter(t => t.length > 1);
  if (terms.length === 0) return escapeHtml(text);

  let result = escapeHtml(text);

  for (const term of terms) {
    const regex = new RegExp(`(${escapeRegex(term)})`, "gi");
    result = result.replace(regex, '<mark class="search-highlight">$1</mark>');
  }

  return result;
}

/**
 * Format date for display
 * @param {string} dateStr - ISO date string
 * @returns {string} - Formatted date
 */
function formatDate(dateStr) {
  const date = new Date(dateStr);
  return date.toLocaleDateString("fr-FR", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text
 */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Escape special regex characters
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text for regex
 */
function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
