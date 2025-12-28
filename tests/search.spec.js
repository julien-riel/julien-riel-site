import { test, expect } from "@playwright/test";

test.describe("Search functionality", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("http://localhost:8081/");
  });

  test("search input is visible in the header", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await expect(searchInput).toBeVisible();
    await expect(searchInput).toHaveAttribute("placeholder", "Rechercher...");
  });

  test("search data is embedded in the page", async ({ page }) => {
    const searchData = page.locator("#search-data");
    await expect(searchData).toBeAttached();

    const content = await searchData.textContent();
    const data = JSON.parse(content);
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBeGreaterThan(0);
  });

  test("typing in search shows results dropdown", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("python");

    // Wait for results to appear
    await page.waitForSelector(".search-results.is-open", { timeout: 5000 });

    const results = page.locator(".search-results");
    await expect(results).toHaveClass(/is-open/);
  });

  test("search results contain matching articles", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("python");

    await page.waitForSelector(".search-result-item", { timeout: 5000 });

    const resultItems = page.locator(".search-result-item");
    const count = await resultItems.count();
    expect(count).toBeGreaterThan(0);

    // Check that results contain Python-related content
    const firstResultTitle = await resultItems.first().locator(".search-result-title").textContent();
    expect(firstResultTitle.toLowerCase()).toContain("python");
  });

  test("search for 'intelligence artificielle' returns results", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("intelligence artificielle");

    await page.waitForSelector(".search-result-item", { timeout: 5000 });

    const resultItems = page.locator(".search-result-item");
    const count = await resultItems.count();
    expect(count).toBeGreaterThan(0);
  });

  test("clicking a search result navigates to the article", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("python");

    await page.waitForSelector(".search-result-item", { timeout: 5000 });

    const firstResult = page.locator(".search-result-item").first();
    const href = await firstResult.getAttribute("href");

    await firstResult.click();
    await page.waitForURL(new RegExp(href.replace(/\//g, "\\/")));

    expect(page.url()).toContain(href);
  });

  test("search highlights matching terms", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("python");

    await page.waitForSelector(".search-result-item", { timeout: 5000 });

    const highlights = page.locator(".search-highlight");
    const count = await highlights.count();
    expect(count).toBeGreaterThan(0);
  });

  test("empty search clears results", async ({ page }) => {
    const searchInput = page.locator("#search-input");

    // First search
    await searchInput.fill("python");
    await page.waitForSelector(".search-results.is-open", { timeout: 5000 });

    // Clear search
    await searchInput.fill("");

    // Results should disappear
    const results = page.locator(".search-results");
    await expect(results).not.toHaveClass(/is-open/);
  });

  test("no results message for non-matching query", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("xyznonexistent123");

    await page.waitForSelector(".search-results.is-open", { timeout: 5000 });

    const noResults = page.locator(".search-no-results");
    await expect(noResults).toBeVisible();
  });

  test("keyboard navigation works in search results", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("python");

    await page.waitForSelector(".search-result-item", { timeout: 5000 });

    // Press down arrow
    await searchInput.press("ArrowDown");

    const firstResult = page.locator(".search-result-item").first();
    await expect(firstResult).toHaveClass(/is-focused/);
  });

  test("Escape key closes search results", async ({ page }) => {
    const searchInput = page.locator("#search-input");
    await searchInput.fill("python");

    await page.waitForSelector(".search-results.is-open", { timeout: 5000 });

    await searchInput.press("Escape");

    const results = page.locator(".search-results");
    await expect(results).not.toHaveClass(/is-open/);
  });
});
