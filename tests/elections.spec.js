import { test, expect } from "@playwright/test";

const URL = "/elections-quebec-2026/";

test.describe("Élections Québec 2026 page", () => {
  test("renders one card per party, each with a platform badge", async ({ page }) => {
    await page.goto(URL);

    await expect(page.locator(".elec-party")).toHaveCount(5);
    await expect(page.locator(".elec-party .elec-badge")).toHaveCount(5);
    await expect(page.locator(".elec-party-short").first()).toHaveText("CAQ");
  });

  test("every theme states a position for the parties it lists", async ({ page }) => {
    await page.goto(URL);

    const themes = page.locator(".elec-theme");
    await expect(themes).toHaveCount(9);

    // A position with no resolved party would render an empty short name.
    const shorts = await page.locator(".elec-position-short").allTextContents();
    expect(shorts.length).toBeGreaterThan(0);
    for (const short of shorts) {
      expect(short.trim()).not.toBe("");
    }
  });

  test("bars are scaled to their own leader, never past 100 %", async ({ page }) => {
    await page.goto(URL);

    const widths = await page.locator(".elec-bar-fill").evaluateAll((nodes) =>
      nodes.map((n) => parseFloat(n.style.width))
    );
    expect(widths.length).toBeGreaterThan(0);
    for (const w of widths) {
      expect(w).toBeGreaterThan(0);
      expect(w).toBeLessThanOrEqual(100);
    }
    // Each chart's leader fills the track.
    expect(widths.filter((w) => w === 100).length).toBeGreaterThanOrEqual(4);
  });

  test("both ridings list their 2022 results and their 2026 candidates", async ({ page }) => {
    await page.goto(URL);

    for (const id of ["vachon", "pierre-laporte"]) {
      const riding = page.locator(`#${id}`);
      await expect(riding).toBeVisible();
      await expect(riding.locator(".elec-candidate")).toHaveCount(5);
      expect(await riding.locator(".elec-bar-row--named").count()).toBeGreaterThanOrEqual(7);
    }

    await expect(page.locator("#vachon .elec-elected")).toHaveCount(1);
  });

  test("section nav anchors all resolve to a section on the page", async ({ page }) => {
    await page.goto(URL);

    const hrefs = await page.locator(".elec-nav a").evaluateAll((links) =>
      links.map((a) => a.getAttribute("href"))
    );
    expect(hrefs.length).toBe(9);
    for (const href of hrefs) {
      await expect(page.locator(href)).toHaveCount(1);
    }
  });

  test("has no English counterpart, so it skips hreflang and links to the EN home", async ({ page }) => {
    await page.goto(URL);

    await expect(page.locator("link[rel=alternate][hreflang]")).toHaveCount(0);
    await expect(page.locator(".lang-switch")).toHaveAttribute("href", "/en/");
  });
});
