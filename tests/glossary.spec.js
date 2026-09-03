import { test, expect } from "@playwright/test";

test.describe("Glossary page", () => {
  test("renders every term with a source link", async ({ page }) => {
    await page.goto("/glossaire/");

    const entries = page.locator("[data-glossary-entry]");
    await expect(entries).toHaveCount(75);

    const sources = page.locator(".glossary-source");
    await expect(sources).toHaveCount(75);
  });

  test("filter narrows the list and reports the count", async ({ page }) => {
    await page.goto("/glossaire/");

    const filter = page.locator("[data-glossary-input]");
    await expect(filter).toBeVisible();

    await filter.fill("jeton");

    // The filter is debounced, so wait for the list to actually narrow.
    const visibleEntries = page.locator("[data-glossary-entry]:not([hidden])");
    await expect(visibleEntries).not.toHaveCount(75);

    await expect(page.locator("#jeton")).toBeVisible();
    const visible = await visibleEntries.count();
    expect(visible).toBeGreaterThan(0);

    await expect(page.locator("[data-glossary-status]")).toContainText(String(visible));
  });

  test("filter ignores accents", async ({ page }) => {
    await page.goto("/glossaire/");

    const filter = page.locator("[data-glossary-input]");
    await filter.fill("evaluation");

    await expect(page.locator("#eval")).toBeVisible();
    await expect(page.locator("#agent")).toBeHidden();
  });

  test("filter shows an empty state when nothing matches", async ({ page }) => {
    await page.goto("/glossaire/");

    await page.locator("[data-glossary-input]").fill("zzzzzz");

    await expect(page.locator("[data-glossary-empty]")).toBeVisible();
    await expect(page.locator("[data-glossary-entry]:not([hidden])")).toHaveCount(0);
  });

  test("language switch maps /glossaire/ to /en/glossary/", async ({ page }) => {
    await page.goto("/glossaire/");
    await page.locator(".lang-switch").click();

    await expect(page).toHaveURL(/\/en\/glossary\/$/);
    await expect(page.locator("[data-glossary-entry]")).toHaveCount(75);

    await page.locator(".lang-switch").click();
    await expect(page).toHaveURL(/\/glossaire\/$/);
  });
});
