import { test, expect } from "@playwright/test";

const URL = "/projets/rouges-contre-bleus/";

test.describe("Rouges contre Bleus", () => {
  test("is listed on the projects page", async ({ page }) => {
    await page.goto("/projets/");
    await expect(page.locator(`a.site-link[href="${URL}"]`)).toContainText("Rouges contre Bleus");
  });

  test("boots the menu and the WebGL scene without errors", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    const manquants = [];
    page.on("response", (r) => {
      if (r.status() >= 400) manquants.push(`${r.status()} ${r.url()}`);
    });
    await page.goto(URL);

    await expect(page).toHaveTitle("Rouges contre Bleus");
    await expect(page.locator("canvas")).toHaveCount(1, { timeout: 15000 });
    await expect(page.getByRole("button").first()).toBeVisible({ timeout: 15000 });
    expect(erreurs).toEqual([]);
    expect(manquants).toEqual([]);
  });
});
