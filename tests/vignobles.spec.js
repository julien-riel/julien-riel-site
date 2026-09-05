import { test, expect } from "@playwright/test";

const URL = "/projets/vignobles-niagara/";

async function attendreCarte(page) {
  await page.waitForFunction(() => window.vignobles && window.vignobles.pret(), null, { timeout: 30000 });
}

test.describe("Vignobles de Niagara-on-the-Lake", () => {
  test("loads the lots, the wineries and the counters without errors", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);
    await attendreCarte(page);

    const lots = await page.evaluate(() => window.vignobles.lots());
    expect(lots).toBeGreaterThan(200);
    const hectares = await page.evaluate(() => window.vignobles.hectares());
    expect(hectares).toBeGreaterThan(1000);

    const vignobles = await page.evaluate(() => window.vignobles.vignobles());
    expect(vignobles.length).toBeGreaterThan(30);
    expect(vignobles).toContain("Inniskillin Wines Inc.");

    await expect(page.locator("#chiffre-lots")).toHaveText(String(lots));
    await expect(page.locator("#chiffre-vignobles")).toHaveText(String(vignobles.length));
    await expect(page.locator("#liste li")).toHaveCount(vignobles.length);
    expect(erreurs).toEqual([]);
  });

  test("picking a winery in the list opens its card and its lots", async ({ page }) => {
    await page.goto(URL);
    await attendreCarte(page);

    await page.locator("#recherche").fill("chateau");
    await expect(page.locator("#liste li")).toHaveCount(1);
    await page.locator("#liste .entree").first().click();

    await expect(page.locator("#fiche h2")).toHaveText("Chateau des Charmes");
    await expect(page.locator("#fiche .fiche-genre")).toHaveText("Vignoble");
    await expect(page.locator("#fiche")).toContainText("Lots rattachés");

    await page.locator("#fiche .fiche-lots summary").click();
    const lots = page.locator("#fiche .fiche-lots li button");
    expect(await lots.count()).toBeGreaterThan(0);
    await lots.first().click();
    await expect(page.locator("#fiche .fiche-genre")).toHaveText("Lot viticole");
    await expect(page.locator("#fiche")).toContainText("Superficie");

    await page.keyboard.press("Escape");
    await expect(page.locator("#fiche")).toHaveClass(/vide/);
  });

  test("the colour modes, the layers and the basemaps switch", async ({ page }) => {
    await page.goto(URL);
    await attendreCarte(page);

    await page.click('[data-teinte="surface"]');
    expect(await page.evaluate(() => window.vignobles.teinte())).toBe("surface");
    await expect(page.locator("#legende li").first()).toContainText("ha");

    await page.click('[data-teinte="cepage"]');
    expect(await page.evaluate(() => window.vignobles.teinte())).toBe("cepage");
    await expect(page.locator("#legende")).toContainText("Cabernet sauvignon");
    await expect(page.locator("#legende")).toContainText("Cépage non relevé");

    await page.click('[data-couche="trame"]');
    expect(await page.evaluate(() => window.vignobles.couches().trame)).toBe(true);
    await expect(page.locator('[data-couche="trame"]')).toHaveAttribute("aria-pressed", "true");

    await page.click('[data-couche="sols"]');
    await page.waitForFunction(() => window.vignobles.solsCharges(), null, { timeout: 30000 });
    await expect(page.locator("#legende")).toContainText("Texture du sol");

    await page.click('[data-fond="clair"]');
    expect(await page.evaluate(() => window.vignobles.fond())).toBe("clair");
    await expect(page.locator('[data-fond="sombre"]')).toHaveAttribute("aria-pressed", "false");
  });

  test("the help dialog explains where the polygons come from", async ({ page }) => {
    await page.goto(URL);
    await attendreCarte(page);
    await page.click("#ouvrir-aide");
    const aide = page.locator("#aide");
    await expect(aide).toBeVisible();
    await expect(aide).toContainText("OpenStreetMap");
    await expect(aide).toContainText("n'est pas une donnée ouverte");
    await aide.locator("button").click();
    await expect(aide).toBeHidden();
  });
});
