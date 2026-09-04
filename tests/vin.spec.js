import { test, expect } from "@playwright/test";

const URL = "/projets/du-cep-a-la-bouteille/";

async function attendrePage(page) {
  await page.waitForFunction(() => window.vin && document.querySelectorAll(".stade").length > 0);
}

test.describe("Du cep à la bouteille", () => {
  test("renders the red wine path with its workshops and no errors", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);
    await attendrePage(page);

    await expect(page.getByTestId("etiquette")).toHaveText("Rouge");
    const stades = await page.evaluate(() => window.vin.stades());
    expect(stades[0]).toBe("plantation");
    expect(stades[stades.length - 1]).toBe("bouteille");
    expect(stades).toContain("eraflage");
    expect(stades).toContain("pressurage-rouge");
    expect(stades).not.toContain("debourbage");
    await expect(page.locator("#etapes li")).toHaveCount(stades.length);
    await expect(page.locator(".atelier")).toHaveCount(10);
    expect(erreurs).toEqual([]);
  });

  test("switching to white reorders the cellar steps and is remembered", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.click('#choixStyle [data-style="blanc"]');
    await expect(page.getByTestId("etiquette")).toHaveText("Blanc");
    const stades = await page.evaluate(() => window.vin.stades());
    expect(stades.indexOf("pressurage-blanc")).toBeLessThan(stades.indexOf("fermentation"));
    expect(stades).toContain("debourbage");
    expect(stades).not.toContain("eraflage");
    await expect(page.locator("#s-fermentation h2")).toHaveText("Fermentation alcoolique");
    await page.reload();
    await attendrePage(page);
    await expect(page.getByTestId("etiquette")).toHaveText("Blanc");
    await page.click('#choixStyle [data-style="rose"]');
    expect(await page.evaluate(() => window.vin.stades())).toContain("maceration-rose");
  });

  test("the maturity workshop feeds the fermentation sugar", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.locator("#mJours").fill("20");
    const sucre = parseInt((await page.getByTestId("sucre").textContent()).replace(/\D/g, ""), 10);
    expect(sucre).toBeGreaterThan(100);
    expect(sucre).toBeLessThan(200);
    await page.click("#btnVendanger");
    const valeur = await page.locator("#fSucre").inputValue();
    expect(Math.abs(parseInt(valeur, 10) - sucre)).toBeLessThanOrEqual(5);
  });

  test("a cooled red tank ferments dry in about a week", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.evaluate(() => window.vin.ateliers.fermentation.avancer(24 * 20));
    await expect(page.getByTestId("etat-fermentation")).toContainText("Vin sec");
    await expect(page.getByTestId("alcool")).toContainText("13,0");
    const jours = await page.evaluate(() => window.vin.ateliers.fermentation.sim.heure / 24);
    expect(jours).toBeGreaterThan(5);
    expect(jours).toBeLessThan(12);
    await expect(page.getByTestId("densite-ferm")).toHaveText(/0\.99\d/);
  });

  test("without cooling the tank overheats and the fermentation sticks", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.evaluate(() => {
      window.vin.ateliers.fermentation.regler("thermo", false);
      window.vin.ateliers.fermentation.avancer(24 * 20);
    });
    await expect(page.getByTestId("etat-fermentation")).toContainText("chaleur");
    const sim = await page.evaluate(() => {
      const s = window.vin.ateliers.fermentation.sim;
      return { fini: s.fini, S: s.S, tmax: Math.max(...s.serie.map((p) => p.T)) };
    });
    expect(sim.fini).toBe("chaleur");
    expect(sim.S).toBeGreaterThan(20);
    expect(sim.tmax).toBeGreaterThan(35);
  });

  test("the simulator runs on its own when launched", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.selectOption("#fVitesse", "240");
    await page.click("#fLancer");
    await expect.poll(() => page.evaluate(() => window.vin.ateliers.fermentation.sim.heure), { timeout: 15000 }).toBeGreaterThan(48);
    await page.click("#fReinit");
    expect(await page.evaluate(() => window.vin.ateliers.fermentation.sim.heure)).toBe(0);
  });

  test("the vine cycle names the phase for a given date", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    const phase = page.getByTestId("phase");
    await page.locator("#cJour").fill("160");
    await expect(phase).toHaveText("Floraison");
    await page.locator("#cJour").fill("20");
    await expect(phase).toHaveText("Dormance");
    await page.locator("#cJour").fill("120");
    await expect(phase).toHaveText("Croissance");
    await page.check("#cQuebec");
    await expect(phase).toHaveText("Pleurs");
  });

  test("the help dialog opens with H and closes", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.keyboard.press("h");
    await expect(page.locator("#aide")).toBeVisible();
    await expect(page.locator("#aide")).toContainText("pressurage");
    await page.click("#aide .fermer");
    await expect(page.locator("#aide")).toBeHidden();
  });
});
