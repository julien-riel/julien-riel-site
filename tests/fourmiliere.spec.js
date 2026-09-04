import { test, expect } from "@playwright/test";

const URL = "/projets/fourmiliere/";

async function attendreSimulation(page) {
  await page.waitForFunction(() => window.fourmiliere && window.fourmiliere.etat && window.fourmiliere.etat.fourmis.length > 0);
}

test.describe("Simulateur de fourmilière", () => {
  test("boots an established colony with a queen, workers and brood", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);
    await attendreSimulation(page);

    const canvas = page.locator("#scene");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    expect(box.width).toBeGreaterThan(300);
    expect(box.height).toBeGreaterThan(300);

    await expect(page.getByTestId("reine")).toContainText("présente");
    const ouvrieres = parseInt((await page.getByTestId("ouvrieres").textContent()).replace(/\D/g, ""), 10);
    expect(ouvrieres).toBeGreaterThan(200);
    await expect(page.getByTestId("date")).toContainText("juin");
    await expect(page.locator("#journal li")).not.toHaveCount(0);
    expect(erreurs).toEqual([]);
  });

  test("clicking on the queen opens her information card", async ({ page }) => {
    await page.goto(URL);
    await attendreSimulation(page);
    await page.locator('[data-vitesse="0"]').click();
    // Centrer la caméra sur la reine, puis cliquer à sa position écran.
    const pos = await page.evaluate(() => {
      const S = window.fourmiliere.etat, cam = window.fourmiliere.cam;
      cam.x = S.reine.x; cam.y = S.reine.y; cam.z = 6;
      return null;
    });
    expect(pos).toBeNull();
    await page.waitForTimeout(150);
    const ecran = await page.evaluate(() => window.fourmiliere.ecran(window.fourmiliere.etat.reine));
    await page.mouse.click(ecran.x, ecran.y);
    await expect(page.getByTestId("titre-fourmi")).toHaveText("La reine");
    await expect(page.locator("#fTable")).toContainText("Spermathèque");
    await expect(page.locator("#ficheContenu")).toBeVisible();
    await page.click("#btnFermerFiche");
    await expect(page.locator("#ficheVide")).toBeVisible();
  });

  test("clicking on a worker shows her task and state", async ({ page }) => {
    await page.goto(URL);
    await attendreSimulation(page);
    await page.locator('[data-vitesse="0"]').click();
    const ecran = await page.evaluate(() => {
      const S = window.fourmiliere.etat, cam = window.fourmiliere.cam;
      const f = S.fourmis.find((a) => a.caste === "ouvriere" && a.mode === "nid");
      cam.x = f.x; cam.y = f.y; cam.z = 8;
      return f.id;
    });
    expect(ecran).toBeGreaterThan(0);
    await page.waitForTimeout(150);
    const p = await page.evaluate((id) => window.fourmiliere.ecran(window.fourmiliere.etat.fourmis.find((a) => a.id === id)), ecran);
    await page.mouse.click(p.x, p.y);
    await expect(page.getByTestId("titre-fourmi")).toContainText("Ouvrière");
    await expect(page.locator("#fBadges span").first()).toHaveText(/Nourrice|Ouvrière d'entretien|Fourrageuse/);
    await expect(page.getByTestId("etat-fourmi")).not.toBeEmpty();
    await expect(page.locator("#fExplication")).not.toBeEmpty();
  });

  test("time runs and speed ×32 makes the calendar advance", async ({ page }) => {
    await page.goto(URL);
    await attendreSimulation(page);
    const heure = await page.getByTestId("heure").textContent();
    await expect.poll(() => page.getByTestId("heure").textContent(), { timeout: 10000 }).not.toBe(heure);
    await page.locator('[data-vitesse="3"]').click();
    const date = await page.getByTestId("date").textContent();
    await expect.poll(() => page.getByTestId("date").textContent(), { timeout: 20000 }).not.toBe(date);
  });

  test("the founding scenario starts with a lone queen and no workers", async ({ page }) => {
    await page.goto(URL);
    await attendreSimulation(page);
    await page.selectOption("#scenario", "fondation");
    await expect(page.getByTestId("ouvrieres")).toHaveText("0");
    await expect(page.getByTestId("reine")).toContainText("présente");
    await expect(page.getByTestId("date")).toContainText("juillet");
    await expect(page.locator("#journal li").first()).toContainText("vol nuptial");
  });

  test("the colony survives a reload", async ({ page }) => {
    await page.goto(URL);
    await attendreSimulation(page);
    await page.selectOption("#scenario", "fondation");
    // La sauvegarde est cadencée toutes les 5 s ; pagehide la force aussi.
    await page.waitForTimeout(5600);
    await page.reload();
    await attendreSimulation(page);
    await expect(page.locator("#scenario")).toHaveValue("fondation");
    await expect(page.getByTestId("ouvrieres")).toHaveText("0");
  });

  test("the help dialog opens from the button and closes", async ({ page }) => {
    await page.goto(URL);
    await page.click("#btnAide");
    await expect(page.locator("#aide")).toBeVisible();
    await expect(page.locator("#aide")).toContainText("polyéthisme");
    await page.click("#aide .fermer");
    await expect(page.locator("#aide")).toBeHidden();
  });
});
