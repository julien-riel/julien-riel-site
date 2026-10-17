import { test, expect } from "@playwright/test";

const URL = "/projets/livrer-a-tout-prix/";

test.describe("Livrer à tout prix", () => {
  test("l'accueil se charge sans erreur et mène à l'open space", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?neuf=1&graine=1");
    await expect(page.locator("h1.titre-jeu")).toHaveText("Livrer à tout prix");
    await expect(page.locator('.hud a[href="/projets/"]')).toBeVisible();
    await page.locator("#btn-commencer").click();
    await expect(page.locator('[data-dev="mathieu"]')).toBeVisible();
    await expect(page.locator('[data-ilot]')).toHaveCount(7);
    expect(erreurs).toEqual([]);
  });

  test("une année complète se simule et mène au bilan", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?neuf=1&graine=2");
    await page.locator("#btn-commencer").click();
    await page.locator("#btn-semaine").click();
    await page.evaluate(() => window.LTP.app.simuler(60));
    await expect(page.locator(".profil-fin")).toBeVisible();
    expect(erreurs).toEqual([]);
  });

  test("chaque état de démonstration s'affiche sans erreur", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    for (const nom of ["debut", "t2-tendu", "triade-appro", "evenement", "comite", "fin-usine"]) {
      await page.goto(URL + "?demo=" + nom);
      await page.waitForFunction(() => window.LTP && window.LTP.app.etat);
    }
    await page.goto(URL + "?demo=evenement");
    await expect(page.locator(".carte-evenement")).toBeVisible();
    await page.locator('[data-choix="decouper"]').click();
    await expect(page.locator(".carte-evenement")).toHaveCount(0);
    expect(erreurs).toEqual([]);
  });

  test("au comité de fin de T1, chaque réponse affiche son texte", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?neuf=1&graine=5");
    await page.evaluate(() => {
      const LTP = window.LTP;
      LTP.app.dispatch({ type: "phase", phase: "planification" });
      LTP.app.dispatch({ type: "phase", phase: "execution" });
      LTP.app.demarrer(LTP.moteur.simuler(LTP.app.etat, 13).etat);
    });
    const reponses = page.locator(".comite .reponses > *");
    await expect(reponses).toHaveCount(3);
    for (const texte of await reponses.allTextContents()) expect(texte.trim().length).toBeGreaterThan(10);
    expect(erreurs).toEqual([]);
  });

  test("au téléphone, pas de défilement horizontal", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    await page.goto(URL + "?demo=t2-tendu");
    await page.waitForFunction(() => window.LTP && window.LTP.app.etat);
    const largeur = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(largeur).toBeLessThanOrEqual(375);
  });
});
