import { test, expect } from "@playwright/test";

const URL = "/projets/livrer-a-tout-prix/";

// Nouvelle partie : accueil → « Commencer » → la note de service de la semaine 1 s'ouvre.
async function commencer(page, graine) {
  await page.goto(URL + "?neuf=1&graine=" + graine);
  await page.locator("#btn-commencer").click();
  await expect(page.locator(".note-service")).toBeVisible();
}
async function fermerNote(page) {
  await page.locator("#btn-note-compris").click();
  await expect(page.locator(".note-service")).toHaveCount(0);
}

test.describe("Livrer à tout prix", () => {
  test("l'accueil se charge sans erreur et mène à l'open space", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?neuf=1&graine=1");
    await expect(page.locator("h1.titre-jeu")).toHaveText("Livrer à tout prix");
    await expect(page.locator('.hud a[href="/projets/"]')).toBeVisible();
    await page.locator("#btn-commencer").click();
    await expect(page.locator('[data-dev="mathieu"]')).toBeVisible();
    await expect(page.locator("[data-ilot]")).toHaveCount(7);
    expect(erreurs).toEqual([]);
  });

  test("une année complète se simule et mène au bilan", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await commencer(page, 2);
    await fermerNote(page);
    await page.locator("#btn-jour").click();
    await page.evaluate(() => window.LTP.app.simuler(60));
    await expect(page.locator(".profil-fin")).toBeVisible();
    expect(erreurs).toEqual([]);
  });

  test("chaque état de démonstration s'affiche sans erreur", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    for (const nom of ["debut", "t2-tendu", "triade-appro", "evenement", "comite", "fin-usine", "reflexions", "note"]) {
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

  test("cinq journées au bouton « +1 jour » font passer la semaine, et le HUD affiche le jour", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await commencer(page, 3);
    await fermerNote(page);
    const quand = page.locator("#hud .hud-sem");
    await expect(quand).toContainText("Semaine 1");
    await expect(quand).toContainText("lundi");
    await page.locator("#btn-jour").click();
    await expect(quand).toContainText("mardi");
    for (let i = 0; i < 4; i++) await page.locator("#btn-jour").click();
    await expect(quand).toContainText("Semaine 2");
    await expect(quand).toContainText("lundi");
    expect(await page.evaluate(() => window.LTP.app.etat.semaine)).toBe(1);
    await expect(page.locator(".tableau-rapport")).toContainText("semaine 1");
    expect(erreurs).toEqual([]);
  });

  test("la note de service de la semaine 1 mène au levier des licences", async ({ page }) => {
    await commencer(page, 4);
    const note = page.locator(".note-service");
    await expect(note.locator(".eyebrow").first()).toContainText("semaine 1");
    // La lecture reste en pause tant que la note est ouverte.
    await expect(page.locator("#btn-jour")).toBeDisabled();
    // Au début, seul le panneau « Utilisation de l'IA » est ouvert.
    await expect(page.locator("#panneau-feuille .onglet")).toHaveCount(1);
    await page.locator("#btn-note-voir").click();
    await expect(note).toHaveCount(0);
    const levier = page.locator('[data-levier="licences"]');
    await expect(levier).toBeVisible();
    await expect(levier).toHaveClass(/is-eclaire/);
    await expect(levier.locator('[aria-checked="true"]')).toBeFocused();
    await expect(page.locator("#btn-jour")).toBeEnabled();
  });

  test("un changement de levier fait réagir l'équipe dans « Ce qu'on entend », avec sa cause", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await commencer(page, 6);
    await fermerNote(page);
    await page.locator('[data-levier="licences"] [data-valeur="tous"]').click();
    const causes = page.locator(".echos .echo-cause--levier");
    await expect(causes.first()).toBeVisible();
    await expect(causes.filter({ hasText: "Licences Copilot" }).first()).toBeVisible();
    await expect(page.locator(".echos .echo.is-immediat").first()).toBeVisible();
    // Les réflexions du jour s'affichent aussi en bulles dans l'open space (la lecture est en pause).
    await expect(page.locator(".os-bulle--vue").first()).toBeVisible();
    expect(erreurs).toEqual([]);
  });

  test("une valeur pas encore permise est visible mais désactivée", async ({ page }) => {
    await page.goto(URL + "?demo=reflexions");
    await page.waitForFunction(() => window.LTP && window.LTP.app.etat);
    const trois = page.locator('[data-levier="autonomie"] [data-valeur="3"]');
    await expect(trois).toBeVisible();
    await expect(trois).toHaveAttribute("aria-disabled", "true");
    await expect(trois).toContainText("S14");
    await expect(trois).toBeDisabled();
    await trois.click({ force: true });
    await expect(trois).toHaveAttribute("aria-checked", "false");
    expect(await page.evaluate(() => window.LTP.q.levier(window.LTP.app.etat, "autonomie"))).toBe(1);
  });

  test("au téléphone, pas de défilement horizontal", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    for (const nom of ["t2-tendu", "reflexions", "note"]) {
      await page.goto(URL + "?demo=" + nom);
      await page.waitForFunction(() => window.LTP && window.LTP.app.etat);
      const largeur = await page.evaluate(() => document.documentElement.scrollWidth);
      expect(largeur).toBeLessThanOrEqual(375);
    }
  });
});
