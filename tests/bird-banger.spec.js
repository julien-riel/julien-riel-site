import { test, expect } from "@playwright/test";

const URL = "/projets/bird-banger/";

async function attendreJeu(page) {
  await page.waitForFunction(() => window.birdBanger && window.birdBanger.etat);
}

test.describe("Bird Banger", () => {
  test("loads the vineyard on September 1st with a full harvest and the starting budget", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);
    await attendreJeu(page);

    const canvas = page.locator("#scene");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    expect(box.width).toBeGreaterThan(300);
    expect(box.height).toBeGreaterThan(300);

    await expect(page.getByTestId("jour")).toHaveText("1 septembre");
    await expect(page.getByTestId("part")).toHaveText("100 %");
    await expect(page.getByTestId("budget")).toContainText("3");
    await expect(page.locator("#journal li").first()).toContainText("véraison");
    expect(erreurs).toEqual([]);
  });

  test("placing a cannon from the shop deducts its price and opens its card", async ({ page }) => {
    await page.goto(URL);
    await attendreJeu(page);
    await page.locator('[data-vitesse="0"]').click();
    await page.getByTestId("placer-simple").click();
    const p = await page.evaluate(() => window.birdBanger.ecran({ x: 200, y: 130 }));
    await page.mouse.click(p.x, p.y);
    await expect(page.getByTestId("fiche-titre")).toHaveText("Canon à propane simple");
    await expect(page.getByTestId("budget")).toHaveText(/2\s?550 \$/);
    const n = await page.evaluate(() => window.birdBanger.etat.canons.length);
    expect(n).toBe(1);
    await expect(page.locator("#journal li").first()).toContainText("posé");
  });

  test("a bang scares birds within range and raises their habituation", async ({ page }) => {
    await page.goto(URL);
    await attendreJeu(page);
    const r = await page.evaluate(() => {
      const B = window.birdBanger;
      B.etat.vitesse = 0;
      const c = B.placer("rotatif", 200, 130, true);
      c.prochain = Infinity; // pas de tir automatique pendant le test
      B.apparaitre("etourneau", 20, { x: 200, y: 130 });
      B.avancer(3); // le temps d'atterrir dans la vigne
      const avant = B.etat.oiseaux.filter((o) => o.etat === "mange" || o.etat === "vol").length;
      const effrayes = B.tirer(c);
      const enFuite = B.etat.oiseaux.filter((o) => o.etat === "fuite").length;
      return { avant, effrayes, enFuite, hab: c.hab.etourneau, tirs: B.etat.stats.tirs };
    });
    expect(r.avant).toBeGreaterThan(10);
    expect(r.effrayes).toBeGreaterThan(5);
    expect(r.enFuite).toBe(r.effrayes);
    expect(r.hab).toBeGreaterThan(0);
    expect(r.tirs).toBe(1);
  });

  test("turkeys ignore a weak cannon but flee a double detonation", async ({ page }) => {
    await page.goto(URL);
    await attendreJeu(page);
    const r = await page.evaluate(() => {
      const B = window.birdBanger;
      B.etat.vitesse = 0;
      const faible = B.placer("simple", 200, 200, true); // dindons à 80 m, en bout de portée
      const fort = B.placer("double", 200, 135, true);
      faible.prochain = fort.prochain = Infinity; // pas de tir automatique pendant le test
      B.apparaitre("dindon", 12, { x: 200, y: 120 });
      B.avancer(6);
      let faibles = 0, forts = 0;
      for (let k = 0; k < 6; k++) faibles += B.tirer(faible);
      for (let k = 0; k < 6; k++) forts += B.tirer(fort);
      return { faibles, forts };
    });
    expect(r.faibles).toBe(0);
    expect(r.forts).toBeGreaterThan(0);
  });

  test("an unprotected vineyard loses most of its harvest by the end of the season", async ({ page }) => {
    test.setTimeout(90000);
    await page.goto(URL);
    await attendreJeu(page);
    const r = await page.evaluate(() => {
      const B = window.birdBanger;
      B.etat.vitesse = 0;
      B.avancer(B.DUREE_JOUR * B.NB_JOURS + 1);
      return { part: B.recolteKg() / B.RECOLTE_TOTALE, fini: B.etat.fini, note: B.etat.bilan.note };
    });
    expect(r.fini).toBe(true);
    expect(r.part).toBeLessThan(0.6);
    expect(["D", "E"]).toContain(r.note);
    await expect(page.getByTestId("bilan")).toBeVisible();
    await expect(page.locator("#bilanTable")).toContainText("Récolte rentrée");
  });

  test("moving a cannon resets most of the habituation", async ({ page }) => {
    await page.goto(URL);
    await attendreJeu(page);
    const r = await page.evaluate(() => {
      const B = window.birdBanger;
      const c = B.placer("simple", 200, 130, true);
      c.hab.etourneau = 0.8;
      B.deplacer(c, 260, 130);
      return { hab: c.hab.etourneau, x: c.x };
    });
    expect(r.x).toBe(260);
    expect(r.hab).toBeLessThan(0.3);
  });

  test("firing near a neighbour's house angers them and eventually costs a fine", async ({ page }) => {
    await page.goto(URL);
    await attendreJeu(page);
    const r = await page.evaluate(() => {
      const B = window.birdBanger;
      const m = B.MAISONS[0];
      const c = B.placer("double", m.x + 30, m.y + 20, true);
      c.dir = 0;
      const budget = B.etat.budget;
      for (let k = 0; k < 40 && B.etat.amendes === 0; k++) B.tirer(c);
      return { amendes: B.etat.amendes, budget, apres: B.etat.budget };
    });
    expect(r.amendes).toBe(1);
    expect(r.apres).toBe(r.budget - 750);
    await expect(page.locator("#journal li").first()).toContainText("amende");
  });

  test("harvesting early ends the season with a report", async ({ page }) => {
    await page.goto(URL);
    await attendreJeu(page);
    await page.click("#btnVendanger");
    await expect(page.locator("#confirmeVendange")).toBeVisible();
    await page.getByTestId("confirmer-vendange").click();
    await expect(page.getByTestId("bilan")).toBeVisible();
    await expect(page.locator("#bilanTable")).toContainText("1 septembre");
    await expect(page.locator("#bilanNote")).toHaveText("A");
    await page.click("#btnRejouer");
    await expect(page.getByTestId("bilan")).toBeHidden();
    await expect(page.getByTestId("part")).toHaveText("100 %");
  });

  test("the help dialog opens from the button and closes", async ({ page }) => {
    await page.goto(URL);
    await page.click("#btnAide");
    await expect(page.locator("#aide")).toBeVisible();
    await expect(page.locator("#aide")).toContainText("accoutumance");
    await page.click("#aide .fermer");
    await expect(page.locator("#aide")).toBeHidden();
  });
});
