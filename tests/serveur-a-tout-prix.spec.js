import { test, expect } from "@playwright/test";

const URL = "/projets/serveur-a-tout-prix/";

test.describe("Serveur à tout prix", () => {
  test("loads the home screen with eight missions and the finale", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);
    await expect(page.locator("h1.title-xl")).toHaveText("Serveur à tout prix");
    await expect(page.locator(".mtile")).toHaveCount(9);
    await expect(page.locator('.hud a[href="/projets/"]')).toBeVisible();
    expect(erreurs).toEqual([]);
  });

  test("Kevin's RTX 5090 fails mission 1, the reference setup passes it", async ({ page }) => {
    await page.goto(URL);
    await page.locator("#btn-start").click();
    await page.getByRole("button", { name: "Passer" }).click();
    await expect(page.locator(".mhead h1")).toHaveText("Le prototype de Kevin");

    await page.locator("#btn-config").click();
    await expect(page.locator(".console")).toContainText("Le modèle ne tient pas en mémoire");
    await page.locator("#btn-run").click();
    await page.locator("#btn-skip").click();
    await page.locator("#btn-debrief").click();
    await expect(page.locator(".verdict-stamp")).toHaveText("À REPRENDRE");

    await page.locator("#btn-retry").click();
    await page.locator('label[for="k-serverId-ws-pro6000"]').click();
    await page.locator('label[for="k-modelId-gptoss120"]').click();
    await page.locator('label[for="k-weightPrec-fp4"]').click();
    await page.locator('label[for="k-kvPrec-fp8"]').click();
    await page.locator("#btn-run").click();
    await page.locator("#btn-skip").click();
    await page.locator("#btn-debrief").click();
    await expect(page.locator(".verdict-stamp")).toHaveText("MISSION RÉUSSIE");
  });

  for (const [id, titre, depart] of [
    ["m7", "Les agents de l'équipe TI", "Agents peu fiables"],
    ["m8", "Des agents pour le bureau", "Agents peu fiables"],
  ]) {
    test(`${id} fails with its starting setup and passes with Ginette's solution`, async ({ page }) => {
      const erreurs = [];
      page.on("pageerror", (e) => erreurs.push(e.message));
      await page.goto(URL);
      await page.getByRole("button", { name: "Tout débloquer (mode animateur)" }).click();
      await page.locator(`[data-act="open-mission"][data-id="${id}"]`).click();
      await expect(page.locator(".mhead h1")).toHaveText(titre);

      await page.locator("#btn-config").click();
      await expect(page.locator(".console")).toContainText("Tâches d'agent menées à bout");
      await expect(page.locator(".console")).toContainText(depart);
      await page.locator("#btn-run").click();
      await page.locator("#btn-skip").click();
      await page.locator("#btn-debrief").click();
      await expect(page.locator(".verdict-stamp")).toHaveText("À REPRENDRE");

      await page.locator("#btn-retry").click();
      await page.locator("#btn-ref").click();
      await page.locator("#btn-run").click();
      await page.locator("#btn-skip").click();
      await page.locator("#btn-debrief").click();
      await expect(page.locator(".verdict-stamp")).toHaveText("MISSION RÉUSSIE");
      await page.locator("#btn-comite").click();
      await page.locator(".answer").first().waitFor();
      expect(erreurs).toEqual([]);
    });
  }
});
