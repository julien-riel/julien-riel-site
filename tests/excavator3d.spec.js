import { test, expect } from "@playwright/test";

const URL = "/projets/simulateur-pelle-3d/";

test.describe("Simulateur de pelle 3D", () => {
  test("boots the WebGL scene and shows the cab HUD and both joysticks", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);

    await expect(page.locator("#chargement")).toHaveCount(0, { timeout: 15000 });
    await expect(page.locator("#manetteG .pouce")).toBeVisible();
    await expect(page.locator("#manetteD .pouce")).toBeVisible();
    await expect(page.locator(".pedale")).toHaveCount(6);
    await expect(page.locator("#chantier option")).toHaveCount(3);
    await expect(page.getByTestId("volume")).toHaveText("0,00 m³");
    expect(erreurs).toEqual([]);
  });

  test("the right joystick lowers the boom until the teeth touch the ground", async ({ page }) => {
    await page.goto(URL);
    await expect(page.locator("#chargement")).toHaveCount(0, { timeout: 15000 });
    await page.selectOption("#chantier", "fosse");
    await expect(page.getByTestId("avancement")).toHaveText("0 %");

    const avant = await page.getByTestId("profondeur").textContent();
    const m = page.locator("#manetteD");
    const box = await m.boundingBox();
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx, cy - 50, { steps: 5 });   // pousser : baisser la flèche
    await page.waitForTimeout(1500);
    await page.mouse.up();

    await expect.poll(() => page.getByTestId("profondeur").textContent()).not.toBe(avant);
    await expect(page.locator("#valChrono")).not.toHaveText("0:00");
  });

  test("keyboard digging fills the bucket", async ({ page }) => {
    await page.goto(URL);
    await expect(page.locator("#chargement")).toHaveCount(0, { timeout: 15000 });
    await page.selectOption("#chantier", "bac");
    await page.click("#btnReset");

    // Le rendu logiciel des tests est lent : on tient chaque touche jusqu'à l'effet attendu
    // plutôt qu'un temps fixe.
    await page.keyboard.down("KeyI");          // flèche vers le bas, jusqu'à ce que les dents entrent dans le sol
    await expect
      .poll(() => page.getByTestId("profondeur").textContent(), { timeout: 20000 })
      .toMatch(/^−0,[2-9]/);
    await page.keyboard.up("KeyI");

    await page.keyboard.down("KeyS");          // rentrer le balancier : les dents raclent
    await expect
      .poll(async () => parseInt((await page.getByTestId("godet").textContent()) || "0", 10), { timeout: 20000 })
      .toBeGreaterThan(0);
    await page.keyboard.up("KeyS");
  });

  test("the help dialog opens and closes", async ({ page }) => {
    await page.goto(URL);
    await expect(page.locator("#chargement")).toHaveCount(0, { timeout: 15000 });
    await page.click("#btnAide");
    await expect(page.locator("#aide")).toBeVisible();
    await page.click("#aide .fermer");
    await expect(page.locator("#aide")).toBeHidden();
  });
});
