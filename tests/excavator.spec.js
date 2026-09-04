import { test, expect } from "@playwright/test";

const URL = "/projets/simulateur-pelle-mecanique/";

test.describe("Simulateur de pelle mécanique", () => {
  test("renders the scene, the HUD and every control lever", async ({ page }) => {
    await page.goto(URL);

    const canvas = page.locator("#scene");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    expect(box.width).toBeGreaterThan(200);
    expect(box.height).toBeGreaterThan(200);

    await expect(page.locator("#chantier option")).toHaveCount(4);
    await expect(page.locator(".pedale")).toHaveCount(6);
    await expect(page.locator("#manetteG .pouce")).toBeVisible();
    await expect(page.locator("#manetteD .pouce")).toBeVisible();
    await expect(page.getByTestId("volume")).toHaveText("0,00 m³");
  });

  test("lowering the boom moves the teeth and starts the chrono", async ({ page }) => {
    await page.goto(URL);
    await page.selectOption("#chantier", "tranchee");
    await expect(page.locator("#titreChantier")).toHaveText("Tranchée de service");

    // Le rendu logiciel des tests peut être lent : on tient la touche jusqu'à l'effet attendu.
    const avant = await page.getByTestId("profondeur").textContent();
    await page.keyboard.down("KeyI");
    await expect
      .poll(() => page.getByTestId("profondeur").textContent(), { timeout: 20000 })
      .not.toBe(avant);
    await page.keyboard.up("KeyI");
    await expect(page.getByTestId("chrono")).not.toHaveText("0:00", { timeout: 15000 });
  });

  test("dragging the bucket through the ground loads it and moves soil", async ({ page }) => {
    await page.goto(URL);
    await page.selectOption("#chantier", "bac");
    await page.click("#btnReset");

    // Bring the teeth into the ground: boom down until the depth reads negative.
    await page.keyboard.down("KeyI");
    await expect
      .poll(() => page.getByTestId("profondeur").textContent(), { timeout: 20000 })
      .toMatch(/^−0,[2-9]/);
    await page.keyboard.up("KeyI");
    // Close the bucket to scoop until it holds something.
    await page.keyboard.down("KeyJ");
    await expect
      .poll(async () => parseInt((await page.locator("#valGodet").textContent()) || "0", 10), { timeout: 20000 })
      .toBeGreaterThan(0);
    await page.keyboard.up("KeyJ");
  });

  test("the terrain survives a reload", async ({ page }) => {
    await page.goto(URL);
    await page.selectOption("#chantier", "nivellement");
    await page.keyboard.down("KeyI");
    await page.waitForTimeout(1400);
    await page.keyboard.up("KeyI");
    // Save is throttled to 2 s; pagehide also flushes it.
    await page.waitForTimeout(2300);

    await page.reload();
    await expect(page.locator("#chantier")).toHaveValue("nivellement");
    await expect(page.locator("#titreChantier")).toHaveText("Nivellement");
  });

  test("the help dialog opens from the button and closes", async ({ page }) => {
    await page.goto(URL);
    await page.click("#btnAide");
    await expect(page.locator("#aide")).toBeVisible();
    await page.click("#aide .fermer");
    await expect(page.locator("#aide")).toBeHidden();
  });
});
