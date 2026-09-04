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
    await expect(page.locator(".levier")).toHaveCount(11);
    await expect(page.getByTestId("volume")).toHaveText("0,00 m³");
  });

  test("lowering the boom moves the teeth and starts the chrono", async ({ page }) => {
    await page.goto(URL);
    await page.selectOption("#chantier", "tranchee");
    await expect(page.locator("#titreChantier")).toHaveText("Tranchée de service");

    const avant = await page.getByTestId("profondeur").textContent();
    await page.keyboard.down("ArrowDown");
    await page.waitForTimeout(600);
    await page.keyboard.up("ArrowDown");

    await expect
      .poll(() => page.getByTestId("profondeur").textContent())
      .not.toBe(avant);
    await expect(page.getByTestId("chrono")).not.toHaveText("0:00");
  });

  test("dragging the bucket through the ground loads it and moves soil", async ({ page }) => {
    await page.goto(URL);
    await page.selectOption("#chantier", "bac");
    await page.click("#btnReset");

    // Bring the teeth into the ground: boom down while the arm stays out.
    await page.keyboard.down("ArrowDown");
    await page.waitForTimeout(1400);
    await page.keyboard.up("ArrowDown");
    // Curl the bucket back to scoop.
    await page.keyboard.down("KeyZ");
    await page.waitForTimeout(700);
    await page.keyboard.up("KeyZ");

    await expect
      .poll(async () => parseInt((await page.locator("#valGodet").textContent()) || "0", 10), { timeout: 5000 })
      .toBeGreaterThan(0);
  });

  test("the terrain survives a reload", async ({ page }) => {
    await page.goto(URL);
    await page.selectOption("#chantier", "nivellement");
    await page.keyboard.down("ArrowDown");
    await page.waitForTimeout(1400);
    await page.keyboard.up("ArrowDown");
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
