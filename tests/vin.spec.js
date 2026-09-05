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
    expect(stades[0]).toBe("climat");
    expect(stades.slice(0, 6)).toEqual(["climat", "geographie", "sol", "vie-du-sol", "rechauffement", "plantation"]);
    expect(stades.indexOf("eau-chaleur")).toBe(stades.indexOf("maturite") - 1);
    expect(stades.indexOf("assemblage")).toBeLessThan(stades.indexOf("mise"));
    expect(stades[stades.length - 1]).toBe("bouteille");
    expect(stades).toContain("eraflage");
    expect(stades).toContain("pressurage-rouge");
    expect(stades).not.toContain("debourbage");
    await expect(page.locator("#etapes li")).toHaveCount(stades.length);
    await expect(page.locator(".atelier")).toHaveCount(17);
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

  test("the terroir workshops react to climate, slope, soil and soil life", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    // climat : 16,5 °C de moyenne → ≈ 1 391 degrés-jours, région II ; 15,5 °C → ≈ 1 177, région Ib
    await expect(page.getByTestId("winkler")).toContainText("II");
    await page.locator("#kTemp").fill("15.5");
    await expect(page.getByTestId("winkler")).toContainText("Ib");
    const dj = parseInt((await page.getByTestId("degres-jours").textContent()).replace(/\D/g, ""), 10);
    expect(dj).toBeGreaterThan(1100);
    expect(dj).toBeLessThan(1250);
    // coteau : un versant sud reçoit plus d'énergie que le plat, un versant nord moins
    await page.selectOption("#gExpo", "S");
    await page.locator("#gPente").fill("20");
    const sud = parseInt((await page.getByTestId("energie").textContent()).replace(/\D/g, ""), 10);
    expect(sud).toBeGreaterThan(105);
    await page.selectOption("#gExpo", "N");
    const nord = parseInt((await page.getByTestId("energie").textContent()).replace(/\D/g, ""), 10);
    expect(nord).toBeLessThan(90);
    // sol : la craie impose un porte-greffe anti-chlorose ; un limon profond ne contraint jamais la vigne
    await page.selectOption("#sType", "craie");
    await expect(page.getByTestId("lecture-sol")).toContainText("chlorotiques");
    await page.selectOption("#sType", "limon");
    await page.locator("#sProf").fill("200");
    await expect(page.getByTestId("contrainte")).toHaveText("aucune");
    // vie du sol : un sol nu appauvrit le moût en azote, un couvert avec légumineuses le nourrit
    await page.selectOption("#vCouvert", "nu");
    const yanNu = parseInt((await page.getByTestId("azote-mout").textContent()).replace(/\D/g, ""), 10);
    await page.selectOption("#vCouvert", "total");
    const yanCouvert = parseInt((await page.getByTestId("azote-mout").textContent()).replace(/\D/g, ""), 10);
    expect(yanCouvert).toBeGreaterThan(yanNu);
    await expect(page.getByTestId("lecture-vie")).toContainText("légumineuses");
  });

  test("warming a vineyard shifts its Winkler region, its harvest date and its alcohol", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    const dj = async () => parseInt((await page.getByTestId("dj-rechauffe").textContent()).replace(/\D/g, "").slice(0, 4), 10);
    const alcool = async () => parseFloat((await page.getByTestId("alcool-rechauffe").textContent()).replace(",", "."));
    await page.locator("#wHausse").fill("0");
    expect(await dj()).toBe(1340);
    await expect(page.getByTestId("dj-rechauffe")).toContainText("Ib");
    const alcoolFroid = await alcool();
    // +3 °C : la Bourgogne prend les degrés-jours du Bordelais, vendange plus tôt, plus d'alcool
    await page.locator("#wHausse").fill("3");
    expect(await dj()).toBeGreaterThan(1850);
    await expect(page.getByTestId("equivalent")).toContainText("Napa");
    await expect(page.getByTestId("avance-vendange")).toContainText("plus tôt");
    expect(await alcool()).toBeGreaterThan(alcoolFroid + 1);
    await expect(page.getByTestId("lecture-rechauffement")).toContainText("gel de printemps");
    // monter en altitude rend une partie du chemin
    await page.check("#wAltitude");
    expect(await dj()).toBeLessThan(1850);
  });

  test("water and heat drive sugar, acidity and tannins", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    const nombre = async (id) => parseFloat((await page.getByTestId(id).textContent()).replace(",", ".").replace(/[^\d.]/g, ""));
    const poidsSec = await (async () => { await page.locator("#qEau").fill("150"); return nombre("poids-baie"); })();
    const sucreSec = await nombre("sucre-baie");
    await page.locator("#qEau").fill("480");
    // une baie gorgée d'eau est plus grosse et son jus plus dilué
    expect(await nombre("poids-baie")).toBeGreaterThan(poidsSec);
    expect(await nombre("sucre-baie")).toBeLessThan(sucreSec);
    await expect(page.getByTestId("lecture-equilibre")).toContainText("Trop d'eau");
    // la chaleur brûle l'acidité
    await page.locator("#qEau").fill("280");
    await page.locator("#qTemp").fill("16");
    const acideFrais = await nombre("acidite-baie");
    await page.locator("#qTemp").fill("26");
    expect(await nombre("acidite-baie")).toBeLessThan(acideFrais);
    // une sécheresse sévère bloque la maturité
    await page.locator("#qEau").fill("110");
    await page.locator("#qTemp").fill("20");
    await expect(page.getByTestId("lecture-equilibre")).toContainText("Blocage de maturité");
  });

  test("the blending bench mixes lots, scores them and respects cellar volumes", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    const note = async () => parseFloat((await page.getByTestId("assemblage-note").textContent()).replace(",", "."));
    // tout sur le merlot : un vin plus alcoolique, moins bien noté pour un vin de garde
    await page.locator("#asLot0").fill("100");
    for (const i of [1, 2, 3]) await page.locator(`#asLot${i}`).fill("0");
    const alcoolMerlot = parseFloat((await page.getByTestId("assemblage-alcool").textContent()).replace(",", "."));
    expect(alcoolMerlot).toBeCloseTo(14.2, 1);
    await expect(page.getByTestId("lecture-assemblage")).toContainText("vin de parcelle");
    const noteMono = await note();
    // l'optimiseur trouve mieux, et la cuvée reste réalisable
    await page.click("#asAuto");
    expect(await note()).toBeGreaterThan(noteMono);
    const volume = parseInt((await page.getByTestId("volume-cuvee").textContent()).split("hL")[0].replace(/\D/g, ""), 10);
    expect(volume).toBeGreaterThan(0);
    expect(volume).toBeLessThanOrEqual(960);
    // changer d'objectif change l'optimum
    const optimumGarde = await page.locator("#asLot1").inputValue();
    await page.selectOption("#asBut", "fruit");
    await page.click("#asAuto");
    expect(await page.locator("#asLot1").inputValue()).not.toBe(optimumGarde);
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
