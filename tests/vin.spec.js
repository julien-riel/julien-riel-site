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
    const chapitres = await page.evaluate(() => window.vin.chapitres());
    // sept chapitres, dans l'ordre des décisions
    expect(chapitres.map((c) => c.id)).toEqual(["comprendre", "planter", "annee", "recolter", "transformer", "cuvee", "bouteille"]);
    expect(chapitres[0].stades).toEqual(["parcours", "sensations"]);
    expect(chapitres[1].stades).toEqual(["climat", "rechauffement", "geographie", "sol", "vie-du-sol", "cepage", "plantation", "jeunesse"]);
    expect(chapitres[2].stades).toEqual(["cycle", "eau-chaleur"]);
    expect(chapitres[3].stades[0]).toBe("maturite");
    expect(chapitres[4].stades).toEqual(["fermentation", "pressurage-rouge", "malo"]);
    // l'assemblage précède la clarification et la stabilisation
    expect(chapitres[5].stades).toEqual(["elevage", "assemblage", "clarification"]);
    expect(chapitres[6].stades).toEqual(["mise", "bouteille", "verre"]);
    expect(stades[0]).toBe("parcours");
    expect(stades[stades.length - 1]).toBe("verre");
    expect(stades).toContain("cepage");
    expect(stades).toContain("eraflage");
    expect(stades).toContain("pressurage-rouge");
    expect(stades).not.toContain("debourbage");
    // navigation : les chapitres, puis les étapes du chapitre en cours
    await expect(page.locator("#chapitres li")).toHaveCount(7);
    await expect(page.locator("#etapes li")).toHaveCount(2);
    await expect(page.locator(".chapitre")).toHaveCount(7);
    await expect(page.locator(".stade")).toHaveCount(stades.length);
    await expect(page.locator("#s-vie-du-sol")).toHaveClass(/approfondissement/);
    await expect(page.locator(".atelier")).toHaveCount(20);
    // chaque étape cite des sources cliquables
    await expect(page.locator("#s-malo .sources a").first()).toHaveAttribute("href", /vignevin/);
    expect(await page.locator(".sources a").count()).toBeGreaterThan(40);
    expect(erreurs).toEqual([]);
  });

  test("the stage bar follows the chapter being read", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.locator("#s-malo").scrollIntoViewIfNeeded();
    await expect(page.locator('#chapitres a[data-chapitre="transformer"]')).toHaveClass(/actif/);
    await expect(page.locator("#etapes li")).toHaveCount(3);
    await expect(page.locator('#etapes a[data-etape="malo"]')).toHaveClass(/actif/);
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
    await expect(page.locator("#s-fermentation .stade-tete h3")).toHaveText("Fermentation alcoolique");
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
    const alcool = async () => parseFloat((await page.getByTestId("alcool-rechauffe").textContent()).replace(",", ".").replace(/[^\d.]/g, ""));
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
    await expect(page.getByTestId("lecture-rechauffement")).not.toContainText("paradoxe");
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

  test("the berry cross-section and the grape variety workshops react to the choices", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    // coupe du raisin : la peau macère dans un rouge, pas dans un blanc
    await expect(page.getByTestId("lecture-raisin")).toContainText("macère");
    await page.selectOption("#rPartie", "pepins");
    await expect(page.getByTestId("lecture-raisin")).toContainText("pépins");
    // cépage : le pinot noir mûrit en Bourgogne, pas le cabernet sauvignon ; le vidal doit être protégé au Québec
    await page.locator("#pDj").fill("1350");
    await expect(page.getByTestId("maturite-cepage")).toHaveText("mûrit bien");
    await page.selectOption("#pCepage", "cabernet-sauvignon");
    await expect(page.getByTestId("maturite-cepage")).toContainText("ne mûrit pas");
    await page.selectOption("#pCepage", "vidal");
    await page.locator("#pHiver").fill("-28");
    await expect(page.getByTestId("hiver-cepage")).toContainText("buttage");
    await page.selectOption("#pCepage", "frontenac");
    await expect(page.getByTestId("hiver-cepage")).toHaveText("sans protection");
    await page.selectOption("#pCepage", "merlot");
    await expect(page.getByTestId("hiver-cepage")).toContainText("ne survit pas");
    // le climat choisi à l'étape 2.1 alimente le curseur du cépage
    await page.locator("#kTemp").fill("19");
    expect(parseInt(await page.locator("#pDj").inputValue(), 10)).toBeGreaterThan(1850);
    // en blanc, la peau est séparée avant la fermentation
    await page.click('#choixStyle [data-style="blanc"]');
    await attendrePage(page);
    await expect(page.getByTestId("lecture-raisin")).toContainText("séparée");
  });

  test("the final tasting summary rereads the decisions left in the workshops", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    const decisions = page.getByTestId("decisions");
    await expect(decisions).toContainText("Climat");
    await expect(decisions).toContainText("Assemblage");
    await expect(decisions).toContainText("Malolactique");
    const alcool = async () => parseFloat((await page.getByTestId("bilan-alcool").textContent()).replace(",", ".").replace(/[^\d.]/g, ""));
    const avant = await alcool();
    // tout sur le merlot : la cuvée monte à 14,2 %, et le bilan suit
    await page.locator("#asLot0").fill("100");
    for (const i of [1, 2, 3]) await page.locator(`#asLot${i}`).fill("0");
    expect(await alcool()).toBeCloseTo(14.2, 1);
    expect(await alcool()).toBeGreaterThan(avant);
    await expect(decisions).toContainText("100 % merlot");
    // le pH n'est plus calculé par moyenne dans l'assemblage
    await expect(page.getByTestId("assemblage-ph")).toContainText("mesurer");
    // un élevage plus long en barrique neuve laisse sa marque
    await page.selectOption("#eContenant", "neuve");
    await page.locator("#eDuree").fill("24");
    await expect(decisions).toContainText("24 mois en barrique neuve");
    await expect(page.getByTestId("lecture-bilan")).toContainText("bois");
  });

  test("the malolactic and bottling workshops no longer promise stability or a buttery taste", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.locator("#mlAvancement").fill("100");
    await expect(page.getByTestId("malo-statut")).toHaveText("Terminée");
    await expect(page.locator("#s-malo .lecture")).not.toContainText("ne refermentera plus");
    await expect(page.locator("#s-malo .lecture")).toContainText("sucres résiduels");
    await page.locator("#mlAvancement").fill("0");
    await expect(page.locator("#s-malo .lecture")).toContainText("co-inoculer");
    // SO₂ : deux formes, deux rôles
    await expect(page.getByTestId("so2-bisulfite")).toContainText("mg/L");
    await page.locator("#bPh").fill("3.9");
    await page.locator("#bSo2").fill("10");
    await expect(page.locator("#s-mise .lecture")).toContainText("antimicrobienne");
    await expect(page.locator("#s-mise .limite")).toContainText("haloanisoles");
  });

  test("orange, sweet and fortified paths render their own cellar steps without errors", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL);
    await attendrePage(page);
    // orange : un blanc traité comme un rouge
    await page.click('#choixStyle [data-style="orange"]');
    await expect(page.getByTestId("etiquette")).toHaveText("Orange");
    let chapitres = await page.evaluate(() => window.vin.chapitres());
    expect(chapitres[3].stades).toEqual(["maturite", "vendanges", "maceration-orange"]);
    expect(chapitres[4].stades).toEqual(["fermentation", "pressurage-rouge", "malo"]);
    await expect(page.locator("#s-fermentation .echantillon")).toContainText("jus doré");
    await expect(page.locator("#s-parcours .atelier canvas")).toBeVisible();
    // doux : concentration, arrêt, pas de malo active
    await page.click('#choixStyle [data-style="doux"]');
    chapitres = await page.evaluate(() => window.vin.chapitres());
    expect(chapitres[3].stades).toEqual(["maturite", "vendanges", "concentration", "pressurage-blanc", "debourbage"]);
    expect(chapitres[4].stades).toEqual(["fermentation", "arret", "malo"]);
    await expect(page.locator("#s-pressurage-blanc h3")).toContainText("rôties");
    expect(await page.locator("#mlAvancement").inputValue()).toBe("0");
    // muté : lagar, mutage, ni débourbage ni malo
    await page.click('#choixStyle [data-style="mute"]');
    chapitres = await page.evaluate(() => window.vin.chapitres());
    expect(chapitres[3].stades).toEqual(["maturite", "vendanges", "lagar"]);
    expect(chapitres[4].stades).toEqual(["fermentation", "mutage", "pressurage-rouge"]);
    const stades = await page.evaluate(() => window.vin.stades());
    expect(stades).not.toContain("malo");
    await expect(page.locator(".stade")).toHaveCount(stades.length);
    // aucune étape n'a perdu son texte au passage
    for (const style of ["orange", "doux", "mute"]) {
      await page.click(`#choixStyle [data-style="${style}"]`);
      const vides = await page.evaluate(() => [...document.querySelectorAll(".stade .intro, .stade h3, .stade .duree")].filter((el) => !el.textContent.trim()).length);
      expect(vides, `textes vides pour ${style}`).toBe(0);
    }
    expect(erreurs).toEqual([]);
  });

  test("fortifying stops the fermentation with spirit and keeps the sugar", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.click('#choixStyle [data-style="mute"]');
    await expect(page.locator("#fMuter")).toBeDisabled();
    await page.evaluate(() => window.vin.ateliers.fermentation.avancer(24 * 2));
    await expect(page.locator("#fMuter")).toBeEnabled();
    const avant = await page.evaluate(() => ({ S: window.vin.ateliers.fermentation.sim.S, E: window.vin.ateliers.fermentation.sim.E }));
    expect(avant.S).toBeGreaterThan(60);
    await page.click("#fMuter");
    const apres = await page.evaluate(() => { const s = window.vin.ateliers.fermentation.sim; return { S: s.S, E: s.E, fini: s.fini }; });
    expect(apres.fini).toBe("mute");
    expect(apres.E).toBeGreaterThan(19);
    expect(apres.E).toBeLessThan(20);
    expect(apres.S).toBeGreaterThan(40);
    expect(apres.S).toBeLessThan(avant.S);
    await expect(page.getByTestId("etat-fermentation")).toContainText("Muté");
    // le bilan relit le sucre gardé
    await expect(page.getByTestId("bilan-sucre")).not.toHaveText("sec");
    await expect(page.getByTestId("lecture-bilan")).toContainText("Le sucre est voulu");
  });

  test("the sweet path concentrates the must and the fermentation stops on its own with sugar left", async ({ page }) => {
    await page.goto(URL);
    await attendrePage(page);
    await page.click('#choixStyle [data-style="doux"]');
    // pourriture noble à 70 % : un moût de liquoreux ; sans brouillard, pas de botrytis
    const sucre = parseInt((await page.getByTestId("sucre-concentre").textContent()).replace(/\D/g, ""), 10);
    expect(sucre).toBeGreaterThan(300);
    await page.selectOption("#cAutomne", "sec");
    await expect(page.getByTestId("lecture-concentration")).toContainText("Pas de botrytis");
    await page.selectOption("#cAutomne", "brumes");
    // le gel : pas assez froid à −6, légal à −9
    await page.selectOption("#cMethode", "gel");
    await page.locator("#cGel").fill("-6");
    await expect(page.getByTestId("lecture-concentration")).toContainText("Pas assez froid");
    await page.locator("#cGel").fill("-9");
    const sucreGel = parseInt((await page.getByTestId("sucre-concentre").textContent()).replace(/\D/g, ""), 10);
    expect(sucreGel).toBeGreaterThan(330);
    // on encuve : la cuve reçoit ce sucre et s'arrête d'elle-même, sucre restant
    await page.click("#cEncuver");
    expect(parseInt(await page.locator("#fSucre").inputValue(), 10)).toBeGreaterThanOrEqual(330);
    await page.evaluate(() => window.vin.ateliers.fermentation.avancer(24 * 90));
    const sim = await page.evaluate(() => { const s = window.vin.ateliers.fermentation.sim; return { S: s.S, E: s.E, fini: s.fini }; });
    expect(sim.fini).toBe("alcool");
    expect(sim.E).toBeLessThan(15);
    expect(sim.S).toBeGreaterThan(60);
    await expect(page.getByTestId("etat-fermentation")).toContainText("vin doux voulu");
    // ou l'on arrête plus tôt, à la main
    await page.click("#fReinit");
    await page.evaluate(() => window.vin.ateliers.fermentation.avancer(24 * 10));
    await page.click("#fArreter");
    expect(await page.evaluate(() => window.vin.ateliers.fermentation.sim.fini)).toBe("arret");
  });

  test("the six style buttons fit a narrow phone without horizontal scrolling", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 700 });
    await page.goto(URL);
    await attendrePage(page);
    for (const style of ["mute", "doux", "orange"]) {
      await page.click(`#choixStyle [data-style="${style}"]`);
      const largeurs = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
      expect(largeurs.scroll, style).toBeLessThanOrEqual(largeurs.client);
      const debordent = await page.evaluate((w) => [...document.querySelectorAll("main *, .bandeau *")].filter((el) => el.getBoundingClientRect().right > w + 1 && el.getBoundingClientRect().width > 0).length, 320);
      expect(debordent, style).toBe(0);
    }
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
