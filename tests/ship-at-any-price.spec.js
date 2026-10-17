import { test, expect } from "@playwright/test";

// Édition anglaise de « Livrer à tout prix » : la page, les liens de langue, et une interface sans français.
const URL = "/en/projets/ship-at-any-price/";

// Français résiduel dans un texte d'interface : mots outils, jours, mots accentués en minuscules, guillemets « ».
// Les noms propres accentués (Régis, Bérubé, Marie-Ève) commencent par une majuscule et sont permis.
const MOTS_FR = /(^|[^\p{L}'’])(avec|pour|dans|une|est|les|des|sont|mais|cette|leur|nous|vous|aussi|peut|faut|rien|chez|sans|sous|toujours|encore|semaine|semaines|trimestre|lundi|mardi|mercredi|jeudi|vendredi|jetons|leviers|réglages|qu'|c'est|n'est|l'|d')(?=$|[^\p{L}])/iu;
const ACCENTS_PERMIS = /^(café|cafés|déjà|résumé|résumés|naïve|naïvely|à|vis-à-vis|façade|rôle|cliché|clichés|décor|entrée|protégé|fiancé|crème|brûlée|tête-à-tête|pâté|soirée|élan|exposé)$/i;
function francais(s) {
  const texte = String(s);
  if (/[«»]/.test(texte)) return "guillemets « »";
  if (/\bMme\b|\bMe [A-ZÉ]/.test(texte)) return "titre Mme ou Me";
  const m = texte.match(MOTS_FR);
  if (m) return `mot « ${m[2]} »`;
  for (const mot of texte.match(/[\p{L}'’-]+/gu) || []) {
    const nu = mot.replace(/^['’-]+|['’-]+$/g, "");
    if (/[à-ÿ]/.test(nu) && /^[a-zà-ÿ]/.test(nu) && !ACCENTS_PERMIS.test(nu)) return `mot « ${nu} »`;
  }
  return null;
}

// Textes visibles, noms accessibles et infobulles de la page, découpés par élément.
async function textesInterface(page) {
  return page.evaluate(() => {
    const res = [];
    const visible = (el) => !!(el.offsetParent || el.getClientRects().length);
    for (const el of document.querySelectorAll("body *")) {
      if (el.closest("script, style, noscript")) continue;
      for (const a of ["aria-label", "title", "placeholder"]) if (el.getAttribute(a)) res.push({ ou: `${el.tagName.toLowerCase()}[${a}]`, texte: el.getAttribute(a) });
      if (!visible(el) && !el.classList.contains("sr-only")) continue;
      for (const n of el.childNodes) {
        if (n.nodeType === 3 && n.textContent.trim()) res.push({ ou: `${el.tagName.toLowerCase()}.${String(el.className && el.className.baseVal != null ? el.className.baseVal : el.className).split(" ")[0]}`, texte: n.textContent.trim() });
      }
    }
    return res;
  });
}

async function verifierAnglais(page, etape) {
  const problemes = [];
  for (const { ou, texte } of await textesInterface(page)) {
    if (texte.includes("⟦")) problemes.push(`${etape} · ${ou} : clé manquante — ${texte.slice(0, 80)}`);
    // Le lien « FR » et son infobulle « Français » sont en français exprès (lang="fr").
    const p = francais(texte);
    if (p) problemes.push(`${etape} · ${ou} : ${p} — ${texte.slice(0, 100)}`);
  }
  const manquants = await page.evaluate(() => Object.keys(window.LTP.textesManquants));
  expect(manquants, `${etape} : textes retombés sur le français`).toEqual([]);
  expect(problemes, problemes.join("\n")).toEqual([]);
}

// Ferme la note de service si elle est ouverte (elle peut se refermer d'elle-même : on ne bloque pas dessus).
async function fermerNoteSiOuverte(page, etape) {
  const b = page.locator("#btn-note-compris");
  if (!(await b.isVisible())) return;
  if (etape) await verifierAnglais(page, etape);
  await b.click({ timeout: 3000 }).catch(() => {});
}

async function commencer(page, graine) {
  await page.goto(URL + "?neuf=1&graine=" + graine);
  await page.locator("#btn-commencer").click();
  await expect(page.locator(".note-service")).toBeVisible();
}

test.describe("Ship at Any Price (édition anglaise)", () => {
  test("la page anglaise s'ouvre en anglais, avec ses liens de langue et de retour", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?neuf=1&graine=1");
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page).toHaveTitle(/^Ship at Any Price/);
    await expect(page.locator("h1.titre-jeu")).toHaveText("Ship at Any Price");
    await expect(page.locator('.hud a.tool--back[href="/en/projets/"]')).toHaveText("← julien-riel.com");
    const fr = page.locator(".hud a.tool--langue");
    await expect(fr).toHaveText("FR");
    await expect(fr).toHaveAttribute("href", "/projets/livrer-a-tout-prix/");
    await expect(fr).toHaveAttribute("hreflang", "fr");
    await expect(fr).toHaveAttribute("lang", "fr");
    expect(await page.evaluate(() => [window.LTP.langue, window.LTP.erreursTraduction.length])).toEqual(["en", 0]);
    await verifierAnglais(page, "accueil");
    expect(erreurs).toEqual([]);
  });

  test("la page française mène à l'édition anglaise", async ({ page }) => {
    await page.goto("/projets/livrer-a-tout-prix/?neuf=1&graine=1");
    await expect(page.locator("html")).toHaveAttribute("lang", "fr");
    const en = page.locator(".hud a.tool--langue");
    await expect(en).toHaveText("EN");
    await expect(en).toHaveAttribute("href", "/en/projets/ship-at-any-price/");
    await expect(en).toHaveAttribute("hreflang", "en");
    await expect(en).toHaveAttribute("lang", "en");
    await expect(page.locator('.hud a.tool--back[href="/projets/"]')).toBeVisible();
    await en.click();
    await expect(page).toHaveURL(/\/en\/projets\/ship-at-any-price\/$/);
    await expect(page.locator("h1.titre-jeu")).toHaveText("Ship at Any Price");
  });

  test("tutoriel, journées, panneaux, fiche, note de service et tableau de bord : tout en anglais", async ({ page }) => {
    test.setTimeout(90000);
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?neuf=1&graine=4");
    await page.locator("#btn-tutoriel").click();
    const bulle = page.locator(".tuto-bulle");
    await expect(bulle).toBeVisible();
    await expect(bulle.locator(".eyebrow")).toContainText("Guided tour · Step 1 of");
    await verifierAnglais(page, "tutoriel 1");
    // La visite : « Next », ou l'action demandée (changer un levier, avancer d'une journée).
    for (let i = 0; i < 20 && (await bulle.count()); i++) {
      const suivant = page.locator("#btn-tuto-suivant");
      if (await suivant.isVisible()) {
        await verifierAnglais(page, `tutoriel ${i + 2}`);
        await suivant.click();
      } else {
        const attendre = await page.evaluate(() => {
          const v = window.LTP.app.ui.visite;
          return v ? (window.LTP.donnees.tutoriel[v.etape] || {}).attendre : null;
        });
        if (attendre === "levier") await page.locator('[data-levier="licences"] [role="radio"][aria-checked="false"]:not([aria-disabled="true"])').first().click();
        else if (attendre) await page.locator("#btn-jour").click();
      }
      await page.waitForTimeout(250);
      await fermerNoteSiOuverte(page, "note de service");
    }
    await expect(bulle).toHaveCount(0);

    // Quelques journées : HUD « Q1 · Week 1 · Tuesday ».
    await fermerNoteSiOuverte(page);
    const quand = page.locator("#hud .hud-quand");
    await expect(quand.locator(".hud-tri")).toHaveText(/^Q\d$/);
    for (let i = 0; i < 6; i++) {
      const choix = page.locator(".carte-evenement:not(.note-service) [data-choix]");
      if (await choix.count()) { await verifierAnglais(page, "carte d'événement"); await choix.first().click(); }
      await fermerNoteSiOuverte(page, "note de service");
      await page.locator("#btn-jour").click();
    }
    await expect(quand).toContainText(/Week \d+/);
    await expect(quand).toContainText(/Monday|Tuesday|Wednesday|Thursday|Friday/);
    await expect(page.locator(".tableau-rapport")).toContainText("report, published on Friday");
    await verifierAnglais(page, "journées");

    // Une note de service (semaine 8 : le mode agent) et ses boutons.
    await page.goto(URL + "?demo=note");
    await expect(page.locator(".note-service")).toBeVisible();
    await expect(page.locator(".note-service .eyebrow").first()).toContainText("Memo · week");
    await verifierAnglais(page, "note de service (démo)");
    await page.locator("#btn-note-compris").click();

    // Chaque onglet de leviers, le registre et la portée par îlot.
    await page.goto(URL + "?demo=t2-tendu");
    for (const onglet of await page.locator("#panneau-feuille .onglet").all()) {
      await onglet.click();
      await verifierAnglais(page, "onglet " + (await onglet.getAttribute("data-panneau")));
    }
    await page.locator('#panneau-feuille .onglet[data-panneau="registre"]').click();
    await page.locator("#portee-ilot").selectOption("portail");
    await verifierAnglais(page, "registre (Portal)");

    // Une fiche de développeur.
    await page.locator('[data-dev="mathieu"]').click();
    await expect(page.locator(".fiche")).toBeVisible();
    await verifierAnglais(page, "fiche");
    await page.keyboard.press("Escape");

    // L'îlot dont la triade est complète : explication des branches.
    await page.goto(URL + "?demo=triade-appro");
    await page.locator('[data-ilot="appro"] .os-etiquette').click();
    await expect(page.locator(".triade-explic")).toBeVisible();
    await verifierAnglais(page, "triade");

    // Tableau de bord : survol d'une courbe (bulle « W12 · … »).
    const spark = page.locator(".metrique:not(.is-cachee) .spark").first();
    await spark.hover();
    await expect(page.locator(".metrique.is-scrub .met-scrub").first()).toHaveText(/^W\d+ · /);
    expect(erreurs).toEqual([]);
  });

  test("démo, comité et bilan en anglais", async ({ page }) => {
    const erreurs = [];
    page.on("pageerror", (e) => erreurs.push(e.message));
    await page.goto(URL + "?demo=evenement");
    await expect(page.locator(".carte-evenement #ev-titre")).toHaveText("The 4,000-line PR");
    await verifierAnglais(page, "événement (démo)");
    await page.locator('[data-choix="decouper"]').click();
    await expect(page.locator(".toast")).toContainText("You chose: ");

    await page.goto(URL + "?demo=comite");
    await expect(page.locator(".comite")).toBeVisible();
    await verifierAnglais(page, "comité (démo)");
    await page.locator(".comite .reponse").first().click();
    await expect(page.locator(".comite .reaction")).toBeVisible();
    await verifierAnglais(page, "comité, réaction");

    await commencer(page, 2);
    await page.locator("#btn-note-compris").click();
    await page.evaluate(() => window.LTP.app.simuler(60));
    await expect(page.locator(".profil-fin")).toBeVisible();
    await expect(page.locator(".bilan .eyebrow").first()).toContainText("Annual review");
    await verifierAnglais(page, "bilan");
    expect(erreurs).toEqual([]);
  });

  test("témoin : le détecteur repère le français de la page française", async ({ page }) => {
    await page.goto("/projets/livrer-a-tout-prix/?neuf=1&graine=1");
    await expect(page.locator("h1.titre-jeu")).toHaveText("Livrer à tout prix");
    const trouves = (await textesInterface(page)).filter(({ texte }) => francais(texte));
    expect(trouves.length).toBeGreaterThan(20);
  });

  test("l'édition anglaise a sa propre sauvegarde", async ({ page }) => {
    await commencer(page, 6);
    const cles = await page.evaluate(() => Object.keys(localStorage));
    expect(cles).toContain("ship-at-any-price:v2");
    expect(cles).not.toContain("livrer-a-tout-prix:v2");
  });

  test("à 375 px, pas de défilement horizontal et un HUD lisible", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    await commencer(page, 3);
    await page.locator("#btn-note-compris").click();
    await page.locator("#btn-jour").click();
    const largeur = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(largeur).toBeLessThanOrEqual(375);
    await expect(page.locator(".hud a.tool--langue")).toBeVisible();
    await expect(page.locator("#hud .hud-sem .hud-court").first()).toHaveText(/^W\d+$/);
    await verifierAnglais(page, "téléphone");
  });
});
