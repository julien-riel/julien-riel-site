// Édition anglaise (en/) : surcouches valides, textes tous traduits, pas de français résiduel,
// et une simulation identique à l'édition française (les textes n'ont aucun effet sur le jeu).
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";
import { jouer, creerStrategie } from "./strategies.js";

const fr = charger();
const en = charger({ langue: "en" });

// PORTEE=evenements,comites : limite les vérifications de contenu à donnees.<x> et aux clés « <x>. » (travail en cours).
const PORTEE = (process.env.PORTEE || "").split(",").map((x) => x.trim()).filter(Boolean);
const dansPortee = (chemin) => !PORTEE.length || PORTEE.some((p) => chemin === p || chemin.startsWith(p + ".") || chemin.startsWith(p + "["));

// Clés qui ne portent jamais de texte affiché.
const STRUCTURELLES = new Set(["id", "v", "cibles", "cible", "ilot", "dev", "devs", "personnage", "theme", "accessoire", "signataire", "levier", "panneau", "metrique", "traits", "niveau", "niveaux", "ton", "motif"]);
// Clés dont la valeur peut rester identique en anglais (piles techniques).
const IDENTIQUES_PERMIS = new Set(["pile"]);
// Textes identiques dans les deux langues, admis tels quels.
const EXCEPTIONS = new Set([
  "SRE", "TDD", "WordPress", "Inspections", "IoT", "Web", "Incidents", "Tests",  "Firmware", "Designer", "Mobile", "Multi-Clauding", "Ship it", "Ginette", "Kevin", "Régis", "Satisfaction", "1 session", "2 sessions", "3 sessions", "4 sessions", "5 sessions", "6 sessions",
]);
// Noms de personnes : identiques dans les deux langues.
for (const p of [...fr.donnees.devs, ...Object.values(fr.donnees.personnages), ...fr.donnees.ilots.map((i) => i.direction || {})]) if (p.nom && !/^M(me|e) /.test(p.nom)) EXCEPTIONS.add(p.nom);
// Fonctions de logique : leur source peut rester la même.
const LOGIQUE = new Set(["condition", "quand", "choisirCible", "evaluer", "trimestreDe", "applicable", "poids"]);

function textuel(s) {
  return /[a-zà-ÿ]/i.test(s) && (/\s/.test(s) || /[à-ÿ]/i.test(s) || /^[A-ZÀ-Ý]/.test(s));
}

// Français résiduel : mots outils du français ou mot en minuscules avec un accent (les noms propres sont permis).
const MOTS_FR = /\b(?:avec|pour|dans|une|est|les|des|sont|mais|cette|leur|nous|vous|aussi|très|même|où|déjà|peut|faut|rien|chez|elle|ils|sans|sous|entre|toujours|encore|plus de)\b|\b(?:qu|c|n|j|l|d)'(?=[a-zà-ÿ])/i;
const ACCENTS_PERMIS = /^(café|cafés|déjà|résumé|résumés|naïve|naïvely|à|vis-à-vis|façade|rôle|cliché|clichés|décor|entrée|protégé|fiancé|crème|brûlée|tête-à-tête|pâté|soirée|élan|exposé)$/i;
function francais(s) {
  const texte = String(s).replace(/\{[^}]*\}/g, "");
  if (/\bMme\b|\bMe [A-ZÉ]/.test(texte)) return "titre français (Mme, Me)";
  const m = texte.match(MOTS_FR);
  if (m) return `mot « ${m[0]} »`;
  for (const mot of texte.match(/[A-Za-zÀ-ÿ'’-]+/g) || []) {
    if (/[à-ÿ]/.test(mot) && /^[a-zà-ÿ]/.test(mot) && !ACCENTS_PERMIS.test(mot.replace(/^['’]|['’]$/g, ""))) return `mot « ${mot} »`;
    if (/[«»]/.test(mot)) return "guillemets français";
  }
  if (/[«»]/.test(texte)) return "guillemets français";
  return null;
}
// Littéraux de chaîne d'une source de fonction.
function litteraux(source) {
  return (source.match(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g) || []).map((l) => l.slice(1, -1).replace(/\\'/g, "'"));
}

function rapport(liste, max = 40) {
  return liste.length ? `\n${liste.length} problème(s) :\n  ` + liste.slice(0, max).join("\n  ") + (liste.length > max ? `\n  … et ${liste.length - max} autres` : "") : "";
}

test("les surcouches anglaises s'appliquent sans erreur", () => {
  assert.deepEqual([...en.erreursTraduction], [], rapport([...en.erreursTraduction]));
  assert.equal(en.langue, "en");
  assert.equal(fr.langue, "fr");
});

test("toutes les clés de textes du moteur et de l'interface ont une version anglaise", () => {
  const manquantes = Object.keys(fr.textes.fr).filter((k) => dansPortee(k) && !Object.prototype.hasOwnProperty.call(en.textes.en || {}, k));
  assert.deepEqual(manquantes, [], rapport(manquantes));
});

test("les textes anglais (t()) ne contiennent pas de français", () => {
  const problemes = [];
  for (const [k, v] of Object.entries(en.textes.en || {})) {
    if (!dansPortee(k)) continue;
    const chaines = typeof v === "function" ? litteraux(v.toString()) : [].concat(v);
    for (const s of chaines) {
      const p = typeof s === "string" && francais(s);
      if (p) problemes.push(`${k} : ${p} — ${String(s).slice(0, 90)}`);
    }
  }
  assert.deepEqual(problemes, [], rapport(problemes));
});

test("tous les textes des données sont traduits, sans français résiduel", () => {
  const nonTraduits = [], residus = [];
  const vus = new Set();
  function marcher(a, b, chemin, cle) {
    if (typeof a === "string") {
      if (STRUCTURELLES.has(cle) || !textuel(a)) return;
      if (a === b) {
        if (!IDENTIQUES_PERMIS.has(cle) && !EXCEPTIONS.has(a)) nonTraduits.push(`${chemin} : ${a.slice(0, 80)}`);
        return;
      }
      const p = typeof b === "string" && francais(b);
      if (p) residus.push(`${chemin} : ${p} — ${String(b).slice(0, 90)}`);
      return;
    }
    if (typeof a === "function") {
      if (LOGIQUE.has(cle) || typeof b !== "function") return;
      for (const s of litteraux(b.toString())) {
        const p = textuel(s) && francais(s);
        if (p) { (a.toString() === b.toString() ? nonTraduits : residus).push(`${chemin}() : ${p} — ${s.slice(0, 80)}`); break; }
      }
      return;
    }
    if (!a || typeof a !== "object" || !b || typeof b !== "object") return;
    if (vus.has(a)) return; // index partagés (leviersParId, metriquesParId)
    vus.add(a);
    if (Array.isArray(a)) a.forEach((x, i) => marcher(x, b[i], `${chemin}[${i}]`, cle));
    else for (const k of Object.keys(a)) marcher(a[k], b[k], `${chemin}.${k}`, k);
  }
  for (const k of Object.keys(fr.donnees)) if (dansPortee(k)) marcher(fr.donnees[k], en.donnees[k], `donnees.${k}`, k);
  assert.deepEqual(nonTraduits, [], "Non traduits :" + rapport(nonTraduits));
  assert.deepEqual(residus, [], "Français résiduel :" + rapport(residus));
});

// Les textes ne changent rien à la simulation : on compare l'état final sans les textes rendus.
function sansTextes(etat) {
  return JSON.stringify(etat, (k, v) => (k === "reflexions" || k === "journal" ? undefined : typeof v === "string" && /\s/.test(v) ? "·" : v));
}

test("même graine, même stratégie : même partie en français et en anglais", () => {
  for (const nom of ["equilibree", "aleatoire"]) {
    const a = jouer(charger(), nom, 7), b = jouer(charger({ langue: "en" }), nom, 7);
    assert.equal(sansTextes(b), sansTextes(a), nom);
    assert.equal(a.phase, "fin");
  }
});

test("en cours de partie : journal, réflexions et textes calculés sont en anglais", () => {
  const L = charger({ langue: "en" });
  const strategie = creerStrategie(L, "equilibree", 3);
  let etat = L.moteur.creer(3);
  const problemes = new Map();
  const noter = (ou, s) => {
    const p = typeof s === "string" && textuel(s) && francais(s);
    if (p && !problemes.has(ou + s)) problemes.set(ou + s, `${ou} : ${p} — ${s.slice(0, 90)}`);
  };
  const ref = () => ({ ilot: etat.ordreIlots[0], dev: Object.keys(etat.devs)[0] });
  function appeler(f, ou) {
    for (const args of [[etat, L.q, ref()], [etat, L.q, etat.devs[Object.keys(etat.devs)[0]], etat.ilots[etat.ordreIlots[0]]], [ref()]]) {
      try { const r = f(...args); if (typeof r === "string") { noter(ou, r); return; } } catch (e) { /* signature différente */ }
    }
  }
  function textesCalcules(x, chemin, cle) {
    if (typeof x === "function") { if (!LOGIQUE.has(cle)) appeler(x, chemin); return; }
    if (!x || typeof x !== "object") return;
    if (Array.isArray(x)) x.forEach((e, i) => textesCalcules(e, `${chemin}[${i}]`, cle));
    else for (const k of Object.keys(x)) if (k !== "leviersParId" && k !== "metriquesParId") textesCalcules(x[k], `${chemin}.${k}`, k);
  }
  for (let s = 0; s < 60 && etat.phase !== "fin"; s++) {
    etat = L.moteur.simuler(etat, 1, strategie).etat;
    for (const j of etat.journal || []) noter(`journal ${j.cle}`, L.t(j.cle, j.vars));
    for (const r of (etat.reflexions && etat.reflexions.liste) || []) noter(`réflexion ${r.motif}`, r.texte);
    if (s % 13 === 6) textesCalcules(L.donnees, "donnees", "");
  }
  assert.equal(etat.phase, "fin");
  assert.deepEqual([...problemes.values()], [], rapport([...problemes.values()]));
  assert.deepEqual(Object.keys(L.textesManquants), [], rapport(Object.keys(L.textesManquants)));
});
