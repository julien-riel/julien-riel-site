// Équilibrage : chaque stratégie scriptée (tests/strategies.js) jouée sur 30 graines.
// Seul « aucune stratégie ne plante » est actif ; les critères d'acceptation d'ARCHITECTURE §5
// sont désactivés tant que l'équilibrage (phase 3) n'est pas fait : EQUILIBRE=1 pour les lancer.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";
import { NOMS_STRATEGIES, jouer } from "./strategies.js";

const LTP = charger();
const GRAINES = Array.from({ length: 30 }, (_, i) => i + 1);
const avecEquilibre = !!process.env.EQUILIBRE;

const moyenne = (l) => (l.length ? l.reduce((a, b) => a + b, 0) / l.length : 0);
const part = (l, f) => (l.length ? l.filter(f).length / l.length : 0);

// Joue toutes les parties une seule fois ; les tests lisent ces résultats.
const resultats = {};
const erreurs = [];
for (const nom of NOMS_STRATEGIES) {
  resultats[nom] = [];
  for (const g of GRAINES) {
    try {
      const etat = jouer(LTP, nom, g);
      const h = etat.historique || {};
      const t1 = (h.pointsLivres || []).slice(1, LTP.K.etat.semainesParTrimestre + 1);
      resultats[nom].push({
        graine: g, phase: etat.phase, semaine: etat.semaine, resultat: etat.resultat,
        profil: etat.resultat && etat.resultat.profil,
        score: (etat.resultat && etat.resultat.score) || 0,
        faits: (etat.resultat && etat.resultat.faits) || {},
        debitT1: moyenne(t1.map((x) => x || 0)),
      });
    } catch (e) {
      erreurs.push(`${nom} / graine ${g} : ${e && e.stack}`);
    }
  }
}

function synthese(nom) {
  const r = resultats[nom];
  const profils = {};
  for (const x of r) profils[x.profil] = (profils[x.profil] || 0) + 1;
  return {
    strategie: nom,
    profils: Object.entries(profils).sort((a, b) => b[1] - a[1]).map(([p, n]) => `${p}:${n}`).join(" "),
    score: Math.round(moyenne(r.map((x) => x.score))),
    debit: Math.round(moyenne(r.map((x) => x.faits.pointsLivres || 0))),
    ratio: +moyenne(r.map((x) => x.faits.ratioDebit || 0)).toFixed(2),
    debitT1: Math.round(moyenne(r.map((x) => x.debitT1))),
    incidents: +moyenne(r.map((x) => x.faits.incidents || 0)).toFixed(1),
    secu: +moyenne(r.map((x) => x.faits.incidentsSecurite || 0)).toFixed(1),
    cout: Math.round(moyenne(r.map((x) => x.faits.budgetDepense || 0))),
    satisfaction: +moyenne(r.map((x) => x.faits.satisfaction || 0)).toFixed(2),
    congedies: r.filter((x) => x.profil === "congedie").length,
  };
}

test("aucune stratégie ne plante et chaque partie va jusqu'au bilan", () => {
  assert.deepEqual(erreurs, []);
  for (const nom of NOMS_STRATEGIES) {
    assert.equal(resultats[nom].length, GRAINES.length, nom);
    for (const x of resultats[nom]) {
      assert.equal(x.phase, "fin", `${nom} / graine ${x.graine} : phase ${x.phase} à la semaine ${x.semaine}`);
      assert.ok(x.resultat && x.profil, `${nom} / graine ${x.graine} : pas de profil`);
      assert.ok(Number.isFinite(x.score) && x.score >= 0 && x.score <= 100, `${nom} / graine ${x.graine} : score ${x.score}`);
    }
  }
  console.log("\nÉquilibrage (30 graines par stratégie)");
  console.table(NOMS_STRATEGIES.map(synthese));
});

// --- Critères d'acceptation (ARCHITECTURE §5), activés en phase 3 ---

test("tout-ia : usine à PR ou Far West dans au moins 80 % des graines, meilleur débit au T1", { skip: !avecEquilibre }, () => {
  const r = resultats["tout-ia"];
  const p = part(r, (x) => x.profil === "usine" || x.profil === "farwest");
  assert.ok(p >= 0.8, `usine ou farwest : ${Math.round(p * 100)} %`);
  const t1 = synthese("tout-ia").debitT1;
  for (const nom of NOMS_STRATEGIES.filter((n) => n !== "tout-ia" && n !== "aleatoire")) {
    assert.ok(t1 > synthese(nom).debitT1, `débit T1 tout-ia (${t1}) ≤ ${nom} (${synthese(nom).debitT1})`);
  }
});

test("musée : profil « musee » dans au moins 70 % des graines, satisfaction la plus basse", { skip: !avecEquilibre }, () => {
  const r = resultats.musee;
  const p = part(r, (x) => x.profil === "musee");
  assert.ok(p >= 0.7, `musee : ${Math.round(p * 100)} %`);
  const s = synthese("musee").satisfaction;
  for (const nom of NOMS_STRATEGIES.filter((n) => n !== "musee" && n !== "aleatoire")) {
    assert.ok(s < synthese(nom).satisfaction, `satisfaction musee (${s}) ≥ ${nom} (${synthese(nom).satisfaction})`);
  }
});

test("équilibrée : transition réussie dans au moins 70 % des graines, meilleur score moyen", { skip: !avecEquilibre }, () => {
  const r = resultats.equilibree;
  const p = part(r, (x) => x.profil === "reussie");
  assert.ok(p >= 0.7, `reussie : ${Math.round(p * 100)} %`);
  const s = synthese("equilibree").score;
  for (const nom of NOMS_STRATEGIES.filter((n) => n !== "equilibree")) {
    assert.ok(s > synthese(nom).score, `score equilibree (${s}) ≤ ${nom} (${synthese(nom).score})`);
  }
});

test("économique : score inférieur à l'équilibrée, plus d'incidents de sécurité", { skip: !avecEquilibre }, () => {
  const e = synthese("economique"), b = synthese("equilibree");
  assert.ok(e.score < b.score, `score economique ${e.score} ≥ equilibree ${b.score}`);
  assert.ok(e.secu > b.secu, `incidents de sécurité economique ${e.secu} ≤ equilibree ${b.secu}`);
});

test("aléatoire : distribution de profils variée", { skip: !avecEquilibre }, () => {
  const profils = new Set(resultats.aleatoire.map((x) => x.profil));
  assert.ok(profils.size >= 3, `profils : ${[...profils].join(", ")}`);
});
