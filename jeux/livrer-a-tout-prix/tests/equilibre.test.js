// Équilibrage : chaque stratégie scriptée (tests/strategies.js) jouée sur 30 graines.
// Les critères d'acceptation d'ARCHITECTURE §5 (et les ordres de grandeur de la phase 3) sont actifs.
// Le tableau de synthèse s'affiche à chaque exécution.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";
import { NOMS_STRATEGIES, creerStrategie } from "./strategies.js";

const LTP = charger();
const GRAINES = Array.from({ length: 30 }, (_, i) => i + 1);
const SPT = LTP.K.etat.semainesParTrimestre;

const moyenne = (l) => (l.length ? l.reduce((a, b) => a + b, 0) / l.length : 0);
const part = (l, f) => (l.length ? l.filter(f).length / l.length : 0);

// Joue une partie semaine par semaine (même déroulement que strategies.jouer) en relevant ce que le bilan ne garde pas.
function jouerDetaille(nom, graine) {
  const strategie = creerStrategie(LTP, nom, graine);
  let etat = LTP.moteur.creer(graine, {});
  const detail = { file: [], journalSemaine: [], tauxEchec: [], secuT12: 0 };
  for (let garde = 0; garde < LTP.K.etat.semainesTotal + 8 && etat.phase !== "fin"; garde++) {
    const avant = etat.semaine;
    const r = LTP.moteur.simuler(etat, 1, strategie);
    etat = r.etat;
    if (etat.semaine === avant) continue;
    detail.file[etat.semaine] = LTP.q.ilots(etat).reduce((a, il) => a + (il.fileRevue || 0), 0);
    detail.journalSemaine.push(r.journal.filter((e) => e.semaine === etat.semaine && e.type !== "levier" && e.type !== "comite").length);
    if (LTP.q.metriqueVisible(etat, "tauxEchec")) detail.tauxEchec.push(etat.metriques.tauxEchec);
    if (etat.trimestre <= 2) detail.secuT12 = (etat.cumuls && etat.cumuls.incidentsSecurite) || 0;
  }
  return { etat, detail };
}

// Joue toutes les parties une seule fois ; les tests lisent ces résultats.
const resultats = {};
const erreurs = [];
const mttrDepart = LTP.systemes.metriques.mttrExact(LTP.moteur.creer(1, {}));
for (const nom of NOMS_STRATEGIES) {
  resultats[nom] = [];
  for (const g of GRAINES) {
    try {
      const { etat, detail } = jouerDetaille(nom, g);
      const h = etat.historique || {};
      const t1 = (h.pointsLivres || []).slice(1, SPT + 1);
      const fileT12 = detail.file.slice(1, 2 * SPT + 1).filter((x) => x != null);
      resultats[nom].push({
        graine: g, phase: etat.phase, semaine: etat.semaine, resultat: etat.resultat,
        profil: etat.resultat && etat.resultat.profil,
        score: (etat.resultat && etat.resultat.score) || 0,
        faits: (etat.resultat && etat.resultat.faits) || {},
        debitT1: moyenne(t1.map((x) => x || 0)),
        fileT12: moyenne(fileT12),
        secuT12: detail.secuT12,
        journalMoyen: moyenne(detail.journalSemaine),
        journalMax: Math.max(0, ...detail.journalSemaine),
        tauxEchec: detail.tauxEchec.length ? moyenne(detail.tauxEchec) : null,
        evenements: etat.evenementsCompte || { tires: 0, parTrimestre: {}, semaines: [] },
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
    fileT12: Math.round(moyenne(r.map((x) => x.fileT12))),
    incidents: +moyenne(r.map((x) => x.faits.incidents || 0)).toFixed(1),
    secu: +moyenne(r.map((x) => x.faits.incidentsSecurite || 0)).toFixed(1),
    secuT12: +moyenne(r.map((x) => x.secuT12)).toFixed(1),
    budget: +moyenne(r.map((x) => x.faits.ratioBudget || 0)).toFixed(2),
    satisfaction: +moyenne(r.map((x) => x.faits.satisfaction || 0)).toFixed(2),
    fantomes: +moyenne(r.map((x) => x.faits.applisFantomes || 0)).toFixed(1),
    comprehension: +moyenne(r.map((x) => x.faits.comprehension || 0)).toFixed(2),
    evenements: +moyenne(r.map((x) => x.evenements.tires || 0)).toFixed(1),
    congedies: r.filter((x) => x.profil === "congedie").length,
  };
}
const S = {};
const s = (nom) => (S[nom] = S[nom] || synthese(nom));
const INTENTIONNELLES = ["tout-ia", "musee", "equilibree", "economique"];

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
  console.table(NOMS_STRATEGIES.map(s));
});

// --- Critères d'acceptation (ARCHITECTURE §5) ---

test("tout-ia : usine, Far West, château ou facture ; l'année se joue jusqu'au bout ; meilleur débit au T1", () => {
  const r = resultats["tout-ia"];
  const ensemble = part(r, (x) => ["usine", "farwest", "chateau", "facture"].includes(x.profil));
  assert.ok(ensemble >= 0.8, `usine, farwest, chateau ou facture : ${Math.round(ensemble * 100)} %`);
  const majoritaires = part(r, (x) => x.profil === "usine" || x.profil === "farwest");
  assert.ok(majoritaires > 0.5, `usine ou farwest : ${Math.round(majoritaires * 100)} %`);
  const congedies = part(r, (x) => x.profil === "congedie");
  assert.ok(congedies <= 0.2, `congédiés avant la fin : ${Math.round(congedies * 100)} %`);
  for (const nom of NOMS_STRATEGIES.filter((n) => n !== "tout-ia" && n !== "aleatoire")) {
    assert.ok(s("tout-ia").debitT1 > s(nom).debitT1, `débit T1 tout-ia (${s("tout-ia").debitT1}) ≤ ${nom} (${s(nom).debitT1})`);
  }
});

test("tout-ia : la demande suit l'appétit, la revue sature, la compréhension s'effondre, la facture dépasse", () => {
  const t = s("tout-ia"), e = s("equilibree");
  assert.ok(t.debit > 1.3 * e.debit, `points livrés tout-ia ${t.debit} ≤ 1,3 × équilibrée ${e.debit}`);
  assert.ok(t.fileT12 >= 10, `file de revue moyenne au T1–T2 : ${t.fileT12} PR`);
  for (const nom of INTENTIONNELLES.filter((n) => n !== "tout-ia")) {
    assert.ok(t.comprehension < s(nom).comprehension, `compréhension tout-ia ${t.comprehension} ≥ ${nom}`);
    assert.ok(t.incidents > s(nom).incidents, `incidents tout-ia ${t.incidents} ≤ ${nom}`);
  }
  assert.ok(t.budget >= 1.3, `budget tout-ia : ${t.budget} × le budget`);
});

test("musée : profil « musee » dans au moins 70 % des graines, satisfaction la plus basse, score nettement sous l'équilibrée", () => {
  const r = resultats.musee;
  const p = part(r, (x) => x.profil === "musee");
  assert.ok(p >= 0.7, `musee : ${Math.round(p * 100)} %`);
  for (const nom of INTENTIONNELLES.filter((n) => n !== "musee")) {
    assert.ok(s("musee").satisfaction < s(nom).satisfaction, `satisfaction musee (${s("musee").satisfaction}) ≥ ${nom} (${s(nom).satisfaction})`);
  }
  assert.ok(s("musee").score <= s("equilibree").score - 12, `score musee ${s("musee").score}, equilibree ${s("equilibree").score}`);
  assert.ok(s("musee").fantomes >= 4, `applications hors TI au musée : ${s("musee").fantomes}`);
});

test("équilibrée : transition réussie dans au moins 70 % des graines, meilleur score moyen, budget de 60 à 100 %", () => {
  const r = resultats.equilibree;
  const p = part(r, (x) => x.profil === "reussie");
  assert.ok(p >= 0.7, `reussie : ${Math.round(p * 100)} %`);
  const sc = s("equilibree").score;
  for (const nom of NOMS_STRATEGIES.filter((n) => n !== "equilibree")) {
    assert.ok(sc > s(nom).score, `score equilibree (${sc}) ≤ ${nom} (${s(nom).score})`);
  }
  const b = s("equilibree").budget;
  assert.ok(b >= 0.6 && b <= 1, `budget dépensé par l'équilibrée : ${Math.round(b * 100)} %`);
});

test("économique : score inférieur à l'équilibrée, plus d'incidents de sécurité et de défauts", () => {
  const e = s("economique"), b = s("equilibree");
  assert.ok(e.score < b.score, `score economique ${e.score} ≥ equilibree ${b.score}`);
  assert.ok(e.secu > b.secu, `incidents de sécurité economique ${e.secu} ≤ equilibree ${b.secu}`);
  assert.ok(e.incidents > b.incidents, `incidents economique ${e.incidents} ≤ equilibree ${b.incidents}`);
});

test("aléatoire : distribution de profils variée", () => {
  const profils = new Set(resultats.aleatoire.map((x) => x.profil));
  assert.ok(profils.size >= 3, `profils : ${[...profils].join(", ")}`);
});

test("par défaut : ne rien toucher ne mène pas à une transition réussie", () => {
  const r = resultats["par-defaut"];
  const p = part(r, (x) => x.profil === "reussie");
  assert.ok(p <= 0.2, `par-defaut réussie : ${Math.round(p * 100)} %`);
  assert.ok(s("par-defaut").score <= s("equilibree").score - 10, `score par-defaut ${s("par-defaut").score}, equilibree ${s("equilibree").score}`);
  assert.ok(s("par-defaut").satisfaction < s("equilibree").satisfaction, "satisfaction par-defaut ≥ équilibrée");
});

// --- Ordres de grandeur ---

test("ordres de grandeur : MTTR de départ, nombre d'événements, journal lisible", () => {
  assert.ok(mttrDepart >= 6 && mttrDepart <= 10, `MTTR de départ ${mttrDepart} h`);
  for (const nom of NOMS_STRATEGIES) {
    const ev = s(nom).evenements;
    assert.ok(ev >= 8 && ev <= 16, `${nom} : ${ev} événements par partie`);
    const r = resultats[nom];
    assert.ok(moyenne(r.map((x) => x.journalMoyen)) <= 8, `${nom} : journal ${moyenne(r.map((x) => x.journalMoyen)).toFixed(1)} entrées par semaine`);
    assert.ok(Math.max(...r.map((x) => x.journalMax)) <= 25, `${nom} : semaine la plus chargée du journal`);
  }
});

test("rythme des événements : rien en semaine 1, au moins 2 semaines d'écart, au moins 3 par trimestre", () => {
  for (const nom of NOMS_STRATEGIES) {
    for (const x of resultats[nom]) {
      const sem = x.evenements.semaines || [];
      assert.ok(!sem.includes(1), `${nom} / graine ${x.graine} : événement en semaine 1`);
      for (let i = 1; i < sem.length; i++) {
        assert.ok(sem[i] - sem[i - 1] >= LTP.K.evenements.espacementMin, `${nom} / graine ${x.graine} : semaines ${sem[i - 1]} et ${sem[i]}`);
      }
      if (x.profil === "congedie") continue;
      for (let t = 1; t <= 4; t++) {
        const n = (x.evenements.parTrimestre || {})[t] || 0;
        assert.ok(n >= 3, `${nom} / graine ${x.graine} : ${n} événements au T${t}`);
      }
    }
  }
});

test("sécurité : risques faibles au T1–T2 pour une stratégie raisonnable (la gouvernance arrive au T3)", () => {
  for (const nom of ["musee", "equilibree", "economique", "par-defaut"]) {
    assert.ok(s(nom).secuT12 <= 2, `${nom} : ${s(nom).secuT12} incidents de sécurité au T1–T2`);
  }
});

test("taux d'échec des changements : plausible (5 à 30 %) pour l'équilibrée", () => {
  const l = resultats.equilibree.map((x) => x.tauxEchec).filter((x) => x != null);
  assert.ok(l.length > 0, "taux d'échec jamais visible");
  const m = moyenne(l);
  assert.ok(m >= 0.05 && m <= 0.3, `taux d'échec moyen ${Math.round(m * 100)} %`);
});
