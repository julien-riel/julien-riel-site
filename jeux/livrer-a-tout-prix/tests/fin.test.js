// Fin de partie et comités (32-fin.js, CONCEPTION §2.3 et §7) : sélection des questions, confiance,
// bilans de trimestre, critères, score, étoiles et profil de fin.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger, json } from "./charger.js";

function question(id, personnage, extra = {}) {
  return {
    id, personnage, trimestres: [1, 2, 3, 4], poids: 1, question: "Question " + id,
    reponses: [
      { id: "vraie", texte: "Vrai", evaluer: () => 1, reaction: "Merci." },
      { id: "floue", texte: "Flou", evaluer: () => -1, reaction: (etat, q, score) => `Score ${score}.` },
      { id: "boum", texte: "Boum", evaluer: () => { throw new Error("x"); }, reaction: "Hein ?" },
    ],
    ...extra,
  };
}
const F = (LTP) => LTP.systemes.fin;
function enComite(LTP, graine = 1) {
  const etat = LTP.moteur.creer(graine);
  etat.semaine = 13;
  etat.phase = "comite";
  F(LTP).preparerComite(etat, LTP.moteur.creerContexte(etat));
  return etat;
}

test("le comité pose trois questions admissibles du trimestre, de personnages variés", () => {
  const LTP = charger();
  LTP.donnees.comites = [
    question("r1", "regis"), question("r2", "regis"), question("r3", "regis"),
    question("b1", "berube"), question("l1", "lavigueur"),
    question("t4", "lavigueur", { trimestres: [4] }),
    question("faux", "berube", { condition: () => false }),
    question("leve", "berube", { condition: () => { throw new Error("x"); } }),
    question("zero", "berube", { poids: 0 }),
  ];
  let varies = 0;
  for (let g = 1; g <= 40; g++) {
    const etat = enComite(LTP, g);
    const ids = etat.comite.questions.map((x) => x.id);
    assert.equal(ids.length, 3);
    assert.equal(new Set(ids).size, 3);
    for (const id of ids) assert.ok(["r1", "r2", "r3", "b1", "l1"].includes(id), id);
    const perso = new Set(etat.comite.questions.map((x) => x.personnage));
    if (perso.size === 3) varies += 1;
    assert.equal(etat.comite.trimestre, 1);
    assert.deepEqual(json(etat.comite.reponses), {});
  }
  assert.ok(varies >= 20, `comités avec trois personnages différents : ${varies} / 40`);
});

test("sans questions, le comité est vide et terminé", () => {
  const LTP = charger();
  LTP.donnees.comites = [];
  const etat = enComite(LTP);
  assert.deepEqual(json(etat.comite.questions), []);
  assert.equal(F(LTP).comiteTermine(etat), true);
});

test("répondre : la bonne réponse monte la confiance, la mauvaise la baisse, la réaction est gardée", () => {
  const LTP = charger();
  LTP.donnees.comites = [question("a", "regis"), question("b", "berube"), question("c", "lavigueur")];
  let etat = enComite(LTP);
  const c0 = etat.org.confiance, K = LTP.K.fin.confianceParPoint;
  assert.equal(F(LTP).comiteTermine(etat), false);
  let r = LTP.moteur.action(etat, { type: "comite", question: "a", reponse: "vraie" });
  assert.equal(r.erreur, undefined);
  etat = r.etat;
  assert.equal(etat.org.confiance, c0 + K);
  assert.equal(etat.comite.scores.a, 1);
  assert.equal(etat.comite.reactions.a, "Merci.");
  assert.equal(LTP.moteur.action(etat, { type: "comite", question: "a", reponse: "floue" }).erreur, "deja-repondu");
  assert.equal(LTP.moteur.action(etat, { type: "comite", question: "b", reponse: "zzz" }).erreur, "reponse-inconnue");
  assert.equal(LTP.moteur.action(etat, { type: "comite", question: "zzz", reponse: "vraie" }).erreur, "question-inconnue");
  etat = LTP.moteur.action(etat, { type: "comite", question: "b", reponse: "floue" }).etat;
  assert.equal(etat.org.confiance, c0);
  assert.equal(etat.comite.reactions.b, "Score -1.");
  assert.equal(LTP.moteur.action(etat, { type: "phase", phase: "planification" }).erreur, "comite-incomplet");
  etat = LTP.moteur.action(etat, { type: "comite", question: "c", reponse: "boum" }).etat;
  assert.equal(etat.comite.scores.c, 0, "un evaluer qui lève vaut 0");
  assert.equal(F(LTP).comiteTermine(etat), true);
  r = LTP.moteur.action(etat, { type: "phase", phase: "planification" });
  assert.equal(r.erreur, undefined);
  assert.equal(r.etat.trimestre, 2);
  assert.equal(r.etat.fin.bilans.length, 1);
  assert.equal(r.etat.fin.bilans[0].trimestre, 1);
});

test("simuler sans stratégie passe les comités (réponses listées dans l'état)", () => {
  const LTP = charger();
  LTP.donnees.comites = [question("a", "regis"), question("b", "berube"), question("c", "lavigueur")];
  const { etat } = LTP.moteur.simuler(LTP.moteur.creer(3), 60);
  assert.equal(etat.phase, "fin");
  assert.ok(etat.fin.bilans.length >= 1);
  assert.ok(etat.fin.questionsPosees.length >= 1);
});

test("résultat d'une partie : forme, bornes et profil connu", () => {
  const LTP = charger();
  const { etat } = LTP.moteur.simuler(LTP.moteur.creer(5), 60);
  const r = etat.resultat;
  assert.ok(["annee", "confiance"].includes(r.raison));
  assert.ok(LTP.donnees.fins.some((f) => f.id === r.profil), r.profil);
  assert.ok(Number.isInteger(r.etoiles) && r.etoiles >= 0 && r.etoiles <= 5);
  assert.ok(r.score >= 0 && r.score <= 100);
  for (const k of ["livraison", "qualite", "couts", "risques", "personnes"]) {
    assert.ok(r.criteres[k] >= 0 && r.criteres[k] <= 1, `${k} = ${r.criteres[k]}`);
  }
  for (const k of ["pointsLivres", "pointsReference", "incidents", "fuites", "budgetDepense", "comprehension", "departs", "volumeCode", "volumeDepart"]) {
    assert.ok(Number.isFinite(r.faits[k]), `faits.${k}`);
  }
  assert.deepEqual(json(r), JSON.parse(JSON.stringify(r)), "résultat en JSON pur");
});

test("confiance sous le seuil → congédié, une étoile au plus", () => {
  const LTP = charger();
  const etat = LTP.moteur.creer(1);
  etat.semaine = 20;
  const r = F(LTP).calculerResultat(etat, null, "confiance");
  assert.equal(r.profil, "congedie");
  assert.ok(r.etoiles <= LTP.K.fin.etoilesCongedieMax);
  assert.equal(F(LTP).seuilConfiance, LTP.K.fin.seuilConfiance);
});

// Faits de référence « moyens », qu'on pousse dans une direction pour chaque profil.
function faitsMoyens(extra = {}) {
  return {
    ratioDebit: 1.1, partLgtm: 0.1, incidentsParSemaine: 0.5, comprehension: 0.6, couvertureReelle: 0.4,
    satisfaction: 0.55, applisFantomes: 1, surfaceAttaque: 0.35, expositionLoi25: 0.3, fuites: 0, fuitesMajeures: 0,
    incidentsSecurite: 1, ratioBudget: 1, ratioVolume: 1.2, releve: 0.5, moral: 0.6, fatigue: 0.3, departs: 0,
    ...extra,
  };
}
function profil(LTP, extra) {
  const f = faitsMoyens(extra);
  return F(LTP).profil("annee", f, F(LTP).criteres(f));
}

test("profils : chaque situation typique donne son profil", () => {
  const LTP = charger();
  assert.equal(profil(LTP, { ratioDebit: 1.45, incidentsParSemaine: 0.2, comprehension: 0.75, surfaceAttaque: 0.15, ratioBudget: 0.9 }), "reussie");
  assert.equal(profil(LTP, { ratioDebit: 1.9, partLgtm: 0.6, incidentsParSemaine: 1.6, comprehension: 0.35 }), "usine");
  assert.equal(profil(LTP, { ratioDebit: 0.75, satisfaction: 0.3, applisFantomes: 7, surfaceAttaque: 0.1, incidentsParSemaine: 0.1, comprehension: 0.8 }), "musee");
  assert.equal(profil(LTP, { fuitesMajeures: 1, fuites: 2, surfaceAttaque: 0.7 }), "farwest");
  assert.equal(profil(LTP, { ratioBudget: 1.8, ratioDebit: 1.4, comprehension: 0.7, incidentsParSemaine: 0.3, surfaceAttaque: 0.2 }), "facture");
  assert.equal(profil(LTP, { ratioVolume: 2, comprehension: 0.25, releve: 0.15, ratioDebit: 1.1, partLgtm: 0.1, incidentsParSemaine: 0.6 }), "chateau");
});

test("critères : de meilleurs chiffres donnent de meilleures notes et un meilleur score", () => {
  const LTP = charger();
  const mauvais = F(LTP).criteres(faitsMoyens({ ratioDebit: 0.8, satisfaction: 0.3, incidentsParSemaine: 1.4, comprehension: 0.35,
    couvertureReelle: 0.2, ratioBudget: 1.7, surfaceAttaque: 0.7, expositionLoi25: 0.6, fuites: 3, incidentsSecurite: 5, applisFantomes: 6,
    moral: 0.3, fatigue: 0.7, releve: 0.2, departs: 4 }));
  const bons = F(LTP).criteres(faitsMoyens({ ratioDebit: 1.6, satisfaction: 0.75, incidentsParSemaine: 0.2, comprehension: 0.75,
    couvertureReelle: 0.65, ratioBudget: 0.8, surfaceAttaque: 0.1, expositionLoi25: 0.1, moral: 0.8, fatigue: 0.2, releve: 0.7 }));
  for (const k of Object.keys(bons)) assert.ok(bons[k] > mauvais[k], `${k} : ${bons[k]} ≤ ${mauvais[k]}`);
});

test("le résultat suit les cumuls de la partie (valeur livrée vs référence)", () => {
  const LTP = charger();
  const a = LTP.moteur.creer(1), b = LTP.moteur.creer(1);
  for (const e of [a, b]) { e.semaine = 52; e.cumuls.semaines = 52; }
  a.cumuls.pointsLivres = a.fin.reference.debit * 52 * 0.8;
  b.cumuls.pointsLivres = b.fin.reference.debit * 52 * 1.5;
  const ra = F(LTP).calculerResultat(a, null, "annee"), rb = F(LTP).calculerResultat(b, null, "annee");
  assert.ok(rb.faits.ratioDebit > ra.faits.ratioDebit);
  assert.ok(rb.criteres.livraison > ra.criteres.livraison);
  assert.ok(rb.score >= ra.score);
});
