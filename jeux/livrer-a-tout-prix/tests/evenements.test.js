// Événements (29-evenements.js, CONCEPTION §6) : tirage, ciblage, retirage, suites, résolution.
// Chaque test remplace le contenu par ses propres cartes : il ne dépend pas des cartes écrites par le contenu.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger, json } from "./charger.js";

function carte(id, extra = {}) {
  return {
    id, theme: "revue", trimestres: [1, 2, 3, 4], poids: 1, delaiRetirage: 0, unique: false, personnage: "ginette",
    titre: "Titre " + id, texte: "Texte",
    choix: [
      { id: "a", libelle: "A", consequence: "Conséquence A.", effets: [{ cible: "ilot", chemin: "satisfaction", op: "add", valeur: 0.1 }] },
      { id: "b", libelle: "B", consequence: "", effets: [{ confiance: -5 }] },
    ],
    ...extra,
  };
}
function neuf(LTP, graine = 1) {
  // Ces tests visent le tirage lui-même : sans espacement ni rattrapage de rythme (testés à part).
  LTP.K.evenements.espacementMin = 0;
  LTP.K.evenements.minimumParTrimestre = 0;
  const etat = LTP.moteur.creer(graine);
  etat.phase = "execution";
  etat.semaine = 2;
  return etat;
}
const E = (LTP) => LTP.systemes.evenements;
// Fait n tirages sur des semaines successives (hors fins de trimestre), en vidant l'événement entre chaque.
function tirages(LTP, etat, n) {
  const ids = [];
  for (let i = 0; i < n; i++) {
    etat.semaine = 2 + (i % 10);
    etat.evenementEnAttente = null;
    E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
    ids.push(etat.evenementEnAttente ? etat.evenementEnAttente.id : null);
  }
  return ids;
}

test("sans cartes, aucun tirage et aucun plantage", () => {
  const LTP = charger();
  LTP.donnees.evenements = [];
  const etat = neuf(LTP);
  assert.ok(tirages(LTP, etat, 200).every((x) => x === null));
});

test("la probabilité hebdomadaire est respectée (≈ K.evenements.probabilite)", () => {
  const LTP = charger();
  LTP.donnees.evenements = [carte("x")];
  const etat = neuf(LTP, 3);
  const n = 2000;
  const p = tirages(LTP, etat, n).filter(Boolean).length / n;
  const attendu = LTP.K.evenements.probabilite;
  assert.ok(Math.abs(p - attendu) < 0.05, `fréquence ${p}, attendu ${attendu}`);
  LTP.K.evenements.probabilite = 0;
  assert.ok(tirages(LTP, etat, 100).every((x) => x === null));
});

test("trimestres et conditions filtrent ; une condition qui lève est ignorée", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 1;
  LTP.donnees.evenements = [
    carte("t3", { trimestres: [3] }),
    carte("faux", { condition: () => false }),
    carte("boum", { condition: () => { throw new Error("contenu fautif"); } }),
    carte("ok", { condition: (etat, q) => q.levier(etat, "autonomie") === 0 }),
  ];
  const etat = neuf(LTP);
  const ids = tirages(LTP, etat, 50);
  assert.ok(ids.every((x) => x === "ok"), ids.join(","));
  etat.trimestre = 3;
  assert.ok(tirages(LTP, etat, 50).includes("t3"));
});

test("choisirCible : cible utilisée ; exception ou cible invalide → carte écartée", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 1;
  LTP.donnees.evenements = [
    carte("cible", { choisirCible: (etat, q, rng) => ({ dev: "amira" }) }),
    carte("boum", { choisirCible: () => { throw new Error("x"); } }),
    carte("vide", { choisirCible: () => null }),
    carte("fantome", { choisirCible: () => ({ ilot: "lune" }) }),
  ];
  const etat = neuf(LTP);
  for (let i = 0; i < 30; i++) {
    etat.evenementEnAttente = null;
    E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
    const att = etat.evenementEnAttente;
    assert.ok(att, "un événement est tiré");
    assert.equal(att.id, "cible");
    assert.deepEqual(json(att.ref), { dev: "amira", ilot: "iot" });
    assert.equal(att.vars.dev, "amira");
  }
});

test("cible par défaut : un îlot et un développeur qui existent ; 'org' : aucune cible", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 1;
  LTP.donnees.evenements = [carte("libre")];
  const etat = neuf(LTP);
  E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
  const ref = etat.evenementEnAttente.ref;
  assert.ok(etat.ilots[ref.ilot]);
  assert.equal(etat.devs[ref.dev].ilot, ref.ilot);
  LTP.donnees.evenements = [carte("org", { cible: "org" })];
  etat.evenementEnAttente = null;
  E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
  assert.deepEqual(json(etat.evenementEnAttente.ref), {});
});

test("délai de retirage et cartes uniques", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 1;
  LTP.donnees.evenements = [carte("rare", { delaiRetirage: 10 }), carte("unique", { unique: true })];
  const etat = neuf(LTP);
  etat.semaine = 5;
  etat.evenementsVus = { rare: 1, unique: 1 };
  E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
  assert.equal(etat.evenementEnAttente, null, "rare vue il y a 4 semaines, unique déjà vue");
  etat.semaine = 12;
  E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
  assert.equal(etat.evenementEnAttente.id, "rare");
});

test("les suites planifiées passent en premier, avec leur cible, même sans hasard", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 0;
  LTP.donnees.evenements = [carte("suite", { trimestres: [4], condition: () => false })];
  const etat = neuf(LTP);
  etat.evenementsForces = [{ id: "inconnue", semaine: 1, ref: {} }, { id: "suite", semaine: 2, ref: { ilot: "web" } }, { id: "suite", semaine: 9, ref: {} }];
  E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
  assert.equal(etat.evenementEnAttente.id, "suite");
  assert.equal(etat.evenementEnAttente.ref.ilot, "web");
  assert.equal(etat.evenementsForces.length, 1, "la suite future reste planifiée, l'inconnue est jetée");
});

test("pas de tirage la semaine du comité", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 1;
  LTP.donnees.evenements = [carte("x")];
  const etat = neuf(LTP);
  etat.semaine = LTP.K.etat.semainesParTrimestre;
  E(LTP).tirage(etat, LTP.moteur.creerContexte(etat));
  assert.equal(etat.evenementEnAttente, null);
});

test("le thème dominant du trimestre est favorisé", () => {
  const LTP = charger();
  LTP.K.evenements.probabilite = 1;
  LTP.donnees.evenements = [carte("revue", { theme: "revue" }), carte("secu", { theme: "Sécurité" })];
  const etat = neuf(LTP, 5);
  let ids = tirages(LTP, etat, 1000);
  const t1 = ids.filter((x) => x === "revue").length / ids.length;
  etat.trimestre = 3;
  ids = tirages(LTP, etat, 1000);
  const t3 = ids.filter((x) => x === "secu").length / ids.length;
  assert.ok(t1 > 0.6, `T1 : revue ${t1}`);
  assert.ok(t3 > 0.6, `T3 : sécurité ${t3} (thème normalisé)`);
});

test("résoudre : effets, evenementsVus, derniereDecision, journal ; choix inconnu refusé", () => {
  const LTP = charger();
  LTP.donnees.evenements = [carte("x")];
  let etat = neuf(LTP);
  etat.evenementEnAttente = { id: "x", ref: { ilot: "web" }, vars: { ilot: "web" } };
  const satAvant = etat.ilots.web.satisfaction;
  const refus = LTP.moteur.action(etat, { type: "choix", choix: "zzz" });
  assert.equal(refus.erreur, "choix-inconnu");
  assert.ok(refus.etat.evenementEnAttente, "l'état est inchangé");
  const r = LTP.moteur.action(etat, { type: "choix", choix: "a" });
  assert.equal(r.erreur, undefined);
  etat = r.etat;
  assert.equal(etat.evenementEnAttente, null);
  assert.ok(Math.abs(etat.ilots.web.satisfaction - satAvant - 0.1) < 1e-9);
  assert.equal(etat.evenementsVus.x, etat.semaine);
  assert.deepEqual(json(etat.derniereDecision), { evenement: "x", choix: "a", semaine: etat.semaine, ref: { ilot: "web" } });
  const note = r.journal.find((e) => e.cle === "evenements.resolu");
  assert.equal(note.vars.consequence, "Conséquence A.");
  // Choix sans conséquence écrite : texte générique.
  etat.evenementEnAttente = { id: "x", ref: {}, vars: {} };
  const r2 = LTP.moteur.action(etat, { type: "choix", choix: "b" });
  assert.ok(r2.journal.some((e) => e.cle === "evenements.resoluSansTexte"));
  assert.equal(r2.etat.org.confiance, etat.org.confiance - 5);
});

test("une carte disparue du contenu ne bloque pas la partie", () => {
  const LTP = charger();
  LTP.donnees.evenements = [];
  const etat = neuf(LTP);
  etat.evenementEnAttente = { id: "disparue", ref: {}, vars: {} };
  const r = LTP.moteur.action(etat, { type: "choix", choix: "a" });
  assert.equal(r.erreur, undefined);
  assert.equal(r.etat.evenementEnAttente, null);
});

test("une partie complète avec des cartes de test tire des événements et les résout", () => {
  const LTP = charger();
  LTP.donnees.evenements = [carte("x", { delaiRetirage: 3 }), carte("y", { theme: "jetons", choix: [{ id: "a", libelle: "A", effets: [{ evenement: "x", dans: 2 }] }] })];
  const { etat } = LTP.moteur.simuler(LTP.moteur.creer(11), 60);
  assert.equal(etat.phase, "fin");
  assert.ok(etat.evenementsCompte.tires >= 5, `tirés : ${etat.evenementsCompte.tires}`);
  assert.ok(Object.keys(etat.evenementsVus).length >= 1);
});
