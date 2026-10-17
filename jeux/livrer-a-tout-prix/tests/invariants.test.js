// Invariants du moteur : des parties aléatoires complètes ne produisent ni valeur hors bornes, ni NaN,
// ni plantage, et restent rapides.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger, parcourir, strategieAleatoire } from "./charger.js";

const LTP = charger();

function verifierEtat(etat, contexte) {
  parcourir(etat, (chemin, v) => {
    if (typeof v === "number") assert.ok(Number.isFinite(v), `${contexte} : ${chemin} = ${v}`);
  });
  const B = LTP.etat.BORNES;
  const verifier = (type, obj, nom) => {
    for (const [chemin, [min, max]] of Object.entries(B[type])) {
      const v = LTP.outils.lireChemin(obj, chemin);
      if (typeof v !== "number") continue;
      assert.ok(v >= min - 1e-9 && v <= max + 1e-9, `${contexte} : ${nom}.${chemin} = ${v} hors [${min}, ${max}]`);
    }
  };
  verifier("org", etat.org, "org");
  for (const il of Object.values(etat.ilots)) {
    verifier("ilot", il, il.id);
    assert.ok(il.backlog >= 0 && il.codebase.volume >= 0 && il.fileRevue >= 0, `${contexte} : ${il.id} négatif`);
  }
  for (const d of Object.values(etat.devs)) verifier("dev", d, d.id);
}

test("une partie neuve respecte les bornes et se sérialise", () => {
  const etat = LTP.moteur.creer(1);
  verifierEtat(etat, "création");
  parcourir(etat, (chemin, v) => {
    assert.ok(typeof v !== "function" && v !== undefined, `création : ${chemin} n'est pas du JSON`);
  });
  assert.equal(Object.keys(etat.devs).length, 21);
  assert.equal(etat.ordreIlots.length, 7);
});

test("le moteur est déterministe pour une graine donnée", () => {
  const a = LTP.moteur.simuler(LTP.moteur.creer(7), 52, strategieAleatoire(LTP, 7)).etat;
  const b = LTP.moteur.simuler(LTP.moteur.creer(7), 52, strategieAleatoire(LTP, 7)).etat;
  assert.deepEqual(JSON.parse(JSON.stringify(a)), JSON.parse(JSON.stringify(b)));
});

test("100 parties aléatoires vont jusqu'au bout sans sortir des bornes, en moins de 200 ms chacune", () => {
  let total = 0;
  for (let g = 1; g <= 100; g++) {
    const debut = performance.now();
    const { etat } = LTP.moteur.simuler(LTP.moteur.creer(g), 60, strategieAleatoire(LTP, g));
    total += performance.now() - debut;
    verifierEtat(etat, `graine ${g}`);
    assert.equal(etat.phase, "fin", `graine ${g} : la partie ne s'est pas terminée (phase ${etat.phase}, semaine ${etat.semaine})`);
    assert.ok(etat.resultat && etat.resultat.profil, `graine ${g} : pas de résultat`);
  }
  assert.ok(total / 100 < 200, `durée moyenne ${Math.round(total / 100)} ms`);
});

test("semaine() ne mute pas l'état reçu", () => {
  let etat = LTP.moteur.creer(3);
  etat = LTP.moteur.action(etat, { type: "phase", phase: "planification" }).etat;
  etat = LTP.moteur.action(etat, { type: "phase", phase: "execution" }).etat;
  const avant = JSON.stringify(etat);
  LTP.moteur.semaine(etat);
  assert.equal(JSON.stringify(etat), avant);
});

test("les leviers à délai prennent effet après le délai", () => {
  let etat = LTP.moteur.creer(4, { toutDebloque: true });
  etat = LTP.moteur.action(etat, { type: "phase", phase: "planification" }).etat;
  const r = LTP.moteur.action(etat, { type: "levier", id: "formation", valeur: "programme" });
  assert.equal(r.erreur, undefined);
  etat = r.etat;
  assert.equal(LTP.q.levier(etat, "formation"), "aucune");
  etat = LTP.moteur.action(etat, { type: "phase", phase: "execution" }).etat;
  for (let i = 0; i < 4; i++) {
    etat = LTP.moteur.semaine(etat).etat;
    if (etat.evenementEnAttente) etat = { ...etat, evenementEnAttente: null };
  }
  assert.equal(LTP.q.levier(etat, "formation"), "programme");
});

test("un levier verrouillé est refusé, une surcharge par îlot est lue", () => {
  let etat = LTP.moteur.creer(5);
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "observabilite", valeur: "base" }).erreur, "levier-verrouille");
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "autonomie", valeur: 1 }).erreur, "levier-verrouille", "autonomie : semaine 3");
  etat = LTP.moteur.creer(5, { toutDebloque: true });
  etat = LTP.moteur.action(etat, { type: "levier", id: "autonomie", valeur: 3, ilot: "web" }).etat;
  assert.equal(LTP.q.levier(etat, "autonomie", "web"), 3);
  assert.equal(LTP.q.levier(etat, "autonomie", "iot"), 0);
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "modele", valeur: "haut", ilot: "web" }).erreur, "levier-non-surchargeable");
});

test("calendrier : les leviers et les niveaux d'autonomie arrivent un à un", () => {
  const cal = LTP.donnees.calendrier;
  const semaines = cal.map((c) => c.semaine);
  assert.equal(new Set(semaines).size, semaines.length, "une nouveauté par semaine au plus");
  for (const l of LTP.donnees.leviers) assert.ok(l.semaine >= 1 && l.semaine <= 52, l.id);
  assert.equal(LTP.q.defLevier("licences").semaine, 1);
  // Semaine 1 : seules les licences ; le panneau de l'IA est le seul ouvert.
  let etat = LTP.moteur.action(LTP.moteur.creer(6), { type: "phase", phase: "planification" }).etat;
  const ouverts = LTP.donnees.leviers.filter((l) => LTP.q.levierDebloque(etat, l.id)).map((l) => l.id);
  assert.equal(ouverts.join(","), "licences");
  assert.equal(LTP.donnees.panneaux.filter((p) => LTP.q.panneauDebloque(etat, p.id)).map((p) => p.id).join(","), "ia");
  assert.equal(etat.nouveaute && etat.nouveaute.semaine, 1, "note de service de la semaine 1");
  // Semaine 3 : autonomie 0 et 1, mais pas 2.
  etat = JSON.parse(JSON.stringify(etat));
  etat.semaine = 2;
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "autonomie", valeur: 1 }).erreur, undefined);
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "autonomie", valeur: 2 }).erreur, "valeur-verrouillee");
  etat.semaine = 9;
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "autonomie", valeur: 2 }).erreur, undefined);
});

test("licences : sans licence, ni session ni jeton ; le groupe pilote compte une personne par îlot", () => {
  const pilotes = LTP.donnees.devs.filter((d) => d.pilote);
  assert.equal(pilotes.length, 7);
  assert.equal(new Set(pilotes.map((d) => d.ilot)).size, 7);
  let etat = LTP.moteur.action(LTP.moteur.creer(8), { type: "phase", phase: "planification" }).etat;
  etat = LTP.moteur.action(etat, { type: "phase", phase: "execution" }).etat;
  etat = LTP.moteur.semaine(etat).etat;
  for (const d of Object.values(etat.devs)) {
    if (d.pilote) assert.equal(d.sessions, 1, d.id);
    else { assert.equal(d.sessions, 0, d.id); assert.equal(d.derniere.jetons, 0, d.id); }
  }
});

test("journées : quatre jours calmes, le vendredi calcule la semaine ; une carte attend un jour de la semaine suivante", () => {
  let etat = LTP.moteur.action(LTP.moteur.creer(9), { type: "phase", phase: "planification" }).etat;
  etat = LTP.moteur.action(etat, { type: "phase", phase: "execution" }).etat;
  for (let j = 0; j < 4; j++) {
    const r = LTP.moteur.jour(etat);
    assert.ok(r.avance && !r.finSemaine);
    etat = r.etat;
    assert.equal(etat.semaine, 0);
    assert.equal(etat.jour, j + 1);
    assert.ok(etat.reflexions.liste.length > 0, "des réflexions chaque jour");
  }
  const r = LTP.moteur.jour(etat);
  assert.ok(r.finSemaine);
  assert.equal(r.etat.semaine, 1);
  assert.equal(r.etat.jour, 0);
  // Une carte tirée le vendredi n'arrive pas tout de suite.
  const e2 = JSON.parse(JSON.stringify(r.etat));
  e2.jour = 4;
  e2.evenementEnAttente = null;
  e2.evenementProgramme = null;
  const LTP2 = charger();
  LTP2.K.evenements.probabilite = 1;
  LTP2.K.evenements.espacementMin = 0;
  LTP2.K.evenements.semainesDebut = 0;
  const r2 = LTP2.moteur.jour(e2);
  assert.equal(r2.etat.evenementEnAttente, null, "pas de carte le lundi matin");
  assert.ok(r2.etat.evenementProgramme && r2.etat.evenementProgramme.jour >= 1, "carte programmée");
  let e3 = r2.etat;
  for (let j = 0; j < 4 && !e3.evenementEnAttente; j++) e3 = LTP2.moteur.jour(e3).etat;
  assert.ok(e3.evenementEnAttente, "la carte arrive dans la semaine");
  assert.equal(e3.semaine, 2, "avant le calcul du vendredi suivant");
});
