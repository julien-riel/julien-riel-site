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
  etat = LTP.moteur.action(etat, { type: "levier", id: "autonomie", valeur: 3, ilot: "web" }).etat;
  assert.equal(LTP.q.levier(etat, "autonomie", "web"), 3);
  assert.equal(LTP.q.levier(etat, "autonomie", "iot"), 1);
  assert.equal(LTP.moteur.action(etat, { type: "levier", id: "modele", valeur: "haut", ilot: "web" }).erreur, "levier-non-surchargeable");
});
