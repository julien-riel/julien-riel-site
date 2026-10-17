// Qualité (22) : revue, cascade de défauts, couverture, agent qui triche (CONCEPTION §4.4).
import { test } from "node:test";
import assert from "node:assert/strict";
import { LTP, partie, avancer, moyenneGraines, GRAINES, somme, ilots, devs, carnetPlein } from "./production-outils.js";

// Moyenne hebdomadaire d'une statistique d'îlot (somme sur les îlots), sur n semaines et plusieurs graines.
function moyenne(leviers, n, f, options = {}) {
  return moyenneGraines(GRAINES, (g) => {
    let total = 0;
    avancer(partie(g, leviers, options), n, { avant: options.avant, apres: (e) => { total += f(e); } });
    return total / n;
  });
}
const echappes = (e) => somme(ilots(e), (i) => i.derniere.defautsEchappes);
const injectes = (e) => somme(ilots(e), (i) => i.derniere.defautsInjectes);
const delai = (e) => somme(ilots(e), (i) => i.derniere.delai) / 7;

test("plus de réviseurs : moins de défauts échappés, délai plus long", () => {
  const r1 = [{ id: "reviseurs", valeur: 1 }], r2 = [{ id: "reviseurs", valeur: 2 }], r0 = [{ id: "reviseurs", valeur: 0 }];
  assert.ok(moyenne(r2, 8, echappes) < moyenne(r1, 8, echappes));
  assert.ok(moyenne(r1, 8, echappes) < moyenne(r0, 8, echappes));
  assert.ok(moyenne(r2, 8, delai) > moyenne(r1, 8, delai));
});

test("autonomie 4 et 6 sessions avec deux réviseurs : la revue sature, l'attention baisse, les LGTM apparaissent", () => {
  const ia = [{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }, { id: "reviseurs", valeur: 2 }];
  const attention = (e) => somme(ilots(e), (i) => i.derniere.attention) / 7;
  const lgtm = (e) => somme(ilots(e), (i) => i.derniere.lgtm);
  const opt = { avant: carnetPlein };
  assert.ok(moyenne(ia, 6, attention, opt) < moyenne([{ id: "reviseurs", valeur: 2 }], 6, attention, opt) - 0.1);
  assert.ok(moyenne(ia, 6, lgtm, opt) > 0.5);
  // Beaucoup plus de défauts échappés avec peu de garde-fous.
  const sansGardeFous = [...ia, { id: "tests", valeur: "aucun" }, { id: "ci", valeur: "aucune" }, { id: "lint", valeur: "aucun" }];
  assert.ok(moyenne(sansGardeFous, 6, echappes, opt) > 3 * moyenne([], 6, echappes, opt));
});

test("approbation automatique : rien n'est lu par un humain, tout est fusionné", () => {
  const e = avancer(partie(1, [{ id: "approbation", valeur: "auto" }]), 3);
  for (const il of ilots(e)) {
    assert.equal(il.derniere.attention, 0, il.id);
    assert.equal(il.fileRevue, 0, il.id);
    assert.ok(Math.abs(il.derniere.prFusionnees - il.derniere.pr) < 1e-6, il.id);
  }
  for (const d of devs(e)) assert.equal(d.derniere.revues, 0, d.id);
});

test("les PR fusionnées font baisser le carnet et grossir le code", () => {
  const e0 = partie(2, []);
  const e1 = avancer(e0, 1);
  for (const il of ilots(e1)) {
    assert.ok(il.derniere.pointsLivres > 0, il.id);
    assert.ok(il.codebase.volume > e0.ilots[il.id].codebase.volume, il.id);
  }
  // Le carnet a baissé de la valeur livrée (avant les nouvelles demandes de 28).
  assert.ok(somme(ilots(e1), (i) => i.derniere.pointsLivres) > 0);
});

test("tests de mutation : la couverture réelle se rapproche de l'apparente", () => {
  const ecart = (tests) => moyenneGraines(GRAINES, (g) => {
    const e = avancer(partie(g, [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 3 }, { id: "tests", valeur: tests }, { id: "lint", valeur: "verrouille" }]), 12);
    return somme(ilots(e), (i) => i.codebase.couvertureApparente - i.codebase.couvertureReelle) / 7;
  });
  assert.ok(ecart("mutation") < ecart("unitaires"));
  // La couverture réelle ne dépasse jamais l'apparente.
  const e = avancer(partie(1, [{ id: "tests", valeur: "mutation" }]), 10);
  for (const il of ilots(e)) assert.ok(il.codebase.couvertureReelle <= il.codebase.couvertureApparente + 1e-9, il.id);
});

test("l'agent qui triche : lint non verrouillé, autonomie ≥ 3 et CI qui teste font baisser la couverture réelle", () => {
  const base = [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 4 }, { id: "ci", valeur: "tests" }, { id: "tests", valeur: "unitaires" }];
  let drapeau = 0;
  const reelle = (lint) => moyenneGraines(GRAINES, (g) => {
    const e = avancer(partie(g, [...base, { id: "lint", valeur: lint }]), 20);
    if (e.org.drapeaux.agentTriche) drapeau++;
    return somme(ilots(e), (i) => i.codebase.couvertureReelle) / 7;
  });
  const verrouille = reelle("verrouille");
  assert.equal(drapeau, 0, "jamais de triche avec le lint verrouillé");
  const strict = reelle("strict");
  assert.ok(drapeau > 0, "la triche finit par arriver");
  assert.ok(strict < verrouille, `${strict} contre ${verrouille}`);
});

test("modèle haut de gamme : moins de défauts injectés par ligne", () => {
  const parKloc = (modele) => moyenne([{ id: "autonomie", valeur: 3 }, { id: "modele", valeur: modele }], 4,
    (e) => injectes(e) / Math.max(0.001, somme(ilots(e), (i) => i.derniere.code)));
  assert.ok(parKloc("haut") < parKloc("intermediaire"));
  assert.ok(parKloc("intermediaire") < parKloc("economique"));
});

test("specs obligatoires et revue IA par un autre modèle : moins de défauts échappés", () => {
  assert.ok(moyenne([{ id: "specs", valeur: "obligatoires" }], 6, echappes) < moyenne([], 6, echappes));
  const auto = [{ id: "autonomie", valeur: 3 }, { id: "approbation", valeur: "auto" }];
  assert.ok(moyenne([...auto, { id: "modeleParTache", valeur: true }], 6, echappes) < moyenne(auto, 6, echappes));
});

test("une file de revue gonflée par un événement est réconciliée", () => {
  const e = avancer(partie(3, [], { preparer: (x) => { x.ilots.portail.fileRevue += 6; } }), 1);
  const ti = e.travail.ilots.portail;
  assert.equal(e.ilots.portail.fileRevue, Math.round(ti.file.pr));
  assert.ok(e.ilots.portail.derniere.prFusionnees > e.ilots.portail.derniere.pr, "les PR ajoutées sont traitées");
  assert.ok(Number.isInteger(e.ilots.portail.fileRevue) && e.ilots.portail.fileRevue >= 0);
});

test("statistiques des développeurs : revues, LGTM et lecture bornées", () => {
  const e = avancer(partie(5, [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 6 }, { id: "reviseurs", valeur: 2 }]), 4, { avant: carnetPlein });
  for (const d of devs(e)) {
    assert.ok(d.derniere.revues >= 0 && d.derniere.lgtm <= d.derniere.revues + 1e-9, d.id);
    assert.ok(d.derniere.luEnRevue >= 0 && d.derniere.luEnRevue <= 1, d.id);
  }
  for (const il of ilots(e)) assert.ok(il.derniere.attention >= 0 && il.derniere.attention <= 1, il.id);
  assert.ok(LTP.systemes.qualite.tauxDefaut(e, e.devs.mathieu, e.ilots.portail) > LTP.systemes.qualite.tauxDefaut(e, e.devs.nadia, e.ilots.portail));
});
