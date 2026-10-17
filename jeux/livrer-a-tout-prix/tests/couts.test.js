// Coûts (24) : jetons, prix des modèles, plafond, coûts fixes, budget (CONCEPTION §4.6).
import { test } from "node:test";
import assert from "node:assert/strict";
import { LTP, partie, avancer, moyenneGraines, GRAINES, somme, ilots, devs, carnetPlein } from "./production-outils.js";

function moyenne(leviers, n, f, options = {}) {
  return moyenneGraines(GRAINES, (g) => {
    let total = 0;
    avancer(partie(g, leviers, options), n, { avant: options.avant, apres: (e) => { total += f(e); } });
    return total / n;
  });
}
const cout = (e) => e.org.coutSemaine;
const jetons = (e) => e.org.jetonsSemaine;
const IA = [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 4 }];

test("les totaux de l'organisation sont cohérents avec les îlots et les développeurs", () => {
  const e0 = partie(1, IA);
  const e1 = avancer(e0, 1);
  const e2 = avancer(e1, 1);
  assert.equal(e2.org.jetonsSemaine, somme(ilots(e2), (i) => i.derniere.jetons));
  for (const il of ilots(e2)) {
    const ds = devs(e2).filter((d) => d.ilot === il.id);
    assert.ok(Math.abs(il.derniere.jetons - somme(ds, (d) => d.derniere.jetons)) <= ds.length, il.id);
  }
  assert.ok(Math.abs(e2.org.budget - (e1.org.budget - e2.org.coutSemaine)) < 0.05);
  assert.ok(Math.abs(e2.org.coutCumul - (e1.org.coutSemaine + e2.org.coutSemaine)) < 0.05);
  assert.equal(e2.org.jetonsCumul, e1.org.jetonsSemaine + e2.org.jetonsSemaine);
  assert.ok(e2.org.coutSemaine >= somme(ilots(e2), (i) => i.derniere.cout) - 0.05, "les coûts fixes d'organisation s'ajoutent");
});

test("modèle haut de gamme : moins de défauts, plus de coût", () => {
  const haut = [...IA, { id: "modele", valeur: "haut" }], inter = [...IA, { id: "modele", valeur: "intermediaire" }];
  assert.ok(moyenne(haut, 6, cout) > 2 * moyenne(inter, 6, cout));
  const echappes = (e) => somme(ilots(e), (i) => i.derniere.defautsEchappes);
  assert.ok(moyenne(haut, 6, echappes) < moyenne(inter, 6, echappes));
});

test("modèle choisi selon la tâche : environ −35 % sur le prix des jetons", () => {
  const e = partie(1, IA);
  const p0 = LTP.systemes.couts.prixModele(e);
  const e2 = LTP.moteur.action(e, { type: "levier", id: "modeleParTache", valeur: true }).etat;
  const p1 = LTP.systemes.couts.prixModele(e2);
  assert.ok(Math.abs(p1 / p0 - 0.65) < 0.01);
});

test("modèle local : jetons presque gratuits, mais un coût fixe", () => {
  const local = [...IA, { id: "modele", valeur: "local" }];
  const jetonsCout = (e) => somme(ilots(e), (i) => i.derniere.cout);
  assert.ok(moyenne(local, 4, jetonsCout) < moyenne(IA, 4, jetonsCout) / 5);
  const e = avancer(partie(1, local), 1);
  assert.ok(LTP.systemes.couts.coutsFixes(e).org > LTP.systemes.couts.coutsFixes(avancer(partie(1, IA), 1)).org);
});

test("plafond strict : moins de jetons, des points perdus notés au journal", () => {
  const lourd = [{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }];
  const opt = { avant: carnetPlein };
  const sans = moyenne(lourd, 4, jetons, opt);
  const strict = moyenne([...lourd, { id: "plafondJetons", valeur: "strict" }], 4, jetons, opt);
  assert.ok(strict < sans * 0.8, `${strict} contre ${sans}`);
  const points = (e) => somme(ilots(e), (i) => i.derniere.points);
  assert.ok(moyenne([...lourd, { id: "plafondJetons", valeur: "strict" }], 4, points, opt) < moyenne(lourd, 4, points, opt));
  const e = avancer(partie(1, [...lourd, { id: "plafondJetons", valeur: "strict" }]), 2, opt);
  assert.ok(e.journal.some((j) => j.cle === "couts.plafondCoupe"));
  for (const il of ilots(e)) assert.ok(il.derniere.jetons <= LTP.systemes.couts.plafondIlot(e, il) + 1, il.id);
});

test("plafond souple : alerte au journal, sans coupure", () => {
  const lourd = [{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }, { id: "plafondJetons", valeur: "souple" }];
  const e = avancer(partie(1, lourd), 2, { avant: carnetPlein });
  assert.ok(e.journal.some((j) => j.cle === "couts.plafondSouple"));
  assert.ok(somme(ilots(e), (i) => i.derniere.jetons) > somme(ilots(e), (i) => LTP.systemes.couts.plafondIlot(e, i)) * 0.9);
});

test("les leviers d'outillage ont un coût fixe hebdomadaire", () => {
  const e0 = partie(1, []);
  const f0 = LTP.systemes.couts.coutsFixes(e0);
  const e1 = partie(1, [{ id: "ci", valeur: "complet" }, { id: "observabilite", valeur: "complete" }, { id: "passerelleMCP", valeur: true }, { id: "bacASable", valeur: "filtre" }]);
  const f1 = LTP.systemes.couts.coutsFixes(e1);
  const total = (f) => f.org + Object.values(f.ilots).reduce((s, x) => s + x, 0);
  assert.ok(total(f1) > total(f0) + 2000);
  // La formation se paie dès la décision, même si elle prend effet dans quatre semaines.
  const e2 = partie(1, [{ id: "formation", valeur: "programme" }]);
  assert.ok(LTP.systemes.couts.coutsFixes(e2).org > f0.org);
});

test("budget : une stratégie prudente dépense peu, le « tout IA » haut de gamme dépasse nettement", () => {
  const projection = (leviers) => moyenneGraines([1, 2, 3], (g) => {
    let total = 0;
    avancer(partie(g, leviers), 8, { apres: (e) => { total += e.org.coutSemaine; } });
    return (total / 8) * 52 / LTP.K.etat.org.budgetAnnuel;
  });
  assert.ok(projection([]) < 0.5);
  assert.ok(projection([{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }, { id: "modele", valeur: "haut" }]) > 1.5);
});

test("budget dépassé : journal et confiance qui s'effrite", () => {
  const e0 = partie(1, IA, { preparer: (x) => { x.org.budget = 10; } });
  const e1 = avancer(e0, 1);
  assert.ok(e1.org.budget < 0);
  assert.ok(e1.journal.some((j) => j.cle === "couts.budgetDepasse"));
  // Confiance comparée à la même partie sans dépassement (les autres systèmes bougent aussi la confiance).
  const t1 = avancer(partie(1, IA), 1);
  assert.ok(e1.org.confiance < t1.org.confiance);
});

test("boucles d'agent : rares mais coûteuses, absentes sans autonomie", () => {
  let boucles = 0;
  for (const g of [1, 2, 3, 4, 5, 6, 7, 8]) {
    const e = avancer(partie(g, [{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }]), 10);
    boucles += e.journal.filter((j) => j.cle === "couts.boucle").length;
  }
  assert.ok(boucles > 0 && boucles < 200, `${boucles} boucles`);
  let aucune = 0;
  for (const g of [1, 2, 3]) aucune += avancer(partie(g, []), 10).journal.filter((j) => j.cle === "couts.boucle").length;
  assert.equal(aucune, 0);
});
