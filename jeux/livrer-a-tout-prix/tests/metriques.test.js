// Métriques (30-metriques.js, CONCEPTION §5) : calcul, historique, visibilité, MTTR grossier ou précis, cumuls.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";

function neuf(LTP, graine = 1) {
  const etat = LTP.moteur.creer(graine);
  etat.phase = "execution";
  return etat;
}
const M = (LTP) => LTP.systemes.metriques;
function calculer(LTP, etat) { M(LTP).semaine(etat, LTP.moteur.creerContexte(etat)); }

test("toutes les métriques des données sont calculées, finies, et l'historique est tenu", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  for (const m of LTP.donnees.metriques) {
    assert.ok(Number.isFinite(etat.metriques[m.id]), `${m.id} = ${etat.metriques[m.id]}`);
    assert.equal(etat.historique[m.id][0], etat.metriques[m.id]);
  }
  etat.semaine = 3;
  calculer(LTP, etat);
  for (const m of LTP.donnees.metriques) {
    assert.equal(etat.historique[m.id].length, 4, m.id);
    assert.ok(etat.historique[m.id].every((v) => Number.isFinite(v)), `${m.id} : trous dans l'historique`);
  }
});

test("une métrique ajoutée au contenu sans calcul ne fait pas planter", () => {
  const LTP = charger();
  LTP.donnees.metriques.push({ id: "nouvelle", groupe: "livraison", unite: "nb", sens: "haut", libelle: "Nouvelle" });
  LTP.donnees.metriquesParId.nouvelle = LTP.donnees.metriques.at(-1);
  const etat = neuf(LTP);
  assert.equal(etat.metriques.nouvelle, 0);
});

test("agrégats : sommes et moyennes des îlots et des développeurs", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.semaine = 1;
  etat.ilots.web.derniere.prFusionnees = 7;
  etat.ilots.iot.derniere.prFusionnees = 3;
  etat.ilots.web.derniere.pointsLivres = 12.5;
  for (const d of Object.values(etat.devs)) d.moral = 0.4;
  etat.devs.luc.present = false;
  etat.devs.luc.moral = 0;
  calculer(LTP, etat);
  assert.equal(etat.metriques.prFusionnees, 10);
  assert.equal(etat.metriques.pointsLivres, 12.5);
  assert.ok(Math.abs(etat.metriques.moral - 0.4) < 1e-9, "les absents ne comptent pas");
  assert.equal(etat.metriques.confiance, etat.org.confiance);
});

test("visibilité : les métriques cachées n'apparaissent qu'avec le bon levier", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  const vis = (id) => M(LTP).visible(etat, id);
  for (const id of ["frequenceDeploiement", "tauxEchec", "couvertureReelle", "comprehension", "surfaceAttaque", "expositionLoi25", "cartesOrphelines"]) {
    assert.equal(vis(id), false, `${id} cachée au départ`);
  }
  assert.equal(vis("prFusionnees"), true);
  assert.equal(vis("inconnue"), false);
  const L = etat.leviers.global;
  L.ci = "tests";
  assert.equal(vis("tauxEchec"), false);
  assert.equal(vis("frequenceDeploiement"), false);
  L.ci = "complet";
  assert.equal(vis("frequenceDeploiement"), true);
  assert.equal(vis("tauxEchec"), true, "même visibilité que la fréquence de déploiement");
  L.tests = "mutation";
  assert.equal(vis("couvertureReelle"), true);
  L.revuesExplication = "mensuelles";
  assert.equal(vis("comprehension"), true);
  L.passerelleMCP = true;
  assert.equal(vis("surfaceAttaque"), true);
  assert.equal(vis("cartesOrphelines"), false);
  L.politiqueRegistre = "proprietaires";
  assert.equal(vis("cartesOrphelines"), true);
  L.classification = "manuelle";
  assert.equal(vis("expositionLoi25"), true);
});

test("visibilité : l'observabilité révèle la couverture réelle et la compréhension ; le mode animateur montre tout", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.leviers.global.observabilite = "base";
  assert.equal(M(LTP).visible(etat, "couvertureReelle"), true);
  assert.equal(M(LTP).visible(etat, "comprehension"), false);
  etat.leviers.global.observabilite = "complete";
  assert.equal(M(LTP).visible(etat, "comprehension"), true);
  const autre = neuf(LTP);
  autre.options.voirCachees = true;
  for (const m of LTP.donnees.metriques) assert.equal(M(LTP).visible(autre, m.id), true, m.id);
});

test("une surcharge par îlot ne suffit pas : il faut la majorité des îlots", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.leviers.surcharges.web = { tests: "mutation" };
  assert.equal(M(LTP).visible(etat, "couvertureReelle"), false);
  for (const id of ["appro", "operations", "bureautique"]) etat.leviers.surcharges[id] = { tests: "mutation" };
  assert.equal(M(LTP).visible(etat, "couvertureReelle"), true);
});

test("MTTR : arrondi grossier sans observabilité, précis avec", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.semaine = 5;
  etat.incidents.push({ id: "t1", ilot: "web", gravite: 0.5, cause: "bogue", debut: 5, mttrHeures: 13.37, finSemaine: 5, resolu: false });
  calculer(LTP, etat);
  assert.equal(etat.metriques.mttr % LTP.K.metriques.arrondiMttr[0], 0, `grossier : ${etat.metriques.mttr}`);
  assert.ok(Math.abs(etat.metriquesDetail.mttrExact - 13.37) < 1e-9);
  etat.leviers.global.observabilite = "complete";
  calculer(LTP, etat);
  assert.ok(Math.abs(etat.metriques.mttr - 13.4) < 1e-6, `précis : ${etat.metriques.mttr}`);
});

test("incidents de la semaine, fuites et cumuls de la partie", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.semaine = 4;
  etat.incidents.push(
    { id: "a", ilot: "appro", gravite: 0.8, cause: "fuiteDonnees", debut: 4, mttrHeures: 10, finSemaine: 4, resolu: true },
    { id: "b", ilot: "web", gravite: 0.3, cause: "bogue", debut: 4, mttrHeures: 4, finSemaine: 4, resolu: true },
    { id: "d", ilot: "web", gravite: 0.9, cause: "paquetHallucine", debut: 4, mttrHeures: 4, finSemaine: 4, resolu: true },
    { id: "c", ilot: "web", gravite: 0.3, cause: "bogue", debut: 1, mttrHeures: 4, finSemaine: 1, resolu: true },
  );
  calculer(LTP, etat);
  assert.equal(etat.metriques.incidents, 3, "les incidents de la semaine comptent même s'ils sont déjà résolus");
  assert.equal(etat.cumuls.incidents, 3);
  assert.equal(etat.cumuls.fuites, 1);
  assert.equal(etat.cumuls.fuitesMajeures, 1);
  assert.equal(etat.cumuls.incidentsSecurite, 2);
  assert.equal(etat.cumuls.incidentsGraves, 2);
  assert.equal(etat.cumuls.semaines, 1);
});

test("les champs des autres systèmes sont lus s'ils existent", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.securite = { surfaceAttaque: 42, expositionLoi25: 73, parIlot: { web: { surface: 80, loi25: 0.25 } } };
  etat.ilots.web.retard = true;
  etat.ilots.iot.retard = 1;
  calculer(LTP, etat);
  assert.equal(etat.metriques.surfaceAttaque, 42);
  assert.ok(Math.abs(etat.metriquesDetail.parIlot.web.surfaceAttaque - 0.8) < 1e-9);
  assert.equal(etat.metriquesDetail.parIlot.web.expositionLoi25, 0.25);
  assert.equal(etat.metriques.expositionLoi25, 73);
  assert.ok(Math.abs(etat.metriques.retard - 2 / 7) < 1e-9);
});

test("sans 25-donnees-secu, surface d'attaque et Loi 25 calculées ici réagissent aux leviers", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  delete a.securite; delete b.securite;
  b.leviers.global.autonomie = 4;
  b.ilots.web.triade = { prive: true, nonFiable: true, sortie: true };
  calculer(LTP, a);
  calculer(LTP, b);
  assert.ok(b.metriquesDetail.parIlot.web.surfaceAttaque > a.metriquesDetail.parIlot.web.surfaceAttaque);
  const c = neuf(LTP);
  delete c.securite;
  c.org.donnees.classification = 0.9;
  calculer(LTP, c);
  assert.ok(c.metriquesDetail.parIlot.portail.expositionLoi25 < a.metriquesDetail.parIlot.portail.expositionLoi25);
});

test("cartes orphelines : équipées sans propriétaire présent", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  const ids = Object.keys(etat.capacites);
  for (const id of ids) { etat.capacites[id].equipeeSur = []; }
  calculer(LTP, etat);
  assert.equal(etat.metriques.cartesOrphelines, 0);
  if (ids.length >= 2) {
    etat.capacites[ids[0]].equipeeSur = ["web"];
    etat.capacites[ids[0]].proprietaire = null;
    etat.capacites[ids[1]].equipeeSur = ["iot"];
    etat.capacites[ids[1]].proprietaire = "luc";
    calculer(LTP, etat);
    assert.equal(etat.metriques.cartesOrphelines, 1);
    etat.devs.luc.present = false;
    calculer(LTP, etat);
    assert.equal(etat.metriques.cartesOrphelines, 2, "le propriétaire est parti");
  }
});
