// Registre des capacités (26-registre.js) : gains, saturation, conflits, péremption, triade,
// facteur données, Goodhart, développement maison, libre-service et politique du registre.
// Les cartes de test sont poussées dans le contenu avant la création de l'état.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger, json } from "./charger.js";

function carte(id, autres) {
  return {
    id, type: "skill", nom: id, provenance: "maison", trimestre: 1, disponible: "depart",
    auteur: null, coutDev: 0, ilotsSuggeres: [], description: "Carte de test.",
    gains: {}, poidsContexte: 0.05, branches: [], risque: { type: "aucun", proba: 0 },
    declencheur: null, prerequis: null, ...autres,
  };
}

const CARTES = [
  carte("t-rapide", { gains: { debit: 0.2, tauxDefaut: 0.8 }, poidsContexte: 0.05 }),
  carte("t-rapide2", { gains: { debit: 0.2 }, poidsContexte: 0.05 }),
  carte("t-lourde1", { poidsContexte: 0.2 }),
  carte("t-lourde2", { poidsContexte: 0.2 }),
  carte("t-tdd", { gains: { couverture: 0.1 }, declencheur: "tests", auteur: "sophie", provenance: "personnelle" }),
  carte("t-shipit", { gains: { debit: 0.1 }, declencheur: "tests", auteur: "mathieu", provenance: "personnelle" }),
  carte("t-local", { gains: { mttr: 0.5 }, ilotsGains: { operations: { mttr: 0.7 } } }),
  carte("t-commu", { type: "mcp", provenance: "communaute", gains: { debit: 0.1 }, branches: ["nonFiable"] }),
  carte("t-sortie", { type: "mcp", provenance: "fournisseur", branches: ["sortie"] }),
  carte("t-cachee", { type: "mcp", provenance: "communaute", disponible: "evenement" }),
  carte("t-maison", { gains: { debit: 0.2 }, disponible: "developpement", coutDev: 3 }),
  carte("t-mcp-catalogue", { type: "mcp", gains: {}, prerequis: { levier: "catalogue", valeurs: ["sources"] } }),
];

function preparer(graine = 1, leviers = {}, cartes = CARTES) {
  const LTP = charger();
  for (const c of cartes) LTP.donnees.capacites.push(c);
  const etat = LTP.moteur.creer(graine, { toutDebloque: true });
  etat.phase = "execution";
  Object.assign(etat.leviers.global, leviers);
  return { LTP, etat, R: LTP.systemes.registre };
}

// Équipe sans passer par la politique (raccourci de test).
function equiper(etat, id, ilot) {
  etat.capacites[id].equipeeSur.push(ilot);
  etat.ilots[ilot].capacites.push(id);
}

function semaine(LTP, etat) {
  etat.semaine += 1;
  for (const d of Object.values(etat.devs)) d.occupation = LTP.etat.occupationVide();
  const ctx = LTP.moteur.creerContexte(etat);
  LTP.systemes.registre.semaine(etat, ctx);
  return ctx;
}

test("une carte active applique ses gains (additifs et multiplicatifs) et ses gains propres à l'îlot", () => {
  const { LTP, etat } = preparer();
  equiper(etat, "t-rapide", "web");
  equiper(etat, "t-local", "operations");
  equiper(etat, "t-local", "web");
  semaine(LTP, etat);
  assert.ok(Math.abs(etat.ilots.web.mods.debit - 0.2) < 1e-9);
  assert.ok(etat.ilots.web.mods.mttr < 0.51 && etat.ilots.web.mods.mttr > 0.49);
  assert.ok(Math.abs(etat.ilots.operations.mods.mttr - 0.7) < 1e-9, "ilotsGains remplace le gain général");
  assert.equal(etat.ilots.iot.mods.debit, 0);
  assert.ok(Math.abs(etat.ilots.web.mods.poidsContexte - 0.1) < 1e-9);
});

test("saturation : au-delà du seuil de contexte, les gains baissent et le mauvais outil devient possible", () => {
  const a = preparer();
  equiper(a.etat, "t-rapide", "web");
  equiper(a.etat, "t-rapide2", "web");
  semaine(a.LTP, a.etat);
  const b = preparer();
  for (const id of ["t-rapide", "t-rapide2", "t-lourde1", "t-lourde2"]) equiper(b.etat, id, "web");
  semaine(b.LTP, b.etat);
  assert.ok(b.etat.ilots.web.mods.debit < a.etat.ilots.web.mods.debit, "gains réduits");
  const ap = b.R.apercu(b.etat, "web");
  assert.ok(ap.exces > 0 && ap.facteurSaturation < 1 && ap.mauvaisOutil > 0);
  assert.ok(ap.incoherence > 0);
  assert.equal(a.R.apercu(a.etat, "web").exces, 0);
});

test("conflit de déclencheurs entre deux skills d'un même îlot → incohérence", () => {
  const a = preparer();
  equiper(a.etat, "t-tdd", "inspections");
  semaine(a.LTP, a.etat);
  const b = preparer();
  equiper(b.etat, "t-tdd", "inspections");
  equiper(b.etat, "t-shipit", "inspections");
  const ctx = semaine(b.LTP, b.etat);
  assert.equal(a.etat.ilots.inspections.mods.incoherence, 0);
  assert.ok(b.etat.ilots.inspections.mods.incoherence > 0);
  const ap = b.R.apercu(b.etat, "inspections");
  assert.equal(ap.conflits.length, 1);
  assert.equal(ap.conflits[0].declencheur, "tests");
  assert.ok(ctx.journal.some((e) => e.cle === "registre.conflit"));
  // Les mêmes skills dans deux îlots différents ne se contredisent pas.
  const c = preparer();
  equiper(c.etat, "t-tdd", "inspections");
  equiper(c.etat, "t-shipit", "portail");
  semaine(c.LTP, c.etat);
  assert.equal(c.etat.ilots.inspections.mods.incoherence, 0);
});

test("péremption : sans propriétaire la carte se périme et une partie de ses gains s'inverse ; un propriétaire l'entretient", () => {
  const a = preparer();
  equiper(a.etat, "t-rapide", "web");
  for (let i = 0; i < 40; i++) semaine(a.LTP, a.etat);
  assert.ok(a.etat.capacites["t-rapide"].fraicheur < a.LTP.K.registre.seuilPeremption);
  assert.ok(a.etat.ilots.web.mods.tauxDefaut > 1, "le gain sur les défauts s'est inversé");
  assert.ok(a.etat.ilots.web.mods.debit < 0.2);
  assert.ok(a.etat.registre.perimees >= 1 && a.etat.registre.orphelines >= 1);

  const b = preparer();
  equiper(b.etat, "t-rapide", "web");
  b.etat.capacites["t-rapide"].proprietaire = "patrick";
  let occupation = 0;
  for (let i = 0; i < 40; i++) { semaine(b.LTP, b.etat); occupation = b.etat.devs.patrick.occupation.registre; }
  assert.ok(b.etat.capacites["t-rapide"].fraicheur > 0.9);
  assert.ok(occupation > 0, "l'entretien prend un peu de temps au propriétaire");
  assert.equal(b.R.apercu(b.etat, "web").cartes.find((c) => c.id === "t-rapide").orpheline, false);
});

test("triade : exposition de l'îlot, branches des cartes, accès à la prod, approbation des sorties", () => {
  const { LTP, etat } = preparer(1, { accesProd: "non" });
  semaine(LTP, etat);
  assert.deepEqual(json(etat.ilots.iot.triade), { prive: false, nonFiable: false, sortie: false });
  assert.deepEqual(json(etat.ilots.web.triade), { prive: false, nonFiable: true, sortie: true });
  etat.leviers.global.accesProd = "oui";
  semaine(LTP, etat);
  assert.equal(etat.ilots.iot.triade.prive, true);
  equiper(etat, "t-commu", "iot");
  equiper(etat, "t-sortie", "iot");
  semaine(LTP, etat);
  assert.ok(LTP.q.triadeComplete(etat, "iot"));
  etat.leviers.surcharges.iot = { approbationSorties: true };
  semaine(LTP, etat);
  assert.equal(etat.ilots.iot.triade.sortie, false);
  assert.equal(etat.ilots.web.triade.sortie, true);
});

test("facteur données : de mauvaises données augmentent les défauts, le MCP du catalogue les neutralise", () => {
  const { LTP, etat, R } = preparer(1, { autonomie: 3 });
  semaine(LTP, etat);
  const avant = etat.ilots.appro.mods.tauxDefaut;
  etat.org.donnees = { qualite: 0.2, duplication: 0.9, fraicheur: 0.2, classification: 0.1 };
  semaine(LTP, etat);
  const pire = etat.ilots.appro.mods.tauxDefaut;
  assert.ok(pire > avant);
  equiper(etat, "t-mcp-catalogue", "appro");
  semaine(LTP, etat);
  assert.equal(R.apercu(etat, "appro").cartes[0].raison, "prerequis", "inactive sans catalogue officiel");
  assert.ok(Math.abs(etat.ilots.appro.mods.tauxDefaut - pire) < 1e-9);
  etat.leviers.global.catalogue = "sources";
  semaine(LTP, etat);
  assert.ok(R.apercu(etat, "appro").catalogue);
  assert.ok(etat.ilots.appro.mods.tauxDefaut < pire);
});

test("Goodhart : l'objectif « nombre de PR » rapetisse les PR, l'objectif « délai » relâche l'attention", () => {
  const { LTP, etat } = preparer(1, { objectif: "pr" });
  semaine(LTP, etat);
  assert.ok(etat.ilots.portail.mods.taillePR < 1);
  assert.equal(etat.ilots.portail.mods.attention, 1);
  etat.leviers.global.objectif = "delai";
  semaine(LTP, etat);
  assert.equal(etat.ilots.portail.mods.taillePR, 1);
  assert.ok(etat.ilots.portail.mods.attention < 1);
});

test("actions : découverte, prérequis, liste blanche et propriétaire obligatoire", () => {
  const { LTP, etat } = preparer();
  const act = (a) => LTP.moteur.action(etat, a);
  assert.equal(act({ type: "equiper", carte: "t-cachee", ilot: "web" }).erreur, "carte-non-decouverte");
  assert.equal(act({ type: "equiper", carte: "t-mcp-catalogue", ilot: "web" }).erreur, "prerequis-absent");
  assert.equal(act({ type: "equiper", carte: "inexistante", ilot: "web" }).erreur, "carte-inconnue");
  assert.equal(act({ type: "equiper", carte: "t-commu", ilot: "web" }).erreur, undefined, "libre-service : permis");

  etat.leviers.global.politiqueRegistre = "listeBlanche";
  assert.equal(act({ type: "equiper", carte: "t-commu", ilot: "iot" }).erreur, "politique-liste-blanche");
  assert.equal(act({ type: "equiper", carte: "t-rapide", ilot: "iot" }).erreur, undefined);

  etat.leviers.global.politiqueRegistre = "proprietaires";
  assert.equal(act({ type: "equiper", carte: "t-rapide2", ilot: "iot" }).erreur, "proprietaire-requis");
  let r = act({ type: "proprietaire", carte: "t-rapide2", dev: "luc" });
  assert.equal(r.erreur, undefined);
  r = LTP.moteur.action(r.etat, { type: "equiper", carte: "t-rapide2", ilot: "iot" });
  assert.equal(r.erreur, undefined);
  assert.ok(r.etat.ilots.iot.capacites.includes("t-rapide2"));
  assert.ok(r.etat.capacites["t-rapide2"].equipeeSur.includes("iot"));
  assert.ok(r.etat.ilots.iot.mods.debit > 0, "les gains sont visibles tout de suite");

  r = LTP.moteur.action(r.etat, { type: "desequiper", carte: "t-rapide2", ilot: "iot" });
  assert.equal(r.erreur, undefined);
  assert.ok(!r.etat.ilots.iot.capacites.includes("t-rapide2"));
  assert.equal(LTP.moteur.action(r.etat, { type: "desequiper", carte: "t-rapide2", ilot: "iot" }).erreur, "carte-non-equipee");
});

test("développement maison : le développeur est réservé pendant coutDev semaines, puis la carte devient active", () => {
  const { LTP, etat } = preparer();
  equiper(etat, "t-maison", "web");
  let r = LTP.moteur.action(etat, { type: "developper", carte: "t-maison", dev: "patrick" });
  assert.equal(r.erreur, undefined);
  let e = r.etat;
  assert.equal(LTP.moteur.action(e, { type: "developper", carte: "t-maison", dev: "chloe" }).erreur, "deja-en-developpement");
  assert.equal(LTP.moteur.action(e, { type: "developper", carte: "t-rapide", dev: "chloe" }).erreur, "carte-non-developpable");
  const journal = [];
  for (let i = 0; i < 3; i++) {
    assert.equal(e.ilots.web.mods.debit, 0, `semaine ${i} : encore inactive`);
    const ctx = semaine(LTP, e);
    journal.push(...ctx.journal);
    assert.ok(e.devs.patrick.occupation.registre >= LTP.K.registre.partDeveloppement);
  }
  assert.ok(journal.some((x) => x.cle === "registre.carteDisponible"));
  assert.ok(e.ilots.web.mods.debit > 0.19);
  assert.equal(e.capacites["t-maison"].proprietaire, "patrick");
  assert.equal(LTP.moteur.action(e, { type: "developper", carte: "t-maison", dev: "chloe" }).erreur, "deja-developpee");
  semaine(LTP, e);
  assert.ok(e.devs.patrick.occupation.registre < LTP.K.registre.partDeveloppement, "plus que l'entretien");
  assert.ok(e.registre.connaissance.patrick > 0, "connaissance capturée");
});

test("libre-service : des développeurs curieux installent des cartes communautaires, pas avec la liste blanche", () => {
  let installes = 0, installesListe = 0;
  for (let g = 1; g <= 5; g++) {
    for (const pol of ["libre", "listeBlanche"]) {
      const { LTP, etat } = preparer(g, { politiqueRegistre: pol });
      let journal = [];
      for (let i = 0; i < 30; i++) journal = journal.concat(semaine(LTP, etat).journal);
      const n = Object.values(etat.ilots).filter((il) => il.capacites.includes("t-commu")).length;
      if (pol === "libre") {
        installes += n;
        if (n) assert.ok(journal.some((x) => x.cle === "registre.installationLibre"));
      } else installesListe += n;
    }
  }
  assert.ok(installes > 0);
  assert.equal(installesListe, 0);
});

test("apercu est pur et décrit gains, poids, triade et conflits", () => {
  const { LTP, etat, R } = preparer();
  equiper(etat, "t-tdd", "inspections");
  equiper(etat, "t-shipit", "inspections");
  equiper(etat, "t-maison", "inspections");
  const avant = JSON.stringify(etat);
  const ap = R.apercu(etat, "inspections");
  assert.equal(JSON.stringify(etat), avant);
  assert.equal(ap.cartes.length, 3);
  assert.equal(ap.cartes.find((c) => c.id === "t-maison").raison, "a-developper");
  assert.ok(ap.gains && ap.triade && Array.isArray(ap.conflits) && typeof ap.poids === "number");
  assert.equal(R.apercu(etat, "inconnu"), null);
});

test("partie complète avec cartes de test : modificateurs finis et bornés", () => {
  const { LTP, etat } = preparer(3);
  for (const id of ["t-rapide", "t-lourde1", "t-lourde2", "t-tdd", "t-shipit", "t-commu"]) equiper(etat, id, "portail");
  const fin = LTP.moteur.simuler(etat, 52).etat;
  for (const il of Object.values(fin.ilots)) {
    for (const [k, v] of Object.entries(il.mods)) assert.ok(Number.isFinite(v), `${il.id}.${k}`);
    assert.ok(il.mods.incoherence >= 0 && il.mods.incoherence <= 1);
    assert.ok(il.mods.tauxDefaut > 0);
  }
  for (const d of Object.values(fin.devs)) assert.ok(d.occupation.registre >= 0 && d.occupation.registre <= 1);
});
