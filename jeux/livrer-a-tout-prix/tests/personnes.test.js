// Personnes (27-personnes.js, CONCEPTION §4.10) : tests directionnels.
// On appelle les étapes du système directement, sur un état neuf, pour ne pas dépendre des autres systèmes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger, json, parcourir } from "./charger.js";

function neuf(LTP, graine = 1, options) {
  const etat = LTP.moteur.creer(graine, options);
  etat.phase = "execution";
  etat.semaine = 1;
  return etat;
}
const P = (LTP) => LTP.systemes.personnes;
const ctxDe = (LTP, etat) => LTP.moteur.creerContexte(etat);
function semaines(LTP, etat, n) {
  for (let i = 0; i < n; i++) {
    etat.semaine += 1;
    const ctx = ctxDe(LTP, etat);
    for (const d of Object.values(etat.devs)) d.occupation = LTP.etat.occupationVide();
    P(LTP).disponibilite(etat, ctx);
    P(LTP).semaine(etat, ctx);
  }
  return etat;
}

test("plus de sessions d'agent → plus de fatigue", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  a.devs.luc.sessions = 1;
  b.devs.luc.sessions = 6;
  P(LTP).semaine(a, ctxDe(LTP, a));
  P(LTP).semaine(b, ctxDe(LTP, b));
  assert.ok(b.devs.luc.fatigue > a.devs.luc.fatigue + 0.05, `${b.devs.luc.fatigue} vs ${a.devs.luc.fatigue}`);
});

test("la garde de nuit et les incidents fatiguent, la file de revue fatigue les réviseurs", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  LTP.q.creerIncident(b, { ilot: "operations", gravite: 0.6 });
  b.incidents[0].finSemaine = 10;
  b.ilots.portail.fileRevue = 30;
  P(LTP).semaine(a, ctxDe(LTP, a));
  P(LTP).semaine(b, ctxDe(LTP, b));
  assert.ok(b.devs.benoit.fatigue > a.devs.benoit.fatigue, "Benoît, de garde");
  assert.ok(b.devs.nadia.fatigue > a.devs.nadia.fatigue, "Nadia, réviseuse");
});

test("fatigue au-delà de 0,85 → congé forcé de deux semaines, puis retour", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.devs.laurence.fatigue = 0.9;
  const ctx = ctxDe(LTP, etat);
  P(LTP).disponibilite(etat, ctx);
  const d = etat.devs.laurence;
  assert.equal(d.etat, "conge");
  assert.equal(d.indisponibleJusqua, etat.semaine + LTP.K.personnes.dureeConge);
  assert.equal(LTP.q.disponible(etat, d), false);
  assert.ok(ctx.journal.some((e) => e.cle === "personnes.conge" && e.dev === "laurence"));
  semaines(LTP, etat, LTP.K.personnes.dureeConge);
  assert.notEqual(d.etat, "conge");
  assert.ok(d.fatigue < 0.85, `fatigue au retour ${d.fatigue}`);
  assert.equal(LTP.q.disponible(etat, d), true);
});

test("les réactions des personnages aux leviers font bouger le moral chaque semaine", () => {
  const LTP = charger();
  LTP.donnees.devs.find((d) => d.id === "raymond").reactions = [{ levier: "autonomie", valeurs: [4], moral: -0.02, bulle: "Pas question." }];
  const a = neuf(LTP), b = neuf(LTP);
  b.leviers.surcharges.appro = { autonomie: 4 };
  semaines(LTP, a, 4);
  semaines(LTP, b, 4);
  assert.ok(b.devs.raymond.moral < a.devs.raymond.moral - 0.03, `${b.devs.raymond.moral} vs ${a.devs.raymond.moral}`);
  // La réaction ne touche que l'îlot dont le levier effectif correspond.
  assert.ok(Math.abs(b.devs.amira.moral - a.devs.amira.moral) < 0.02);
});

test("changer un levier : choc de moral immédiat et bulle du personnage au journal", () => {
  const LTP = charger();
  for (const d of LTP.donnees.devs) d.reactions = [];
  LTP.donnees.devs.find((d) => d.id === "sophie").reactions = [{ levier: "tests", valeurs: ["mutation"], moral: 0.015, bulle: "Enfin des vrais tests !" }];
  LTP.donnees.devs.find((d) => d.id === "mathieu").reactions = [{ levier: "tests", valeurs: ["mutation"], moral: -0.005, bulle: "Ça va ralentir." }];
  let etat = LTP.moteur.creer(2);
  etat = LTP.moteur.action(etat, { type: "phase", phase: "planification" }).etat;
  const avant = etat.devs.sophie.moral;
  const r = LTP.moteur.action(etat, { type: "levier", id: "tests", valeur: "mutation" });
  assert.equal(r.erreur, undefined);
  assert.ok(r.etat.devs.sophie.moral > avant);
  assert.ok(r.etat.devs.mathieu.moral < etat.devs.mathieu.moral);
  const note = r.journal.find((e) => e.cle === "personnes.reaction");
  assert.ok(note, "note de réaction");
  assert.equal(note.dev, "sophie", "le personnage le plus concerné parle");
  assert.equal(note.vars.bulle, "Enfin des vrais tests !");
  // Sans réaction définie, aucune note.
  const r2 = LTP.moteur.action(etat, { type: "levier", id: "lint", valeur: "strict" });
  assert.ok(!r2.journal.some((e) => e.cle === "personnes.reaction"));
});

test("moral sous 0,15 → démission ; la compréhension des collègues baisse", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  etat.devs.patrick.moral = 0.05;
  const avantChloe = etat.devs.chloe.comprehension;
  const ctx = ctxDe(LTP, etat);
  P(LTP).semaine(etat, ctx);
  assert.equal(etat.devs.patrick.present, false);
  assert.ok(ctx.journal.some((e) => e.cle === "personnes.demission" && e.dev === "patrick"));
  assert.ok(etat.devs.chloe.comprehension < avantChloe);
  assert.equal(etat.personnes.departs[0].dev, "patrick");
});

test("la connaissance capturée dans une skill maison réduit la perte au départ", () => {
  const LTP = charger();
  // Seule notre carte de test, pour ne pas dépendre du contenu du registre.
  LTP.donnees.capacites = [];
  LTP.donnees.capacites.push({
    id: "test-skill-patrick", type: "skill", nom: "test", provenance: "maison", trimestre: 1, disponible: "depart",
    auteur: "patrick", coutDev: 0, ilotsSuggeres: ["web"], description: "", gains: {}, poidsContexte: 0, branches: [],
    risque: { type: "aucun", proba: 0 }, declencheur: null, prerequis: null,
  });
  const sans = neuf(LTP), avec = neuf(LTP);
  delete sans.capacites["test-skill-patrick"];
  assert.ok(avec.capacites["test-skill-patrick"], "carte créée avec la partie");
  assert.ok(P(LTP).capture(avec, "patrick") > 0);
  assert.equal(P(LTP).capture(sans, "patrick"), 0);
  for (const e of [sans, avec]) {
    e.devs.patrick.moral = 0.05;
    P(LTP).semaine(e, ctxDe(LTP, e));
  }
  assert.ok(avec.devs.chloe.comprehension > sans.devs.chloe.comprehension);
});

test("le contractuel part vers la semaine 26, sauf contrat prolongé", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  b.org.drapeaux[LTP.K.personnes.contractuel.drapeauProlonge] = true;
  for (const e of [a, b]) {
    e.semaine = LTP.K.personnes.contractuel.semaine - 1;
    P(LTP).disponibilite(e, ctxDe(LTP, e));
    assert.equal(e.devs.olivier.present, true);
    e.semaine += 1;
    P(LTP).disponibilite(e, ctxDe(LTP, e));
  }
  assert.equal(a.devs.olivier.present, false);
  assert.equal(a.personnes.departs[0].motif, "finContrat");
  assert.equal(b.devs.olivier.present, true);
});

test("compétence d'un junior : stagne avec autonomie ≥ 3 sans pair, monte avec le pair humain", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  a.leviers.surcharges.operations = { autonomie: 4 };
  a.leviers.global.pair = "aucun";
  b.leviers.surcharges.operations = { autonomie: 1 };
  b.leviers.global.pair = "humain";
  const depart = a.devs.tommy.competence;
  semaines(LTP, a, 10);
  semaines(LTP, b, 10);
  const gainA = a.devs.tommy.competence - depart, gainB = b.devs.tommy.competence - depart;
  assert.ok(gainB > 3 * gainA, `pair humain ${gainB} vs autonomie seule ${gainA}`);
  assert.ok(gainA >= 0);
});

test("formation : temps réservé pendant le délai, compétence IA qui monte plus vite", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  b.leviers.enAttente.push({ id: "formation", valeur: "programme", ilot: null, semaine: 99 });
  P(LTP).disponibilite(b, ctxDe(LTP, b));
  const occ = Object.values(b.devs).map((d) => d.occupation.formation);
  assert.ok(occ.every((x) => x > 0), "tout le monde a du temps de formation");
  assert.ok(Object.values(b.devs).some((d) => d.etat === "formation"), "certains sont en salle cette semaine");
  semaines(LTP, a, 4);
  semaines(LTP, b, 4);
  assert.ok(b.devs.karim.competenceIA > a.devs.karim.competenceIA + 0.02);
  // Une fois en vigueur, le programme continue de rapporter.
  const c = neuf(LTP), d = neuf(LTP);
  d.leviers.global.formation = "mentorat";
  semaines(LTP, c, 8);
  semaines(LTP, d, 8);
  assert.ok(d.devs.karim.competenceIA > c.devs.karim.competenceIA);
});

test("adhésion d'une réfractaire plafonnée, sauf avec le mentorat", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  b.leviers.global.formation = "mentorat";
  a.devs.diane.adhesion = 0.6;
  b.devs.diane.adhesion = 0.6;
  semaines(LTP, a, 30);
  semaines(LTP, b, 30);
  assert.ok(a.devs.diane.adhesion <= 0.3 + 1e-9, `sans mentorat ${a.devs.diane.adhesion}`);
  assert.ok(b.devs.diane.adhesion > 0.35, `avec mentorat ${b.devs.diane.adhesion}`);
});

test("la rigueur baisse avec la fatigue", () => {
  const LTP = charger();
  const a = neuf(LTP), b = neuf(LTP);
  a.devs.nadia.fatigue = 0.2;
  b.devs.nadia.fatigue = 0.84;
  b.devs.nadia.sessions = 6;
  for (let i = 0; i < 5; i++) { b.devs.nadia.fatigue = 0.84; semaines(LTP, a, 1); semaines(LTP, b, 1); }
  assert.ok(b.devs.nadia.rigueur < a.devs.nadia.rigueur);
});

test("relève : le programme de relève la fait monter, le gel la fait baisser", () => {
  const LTP = charger();
  const a = neuf(LTP, 1, { toutDebloque: true }), b = neuf(LTP, 1, { toutDebloque: true });
  a.leviers.global.embauche = "gel";
  b.leviers.global.embauche = "releve";
  const depart = a.org.releve;
  semaines(LTP, a, 20);
  semaines(LTP, b, 20);
  assert.ok(a.org.releve < depart);
  assert.ok(b.org.releve > depart);
});

test("bulle : propre au personnage, sinon générique, déterministe", () => {
  const LTP = charger();
  LTP.donnees.devs.find((d) => d.id === "benoit").bulles = { code: ["Encore un script bash."] };
  LTP.donnees.devs.find((d) => d.id === "benoit").reactions = [];
  LTP.donnees.devs.find((d) => d.id === "luc").bulles = {};
  LTP.donnees.devs.find((d) => d.id === "luc").reactions = [];
  const etat = neuf(LTP);
  etat.devs.benoit.etat = "code";
  etat.devs.luc.etat = "revue";
  assert.equal(P(LTP).bulle(etat, "benoit"), "Encore un script bash.");
  assert.ok(LTP.donnees.bulles.revue.includes(P(LTP).bulle(etat, "luc")));
  assert.equal(P(LTP).bulle(etat, "luc"), P(LTP).bulle(json(etat), "luc"));
  etat.devs.luc.etat = "conge";
  assert.ok(LTP.donnees.bulles.conge.includes(P(LTP).bulle(etat, "luc")));
  assert.equal(P(LTP).bulle(etat, "inconnu"), null);
});

test("des valeurs extrêmes restent bornées, sans NaN, et l'état reste du JSON", () => {
  const LTP = charger();
  const etat = neuf(LTP);
  for (const d of Object.values(etat.devs)) {
    d.fatigue = 1; d.moral = 0.2; d.sessions = 12; d.adhesion = 1; d.competence = 1; d.competenceIA = 0;
  }
  etat.ilots.portail.fileRevue = 1e6;
  semaines(LTP, etat, 30);
  parcourir(json(etat), (chemin, v) => {
    if (typeof v === "number") assert.ok(Number.isFinite(v), `${chemin} = ${v}`);
  });
  for (const d of Object.values(etat.devs)) {
    for (const [k, [min, max]] of Object.entries(LTP.etat.BORNES.dev)) assert.ok(d[k] >= min && d[k] <= max, `${d.id}.${k} = ${d[k]}`);
  }
  assert.ok(etat.org.releve >= 0 && etat.org.releve <= 1);
});
