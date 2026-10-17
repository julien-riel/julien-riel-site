// Références croisées du contenu : tout identifiant cité dans les données existe,
// et tout effet déclaratif vise un chemin réel de l'état.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";

const LTP = charger();
const D = LTP.donnees;
const etat = LTP.moteur.creer(1);
const ilots = new Set(D.ilots.map((i) => i.id));
const devs = new Set(D.devs.map((d) => d.id));
const cartes = new Set((D.capacites || []).map((c) => c.id));
const evenements = new Set((D.evenements || []).map((e) => e.id));
const personnages = new Set([...Object.keys(D.personnages), ...devs, "direction"]);
const chemins = {
  org: new Set(LTP.effets.cheminsNumeriques(etat.org)),
  ilot: new Set(LTP.effets.cheminsNumeriques(Object.values(etat.ilots)[0])),
  dev: new Set(LTP.effets.cheminsNumeriques(Object.values(etat.devs)[0])),
};

function typeCible(cible) {
  const c = String(cible || "org");
  if (c === "org") return "org";
  if (c === "ilot" || c === "ilots" || c.startsWith("ilot:")) return "ilot";
  if (c === "dev" || c === "devs" || c.startsWith("dev:")) return "dev";
  return null;
}

function verifierEffets(effets, ou) {
  assert.ok(Array.isArray(effets), `${ou} : effets doit être une liste`);
  for (const e of effets) {
    const lieu = `${ou} → ${JSON.stringify(e)}`;
    if (e.chemin) {
      const type = typeCible(e.cible);
      assert.ok(type, `${lieu} : cible inconnue`);
      if (String(e.cible).startsWith("ilot:")) assert.ok(ilots.has(e.cible.slice(5)), `${lieu} : îlot inconnu`);
      if (String(e.cible).startsWith("dev:")) assert.ok(devs.has(e.cible.slice(4)), `${lieu} : dev inconnu`);
      assert.ok(chemins[type].has(e.chemin), `${lieu} : chemin « ${e.chemin} » absent de ${type}`);
      assert.ok(["add", "mul", "set", undefined].includes(e.op), `${lieu} : op inconnue`);
      if (e.op !== "set") assert.equal(typeof e.valeur, "number", `${lieu} : valeur numérique attendue`);
      if (e.filtre && e.filtre.ilot && e.filtre.ilot !== "courant") assert.ok(ilots.has(e.filtre.ilot), `${lieu} : filtre îlot inconnu`);
      if (e.filtre && e.filtre.trait) assert.ok(D.traits[e.filtre.trait], `${lieu} : trait inconnu`);
    } else if (e.carte) {
      assert.ok(cartes.has(e.carte), `${lieu} : carte inconnue`);
      assert.ok(["decouvrir", "retirer", "equiper"].includes(e.op), `${lieu} : op de carte inconnue`);
    } else if (e.levier) {
      const l = D.leviersParId[e.levier];
      assert.ok(l, `${lieu} : levier inconnu`);
      assert.ok(l.valeurs.some((x) => x.v === e.valeur), `${lieu} : valeur de levier invalide`);
    } else if (e.evenement) {
      assert.ok(evenements.has(e.evenement), `${lieu} : événement de suite inconnu`);
    } else if (e.incident) {
      assert.equal(typeof e.incident.gravite, "number", `${lieu} : gravité manquante`);
    } else {
      assert.ok(e.drapeau || e.journal || e.confiance != null || e.budget != null, `${lieu} : effet non reconnu`);
    }
    if (e.ilot && e.ilot !== "ilot") assert.ok(ilots.has(e.ilot), `${lieu} : îlot inconnu`);
  }
}

test("îlots et développeurs : trois développeurs par îlot, traits connus", () => {
  for (const i of ilots) assert.equal(D.devs.filter((d) => d.ilot === i).length, 3, i);
  for (const d of D.devs) {
    assert.ok(ilots.has(d.ilot), d.id);
    assert.ok(["stagiaire", "junior", "inter", "senior"].includes(d.niveau), d.id);
    for (const t of d.traits) assert.ok(D.traits[t], `${d.id} : trait ${t}`);
    for (const r of d.reactions || []) {
      const l = D.leviersParId[r.levier];
      assert.ok(l, `${d.id} : réaction à un levier inconnu ${r.levier}`);
      for (const v of r.valeurs) assert.ok(l.valeurs.some((x) => x.v === v), `${d.id} : valeur ${v} de ${r.levier}`);
    }
  }
});

test("leviers : valeur par défaut valide, panneau connu", () => {
  const panneaux = new Set(D.panneaux.map((p) => p.id));
  for (const l of D.leviers) {
    assert.ok(panneaux.has(l.panneau), l.id);
    assert.ok(l.valeurs.some((x) => x.v === l.defaut), `${l.id} : défaut`);
    assert.ok(l.libelle && l.aide, `${l.id} : libellé et aide`);
  }
});

test("capacités : forme valide", () => {
  const cles = new Set(Object.keys(LTP.etat.modsIlotNeutres()));
  for (const c of D.capacites || []) {
    assert.ok(["skill", "mcp"].includes(c.type), c.id);
    assert.ok(["maison", "fournisseur", "communaute", "personnelle"].includes(c.provenance), c.id);
    assert.ok(["depart", "evenement", "developpement"].includes(c.disponible), c.id);
    if (c.auteur) assert.ok(devs.has(c.auteur), `${c.id} : auteur ${c.auteur}`);
    for (const k of Object.keys(c.gains || {})) assert.ok(cles.has(k), `${c.id} : gain ${k}`);
    for (const [i, g] of Object.entries(c.ilotsGains || {})) {
      assert.ok(ilots.has(i), `${c.id} : îlot ${i}`);
      for (const k of Object.keys(g)) assert.ok(cles.has(k), `${c.id} : gain ${k}`);
    }
    for (const b of c.branches || []) assert.ok(["prive", "nonFiable", "sortie"].includes(b), `${c.id} : branche ${b}`);
    for (const i of c.ilotsSuggeres || []) assert.ok(ilots.has(i), `${c.id} : îlot suggéré ${i}`);
    if (c.prerequis) assert.ok(D.leviersParId[c.prerequis.levier], `${c.id} : prérequis`);
  }
});

test("événements : effets valides, choix identifiés, personnages connus", () => {
  const ids = new Set();
  for (const e of D.evenements || []) {
    assert.ok(!ids.has(e.id), `événement en double : ${e.id}`);
    ids.add(e.id);
    assert.ok(e.titre && e.texte, `${e.id} : titre et texte`);
    assert.ok(personnages.has(e.personnage), `${e.id} : personnage ${e.personnage}`);
    assert.ok(e.choix && e.choix.length >= 1, `${e.id} : au moins un choix`);
    const choix = new Set();
    for (const c of e.choix) {
      assert.ok(c.id && !choix.has(c.id), `${e.id} : choix sans id ou en double`);
      choix.add(c.id);
      assert.ok(c.libelle, `${e.id}.${c.id} : libellé`);
      verifierEffets(c.effets || [], `${e.id}.${c.id}`);
    }
  }
});

test("comités : réponses évaluables", () => {
  for (const c of D.comites || []) {
    assert.ok(personnages.has(c.personnage), `${c.id} : personnage`);
    assert.ok(c.question && c.reponses && c.reponses.length >= 2, c.id);
    for (const r of c.reponses) {
      assert.ok(r.id && r.texte, `${c.id} : réponse`);
      assert.equal(typeof r.evaluer, "function", `${c.id}.${r.id} : evaluer`);
      const s = r.evaluer(etat, LTP.q);
      assert.ok(typeof s === "number" && s >= -1 && s <= 1, `${c.id}.${r.id} : score ${s}`);
    }
  }
});

test("fins : les profils attendus existent", () => {
  const fins = new Set(D.fins.map((f) => f.id));
  for (const id of ["reussie", "usine", "musee", "farwest", "facture", "chateau", "congedie"]) assert.ok(fins.has(id), id);
});

test("métriques : groupes connus", () => {
  const groupes = new Set(D.groupesMetriques.map((g) => g.id));
  for (const m of D.metriques) assert.ok(groupes.has(m.groupe), m.id);
});

test("textes enregistrés : aucune valeur vide", () => {
  for (const [k, v] of Object.entries(LTP.textes.fr)) assert.ok(v !== "" && v != null, k);
});
