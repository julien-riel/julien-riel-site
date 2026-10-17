// Données et sécurité (25-donnees-secu.js) : évolution des données, menace, exposition Loi 25,
// surface d'attaque et risques hebdomadaires. Tests directionnels, en moyenne sur plusieurs graines.
// On appelle le système seul : la triade est posée à la main et reste fixe.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";

const GRAINES = 20;
const COMPLETE = { prive: true, nonFiable: true, sortie: true };
const CASSEE = { prive: true, nonFiable: true, sortie: false };

function carte(id, autres) {
  return {
    id, type: "mcp", nom: id, provenance: "communaute", trimestre: 1, disponible: "depart",
    auteur: null, coutDev: 0, ilotsSuggeres: [], description: "Carte de test.",
    gains: {}, poidsContexte: 0.05, branches: [], risque: { type: "aucun", proba: 0 },
    declencheur: null, prerequis: null, ...autres,
  };
}
const CARTES = [
  carte("t-exfil", { risque: { type: "exfiltration", proba: 0.05 } }),
  carte("t-fournisseur", { provenance: "fournisseur", risque: { type: "changementSilencieux", proba: 0.05 } }),
];

// Joue n semaines du seul système de sécurité ; reglages(etat) est rappelé chaque semaine.
function jouer(graine, leviers, reglages, n = 52) {
  const LTP = charger();
  for (const c of CARTES) LTP.donnees.capacites.push(c);
  const etat = LTP.moteur.creer(graine, { toutDebloque: true });
  etat.phase = "execution";
  Object.assign(etat.leviers.global, leviers);
  const journal = [];
  for (let i = 0; i < n; i++) {
    etat.semaine += 1;
    for (const il of Object.values(etat.ilots)) il.derniere = LTP.etat.derniereIlotVide();
    if (reglages) reglages(etat, LTP);
    const ctx = LTP.moteur.creerContexte(etat);
    LTP.systemes.donneesSecu.semaine(etat, ctx);
    journal.push(...ctx.journal);
  }
  return { LTP, etat, journal };
}

function compter(etat, cause) {
  return etat.incidents.filter((i) => !cause || i.cause === cause).length;
}
function total(leviers, reglages, cause) {
  let n = 0;
  for (let g = 1; g <= GRAINES; g++) n += compter(jouer(g, leviers, reglages).etat, cause);
  return n;
}
function triade(t) {
  return (etat) => { for (const il of Object.values(etat.ilots)) il.triade = { ...t }; };
}

test("triade complète → beaucoup plus d'injections de prompt que la triade cassée", () => {
  const leviers = { autonomie: 3 };
  const complete = total(leviers, triade(COMPLETE), "injection");
  const cassee = total(leviers, triade(CASSEE), "injection");
  assert.ok(complete > 3 * cassee + 5, `complète ${complete}, cassée ${cassee}`);
});

test("bac à sable filtré → moins d'incidents de sécurité", () => {
  const base = { autonomie: 4, modele: "economique" };
  const sans = total({ ...base, bacASable: "aucun" }, triade(COMPLETE));
  const filtre = total({ ...base, bacASable: "filtre" }, triade(COMPLETE));
  assert.ok(filtre < sans * 0.6, `filtré ${filtre}, aucun ${sans}`);
  assert.ok(total({ ...base, bacASable: "filtre" }, triade(COMPLETE), "paquetHallucine") <
    total({ ...base, bacASable: "aucun" }, triade(COMPLETE), "paquetHallucine"));
});

test("approbation humaine des sorties et formation → moins d'injections", () => {
  const base = { autonomie: 3 };
  const sans = total(base, triade(COMPLETE), "injection");
  assert.ok(total({ ...base, approbationSorties: true }, triade(COMPLETE), "injection") < sans * 0.4);
  assert.ok(total({ ...base, formation: "mentorat" }, triade(COMPLETE), "injection") < sans);
});

test("destruction en production : seulement avec accès en écriture et autonomie ≥ 3", () => {
  assert.equal(total({ autonomie: 2, accesProd: "oui" }, null, "destructionProd"), 0);
  assert.equal(total({ autonomie: 4, accesProd: "lecture" }, null, "destructionProd"), 0);
  assert.ok(total({ autonomie: 4, accesProd: "oui" }, null, "destructionProd") > 0);
});

test("paquet halluciné : plus fréquent avec le modèle économique qu'avec le haut de gamme", () => {
  const eco = total({ autonomie: 4, modele: "economique" }, null, "paquetHallucine");
  const haut = total({ autonomie: 4, modele: "haut" }, null, "paquetHallucine");
  assert.ok(eco > haut, `économique ${eco}, haut ${haut}`);
});

test("MCP communautaire empoisonné : la passerelle réduit le risque", () => {
  const equiper = (etat) => {
    if (!etat.ilots.portail.capacites.includes("t-exfil")) {
      etat.ilots.portail.capacites.push("t-exfil");
      etat.capacites["t-exfil"].equipeeSur.push("portail");
    }
  };
  const sans = total({ autonomie: 3 }, equiper, "mcpEmpoisonne");
  const avec = total({ autonomie: 3, passerelleMCP: true }, equiper, "mcpEmpoisonne");
  assert.ok(sans > 0 && avec < sans * 0.6, `sans ${sans}, avec ${avec}`);
});

test("changement silencieux d'un MCP de fournisseur : la passerelle le détecte au lieu d'un incident", () => {
  const equiper = (etat) => {
    if (!etat.ilots.operations.capacites.includes("t-fournisseur")) {
      etat.ilots.operations.capacites.push("t-fournisseur");
      etat.capacites["t-fournisseur"].equipeeSur.push("operations");
    }
  };
  const sans = total({}, equiper, "changementSilencieux");
  let avec = 0, detections = 0;
  for (let g = 1; g <= GRAINES; g++) {
    const { etat, journal } = jouer(g, { passerelleMCP: true }, equiper);
    avec += compter(etat, "changementSilencieux");
    detections += etat.securite.detections;
    if (etat.securite.detections) {
      assert.ok(etat.org.drapeaux.changementDetecte);
      assert.ok(journal.some((e) => e.cle === "donneesSecu.changementDetecte"));
    }
  }
  assert.ok(sans > 0);
  assert.equal(avec, 0);
  assert.ok(detections > 0);
});

test("fuite de données : le modèle local et la classification réduisent l'exposition Loi 25 et les fuites", () => {
  const ext = jouer(1, { autonomie: 3 }).etat.securite.expositionLoi25;
  const loc = jouer(1, { autonomie: 3, modele: "local" }).etat.securite.expositionLoi25;
  const cla = jouer(1, { autonomie: 3, classification: "auto" }).etat.securite.expositionLoi25;
  assert.ok(loc < ext && cla < ext, `externe ${ext}, local ${loc}, classifié ${cla}`);
  const fExt = total({ autonomie: 4 }, null, "fuiteDonnees");
  const fLoc = total({ autonomie: 4, modele: "local" }, null, "fuiteDonnees");
  assert.ok(fExt > fLoc, `externe ${fExt}, local ${fLoc}`);
});

test("données : la duplication monte avec le code et les applications fantômes, le catalogue la contient", () => {
  const code = (etat) => {
    for (const il of Object.values(etat.ilots)) { il.derniere.code = 1; il.applisFantomes = 1; }
  };
  const sans = jouer(1, {}, code, 20).etat.org.donnees;
  const avec = jouer(1, { catalogue: "sources" }, code, 20).etat.org.donnees;
  const calme = jouer(1, {}, null, 20).etat.org.donnees;
  const depart = charger().K.etat.org.donnees;
  assert.ok(sans.duplication > depart.duplication);
  assert.ok(sans.duplication > calme.duplication);
  assert.ok(avec.duplication < sans.duplication);
  assert.ok(sans.fraicheur < depart.fraicheur);
  assert.ok(avec.fraicheur > sans.fraicheur);
  assert.ok(avec.qualite > sans.qualite);
});

test("menace en hausse, champs exposés bornés, drapeaux et confiance après les incidents", () => {
  const { LTP, etat } = jouer(2, { autonomie: 4, accesProd: "oui", modele: "economique" }, triade(COMPLETE));
  assert.ok(etat.org.menace > LTP.K.etat.org.menace);
  const s = etat.securite;
  assert.ok(s.surfaceAttaque >= 0 && s.surfaceAttaque <= 100);
  assert.ok(s.expositionLoi25 >= 0 && s.expositionLoi25 <= 100);
  for (const il of Object.values(s.parIlot)) {
    assert.ok(il.surface >= 0 && il.surface <= 100 && il.loi25 >= 0 && il.loi25 <= 1);
    for (const p of Object.values(il.risques)) assert.ok(p >= 0 && p <= 1);
  }
  assert.ok(compter(etat) > 0);
  assert.ok(etat.org.confiance < LTP.K.etat.org.confiance);
  assert.ok(etat.org.drapeaux.injectionPrompt);
  const parCause = Object.values(s.incidents).reduce((a, b) => a + b, 0);
  assert.equal(parCause, compter(etat));
});

test("surface d'attaque : plus grande en mode « tout-IA » qu'avec les garde-fous", () => {
  const farWest = jouer(1, { autonomie: 4, accesProd: "oui" }, null, 1).etat.securite.surfaceAttaque;
  const sage = jouer(1, { autonomie: 2, accesProd: "lecture", bacASable: "filtre", passerelleMCP: true, approbationSorties: true }, null, 1)
    .etat.securite.surfaceAttaque;
  assert.ok(farWest > sage * 1.5, `far west ${farWest}, sage ${sage}`);
});
