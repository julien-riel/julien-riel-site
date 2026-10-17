// Demandes (28-demandes.js) : backlog, appétit, discipline du « non », portée, gonflement,
// satisfaction, retards, applications hors TI et Goodhart. Tests directionnels sur plusieurs graines.
// On appelle le système seul ; la production de la semaine est posée à la main.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";

const GRAINES = 15;

// Joue n semaines du seul système de demandes. production(il) → { pointsLivres, code, bogues, incidents }.
function jouer(graine, leviers, { n = 30, production, reglages, options = { toutDebloque: true } } = {}) {
  const LTP = charger();
  const etat = LTP.moteur.creer(graine, options);
  etat.phase = "execution";
  Object.assign(etat.leviers.global, leviers);
  const cumul = { demandes: 0, refusees: 0 };
  const journal = [];
  for (let i = 0; i < n; i++) {
    etat.semaine += 1;
    for (const il of Object.values(etat.ilots)) {
      il.derniere = LTP.etat.derniereIlotVide();
      Object.assign(il.derniere, production ? production(il) : {});
    }
    if (reglages) reglages(etat, LTP);
    const ctx = LTP.moteur.creerContexte(etat);
    LTP.systemes.demandes.semaine(etat, ctx);
    journal.push(...ctx.journal);
    for (const il of Object.values(etat.ilots)) {
      cumul.demandes += il.derniere.demandes;
      cumul.refusees += il.derniere.demandesRefusees;
    }
  }
  return { LTP, etat, cumul, journal };
}

function somme(f) {
  let s = 0;
  for (let g = 1; g <= GRAINES; g++) s += f(g);
  return s;
}
const acceptees = (r) => r.cumul.demandes - r.cumul.refusees;
const ilots = (etat) => Object.values(etat.ilots);
const fantomes = (etat) => ilots(etat).reduce((a, il) => a + il.applisFantomes, 0);

test("discipline ferme → moins de demandes acceptées et plus de refus", () => {
  const ferme = somme((g) => acceptees(jouer(g, { disciplineNon: "ferme" })));
  const laxiste = somme((g) => acceptees(jouer(g, { disciplineNon: "laxiste" })));
  assert.ok(ferme < laxiste * 0.8, `ferme ${ferme}, laxiste ${laxiste}`);
  const r = jouer(1, { disciplineNon: "laxiste" });
  assert.equal(r.cumul.refusees, 0);
});

test("appétit : avec un agent autonome, les directions demandent davantage", () => {
  const auto = somme((g) => jouer(g, { autonomie: 4 }).cumul.demandes);
  const manuel = somme((g) => jouer(g, { autonomie: 0 }).cumul.demandes);
  assert.ok(auto > manuel * 1.2, `autonomie 4 : ${auto}, autonomie 0 : ${manuel}`);
});

test("gonflement fonctionnel : la discipline laxiste ajoute du code sans valeur", () => {
  const prod = () => ({ pointsLivres: 12, code: 2 });
  const lax = jouer(1, { disciplineNon: "laxiste" }, { production: prod }).etat;
  const ferme = jouer(1, { disciplineNon: "ferme" }, { production: prod }).etat;
  for (const il of ilots(lax)) {
    assert.ok(il.codebase.volume > ferme.ilots[il.id].codebase.volume, il.id);
    assert.ok(il.produit.gonflement > 0 && ferme.ilots[il.id].produit.gonflement === 0);
  }
});

test("changements de portée : les specs obligatoires réduisent la reprise", () => {
  const prod = () => ({ pointsLivres: 12 });
  const reprise = (specs) => somme((g) => ilots(jouer(g, { specs }, { production: prod, n: 40 }).etat)
    .reduce((a, il) => a + il.produit.reprise, 0));
  const sans = reprise("non"), avec = reprise("obligatoires");
  assert.ok(sans > 0 && avec < sans * 0.6, `sans specs ${sans}, avec ${avec}`);
});

test("satisfaction : la valeur livrée la fait monter, les bogues et les incidents la font baisser", () => {
  const sat = (prod) => somme((g) => ilots(jouer(g, {}, { production: prod }).etat).reduce((a, il) => a + il.satisfaction, 0));
  const bien = sat(() => ({ pointsLivres: 16 }));
  const peu = sat(() => ({ pointsLivres: 4 }));
  const bogue = sat(() => ({ pointsLivres: 16, bogues: 4, incidents: 1 }));
  assert.ok(bien > peu, `bien ${bien}, peu ${peu}`);
  assert.ok(bogue < bien, `avec bogues ${bogue}, sans ${bien}`);
});

test("retards : un backlog qui s'accumule fait monter ilot.retard", () => {
  const r = jouer(1, {}, { production: () => ({ pointsLivres: 2 }), n: 30 }).etat;
  const ok = jouer(1, {}, { production: () => ({ pointsLivres: 30 }), n: 30 }).etat;
  for (const il of ilots(r)) {
    assert.ok(il.retard >= 0 && il.retard <= 1);
    assert.ok(il.retard >= ok.ilots[il.id].retard);
  }
  assert.ok(ilots(r).some((il) => il.retard > 0.5));
});

test("non-TI : la demande insatisfaite crée des applications ; encadrer → moins d'applications risquées", () => {
  const prod = () => ({ pointsLivres: 2 });
  const opts = { production: prod, n: 40 };
  const ignorer = somme((g) => fantomes(jouer(g, { politiqueNonTI: "ignorer", autonomie: 3 }, opts).etat));
  const encadrer = somme((g) => fantomes(jouer(g, { politiqueNonTI: "encadrer", autonomie: 3 }, opts).etat));
  const interdire = jouer(1, { politiqueNonTI: "interdire", autonomie: 3 }, opts).etat;
  assert.ok(ignorer > 0, "des applications apparaissent");
  assert.ok(encadrer < ignorer * 0.5, `encadrer ${encadrer}, ignorer ${ignorer}`);
  const encadrees = somme((g) => ilots(jouer(g, { politiqueNonTI: "encadrer", autonomie: 3 }, opts).etat)
    .reduce((a, il) => a + il.applisEncadrees, 0));
  assert.ok(encadrees > 0);
  // Interdire fâche : satisfaction plus basse qu'en encadrant.
  const enc1 = jouer(1, { politiqueNonTI: "encadrer", autonomie: 3 }, opts).etat;
  const satI = ilots(interdire).reduce((a, il) => a + il.satisfaction, 0);
  const satE = ilots(enc1).reduce((a, il) => a + il.satisfaction, 0);
  assert.ok(satI < satE);
});

test("Kevin : la première application hors TI côté Bureautique ou Web pose le drapeau kevinAppli", () => {
  let vu = 0;
  for (let g = 1; g <= GRAINES; g++) {
    const { etat, journal } = jouer(g, { autonomie: 3 }, { production: () => ({ pointsLivres: 1 }), n: 40 });
    const kevin = etat.ilots.bureautique.applisFantomes + etat.ilots.web.applisFantomes > 0;
    if (kevin) {
      vu++;
      assert.ok(etat.org.drapeaux.kevinAppli, `graine ${g}`);
      assert.equal(journal.filter((e) => e.cle === "demandes.kevinAppli").length, 1);
    }
  }
  assert.ok(vu > 0);
});

test("non-TI : rien avant le deuxième trimestre", () => {
  const { etat } = jouer(1, { autonomie: 3 }, { production: () => ({ pointsLivres: 0 }), n: 12, options: {} });
  assert.equal(fantomes(etat), 0);
});

test("Goodhart « satisfaction » : la discipline du « non » est forcée à laxiste", () => {
  const r = jouer(1, { disciplineNon: "ferme", objectif: "satisfaction" });
  assert.equal(r.cumul.refusees, 0);
});

test("Goodhart « incidents » : une partie des incidents est reclassée en demandes", () => {
  let reclasses = 0;
  const creer = (etat, LTP) => {
    LTP.q.creerIncident(etat, { ilot: "portail", gravite: 0.5, cause: "bogue" });
  };
  for (let g = 1; g <= 5; g++) {
    const { etat } = jouer(g, { objectif: "incidents" }, { reglages: creer, n: 10 });
    reclasses += etat.produit.incidentsReclasses;
    assert.equal(etat.incidents.filter((i) => i.reclasse).length, etat.produit.incidentsReclasses);
  }
  assert.ok(reclasses > 0);
  const { etat } = jouer(1, {}, { reglages: creer, n: 10 });
  assert.equal(etat.produit.incidentsReclasses, 0);
});

test("partie complète : champs propres présents, finis et bornés", () => {
  const LTP = charger();
  const fin = LTP.moteur.simuler(LTP.moteur.creer(9), 52).etat;
  for (const il of ilots(fin)) {
    assert.ok(il.retard >= 0 && il.retard <= 1);
    assert.ok(il.applisEncadrees >= 0 && il.applisFantomes >= 0);
    for (const v of Object.values(il.produit)) assert.ok(Number.isFinite(v) && v >= 0);
  }
});
