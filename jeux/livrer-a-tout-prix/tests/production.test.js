// Production (21) : points, code, PR, sessions, état des développeurs (CONCEPTION §4.4).
import { test } from "node:test";
import assert from "node:assert/strict";
import { LTP, partie, avancer, moyenneGraines, GRAINES, somme, ilots, devs, carnetPlein } from "./production-outils.js";

const IA_MAX = [{ id: "autonomie", valeur: 4 }, { id: "sessionsMax", valeur: 6 }];

// Points produits par semaine (moyenne sur n semaines), carnet plein pour mesurer le débit brut.
function debit(leviers, n = 4, options = {}) {
  return moyenneGraines(GRAINES, (g) => {
    let total = 0;
    avancer(partie(g, leviers, options), n, { avant: carnetPlein, apres: (e) => { total += somme(ilots(e), (i) => i.derniere.points); } });
    return total / n;
  });
}

test("au départ, un inter livre environ 5 points et un îlot environ 15", () => {
  const e = avancer(partie(1, []), 1);
  const inters = devs(e).filter((d) => d.niveau === "inter");
  const moyInter = somme(inters, (d) => d.derniere.points) / inters.length;
  assert.ok(moyInter > 3.5 && moyInter < 6.5, `inter : ${moyInter}`);
  const moyIlot = somme(ilots(e), (i) => i.derniere.points) / 7;
  assert.ok(moyIlot > 10 && moyIlot < 20, `îlot : ${moyIlot}`);
  for (const d of devs(e)) {
    assert.equal(d.sessions, 1);
    assert.ok(["code", "revue", "attente"].includes(d.etat), `${d.id} : ${d.etat}`);
  }
  // Totaux d'îlot cohérents avec ceux des développeurs.
  for (const il of ilots(e)) {
    const ds = devs(e).filter((d) => d.ilot === il.id);
    assert.ok(Math.abs(il.derniere.points - somme(ds, (d) => d.derniere.points)) < 0.01, il.id);
    assert.equal(il.derniere.pr, somme(ds, (d) => d.derniere.pr), il.id);
  }
});

test("autonomie 4 et 6 sessions : débit brut multiplié par 2 à 4", () => {
  const base = debit([]);
  const ia = debit(IA_MAX);
  assert.ok(ia / base > 2 && ia / base < 4, `rapport ${ia / base}`);
});

test("plus d'autonomie : plus de points, moins de code écrit à la main", () => {
  const ecrit = (a) => moyenneGraines(GRAINES, (g) => {
    const e = avancer(partie(g, [{ id: "autonomie", valeur: a }]), 2, { avant: carnetPlein });
    return somme(devs(e), (d) => d.derniere.ecritSoiMeme) / 21;
  });
  assert.ok(debit([{ id: "autonomie", valeur: 3 }]) > debit([{ id: "autonomie", valeur: 1 }]));
  assert.ok(ecrit(3) < ecrit(1) && ecrit(1) < ecrit(0));
});

test("les sessions respectent le plafond du levier et l'envie de chacun", () => {
  const e = avancer(partie(2, [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 3 }]), 1);
  for (const d of devs(e)) assert.ok(d.sessions >= 1 && d.sessions <= 3, `${d.id} : ${d.sessions}`);
  const e6 = avancer(partie(2, [{ id: "autonomie", valeur: 3 }, { id: "sessionsMax", valeur: 6 }]), 1);
  // Les adeptes du multi-Clauding en veulent plus que les réfractaires.
  assert.ok(e6.devs.laurence.sessions > e6.devs.diane.sessions);
  assert.ok(e6.devs.mathieu.sessions >= 4);
});

test("les points produits ne dépassent pas le carnet disponible", () => {
  const e = avancer(partie(3, IA_MAX, { preparer: (x) => { for (const il of Object.values(x.ilots)) il.backlog = 4; } }), 1);
  for (const il of ilots(e)) assert.ok(il.derniere.points <= 4 + 1e-6, `${il.id} : ${il.derniere.points}`);
  assert.ok(devs(e).some((d) => d.etat === "attente"));
});

test("un développeur en congé ne produit rien et garde son état", () => {
  // Les congés sont posés par 27 à l'étape 2 : on enveloppe cette étape pour en imposer un.
  const pers = LTP.systemes.personnes, original = pers.disponibilite;
  pers.disponibilite = function (etat, ctx) {
    if (original) original.call(this, etat, ctx);
    etat.devs.lucie.etat = "conge";
    etat.devs.lucie.indisponibleJusqua = etat.semaine + 1;
  };
  let e;
  try {
    e = avancer(partie(4, []), 1);
  } finally {
    pers.disponibilite = original;
  }
  assert.equal(e.devs.lucie.derniere.points, 0);
  assert.equal(e.devs.lucie.sessions, 0);
  assert.equal(e.devs.lucie.derniere.code, 0);
});

test("l'occupation (registre, formation) réduit les points", () => {
  // L'occupation est posée par 26 et 27 avant la production : on enveloppe l'étape 3 pour l'imposer.
  const reg = LTP.systemes.registre, original = reg.semaine;
  const mesure = (occ) => {
    reg.semaine = function (etat, ctx) {
      original.call(this, etat, ctx);
      for (const d of Object.values(etat.devs)) d.occupation.formation = occ;
    };
    try {
      return moyenneGraines(GRAINES, (g) => somme(ilots(avancer(partie(g, []), 1, { avant: carnetPlein })), (i) => i.derniere.points));
    } finally {
      reg.semaine = original;
    }
  };
  assert.ok(mesure(0.5) < mesure(0) * 0.7);
});

test("taille maximale des PR : plus de PR, plus petites, à code comparable", () => {
  const prs = (taille) => moyenneGraines(GRAINES, (g) => {
    const e = avancer(partie(g, [{ id: "autonomie", valeur: 3 }, { id: "taillePR", valeur: taille }]), 3, { avant: carnetPlein });
    return somme(ilots(e), (i) => i.derniere.pr);
  });
  assert.ok(prs(200) > prs("libre"));
});

test("modèle local : le serveur acheté rend l'agent plus efficace", () => {
  const lev = [{ id: "autonomie", valeur: 3 }, { id: "modele", valeur: "local" }];
  assert.ok(debit(lev, 3, { serveurAchete: true }) > debit(lev, 3));
  assert.ok(debit([{ id: "autonomie", valeur: 3 }, { id: "modele", valeur: "haut" }], 3) > debit(lev, 3));
});

test("le pair humain coûte du débit", () => {
  assert.ok(debit([{ id: "pair", valeur: "humain" }]) < debit([]));
});
