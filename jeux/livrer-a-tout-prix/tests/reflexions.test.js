// Système des réflexions (33-reflexions.js) : priorités, déterminisme, aucun effet sur la simulation.
// Le contenu est remplacé par un contenu d'essai : ces tests ne dépendent pas des textes du jeu.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";

function avecContenu() {
  const LTP = charger();
  LTP.donnees.reflexions = {
    leviers: {
      reviseurs: {
        2: [
          { texte: "Deux réviseurs ? Ma PR va attendre.", effet: false, ton: "moins" },
          { texte: "Deux réviseurs : la file s'allonge pour vrai.", effet: true, ton: "moins" },
        ],
      },
    },
    nouveautes: { licences: [{ texte: "Une note de service sur Copilot.", pilote: true }] },
    situations: [
      { id: "sansLicence", libelle: "sans licence", quand: (etat, q, dev) => !q.aLicence(etat, dev), lignes: ["Pas de licence pour moi."] },
      { id: "boum", libelle: "contenu fautif", quand: () => { throw new Error("x"); }, lignes: ["Jamais."] },
    ],
  };
  return LTP;
}

function enExecution(LTP, graine, options = {}) {
  let e = LTP.moteur.creer(graine, options);
  e = LTP.moteur.action(e, { type: "phase", phase: "planification" }).etat;
  return LTP.moteur.action(e, { type: "phase", phase: "execution" }).etat;
}

test("chaque jour : cinq réflexions au plus, deux par îlot au plus, des personnes présentes", () => {
  const LTP = avecContenu();
  let e = enExecution(LTP, 11);
  for (let j = 0; j < 15; j++) {
    const l = e.reflexions.liste;
    assert.ok(l.length > 0 && l.length <= LTP.K.reflexions.parJour, `jour ${j} : ${l.length}`);
    const parIlot = {};
    for (const r of l) {
      assert.ok(e.devs[r.dev] && e.devs[r.dev].present, r.dev);
      assert.equal(r.ilot, e.devs[r.dev].ilot);
      assert.ok(typeof r.texte === "string" && r.texte.length > 0);
      parIlot[r.ilot] = (parIlot[r.ilot] || 0) + 1;
    }
    for (const n of Object.values(parIlot)) assert.ok(n <= LTP.K.reflexions.parIlotMax);
    e = LTP.moteur.jour(e).etat;
    if (e.evenementEnAttente) e = { ...e, evenementEnAttente: null };
  }
  assert.ok(e.reflexions.fil.length <= LTP.K.reflexions.filMax);
});

test("une situation qui lève une exception est ignorée ; le groupe pilote fait parler les sans-licence", () => {
  const LTP = avecContenu();
  let e = enExecution(LTP, 12);
  // Le lundi et le mardi, la note de service des licences fait surtout parler le groupe pilote.
  assert.ok(e.reflexions.liste.some((r) => r.motif === "nouveaute" && e.devs[r.dev].pilote));
  const vues = [];
  for (let j = 0; j < 4; j++) {
    vues.push(...e.reflexions.liste);
    e = LTP.moteur.jour(e).etat;
  }
  assert.ok(!vues.some((r) => r.motif === "situation:boum"));
  assert.ok(vues.some((r) => r.motif === "situation:sansLicence" && !e.devs[r.dev].pilote));
});

test("un changement de levier fait réagir tout de suite, puis domine les jours suivants ; l'effet vient après le vendredi", () => {
  const LTP = avecContenu();
  let e = enExecution(LTP, 13, { toutDebloque: true });
  const r = LTP.moteur.action(e, { type: "levier", id: "reviseurs", valeur: 2 });
  assert.equal(r.erreur, undefined);
  e = r.etat;
  const immediates = e.reflexions.liste.filter((x) => x.immediat);
  assert.ok(immediates.length >= 1 && immediates.length <= LTP.K.reflexions.immediatesMax);
  assert.ok(immediates.every((x) => x.levier === "reviseurs" && x.texte.startsWith("Deux réviseurs ?")));
  // Les jours suivants : des réflexions sur le levier, avant puis après le calcul du vendredi.
  const textes = [];
  for (let j = 0; j < 8; j++) {
    e = LTP.moteur.jour(e).etat;
    if (e.evenementEnAttente) e = { ...e, evenementEnAttente: null };
    for (const x of e.reflexions.liste) if (x.levier === "reviseurs") textes.push([e.semaine, x.texte]);
  }
  assert.ok(textes.length >= 4, `réflexions sur le levier : ${textes.length}`);
  assert.ok(textes.some(([s, t]) => s === 0 && t.startsWith("Deux réviseurs ?")), "avant le vendredi");
  assert.ok(textes.some(([s, t]) => s >= 1 && t.endsWith("pour vrai.")), "après le vendredi");
  assert.ok(!textes.some(([s, t]) => s === 0 && t.endsWith("pour vrai.")), "pas d'effet avant le vendredi");
  assert.ok(e.reflexions.liste.filter((x) => x.levier === "reviseurs").length <= LTP.K.reflexions.leviersMaxParJour);
});

test("les réflexions sont déterministes et ne changent rien à la simulation", () => {
  const jouer = (LTP, sansReflexions) => {
    if (sansReflexions) LTP.systemes.reflexions.assurer = () => [];
    let e = enExecution(LTP, 21, { toutDebloque: true });
    e = LTP.moteur.action(e, { type: "levier", id: "licences", valeur: "tous" }).etat;
    for (let j = 0; j < 5 * 14; j++) {
      if (e.phase === "comite") {
        for (const q of e.comite.questions) e = LTP.moteur.action(e, { type: "comite", question: q.id, reponse: q.reponses[0].id }).etat;
        e = LTP.moteur.action(e, { type: "phase", phase: "planification" }).etat;
        e = LTP.moteur.action(e, { type: "phase", phase: "execution" }).etat;
      }
      if (e.evenementEnAttente) {
        const def = LTP.q.defEvenement(e.evenementEnAttente.id);
        e = LTP.moteur.action(e, { type: "choix", choix: def.choix[0].id }).etat;
      }
      e = LTP.moteur.jour(e).etat;
    }
    return e;
  };
  const a = jouer(avecContenu(), false), b = jouer(avecContenu(), false), c = jouer(avecContenu(), true);
  assert.equal(JSON.stringify(a), JSON.stringify(b), "même graine, mêmes réflexions");
  const sans = (e) => JSON.stringify({ ...e, reflexions: null, journal: e.journal.filter((x) => x.cle !== "personnes.reaction") });
  assert.equal(sans(a), sans(c), "les réflexions n'ont pas d'effet sur la simulation");
  assert.ok(a.semaine >= 13);
});
