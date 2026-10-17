/* 32-fin.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent C. Comités trimestriels, score et profil de fin
   (CONCEPTION §2.3, §7 ; contenu dans 15-donnees-comites.js et 17-donnees-fins.js). */
(function (LTP) {
  'use strict';
  LTP.K.fin = {};
  LTP.systemes.fin = {
    seuilConfiance: 15,
    initialiser: function (etat) {},
    // Remplit etat.comite = { trimestre, questions: [{ id, ... }], reponses: { [questionId]: reponseId } }.
    preparerComite: function (etat, ctx) {
      etat.comite = { trimestre: etat.trimestre, questions: [], reponses: {} };
    },
    repondreComite: function (etat, questionId, reponseId, ctx) {
      etat.comite.reponses[questionId] = reponseId;
      return null;
    },
    comiteTermine: function (etat) {
      return !etat.comite || etat.comite.questions.every(function (x) { return etat.comite.reponses[x.id] != null; });
    },
    debutTrimestre: function (etat, ctx) {},
    // Renvoie { raison, profil, etoiles (0–5), score (0–100), criteres: { [nom]: 0–1 } }.
    calculerResultat: function (etat, ctx, raison) {
      return { raison: raison, profil: 'reussie', etoiles: 3, score: 50, criteres: {} };
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
