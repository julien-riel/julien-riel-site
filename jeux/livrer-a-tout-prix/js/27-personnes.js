/* 27-personnes.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent C. Disponibilité, fatigue, moral, compétence,
   départs, relève, réactions aux leviers (CONCEPTION §4.10). */
(function (LTP) {
  'use strict';
  LTP.K.personnes = {};
  LTP.systemes.personnes = {
    initialiser: function (etat) {},
    // Étape 2 : congés, formation, retours.
    disponibilite: function (etat, ctx) {},
    // Étape 11 : fatigue, moral, compétence, départs.
    semaine: function (etat, ctx) {},
    // Appelé par la boucle quand le joueur change un levier.
    reactionLevier: function (etat, action, ctx) {},
  };
})(globalThis.LTP = globalThis.LTP || {});
