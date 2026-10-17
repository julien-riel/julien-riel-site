/* 23-dette.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent A. Compréhension, MTTR et cycle de vie des incidents (CONCEPTION §4.5). */
(function (LTP) {
  'use strict';
  LTP.K.dette = {};
  LTP.systemes.dette = {
    initialiser: function (etat) {},
    semaine: function (etat, ctx) {},
    // Étape 9 : transforme des défauts latents en bogues et incidents, résout les incidents échus.
    incidents: function (etat, ctx) {
      etat.incidents.forEach(function (i) { if (!i.resolu && i.finSemaine <= etat.semaine) i.resolu = true; });
    },
    // Durée de rétablissement prévue, en heures.
    mttr: function (etat, ilotId, gravite) { return 8; },
  };
})(globalThis.LTP = globalThis.LTP || {});
