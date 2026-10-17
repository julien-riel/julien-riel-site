/* 29-evenements.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent C. Tirage des cartes d'événement et résolution des choix
   (CONCEPTION §6). Les effets sont appliqués par LTP.effets (19-effets.js). */
(function (LTP) {
  'use strict';
  LTP.K.evenements = {};
  LTP.systemes.evenements = {
    initialiser: function (etat) {},
    // Étape 12 : au plus un événement par semaine dans etat.evenementEnAttente = { id, ref: { ilot, dev }, vars }.
    tirage: function (etat, ctx) {},
    // Résout le choix du joueur. Renvoie un code d'erreur ou null.
    resoudre: function (etat, choixId, ctx) {
      var att = etat.evenementEnAttente;
      var def = LTP.q.defEvenement(att.id);
      var choix = def && (def.choix || []).filter(function (c) { return c.id === choixId; })[0];
      if (!choix) return 'choix-inconnu';
      LTP.effets.appliquer(etat, choix.effets, att.ref || {}, ctx);
      etat.evenementsVus[att.id] = etat.semaine;
      etat.evenementEnAttente = null;
      return null;
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
