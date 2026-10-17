/* 26-registre.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent B. Capacités (skills et MCP) : modificateurs d'îlot,
   triade, conflits, saturation, péremption, développement maison (CONCEPTION §4.9). */
(function (LTP) {
  'use strict';
  LTP.K.registre = {};
  LTP.systemes.registre = {
    initialiser: function (etat) {},
    // Étape 3 : recalcule ilot.mods et ilot.triade.
    semaine: function (etat, ctx) {
      LTP.q.ilots(etat).forEach(function (il) { il.mods = LTP.etat.modsIlotNeutres(); });
    },
    // Actions du joueur : equiper, desequiper, developper, proprietaire. Renvoie un code d'erreur ou null.
    action: function (etat, a, ctx) {
      var c = etat.capacites[a.carte];
      if (!c) return 'carte-inconnue';
      if (a.type === 'equiper') {
        if (!etat.ilots[a.ilot]) return 'ilot-inconnu';
        if (c.equipeeSur.indexOf(a.ilot) < 0) c.equipeeSur.push(a.ilot);
        if (etat.ilots[a.ilot].capacites.indexOf(a.carte) < 0) etat.ilots[a.ilot].capacites.push(a.carte);
        return null;
      }
      if (a.type === 'desequiper') { LTP.effets.retirerCarte(etat, a.carte, a.ilot); return null; }
      if (a.type === 'proprietaire') { c.proprietaire = a.dev || null; return null; }
      return 'non-implemente';
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
