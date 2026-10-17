/* 49-ui-sauvegarde.js */
/* Sauvegarde automatique dans localStorage, protégée (navigation privée, stockage bloqué). Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var CLE = 'livrer-a-tout-prix:v1';
  LTP.sauvegarde = {
    enregistrer: function (etat) {
      try { if (etat && etat.phase !== 'accueil') localStorage.setItem(CLE, JSON.stringify(etat)); } catch (e) { /* ignoré */ }
    },
    charger: function () {
      try {
        var s = localStorage.getItem(CLE);
        if (!s) return null;
        var etat = JSON.parse(s);
        return etat && etat.version === 1 ? etat : null;
      } catch (e) { return null; }
    },
    effacer: function () { try { localStorage.removeItem(CLE); } catch (e) { /* ignoré */ } },
  };
})(globalThis.LTP = globalThis.LTP || {});
