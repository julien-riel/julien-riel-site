/* 49-ui-sauvegarde.js */
/* Sauvegarde automatique dans localStorage, protégée (navigation privée, stockage bloqué). Propriétaire : agent F.
   API : enregistrer(etat), charger() → etat | null, effacer(), existe() → booléen. */
(function (LTP) {
  'use strict';
  var CLE = 'livrer-a-tout-prix:v2';
  LTP.sauvegarde = {
    enregistrer: function (etat) {
      try { if (etat && etat.phase !== 'accueil') localStorage.setItem(CLE, JSON.stringify(etat)); } catch (e) { /* ignoré */ }
    },
    charger: function () {
      try {
        var s = localStorage.getItem(CLE);
        if (!s) return null;
        var etat = JSON.parse(s);
        return etat && etat.version === 2 && etat.devs && etat.ilots ? etat : null;
      } catch (e) { return null; }
    },
    effacer: function () { try { localStorage.removeItem(CLE); } catch (e) { /* ignoré */ } },
    existe: function () { return !!LTP.sauvegarde.charger(); },
  };
})(globalThis.LTP = globalThis.LTP || {});
