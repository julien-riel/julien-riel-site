/* en/28-demandes.js */
/* Édition anglaise : textes du journal des demandes (portée, applications hors TI). */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }

  LTP.enregistrerTextes({
    'demandes.portee': function (v) {
      var n = Number(v.points);
      return 'Scope change on the ' + val(v, 'ilot') + ' pod: “actually, we\'d like it done differently.” '
        + (isNaN(n) ? val(v, 'points') : LTP.format.nombre(n, 0)) + (n === 1 ? ' point' : ' points') + ' to redo.';
    },
    'demandes.kevinAppli': 'Kevin, from Marketing, has put his own app online. It works. Nobody knows where the data lives.',
    'demandes.kevinAppliEncadree': 'Kevin, from Marketing, built his own app, from the approved registry and with a hand from the team.',
    'demandes.appliFantome': 'A new app built outside IT has turned up over by the {ilot} pod.',
    'demandes.appliEncadree': 'A business line built a supported app with the {ilot} pod.',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
