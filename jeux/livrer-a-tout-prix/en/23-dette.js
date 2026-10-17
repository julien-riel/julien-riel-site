/* en/23-dette.js */
/* Édition anglaise : textes du journal de la dette cognitive et technique. */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }
  function compte(v, k, un, plusieurs) {
    var n = Number(v[k]);
    return (isNaN(n) ? val(v, k) : LTP.format.nombre(n, 0)) + ' ' + (n === 1 ? un : plusieurs);
  }

  LTP.enregistrerTextes({
    'dette.personneNeComprend': '{ilot}: nobody really knows anymore how a good chunk of the code works.',
    'dette.resolutionLongue': function (v) {
      return val(v, 'ilot') + ': ' + compte(v, 'heures', 'hour', 'hours') + ' to restore service. First, someone had to figure out what the code was doing.';
    },
    'dette.bogues': function (v) {
      return val(v, 'ilot') + ': ' + compte(v, 'bogues', 'bug', 'bugs') + ' reported in production this week.';
    },
    // Incident grave : remplace la ligne « incident.bogue » et dit pourquoi la confiance baisse.
    'dette.incidentGrave': function (v) {
      return val(v, 'ilot') + ': major production outage (about ' + compte(v, 'heures', 'hour', 'hours') + ' to fix'
        + '). Management saw it happen: executive confidence −' + val(v, 'delta') + '.';
    },
    'dette.incidentGraveDette': function (v) {
      return val(v, 'ilot') + ': major outage (about ' + compte(v, 'heures', 'hour', 'hours') + ' to fix'
        + '), and nobody understands this code well enough to fix it fast. Executive confidence −' + val(v, 'delta') + '.';
    },
    // Clé produite par LTP.q.creerIncident pour la cause « bogue » (20-etat.js).
    'incident.bogue': '{ilot}: a production bug turns into an incident.',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
