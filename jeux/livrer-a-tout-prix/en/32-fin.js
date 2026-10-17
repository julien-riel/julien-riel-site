/* en/32-fin.js */
/* Édition anglaise : textes du journal du comité de direction et des bilans. */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }
  function compte(v, k, un, plusieurs) {
    var n = Number(v[k]);
    return (isNaN(n) ? val(v, k) : LTP.format.nombre(n, 0)) + ' ' + (n === 1 ? un : plusieurs);
  }

  LTP.enregistrerTextes({
    'fin.reponseBonne': 'The committee nods. Confidence: +{delta}.',
    'fin.reponseMauvaise': 'The committee frowns. Confidence: {delta}.',
    'fin.bilanTrimestre': function (v) {
      return 'Quarter ' + val(v, 'trimestre') + ' review: ' + compte(v, 'pointsLivres', 'point', 'points') + ' delivered and '
        + compte(v, 'prFusionnees', 'PR', 'PRs') + ' merged. Incidents: ' + val(v, 'incidents') + '.';
    },
    'fin.resultat': 'Year-end review: {etoiles} out of 5 stars.',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
