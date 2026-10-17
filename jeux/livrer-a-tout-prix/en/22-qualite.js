/* en/22-qualite.js */
/* Édition anglaise : textes du journal de la qualité (revue de code). */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }
  function compte(v, k, un, plusieurs) {
    var n = Number(v[k]);
    return (isNaN(n) ? val(v, k) : LTP.format.nombre(n, 0)) + ' ' + (n === 1 ? un : plusieurs);
  }

  LTP.enregistrerTextes({
    'qualite.revueSaturee': function (v) {
      return val(v, 'ilot') + ': ' + compte(v, 'pr', 'PR is', 'PRs are') + ' waiting for review. The reviewers are skimming to keep up.';
    },
    'qualite.lgtm': function (v) {
      return val(v, 'ilot') + ': ' + compte(v, 'lgtm', 'PR', 'PRs') + ' approved with a “LGTM” this week, without a real read.';
    },
    'qualite.triche': '{ilot}: an agent commit switches off a check rule. CI is green again.',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
