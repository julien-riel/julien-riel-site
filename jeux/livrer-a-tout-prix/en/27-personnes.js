/* en/27-personnes.js */
/* Édition anglaise : textes du journal des personnes (départs, congés, réactions). */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }

  LTP.enregistrerTextes({
    'personnes.conge': function (v) {
      var n = Number(v.semaines);
      return val(v, 'dev') + ' can\'t take any more: forced leave for ' + val(v, 'semaines') + (n === 1 ? ' week.' : ' weeks.');
    },
    'personnes.demission': '{dev} resigns. What this person knew about the code walks out the door too ({capture}% captured in skills).',
    'personnes.finContrat': '{dev}\'s contract is over. What they knew leaves with them ({capture}% captured in skills).',
    'personnes.formationDebut': 'Training begins: the team spends part of every week in a classroom.',
    'personnes.reaction': '{dev}: “{bulle}”',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
