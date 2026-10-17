/* en/24-couts.js */
/* Édition anglaise : textes du journal des coûts (jetons, plafond, budget). */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }

  LTP.enregistrerTextes({
    'couts.boucle': '{dev}: an agent got stuck in a loop and burned through {jetons} tokens ({cout}).',
    'couts.plafondSouple': '{ilot}: token cap alert, {jetons} tokens this week.',
    'couts.plafondCoupe': function (v) {
      var n = Number(v.points);
      return val(v, 'ilot') + ': the hard cap cut the agents off mid-task. About '
        + (isNaN(n) ? val(v, 'points') : LTP.format.nombre(n, 0)) + (n === 1 ? ' point' : ' points') + ' of work lost.';
    },
    'couts.budget': '{pourcent}% of the annual AI and tooling budget is spent.',
    'couts.budgetDepasse': 'The annual budget is blown. Régis has booked a meeting room to talk about it.',
    'couts.hausseFournisseur': 'The model vendor is raising its prices by {pourcent}%. Every token now costs more.',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
