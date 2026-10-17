/* 47-ui-comite.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent F. Comité trimestriel (région) et bilan annuel (écran « fin »). */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h;
  LTP.ui.regions.comite = {
    monter: function (racine) {},
    rendre: function () {},
  };
  LTP.ui.ecrans.fin = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat, ui, precedent, app) {
      var res = etat.resultat || {};
      var fin = (LTP.donnees.fins || []).filter(function (f) { return f.id === res.profil; })[0] || {};
      LTP.ui.remplacer(this._racine, h('section', { class: 'bilan' },
        h('h1', { class: 'profil-fin' }, fin.titre || 'Fin'),
        h('p', null, (res.etoiles || 0) + ' / 5'),
        h('button', { type: 'button', onclick: function () { app.nouvellePartie(); } }, LTP.t('action.recommencer'))));
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
