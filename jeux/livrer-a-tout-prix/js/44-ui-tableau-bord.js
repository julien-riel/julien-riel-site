/* 44-ui-tableau-bord.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent F. Tableau de bord des métriques. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h;
  LTP.ui.regions.tableau = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat) {
      LTP.ui.remplacer(this._racine, h('dl', { class: 'tableau-provisoire' }, LTP.donnees.metriques.map(function (m) {
        var vis = LTP.q.metriqueVisible(etat, m.id), v = etat.metriques[m.id];
        return [h('dt', null, m.libelle), h('dd', { 'data-metrique': m.id }, vis ? LTP.format.nombre(v) : 'non mesuré')];
      })));
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
