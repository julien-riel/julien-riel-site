/* 43-ui-leviers.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent F. Panneau des leviers (et onglet Registre via 45). */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h;
  LTP.ui.regions.panneau = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat, ui, precedent, app) {
      var leviers = LTP.donnees.leviers.filter(function (l) { return LTP.q.levierDebloque(etat, l.id); });
      LTP.ui.remplacer(this._racine, h('div', { class: 'leviers-provisoire' }, leviers.map(function (l) {
        var v = LTP.q.levier(etat, l.id);
        return h('label', { class: 'levier', 'data-levier': l.id }, l.libelle, ' ',
          h('select', { onchange: function (ev) {
              var val = l.valeurs[ev.target.selectedIndex].v;
              app.dispatch({ type: 'levier', id: l.id, valeur: val });
            } },
            l.valeurs.map(function (x) { return h('option', { selected: x.v === v }, x.libelle); })));
      })));
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
