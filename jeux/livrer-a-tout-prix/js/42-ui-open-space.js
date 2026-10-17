/* 42-ui-open-space.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent E. Vue de haut de l'open space, signaux d'état. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h;
  LTP.ui.regions.openspace = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat, ui, precedent, app) {
      LTP.ui.remplacer(this._racine,
        h('div', { class: 'os-provisoire' }, LTP.q.ilots(etat).map(function (il) {
          return h('div', { class: 'os-ilot', 'data-ilot': il.id },
            h('strong', null, LTP.ui.nomIlot(il.id)),
            LTP.q.devsDe(etat, il.id).map(function (d) {
              return h('button', { type: 'button', class: 'os-dev', 'data-dev': d.id,
                onclick: function () { app.setUI({ selection: { type: 'dev', id: d.id }, modale: { type: 'fiche', id: d.id } }); } },
                LTP.ui.avatar(d.id, { taille: 24 }), ' ', LTP.ui.nomDev(d.id), ' · ', d.sessions, ' écran(s)');
            }));
        })));
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
