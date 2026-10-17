/* 46-ui-modales.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent F. Fiche de développeur et carte d'événement. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h;
  LTP.ui.regions.modale = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat, ui, precedent, app) {
      var r = this._racine;
      var att = etat.evenementEnAttente;
      if (att && etat.phase === 'execution') {
        var def = LTP.q.defEvenement(att.id) || { titre: att.id, texte: '', choix: [] };
        LTP.ui.remplacer(r, h('div', { class: 'modale carte-evenement', role: 'dialog', 'aria-modal': 'true' },
          h('h2', null, LTP.ui.texteContenu(def.titre, etat, att.ref)),
          h('p', null, LTP.ui.texteContenu(def.texte, etat, att.ref)),
          (def.choix || []).map(function (c) {
            return h('button', { type: 'button', 'data-choix': c.id, onclick: function () { app.dispatch({ type: 'choix', choix: c.id }); } }, c.libelle);
          })));
        return;
      }
      if (ui.modale && ui.modale.type === 'fiche') {
        var d = etat.devs[ui.modale.id], def2 = LTP.q.defDev(ui.modale.id);
        LTP.ui.remplacer(r, h('div', { class: 'modale fiche', role: 'dialog', 'aria-modal': 'true' },
          h('h2', null, def2.nom), h('p', null, def2.tagline),
          h('button', { type: 'button', class: 'fermer', onclick: function () { app.setUI({ modale: null }); } }, 'Fermer')));
        return;
      }
      LTP.ui.vider(r);
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
