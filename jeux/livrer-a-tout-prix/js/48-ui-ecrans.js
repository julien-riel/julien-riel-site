/* 48-ui-ecrans.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent F. Accueil, tutoriel, HUD et journal. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h;
  LTP.ui.ecrans.accueil = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat, ui, precedent, app) {
      LTP.ui.remplacer(this._racine, h('section', { class: 'accueil' },
        h('h1', { class: 'titre-jeu' }, LTP.t('jeu.titre')),
        h('p', null, LTP.t('jeu.sousTitre')),
        h('button', { type: 'button', id: 'btn-commencer', onclick: function () { app.dispatch({ type: 'phase', phase: 'planification' }); } }, LTP.t('action.commencer'))));
    },
  };
  LTP.ui.regions.hud = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat, ui, precedent, app) {
      var enJeu = etat.phase === 'planification' || etat.phase === 'execution';
      LTP.ui.remplacer(this._racine, h('div', { class: 'hud-inner' },
        h('a', { class: 'tool tool--back', href: '/projets/' }, '← julien-riel.com'),
        h('span', { class: 'hud-titre' }, LTP.t('jeu.titre')),
        h('span', { class: 'hud-semaine' }, LTP.t('hud.trimestre', etat), ' · ', LTP.t('hud.semaine', etat)),
        enJeu ? h('button', { type: 'button', id: 'btn-lecture', onclick: function () {
          if (etat.phase === 'planification') app.dispatch({ type: 'phase', phase: 'execution' });
          app.setUI({ vitesse: ui.vitesse ? 0 : 1 });
        } }, ui.vitesse ? LTP.t('action.pause') : LTP.t('action.jouer')) : null,
        enJeu ? h('button', { type: 'button', id: 'btn-semaine', onclick: function () {
          if (etat.phase === 'planification') app.dispatch({ type: 'phase', phase: 'execution' });
          app.avancer();
        } }, LTP.t('action.semaine')) : null,
        etat.phase === 'comite' ? h('button', { type: 'button', id: 'btn-continuer', onclick: function () { app.dispatch({ type: 'phase', phase: 'planification' }); } }, LTP.t('action.continuer')) : null));
    },
  };
  LTP.ui.regions.journal = {
    monter: function (racine) { this._racine = racine; },
    rendre: function (etat) {
      LTP.ui.remplacer(this._racine, h('ol', { class: 'journal', reversed: true },
        etat.journal.slice(-12).reverse().map(function (e) { return h('li', null, 'S' + e.semaine + ' · ', LTP.ui.texteJournal(e)); })));
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
