/* 90-main.js */
/* Démarrage : paramètres d'URL, sauvegarde, horloge de lecture, raccourcis clavier globaux.
   ?graine=123 · ?animateur=1 · ?demo=<nom> · ?neuf=1 (ignore la sauvegarde) */
(function (LTP) {
  'use strict';

  function parametres() {
    var p = {};
    try { new URLSearchParams(location.search).forEach(function (v, k) { p[k] = v; }); } catch (e) { /* ignoré */ }
    return p;
  }

  var minuterie = null;
  function planifierTic() {
    clearTimeout(minuterie);
    var ui = LTP.app.ui;
    var delai = ui.vitesse ? LTP.K.ui.dureeSemaineMs / ui.vitesse : 250;
    minuterie = setTimeout(tic, delai);
  }
  function tic() {
    var app = LTP.app, etat = app.etat;
    if (app.ui.vitesse && etat && etat.phase === 'execution' && !etat.evenementEnAttente && !app.ui.modaleBloquante) app.avancer();
    planifierTic();
  }

  function demarrer() {
    var p = parametres();
    var etat = null;
    var options = { animateur: p.animateur === '1', toutDebloque: p.animateur === '1' && p.tout === '1' };
    if (p.demo && LTP.demo) etat = LTP.demo.etat(p.demo);
    else if (!p.graine && p.neuf !== '1' && LTP.sauvegarde) etat = LTP.sauvegarde.charger();
    if (!etat) etat = LTP.moteur.creer(p.graine != null ? p.graine : Date.now(), options);
    LTP.app.demarrer(etat);

    document.addEventListener('keydown', function (ev) {
      if (ev.target && /input|select|textarea|button/i.test(ev.target.tagName)) return;
      if (ev.key === ' ' && LTP.app.etat.phase === 'execution') {
        ev.preventDefault();
        LTP.app.setUI({ vitesse: LTP.app.ui.vitesse ? 0 : 1 });
      }
      if (ev.key === 'Escape' && LTP.app.ui.modale) LTP.app.setUI({ modale: null });
    });
    planifierTic();
  }

  if (typeof document === 'undefined') return;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', demarrer);
  else demarrer();
})(globalThis.LTP = globalThis.LTP || {});
