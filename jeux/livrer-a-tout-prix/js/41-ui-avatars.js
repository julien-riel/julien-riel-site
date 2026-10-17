/* 41-ui-avatars.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent E. Avatars SVG paramétriques et déterministes. */
(function (LTP) {
  'use strict';
  // avatar(devId, { taille }) → élément SVG
  LTP.ui.avatar = function (devId, opts) {
    var s = LTP.ui.svg, t = (opts && opts.taille) || 32, hue = Math.round(LTP.outils.hachage(devId) * 360);
    return s('svg', { viewBox: '0 0 32 32', width: t, height: t, class: 'avatar', 'aria-hidden': 'true' },
      s('circle', { cx: 16, cy: 12, r: 7, fill: 'hsl(30 40% 70%)' }),
      s('rect', { x: 6, y: 20, width: 20, height: 12, rx: 6, fill: 'hsl(' + hue + ' 50% 50%)' }));
  };
})(globalThis.LTP = globalThis.LTP || {});
