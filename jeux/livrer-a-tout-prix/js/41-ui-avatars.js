/* 41-ui-avatars.js */
/* Avatars SVG paramétriques et déterministes (CONCEPTION §3.3) et accessoires signature des bureaux.
   Propriétaire : agent E.

   LTP.ui.avatar(id, { taille, etat, vue, fond, libelle }) → <svg> autonome (aucune référence externe)
     taille  : largeur et hauteur en px (32 par défaut)
     etat    : expression — 'incident' (inquiet), 'fatigue' (yeux mi-clos), 'conge' (lunettes de soleil),
               'heureux', 'triste' ; tout autre état du développeur (code, revue, attente…) : neutre
     vue     : 'face' (portrait, par défaut) ou 'dos' (vue par-dessus l'épaule, utilisée dans l'open space)
     fond    : cercle de fond et découpe ronde (vrai par défaut en vue de face)
     libelle : si fourni, l'avatar devient role="img" avec ce nom accessible ; sinon aria-hidden
   LTP.ui.apparence(id) → { teint, cheveux, coiffure, chandail, col, lunettes, fond } (partagé avec l'open space)
   LTP.ui.accessoire(nom) → <g> de 24 × 28 unités, dessiné de haut, à placer sur un bureau */
(function (LTP) {
  'use strict';
  var s = LTP.ui.svg;
  var hachage = LTP.outils.hachage;

  // Huit teints, du plus clair au plus foncé.
  var TEINTS = ['#f6dcc8', '#eec5a4', '#e0ad86', '#c99069', '#ad7350', '#8c5537', '#6c3e25', '#4a2b1a'];
  var CHEVEUX = {
    noir: '#1b1512', brunFonce: '#34241a', chatain: '#664730', auburn: '#82391f',
    roux: '#b2582b', blond: '#cfa767', gris: '#9b958d', blanc: '#dcd7ce',
  };
  var COIFFURES = ['courte', 'raie', 'longue', 'chignon', 'boucles', 'queue', 'rasee', 'carre', 'tresses', 'frisee', 'degarnie'];
  // Chandails : moutarde, bordeaux, marine, sapin, gris chiné, terracotta, sarcelle, lavande, crème, charbon, framboise, olive
  var CHANDAILS = ['#c8962e', '#8c2f39', '#2f4b7c', '#2f6b4f', '#8a8f96', '#c0603a', '#2a8585', '#7d6aa8',
    '#e6d9bf', '#2b2b2e', '#b84a6e', '#5b7f3a'];
  var COLS = ['rond', 'rond', 'v', 'chemise', 'capuchon'];

  // Quelques personnages dont le texte fixe l'âge ou le style. Un champ `apparence` dans les données a priorité.
  var PRESETS = {
    raymond:   { cheveux: 'gris', coiffure: 'degarnie', lunettes: 'rect', col: 'chemise' },
    diane:     { cheveux: 'gris', coiffure: 'carre', lunettes: 'ronde' },
    benoit:    { coiffure: 'courte', col: 'capuchon' },
    ginette:   { cheveux: 'gris', coiffure: 'chignon', lunettes: 'ronde', chandail: '#7d6aa8' },
    regis:     { cheveux: 'gris', coiffure: 'degarnie', lunettes: 'rect', chandail: '#2f4b7c', col: 'chemise' },
    lavigueur: { cheveux: 'blanc', coiffure: 'carre', chandail: '#8c2f39', col: 'chemise' },
    berube:    { lunettes: 'rect', chandail: '#2b2b2e', col: 'chemise' },
    kevin:     { coiffure: 'raie', chandail: '#c0603a', col: 'capuchon', lunettes: null },
  };

  function h(id, sel) { return hachage(String(id) + '|' + sel); }
  function choisir(liste, x) { return liste[Math.min(liste.length - 1, Math.floor(x * liste.length))]; }
  function pondere(paires, x) {
    var total = 0, i;
    for (i = 0; i < paires.length; i++) total += paires[i][1];
    var v = x * total;
    for (i = 0; i < paires.length; i++) { v -= paires[i][1]; if (v < 0) return paires[i][0]; }
    return paires[paires.length - 1][0];
  }

  var cache = {};
  function apparence(id) {
    id = String(id);
    if (cache[id]) return cache[id];
    var def = (LTP.q && LTP.q.defDev && LTP.q.defDev(id)) || (LTP.donnees.personnages || {})[id] || null;
    var senior = def && def.niveau === 'senior';
    var iTeint = Math.floor(h(id, 'teint') * TEINTS.length);
    var fonce = iTeint >= 5;
    var nomCheveux;
    if (senior && h(id, 'age') < 0.3) nomCheveux = h(id, 'gris') < 0.7 ? 'gris' : 'blanc';
    else if (fonce) nomCheveux = pondere([['noir', 6], ['brunFonce', 3], ['chatain', 1]], h(id, 'cheveux'));
    else nomCheveux = pondere([['noir', 2], ['brunFonce', 3], ['chatain', 3], ['auburn', 1], ['roux', 1], ['blond', 2]], h(id, 'cheveux'));
    var coiffure = choisir(COIFFURES, h(id, 'coiffure'));
    if (coiffure === 'degarnie' && !senior) coiffure = choisir(COIFFURES.slice(0, -1), h(id, 'coiffure2'));
    var lx = h(id, 'lunettes');
    var a = {
      teint: TEINTS[iTeint],
      cheveux: nomCheveux,
      coiffure: coiffure,
      chandail: choisir(CHANDAILS, h(id, 'chandail')),
      col: choisir(COLS, h(id, 'col')),
      lunettes: lx < 0.2 ? 'ronde' : lx < 0.38 ? 'rect' : null,
      fond: 'hsl(' + Math.round(h(id, 'fond') * 360) + ' 45% 86%)',
    };
    var p = PRESETS[id], k;
    if (p) for (k in p) a[k] = p[k];
    if (def && def.apparence) for (k in def.apparence) a[k] = def.apparence[k];
    a.couleurCheveux = CHEVEUX[a.cheveux] || a.cheveux;
    if (typeof a.teint === 'number') a.teint = TEINTS[Math.max(0, Math.min(TEINTS.length - 1, a.teint))];
    cache[id] = a;
    return a;
  }

  function assombrir(hex, f) {
    var m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    var n = parseInt(m[1], 16), r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    r = Math.round(r * f); g = Math.round(g * f); b = Math.round(b * f);
    return '#' + ((1 << 24) | (Math.min(255, r) << 16) | (Math.min(255, g) << 8) | Math.min(255, b)).toString(16).slice(1);
  }

  // --- Cheveux, vue de face : [couche arrière, couche avant] ---
  function cheveuxFace(a) {
    var c = a.couleurCheveux, arr = [], av = [];
    var calotte = 'M21.2 26.5 C20.4 17 25 13.4 32 13.4 C39 13.4 43.6 17 42.8 26.5 C41 21.6 37.6 19.6 32.4 19.8 C27.4 20 23.4 22 21.2 26.5 Z';
    switch (a.coiffure) {
      case 'raie':
        av.push(s('path', { d: 'M21 27.5 C20 17 25 12.8 32 12.8 C40 12.8 44.2 18 43 27.5 C42 22 40.2 19.4 36.4 18.4 C32 21 26 21.6 21 27.5 Z', fill: c }));
        break;
      case 'longue':
        arr.push(s('path', { d: 'M19 24 C19 13.5 25 11 32 11 C39 11 45 13.5 45 24 L46.5 47 C41 50 23 50 17.5 47 Z', fill: c }));
        av.push(s('path', { d: 'M21 29 C20 16 26 12.4 32 12.4 C38 12.4 44 16 43 29 C41.4 21 37.4 18 32.6 17.6 C29.6 20 25.2 23 21 29 Z', fill: c }));
        break;
      case 'chignon':
        arr.push(s('circle', { cx: 32, cy: 11.5, r: 5.6, fill: c }));
        av.push(s('path', { d: calotte, fill: c }));
        break;
      case 'boucles':
        arr.push(s('circle', { cx: 32, cy: 23, r: 15.5, fill: c }));
        [[22, 18.5, 5.2], [27, 15.5, 5.6], [32, 14.6, 5.8], [37, 15.5, 5.6], [42, 18.5, 5.2]].forEach(function (b) {
          av.push(s('circle', { cx: b[0], cy: b[1], r: b[2], fill: c }));
        });
        break;
      case 'queue':
        arr.push(s('path', { d: 'M41 19 C50 20 51 35 46.5 43 C45.4 34.5 44.4 27.5 40 23.5 Z', fill: c }));
        av.push(s('path', { d: calotte, fill: c }));
        break;
      case 'rasee':
        av.push(s('path', { d: calotte, fill: c, opacity: 0.55 }));
        break;
      case 'degarnie':
        av.push(s('path', { d: 'M20.6 29 C20 24 20.6 20.6 23 19 L24.6 27.6 Z', fill: c }));
        av.push(s('path', { d: 'M43.4 29 C44 24 43.4 20.6 41 19 L39.4 27.6 Z', fill: c }));
        break;
      case 'carre':
        arr.push(s('path', { d: 'M19.4 24 C19 13.6 25 11.4 32 11.4 C39 11.4 45 13.6 44.6 24 L45.2 39.5 C41 41.6 23 41.6 18.8 39.5 Z', fill: c }));
        av.push(s('path', { d: 'M21 24.5 C21 16 26 12.8 32 12.8 C38 12.8 43 16 43 24.5 L43 22.2 C38 20.6 26 20.6 21 22.2 Z', fill: c }));
        break;
      case 'tresses':
        arr.push(s('path', { d: 'M19 24 C19 13.5 25 11 32 11 C39 11 45 13.5 45 24 L46 50 L41 50 L40 34 L24 34 L23 50 L18 50 Z', fill: c }));
        av.push(s('path', { d: calotte, fill: c }));
        [24, 28, 32, 36, 40].forEach(function (x) {
          av.push(s('path', { d: 'M' + x + ' 14.5 L' + (x + (x - 32) * 0.15) + ' 20', stroke: assombrir(c, 0.7), 'stroke-width': 0.8, fill: 'none' }));
        });
        break;
      case 'frisee':
        [[23, 21, 3.6], [26.5, 17.4, 3.8], [31, 15.6, 4], [35.6, 16.2, 3.9], [39.6, 18.6, 3.7], [42, 22.4, 3.2], [22, 24.6, 2.6]].forEach(function (b) {
          av.push(s('circle', { cx: b[0], cy: b[1], r: b[2], fill: c }));
        });
        break;
      default: // courte
        av.push(s('path', { d: calotte, fill: c }));
    }
    return [arr, av];
  }

  // --- Cheveux, vue de dos : une seule couche, par-dessus la tête ---
  function cheveuxDos(a) {
    var c = a.couleurCheveux, el = [];
    var arriere = 'M21.4 27 C21 17 26 14.4 32 14.4 C38 14.4 43 17 42.6 27 C42.2 33 40 36.6 32 37 C24 36.6 21.8 33 21.4 27 Z';
    switch (a.coiffure) {
      case 'longue':
        el.push(s('path', { d: 'M20.6 26 C20.6 16.6 26 14 32 14 C38 14 43.4 16.6 43.4 26 L45 50 C40 52 24 52 19 50 Z', fill: c }));
        break;
      case 'carre':
        el.push(s('path', { d: 'M20.8 26 C20.8 16.6 26 14 32 14 C38 14 43.2 16.6 43.2 26 L44 40.5 C40 42 24 42 20 40.5 Z', fill: c }));
        break;
      case 'tresses':
        el.push(s('path', { d: arriere, fill: c }));
        [25, 29.4, 34.6, 39].forEach(function (x) {
          el.push(s('rect', { x: x - 1.6, y: 33, width: 3.2, height: 17, rx: 1.6, fill: c }));
        });
        break;
      case 'chignon':
        el.push(s('path', { d: arriere, fill: c }));
        el.push(s('circle', { cx: 32, cy: 22, r: 5.4, fill: assombrir(c, 0.85) }));
        break;
      case 'queue':
        el.push(s('path', { d: arriere, fill: c }));
        el.push(s('path', { d: 'M30 29 C29 38 30.6 44 32 48 C33.4 44 35 38 34 29 Z', fill: assombrir(c, 0.9) }));
        break;
      case 'boucles':
        el.push(s('circle', { cx: 32, cy: 24, r: 14.6, fill: c }));
        break;
      case 'frisee':
        [[24, 22, 4.4], [29, 18.6, 4.8], [35, 18.6, 4.8], [40, 22, 4.4], [26, 28, 4.2], [32, 25, 5], [38, 28, 4.2], [32, 32, 4]].forEach(function (b) {
          el.push(s('circle', { cx: b[0], cy: b[1], r: b[2], fill: c }));
        });
        break;
      case 'rasee':
        el.push(s('path', { d: arriere, fill: c, opacity: 0.5 }));
        break;
      case 'degarnie':
        el.push(s('path', { d: 'M21.6 28 C22 33.4 25 36.6 32 37 C39 36.6 42 33.4 42.4 28 C39 30.4 25 30.4 21.6 28 Z', fill: c }));
        break;
      default: // courte, raie
        el.push(s('path', { d: arriere, fill: c }));
    }
    return el;
  }

  function corps(a, dos) {
    var el = [], ch = a.chandail, ombre = assombrir(ch, 0.78);
    el.push(s('path', { d: dos ? 'M6 64 C6 49 17 42.4 32 42.4 C47 42.4 58 49 58 64 Z' : 'M8 64 C8 50.5 18 44.2 32 44.2 C46 44.2 56 50.5 56 64 Z', fill: ch }));
    el.push(s('rect', { x: 28, y: dos ? 33 : 35, width: 8, height: dos ? 11 : 10.5, rx: 3, fill: assombrir(a.teint, 0.9) }));
    if (a.col === 'chemise') {
      el.push(s('path', { d: dos ? 'M26.6 42.6 L32 40.8 L37.4 42.6 L37 45 L27 45 Z' : 'M26.6 43.6 L32 49 L37.4 43.6 L36 46.8 L32 52 L28 46.8 Z', fill: '#f4f1ea' }));
    } else if (a.col === 'v' && !dos) {
      el.push(s('path', { d: 'M27 44.6 L32 51 L37 44.6 Z', fill: assombrir(a.teint, 0.9) }));
      el.push(s('path', { d: 'M27 44.6 L32 51 L37 44.6', fill: 'none', stroke: ombre, 'stroke-width': 1.2 }));
    } else if (a.col === 'capuchon') {
      el.push(dos
        ? s('path', { d: 'M21 44 C21 40 26 38.6 32 38.6 C38 38.6 43 40 43 44 C43 50 38 53 32 53 C26 53 21 50 21 44 Z', fill: ombre })
        : s('path', { d: 'M22 45 C24 49.5 28 51 32 51 C36 51 40 49.5 42 45 C38 47 26 47 22 45 Z', fill: ombre }));
    } else {
      el.push(s('path', { d: dos ? 'M27 42.8 C29 41.4 35 41.4 37 42.8' : 'M27 44.6 C29 47 35 47 37 44.6', fill: 'none', stroke: ombre, 'stroke-width': 1.3 }));
    }
    return el;
  }

  function visage(a, etat) {
    var el = [], yeux = '#1d1714', peauOmbre = assombrir(a.teint, 0.8);
    var yL = 28.6, xL = 27.6, xR = 36.4;
    // Sourcils
    var sourcil = etat === 'incident'
      ? ['M25 24.6 L29.6 23.6', 'M34.4 23.6 L39 24.6']
      : etat === 'triste' ? ['M25.2 24 L29.4 25', 'M34.6 25 L38.8 24'] : ['M25.4 24.6 C26.8 23.6 28.6 23.6 29.8 24.2', 'M34.2 24.2 C35.4 23.6 37.2 23.6 38.6 24.6'];
    var cs = a.cheveux === 'blanc' || a.cheveux === 'gris' ? assombrir(a.couleurCheveux, 0.7) : a.couleurCheveux;
    sourcil.forEach(function (d) { el.push(s('path', { d: d, fill: 'none', stroke: cs, 'stroke-width': 1.1, 'stroke-linecap': 'round' })); });
    // Yeux
    if (etat === 'conge') {
      el.push(s('rect', { x: 24.4, y: 26.6, width: 6.4, height: 4, rx: 1.8, fill: '#1d1d22' }));
      el.push(s('rect', { x: 33.2, y: 26.6, width: 6.4, height: 4, rx: 1.8, fill: '#1d1d22' }));
      el.push(s('path', { d: 'M30.8 27.6 L33.2 27.6', stroke: '#1d1d22', 'stroke-width': 0.9 }));
    } else if (etat === 'fatigue') {
      el.push(s('path', { d: 'M26 29 C27 29.8 28.4 29.8 29.4 29', fill: 'none', stroke: yeux, 'stroke-width': 1.1, 'stroke-linecap': 'round' }));
      el.push(s('path', { d: 'M34.6 29 C35.6 29.8 37 29.8 38 29', fill: 'none', stroke: yeux, 'stroke-width': 1.1, 'stroke-linecap': 'round' }));
      el.push(s('path', { d: 'M26 31.4 C27 32.2 28.4 32.2 29.4 31.4 M34.6 31.4 C35.6 32.2 37 32.2 38 31.4', fill: 'none', stroke: peauOmbre, 'stroke-width': 0.8 }));
    } else if (etat === 'heureux') {
      el.push(s('path', { d: 'M26 29.4 C26.8 28 28.4 28 29.2 29.4 M34.8 29.4 C35.6 28 37.2 28 38 29.4', fill: 'none', stroke: yeux, 'stroke-width': 1.2, 'stroke-linecap': 'round' }));
    } else {
      var r = etat === 'incident' ? 1.6 : 1.35;
      el.push(s('ellipse', { cx: xL, cy: yL, rx: r, ry: r * 1.15, fill: yeux }));
      el.push(s('ellipse', { cx: xR, cy: yL, rx: r, ry: r * 1.15, fill: yeux }));
    }
    // Joues et nez
    el.push(s('circle', { cx: 25.2, cy: 32.4, r: 2, fill: '#e2786a', opacity: 0.18 }));
    el.push(s('circle', { cx: 38.8, cy: 32.4, r: 2, fill: '#e2786a', opacity: 0.18 }));
    el.push(s('path', { d: 'M32 29.6 L31 32.6 L32.6 32.8', fill: 'none', stroke: peauOmbre, 'stroke-width': 0.9, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
    // Bouche
    var bouche = {
      incident: s('ellipse', { cx: 32, cy: 35.6, rx: 1.6, ry: 1.3, fill: '#5a2a24' }),
      triste: s('path', { d: 'M29.4 36.4 C31 35 33 35 34.6 36.4', fill: 'none', stroke: '#5a2a24', 'stroke-width': 1.1, 'stroke-linecap': 'round' }),
      fatigue: s('path', { d: 'M30 35.6 L34 35.6', fill: 'none', stroke: '#5a2a24', 'stroke-width': 1.1, 'stroke-linecap': 'round' }),
      heureux: s('path', { d: 'M28.6 34.6 C30 37.6 34 37.6 35.4 34.6 Z', fill: '#5a2a24' }),
    }[etat] || s('path', { d: 'M29.4 34.8 C31 36.4 33 36.4 34.6 34.8', fill: 'none', stroke: '#5a2a24', 'stroke-width': 1.1, 'stroke-linecap': 'round' });
    el.push(bouche);
    // Lunettes
    if (a.lunettes && etat !== 'conge') {
      var m = '#2a2420';
      if (a.lunettes === 'ronde') {
        el.push(s('circle', { cx: xL, cy: yL, r: 3.3, fill: 'none', stroke: m, 'stroke-width': 0.9 }));
        el.push(s('circle', { cx: xR, cy: yL, r: 3.3, fill: 'none', stroke: m, 'stroke-width': 0.9 }));
      } else {
        el.push(s('rect', { x: 23.8, y: 26.2, width: 7.6, height: 5, rx: 1.4, fill: 'none', stroke: m, 'stroke-width': 0.9 }));
        el.push(s('rect', { x: 32.6, y: 26.2, width: 7.6, height: 5, rx: 1.4, fill: 'none', stroke: m, 'stroke-width': 0.9 }));
      }
      el.push(s('path', { d: 'M30.9 28.2 C31.6 27.6 32.4 27.6 33.1 28.2', fill: 'none', stroke: m, 'stroke-width': 0.9 }));
    }
    if (etat === 'incident') {
      el.push(s('path', { d: 'M43.6 20 C43.6 20 41.6 23 41.6 24.2 A2 2 0 0 0 45.6 24.2 C45.6 23 43.6 20 43.6 20 Z', fill: '#7cc4f0' }));
    }
    return el;
  }

  var compteur = 0;
  LTP.ui.apparence = apparence;
  LTP.ui.avatar = function (devId, opts) {
    opts = opts || {};
    var a = apparence(devId);
    var taille = opts.taille || 32;
    var dos = opts.vue === 'dos';
    var fond = opts.fond == null ? !dos : !!opts.fond;
    var attrs = { viewBox: '0 0 64 64', width: taille, height: taille, class: 'avatar' + (dos ? ' avatar--dos' : ''), focusable: 'false' };
    if (opts.libelle) { attrs.role = 'img'; attrs['aria-label'] = opts.libelle; } else attrs['aria-hidden'] = 'true';
    var racine = s('svg', attrs);
    var contenu = racine;
    if (fond) {
      var idClip = 'av-clip-' + (++compteur);
      racine.appendChild(s('defs', null, s('clipPath', { id: idClip }, s('circle', { cx: 32, cy: 32, r: 32 }))));
      racine.appendChild(s('circle', { cx: 32, cy: 32, r: 32, fill: a.fond }));
      contenu = s('g', { 'clip-path': 'url(#' + idClip + ')' });
      racine.appendChild(contenu);
    }
    var tete = s('g', { class: 'av-tete' });
    if (dos) {
      corps(a, true).forEach(function (e) { contenu.appendChild(e); });
      tete.appendChild(s('ellipse', { cx: 21.4, cy: 28, rx: 2, ry: 3, fill: assombrir(a.teint, 0.92) }));
      tete.appendChild(s('ellipse', { cx: 42.6, cy: 28, rx: 2, ry: 3, fill: assombrir(a.teint, 0.92) }));
      tete.appendChild(s('ellipse', { cx: 32, cy: 26.6, rx: 10.6, ry: 12.2, fill: a.teint }));
      cheveuxDos(a).forEach(function (e) { tete.appendChild(e); });
      if (a.lunettes) {
        tete.appendChild(s('path', { d: 'M20 26.4 L23 26 M44 26.4 L41 26', stroke: '#2a2420', 'stroke-width': 1, 'stroke-linecap': 'round' }));
      }
    } else {
      var ch = cheveuxFace(a);
      ch[0].forEach(function (e) { contenu.appendChild(e); });
      corps(a, false).forEach(function (e) { contenu.appendChild(e); });
      tete.appendChild(s('ellipse', { cx: 21.4, cy: 29, rx: 2.2, ry: 3, fill: assombrir(a.teint, 0.92) }));
      tete.appendChild(s('ellipse', { cx: 42.6, cy: 29, rx: 2.2, ry: 3, fill: assombrir(a.teint, 0.92) }));
      tete.appendChild(s('ellipse', { cx: 32, cy: 27.2, rx: 10.6, ry: 12.6, fill: a.teint }));
      ch[1].forEach(function (e) { tete.appendChild(e); });
      visage(a, opts.etat).forEach(function (e) { tete.appendChild(e); });
    }
    contenu.appendChild(tete);
    return racine;
  };

  // --- Accessoires signature (vus de haut, boîte de 24 × 28) ---
  function papier(x, y, w, hh, rot) {
    return s('rect', { x: x, y: y, width: w, height: hh, rx: 0.8, fill: '#fbf8f1', stroke: '#cfc6b4', 'stroke-width': 0.6,
      transform: rot ? 'rotate(' + rot + ' ' + (x + w / 2) + ' ' + (y + hh / 2) + ')' : null });
  }
  var DESSINS = {
    'terminal-vert': function () {
      return [s('rect', { x: 0, y: 5, width: 24, height: 20, rx: 2.4, fill: '#d6ccb4', stroke: '#a89c82', 'stroke-width': 0.6 }),
        s('rect', { x: 3, y: 8, width: 18, height: 12, rx: 1.2, fill: '#0b1a0e' }),
        s('rect', { x: 5, y: 10.2, width: 10, height: 1.3, fill: '#4cff7a' }),
        s('rect', { x: 5, y: 13, width: 13, height: 1.3, fill: '#4cff7a', opacity: 0.8 }),
        s('rect', { x: 5, y: 15.8, width: 6, height: 1.3, fill: '#4cff7a' }),
        s('rect', { x: 12, y: 15.8, width: 2.2, height: 1.3, fill: '#4cff7a', class: 'acc-clignote' }),
        s('rect', { x: 9, y: 25, width: 6, height: 2, fill: '#a89c82' })];
    },
    classeurs: function () {
      return [[1, 5, '#2f5d9a'], [8.5, 3, '#c0392b'], [16, 6, '#e2b33c']].map(function (c) {
        return s('g', null, s('rect', { x: c[0], y: c[1], width: 7, height: 22, rx: 1, fill: c[2] }),
          s('rect', { x: c[0] + 1.5, y: c[1] + 4, width: 4, height: 6, rx: 0.5, fill: '#f6f1e6' }),
          s('circle', { cx: c[0] + 3.5, cy: c[1] + 17, r: 1.2, fill: 'none', stroke: '#f6f1e6', 'stroke-width': 0.6 }));
      });
    },
    cables: function () {
      return [s('path', { d: 'M1 22 C6 4 13 30 18 12 S 9 2 4 9 S 18 26 23 18', fill: 'none', stroke: '#2b2b2b', 'stroke-width': 1.6, 'stroke-linecap': 'round' }),
        s('path', { d: 'M3 26 C10 18 6 8 16 6 S 22 16 20 24', fill: 'none', stroke: '#c0392b', 'stroke-width': 1.4, 'stroke-linecap': 'round' }),
        s('rect', { x: 19, y: 23, width: 4, height: 3, rx: 0.6, fill: '#9aa0a6' }),
        s('rect', { x: 0, y: 20, width: 3, height: 4, rx: 0.6, fill: '#3a3a3a' })];
    },
    tasse: function () {
      return [s('circle', { cx: 19.5, cy: 15, r: 3.4, fill: 'none', stroke: '#efe9df', 'stroke-width': 2.2 }),
        s('circle', { cx: 11, cy: 15, r: 8.2, fill: '#f3efe8', stroke: '#c9c1b3', 'stroke-width': 0.7 }),
        s('circle', { cx: 11, cy: 15, r: 6.2, fill: '#5a3a22' }),
        s('ellipse', { cx: 9.4, cy: 13.4, rx: 2.2, ry: 1.2, fill: '#8a5e3c', opacity: 0.7 }),
        s('path', { d: 'M9 6 C7 3 11 2 9 -2 M13 6 C11 3 15 2 13 -2', fill: 'none', stroke: '#ffffff', 'stroke-width': 0.9, opacity: 0.7, class: 'acc-vapeur' })];
    },
    'tableau-bord': function () {
      return [s('rect', { x: 0, y: 6, width: 24, height: 17, rx: 2, fill: '#232a33' }),
        s('rect', { x: 3, y: 15, width: 3, height: 5, fill: '#4cc38a' }), s('rect', { x: 7.5, y: 12, width: 3, height: 8, fill: '#4cc38a' }),
        s('rect', { x: 12, y: 10, width: 3, height: 10, fill: '#e5b84a' }), s('rect', { x: 16.5, y: 13, width: 3, height: 7, fill: '#e2665a' }),
        s('path', { d: 'M3 9.5 L8 11 L13 8.5 L21 9', fill: 'none', stroke: '#7fd3ff', 'stroke-width': 0.8 })];
    },
    carnet: function () {
      var el = [s('rect', { x: 3, y: 2, width: 17, height: 24, rx: 1.4, fill: '#e9e2cf', stroke: '#c2b79c', 'stroke-width': 0.6 })];
      for (var i = 0; i < 6; i++) el.push(s('circle', { cx: 3, cy: 5 + i * 3.8, r: 1, fill: 'none', stroke: '#6b6b6b', 'stroke-width': 0.6 }));
      for (var j = 0; j < 5; j++) el.push(s('path', { d: 'M6 ' + (7 + j * 3.6) + ' L18 ' + (7 + j * 3.6), stroke: '#9fb6d6', 'stroke-width': 0.5 }));
      el.push(s('rect', { x: 20, y: 6, width: 2.2, height: 16, rx: 0.6, fill: '#f0c419', transform: 'rotate(12 21 14)' }));
      return el;
    },
    organigramme: function () {
      return [papier(1, 2, 22, 25, -4),
        s('rect', { x: 9, y: 5, width: 6, height: 4, fill: '#3b82c4' }),
        s('rect', { x: 3.5, y: 14, width: 6, height: 4, fill: '#3b82c4' }), s('rect', { x: 14.5, y: 14, width: 6, height: 4, fill: '#e2665a' }),
        s('rect', { x: 9, y: 21, width: 6, height: 3.6, fill: '#4cc38a' }),
        s('path', { d: 'M12 9 L12 11.5 L6.5 11.5 L6.5 14 M12 11.5 L17.5 11.5 L17.5 14 M6.5 18 L12 21 M17.5 18 L12 21', fill: 'none', stroke: '#555', 'stroke-width': 0.6 })];
    },
    plante: function () {
      var el = [s('circle', { cx: 12, cy: 16, r: 7, fill: '#b5643a' }), s('circle', { cx: 12, cy: 16, r: 5.6, fill: '#6b4128' })];
      for (var i = 0; i < 7; i++) {
        var ang = i * 360 / 7;
        el.push(s('ellipse', { cx: 12, cy: 9, rx: 2.6, ry: 6.5, fill: i % 2 ? '#4f8a3c' : '#3f7a33', transform: 'rotate(' + ang + ' 12 16)' }));
      }
      el.push(s('circle', { cx: 12, cy: 16, r: 2, fill: '#5d9c45' }));
      return el;
    },
    'sac-a-dos': function () {
      return [s('rect', { x: 2, y: 3, width: 20, height: 24, rx: 7, fill: '#3c6e71' }),
        s('rect', { x: 5.5, y: 14, width: 13, height: 9, rx: 3, fill: '#2e5658' }),
        s('path', { d: 'M5.5 16 L18.5 16', stroke: '#e5b84a', 'stroke-width': 0.8 }),
        s('rect', { x: 8, y: 1, width: 8, height: 4, rx: 2, fill: 'none', stroke: '#2e5658', 'stroke-width': 1.6 })];
    },
    cactus: function () {
      var el = [s('circle', { cx: 12, cy: 15, r: 7, fill: '#c8763f' }), s('circle', { cx: 12, cy: 15, r: 5.4, fill: '#4f8a3c' })];
      for (var i = 0; i < 10; i++) {
        var a = i * Math.PI / 5;
        el.push(s('circle', { cx: 12 + Math.cos(a) * 3.6, cy: 15 + Math.sin(a) * 3.6, r: 0.45, fill: '#f3efe0' }));
      }
      el.push(s('circle', { cx: 13.4, cy: 13.4, r: 1.6, fill: '#e86a9a' }));
      return el;
    },
    telephones: function () {
      return [s('rect', { x: 1, y: 3, width: 10, height: 19, rx: 2, fill: '#222', transform: 'rotate(-8 6 12)' }),
        s('rect', { x: 2.2, y: 5, width: 7.6, height: 14, rx: 1, fill: '#6fb3ff', transform: 'rotate(-8 6 12)' }),
        s('rect', { x: 12, y: 7, width: 10, height: 19, rx: 2, fill: '#e2e2e2', transform: 'rotate(6 17 16)' }),
        s('rect', { x: 13.2, y: 9, width: 7.6, height: 14, rx: 1, fill: '#2b2b33', transform: 'rotate(6 17 16)' })];
    },
    'boisson-energie': function () {
      return [s('circle', { cx: 8, cy: 9, r: 6, fill: '#c9cdd1' }), s('circle', { cx: 8, cy: 9, r: 4.8, fill: '#9be15d' }),
        s('ellipse', { cx: 8, cy: 7.6, rx: 1.6, ry: 1, fill: '#7a7f85' }),
        s('rect', { x: 6, y: 17, width: 17, height: 8, rx: 4, fill: '#9be15d', transform: 'rotate(-14 14 21)' }),
        s('path', { d: 'M15 17.6 L12.6 21.4 L14.8 21.4 L13 24.6', fill: 'none', stroke: '#1d1d22', 'stroke-width': 0.9, transform: 'rotate(-14 14 21)' })];
    },
    oscilloscope: function () {
      return [s('rect', { x: 0, y: 5, width: 24, height: 19, rx: 2, fill: '#b9bdc1' }),
        s('rect', { x: 2, y: 7, width: 14, height: 12, rx: 1, fill: '#0d1f17' }),
        s('path', { d: 'M3 13 C5 8 7 8 9 13 S 13 18 15 13', fill: 'none', stroke: '#4cff7a', 'stroke-width': 0.9 }),
        s('circle', { cx: 20, cy: 10, r: 1.8, fill: '#444' }), s('circle', { cx: 20, cy: 16, r: 1.8, fill: '#c0392b' })];
    },
    'fer-a-souder': function () {
      return [s('circle', { cx: 6, cy: 21, r: 5, fill: '#555' }), s('circle', { cx: 6, cy: 21, r: 2.6, fill: '#c9b458' }),
        s('path', { d: 'M8 19 L19 5', stroke: '#c0392b', 'stroke-width': 3.2, 'stroke-linecap': 'round' }),
        s('path', { d: 'M19 5 L22.5 1', stroke: '#c0c0c0', 'stroke-width': 1.4, 'stroke-linecap': 'round' }),
        s('path', { d: 'M22 1 C20 -2 24 -3 22 -6', fill: 'none', stroke: '#ffffff', 'stroke-width': 0.8, opacity: 0.6, class: 'acc-vapeur' })];
    },
    graphique: function () {
      return [papier(1, 2, 22, 24, 3),
        s('path', { d: 'M4 23 L4 6 M4 23 L21 23', stroke: '#555', 'stroke-width': 0.6, fill: 'none' }),
        s('path', { d: 'M5 20 L9 16 L12 18 L16 10 L20 7', fill: 'none', stroke: '#3b82c4', 'stroke-width': 1.1 }),
        s('circle', { cx: 16, cy: 10, r: 1, fill: '#e2665a' })];
    },
    'tablette-dessin': function () {
      return [s('rect', { x: 0, y: 4, width: 24, height: 18, rx: 2, fill: '#2b2b33' }),
        s('rect', { x: 3, y: 6.5, width: 18, height: 13, rx: 1, fill: '#3d3d4a' }),
        s('path', { d: 'M5 16 C8 9 12 18 18 9', fill: 'none', stroke: '#ff7eb6', 'stroke-width': 1.2, 'stroke-linecap': 'round' }),
        s('path', { d: 'M16 27 L23 20', stroke: '#e6e6e6', 'stroke-width': 1.4, 'stroke-linecap': 'round' })];
    },
    'mug-wp': function () {
      return [s('circle', { cx: 20, cy: 15, r: 3.2, fill: 'none', stroke: '#21759b', 'stroke-width': 2.2 }),
        s('circle', { cx: 11, cy: 15, r: 8.2, fill: '#21759b' }), s('circle', { cx: 11, cy: 15, r: 6, fill: '#3a2414' }),
        s('circle', { cx: 11, cy: 15, r: 3.4, fill: 'none', stroke: '#ffffff', 'stroke-width': 0.8, opacity: 0.9 }),
        s('path', { d: 'M9 13.6 L10 16.6 L11 14.4 L12 16.6 L13 13.6', fill: 'none', stroke: '#ffffff', 'stroke-width': 0.7 })];
    },
    ecrans: function () {
      return [s('path', { d: 'M2 9 L22 9 L20 1 L4 1 Z', fill: '#2b2f36' }), s('path', { d: 'M4 8 L20 8 L18.6 2.2 L5.4 2.2 Z', fill: '#7fd3ff', opacity: 0.85 }),
        s('rect', { x: 1, y: 9, width: 22, height: 13, rx: 1.4, fill: '#b9bdc1' }),
        s('rect', { x: 3, y: 11, width: 18, height: 6, rx: 0.6, fill: '#8d9298' }), s('rect', { x: 8, y: 18, width: 8, height: 2.6, rx: 0.6, fill: '#8d9298' }),
        s('path', { d: 'M12 22 C12 26 4 24 2 27', fill: 'none', stroke: '#2b2b2b', 'stroke-width': 1 })];
    },
    casque: function () {
      return [s('path', { d: 'M4 16 C4 4 20 4 20 16', fill: 'none', stroke: '#222', 'stroke-width': 3, 'stroke-linecap': 'round' }),
        s('ellipse', { cx: 4, cy: 18, rx: 3.6, ry: 5, fill: '#333' }), s('ellipse', { cx: 20, cy: 18, rx: 3.6, ry: 5, fill: '#333' }),
        s('ellipse', { cx: 4, cy: 18, rx: 1.6, ry: 2.6, fill: '#e2443a' }), s('ellipse', { cx: 20, cy: 18, rx: 1.6, ry: 2.6, fill: '#e2443a' })];
    },
    'pile-papiers': function () {
      return [s('rect', { x: 0, y: 3, width: 23, height: 24, rx: 1.6, fill: '#8a8f96' }),
        papier(2.5, 5, 18, 20, -3), papier(2.5, 5, 18, 20, 2), papier(3, 5.5, 17, 19, 0),
        s('rect', { x: 6, y: 9, width: 11, height: 4.4, rx: 0.6, fill: 'none', stroke: '#c0261c', 'stroke-width': 0.9, transform: 'rotate(-8 11.5 11)' }),
        s('path', { d: 'M7.4 10.4 L7.4 12.2 L9 12.2 M10.6 10.4 L10.6 12.2 L12.2 12.2 M13.6 10.4 L15.6 10.4 M14.6 10.4 L14.6 12.4', fill: 'none', stroke: '#c0261c', 'stroke-width': 0.6, transform: 'rotate(-8 11.5 11)' }),
        s('path', { d: 'M4 26 L20 22', stroke: '#c0261c', 'stroke-width': 1.4, 'stroke-linecap': 'round' })];
    },
    cadenas: function () {
      return [s('path', { d: 'M7 13 L7 8 C7 3 17 3 17 8 L17 13', fill: 'none', stroke: '#8a8f96', 'stroke-width': 2.6 }),
        s('rect', { x: 3.5, y: 12, width: 17, height: 13, rx: 2.2, fill: '#e0b53a' }),
        s('circle', { cx: 12, cy: 17.4, r: 1.8, fill: '#6b5418' }), s('rect', { x: 11.2, y: 18, width: 1.6, height: 3.6, fill: '#6b5418' })];
    },
  };

  LTP.ui.accessoire = function (nom) {
    var f = DESSINS[nom];
    return s('g', { class: 'acc acc-' + (nom || 'aucun') }, f ? f() : []);
  };
  LTP.ui.accessoires = Object.keys(DESSINS);
})(globalThis.LTP = globalThis.LTP || {});
