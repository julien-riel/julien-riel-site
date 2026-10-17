/* 42-ui-open-space.js */
/* L'open space vu de haut (CONCEPTION §3.3 et §8). Propriétaire : agent E.
   Le DOM est construit une fois au montage : 7 îlots (div[data-ilot]) de 3 bureaux (button[data-dev]),
   chacun avec une petite scène SVG vue par-dessus l'épaule. Chaque rendu ne fait que des différences :
   classes, attributs, transformations et textes. Les écouteurs sont délégués à la racine de la région.

   Signaux par bureau (classes sur le bouton) :
     écrans allumés = dev.sessions · pile de papiers = PR à relire (ilot.fileRevue réparti sur les réviseurs)
     brouillard = 1 − compréhension (uniforme si la métrique n'est pas visible) · flamme = incident
     goutte, puis tête sur le bureau = fatigue · valise = congé · boîte = départ ou absence · livre = formation
   Au téléphone (< 700 px), la même structure devient une liste d'îlots repliables (voir 30-open-space.css). */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, s = LTP.ui.svg, O = LTP.outils, F = LTP.format;

  LTP.K.openspace = {
    seuilSueur: 0.55,       // fatigue à partir de laquelle la goutte de sueur apparaît
    seuilEpuise: 0.8,       // fatigue à partir de laquelle la tête tombe sur le bureau
    seuilBulleFatigue: 0.7, // au-delà, la bulle parle de fatigue plutôt que de l'activité
    feuillesMax: 7,         // feuilles dessinées dans une pile (le nombre exact est écrit sur le post-it)
    bullesMax: 2,
    brouillardInconnu: 0.32,
  };
  var K = LTP.K.openspace;

  var ETATS_ACTIFS = { code: 1, revue: 1, attente: 1, incident: 1 };
  var LIBELLES_ETAT = {
    code: 'code avec ses agents', revue: 'fait de la revue', attente: 'attend une revue', incident: 'gère un incident',
    formation: 'en formation', conge: 'en congé', absent: 'hors du bureau',
  };

  // --- Petits outils de mise à jour par différence ---
  function attr(el, k, v) {
    var c = el.__osA || (el.__osA = {});
    if (c[k] === v) return;
    c[k] = v;
    if (v == null) el.removeAttribute(k); else el.setAttribute(k, v);
  }
  function texte(el, v) { v = v == null ? '' : String(v); if (el.__osT === v) return; el.__osT = v; el.textContent = v; }
  function style(el, k, v) {
    var c = el.__osS || (el.__osS = {});
    if (c[k] === v) return;
    c[k] = v;
    el.style.setProperty(k, v);
  }
  function classe(el, nom, oui) { if (el.classList.contains(nom) !== !!oui) el.classList.toggle(nom, !!oui); }
  function pluriel(n, un, plusieurs) { return n + ' ' + (n > 1 ? plusieurs : un); }

  // --- Icônes ---
  function icone(nom) {
    var p = {
      prive: [s('rect', { x: 3, y: 7, width: 10, height: 7, rx: 1.4 }), s('path', { d: 'M5.4 7 V5 a2.6 2.6 0 0 1 5.2 0 V7', fill: 'none', 'stroke-width': 1.6 })],
      nonFiable: [s('path', { d: 'M8 1.6 L15 14 H1 Z' }), s('rect', { x: 7.2, y: 5.6, width: 1.6, height: 4.4, class: 'creux' }), s('rect', { x: 7.2, y: 11, width: 1.6, height: 1.6, class: 'creux' })],
      sortie: [s('path', { d: 'M2 8 H11 M8 4 L12 8 L8 12', fill: 'none', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }), s('rect', { x: 13, y: 2, width: 1.8, height: 12, rx: 0.9 })],
      prise: [s('rect', { x: 4, y: 6, width: 8, height: 6, rx: 1.6 }), s('rect', { x: 5.4, y: 2, width: 1.6, height: 4.4 }), s('rect', { x: 9, y: 2, width: 1.6, height: 4.4 }), s('rect', { x: 7.2, y: 12, width: 1.6, height: 3 })],
      flamme: [s('path', { d: 'M8 15 C3 13 3 8 6 4 C6 7 8 7 8.4 5 C11 7 13 11 8 15 Z' })],
      losange: [s('path', { d: 'M8 1.5 L14.5 8 L8 14.5 L1.5 8 Z' })],
      chevron: [s('path', { d: 'M4 6 L8 10 L12 6', fill: 'none', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' })],
    }[nom] || [];
    return s('svg', { viewBox: '0 0 16 16', class: 'os-ico os-ico--' + nom, 'aria-hidden': 'true', focusable: 'false' }, p);
  }

  // --- Définitions partagées (dégradés) : un seul petit SVG invisible par région ---
  function definitions() {
    return s('svg', { class: 'os-defs', width: 0, height: 0, 'aria-hidden': 'true', focusable: 'false' },
      s('defs', null,
        s('radialGradient', { id: 'os-g-brouillard' },
          s('stop', { offset: '0', class: 'os-stop-brouillard', 'stop-opacity': '0.95' }),
          s('stop', { offset: '0.55', class: 'os-stop-brouillard', 'stop-opacity': '0.6' }),
          s('stop', { offset: '1', class: 'os-stop-brouillard', 'stop-opacity': '0' })),
        s('radialGradient', { id: 'os-g-lueur' },
          s('stop', { offset: '0', class: 'os-stop-lueur', 'stop-opacity': '0.9' }),
          s('stop', { offset: '1', class: 'os-stop-lueur', 'stop-opacity': '0' })),
        s('linearGradient', { id: 'os-g-flaque', x1: '0', y1: '0', x2: '0', y2: '1' },
          s('stop', { offset: '0', class: 'os-stop-lueur', 'stop-opacity': '0.55' }),
          s('stop', { offset: '1', class: 'os-stop-lueur', 'stop-opacity': '0' })),
        s('radialGradient', { id: 'os-g-alerte' },
          s('stop', { offset: '0', class: 'os-stop-alerte', 'stop-opacity': '0.7' }),
          s('stop', { offset: '1', class: 'os-stop-alerte', 'stop-opacity': '0' }))));
  }

  // --- Disposition des écrans selon leur nombre (rangée du bas sur le bureau, rangée du haut sur bras) ---
  var X_RANG = { 1: [29], 2: [16, 42], 3: [4, 29, 54] };
  function positionsEcrans(n) {
    var pos = [], bas = Math.min(3, n), haut = Math.max(0, n - 3), i;
    for (i = 0; i < bas; i++) pos.push([X_RANG[bas][i], 24]);
    for (i = 0; i < haut; i++) pos.push([X_RANG[haut][i], 9]);
    return pos;
  }

  // --- Scène d'un bureau (viewBox 80 × 132) ---
  function construireScene(graine) {
    var r = {};
    var hx = function (k) { return O.hachage(graine + ':' + k); };
    var svg = s('svg', { class: 'os-scene', viewBox: '0 0 80 120', 'aria-hidden': 'true', focusable: 'false' });
    r.svg = svg;
    // Cloison de tissu et son post-it
    svg.appendChild(s('rect', { class: 's-cloison', x: -1, y: 0, width: 82, height: 9 }));
    svg.appendChild(s('rect', { class: 's-postit-cloison', x: 6 + Math.round(hx('px') * 58), y: 1.4, width: 7, height: 6.4, rx: 0.4,
      style: 'fill: hsl(' + [52, 18, 190, 95, 330][Math.floor(hx('pc') * 5)] + ' 85% 72%)',
      transform: 'rotate(' + Math.round(hx('pr') * 16 - 8) + ' 10 4.6)' }));
    // Plan de travail
    svg.appendChild(s('rect', { class: 's-bureau', x: 0, y: 9, width: 80, height: 71 }));
    svg.appendChild(s('rect', { class: 's-bureau-bord', x: 0, y: 78, width: 80, height: 3 }));
    svg.appendChild(s('path', { class: 's-joint', d: 'M0.5 9 V81' }));
    r.alerte = s('ellipse', { class: 's-alerte', cx: 52, cy: 30, rx: 44, ry: 34, fill: 'url(#os-g-alerte)' });
    svg.appendChild(r.alerte);
    r.flaque = s('path', { class: 's-flaque', d: 'M6 34 L74 34 L60 74 L20 74 Z', fill: 'url(#os-g-flaque)' });
    svg.appendChild(r.flaque);
    // Écrans : lueurs d'abord, puis les écrans eux-mêmes
    var lueurs = s('g', { class: 's-lueurs' }), ecrans = s('g', { class: 's-ecrans' });
    r.ecrans = [];
    for (var i = 0; i < 6; i++) {
      var lu = s('ellipse', { class: 's-lueur', cx: 11, cy: 10, rx: 15, ry: 11, fill: 'url(#os-g-lueur)' });
      var w1 = 6 + Math.round(hx('l1' + i) * 8), w2 = 4 + Math.round(hx('l2' + i) * 10), w3 = 5 + Math.round(hx('l3' + i) * 9);
      var teinteCode = 's-code--' + (1 + Math.floor(hx('c' + i) * 3));
      var frappe = s('rect', { class: 's-code s-frappe ' + teinteCode, x: 3.2, y: 8.6, width: w3, height: 1.3 });
      frappe.style.animationDelay = (-hx('d' + i) * 2).toFixed(2) + 's';
      var lueGrp = s('g', { class: 's-ecran-lueur' }, lu);
      lueGrp.style.animationDelay = (-hx('g' + i) * 3).toFixed(2) + 's';
      var g = s('g', { class: 's-ecran' },
        s('rect', { class: 's-pied', x: 9, y: 12.6, width: 4, height: 2.4 }),
        s('rect', { class: 's-cadre', x: 0, y: 0, width: 22, height: 13.4, rx: 1.2 }),
        s('rect', { class: 's-dalle', x: 1.2, y: 1.2, width: 19.6, height: 10.8, rx: 0.5 }),
        s('rect', { class: 's-code ' + teinteCode, x: 3.2, y: 3.2, width: w1, height: 1.3 }),
        s('rect', { class: 's-code s-code--3', x: 5.2, y: 5.9, width: w2, height: 1.3 }),
        frappe);
      lueurs.appendChild(lueGrp);
      ecrans.appendChild(g);
      r.ecrans.push({ g: g, lueur: lueGrp });
    }
    svg.appendChild(lueurs);
    svg.appendChild(ecrans);
    // Clavier et souris
    svg.appendChild(s('rect', { class: 's-clavier', x: 29, y: 63, width: 22, height: 8, rx: 1.4 }));
    svg.appendChild(s('path', { class: 's-touches', d: 'M31 65.6 H49 M31 68.4 H49' }));
    svg.appendChild(s('ellipse', { class: 's-souris', cx: 54.6, cy: 67, rx: 2, ry: 2.8 }));
    // Accessoire signature (rempli par l'occupant)
    r.accessoire = s('g', { class: 's-accessoire', transform: 'translate(3 47) scale(0.95)' });
    svg.appendChild(r.accessoire);
    // Pile de PR à relire
    var pile = s('g', { class: 's-pile', transform: 'translate(58 50)' });
    r.feuilles = [];
    for (var f = 0; f < K.feuillesMax; f++) {
      var feuille = s('rect', { class: 's-feuille', x: 0, y: 6, width: 17, height: 21, rx: 0.8 });
      feuille.style.transform = 'translate(0px, ' + (-f * 1.7) + 'px) rotate(' + ((f % 2 ? 1 : -1) * (1 + hx('f' + f) * 5)).toFixed(1) + 'deg)';
      pile.appendChild(feuille);
      r.feuilles.push(feuille);
    }
    r.etiquettePile = s('text', { class: 's-pile-n', x: 14.6, y: 2.6 });
    r.badgePile = s('g', { class: 's-pile-badge' }, s('rect', { class: 's-pile-postit', x: 9, y: -4, width: 11.4, height: 9, rx: 0.6, transform: 'rotate(6 14.6 0.5)' }), r.etiquettePile);
    pile.appendChild(r.badgePile);
    svg.appendChild(pile);
    // Livre ouvert (formation)
    svg.appendChild(s('g', { class: 's-livre' },
      s('path', { class: 's-livre-couv', d: 'M25 50 L40 52 L55 50 L55 68 L40 70 L25 68 Z' }),
      s('path', { class: 's-livre-page', d: 'M26.6 51.4 L39.4 53.2 L39.4 68.4 L26.6 66.6 Z' }),
      s('path', { class: 's-livre-page', d: 'M53.4 51.4 L40.6 53.2 L40.6 68.4 L53.4 66.6 Z' }),
      s('path', { class: 's-livre-lignes', d: 'M29 55 L37 56 M29 58 L37 59 M29 61 L35 62 M43 56 L51 55 M43 59 L51 58 M43 62 L49 61' })));
    // Boîte de carton (départ ou absence)
    svg.appendChild(s('g', { class: 's-boite' },
      s('rect', { class: 's-boite-corps', x: 22, y: 40, width: 36, height: 30, rx: 1 }),
      s('path', { class: 's-boite-rabat', d: 'M22 40 L40 46 L58 40 M40 46 V70' }),
      s('rect', { class: 's-boite-ruban', x: 36.6, y: 40, width: 6.8, height: 30 }),
      s('rect', { class: 's-postit-boite', x: 44, y: 55, width: 9, height: 8, rx: 0.4, transform: 'rotate(-6 48 59)' })));
    // Flamme (incident)
    var flamme = s('g', { class: 's-flamme', transform: 'translate(58 -10)' },
      s('g', { class: 's-flamme-anim' },
        s('path', { class: 's-flamme-ext', d: 'M11 30 C1 26 1 16 7 7 C7 13 10 14 11.5 9 C14 3 13 1 13 1 C22 9 24 24 11 30 Z' }),
        s('path', { class: 's-flamme-int', d: 'M11.6 28 C6 26 6.4 20 9.4 15.6 C10 19 12 19.4 12.6 16.6 C16.6 20 17 26 11.6 28 Z' })));
    svg.appendChild(flamme);
    // Chaise, bras, personne
    r.chaise = [s('circle', { class: 's-chaise s-chaise-siege', cx: 40, cy: 99, r: 15 })];
    svg.appendChild(r.chaise[0]);
    r.personne = s('g', { class: 's-personne' });
    r.bras = s('g', { class: 's-bras' });
    r.brasPlies = s('g', { class: 's-bras-plies' });
    r.corps = s('g', { class: 's-corps' });
    r.personne.appendChild(r.bras);
    r.personne.appendChild(r.brasPlies);
    r.personne.appendChild(r.corps);
    svg.appendChild(r.personne);
    r.chaise.push(s('rect', { class: 's-chaise s-chaise-dossier', x: 24, y: 106.5, width: 32, height: 7, rx: 3.5 }));
    svg.appendChild(r.chaise[1]);
    // Valise (congé)
    svg.appendChild(s('g', { class: 's-valise' },
      s('rect', { class: 's-valise-poignee', x: 64, y: 86, width: 8, height: 4, rx: 1.6 }),
      s('rect', { class: 's-valise-corps', x: 59, y: 89, width: 18, height: 24, rx: 3 }),
      s('path', { class: 's-valise-sangle', d: 'M59 97 H77 M59 105 H77' }),
      s('circle', { class: 's-valise-autocollant', cx: 72, cy: 109.4, r: 2.4 })));
    // Fatigue : goutte et « z »
    svg.appendChild(s('path', { class: 's-sueur', d: 'M3 0 C3 0 0 4 0 6 A3 3 0 0 0 6 6 C6 4 3 0 3 0 Z', transform: 'translate(55 76)' }));
    svg.appendChild(s('g', { class: 's-zzz' },
      s('text', { x: 46, y: 60, class: 's-z s-z1' }, 'z'), s('text', { x: 51, y: 53, class: 's-z s-z2' }, 'z'), s('text', { x: 57, y: 45, class: 's-z s-z3' }, 'Z')));
    // Brouillard (dette cognitive)
    r.brouillard = s('g', { class: 's-brouillard' },
      s('g', { class: 's-nappe s-nappe1' }, s('circle', { cx: 18, cy: 44, r: 34, fill: 'url(#os-g-brouillard)' }), s('circle', { cx: 64, cy: 30, r: 30, fill: 'url(#os-g-brouillard)' })),
      s('g', { class: 's-nappe s-nappe2' }, s('circle', { cx: 42, cy: 90, r: 36, fill: 'url(#os-g-brouillard)' }), s('circle', { cx: 74, cy: 100, r: 22, fill: 'url(#os-g-brouillard)' })));
    svg.appendChild(r.brouillard);
    return r;
  }

  // Manches un peu plus sombres que le chandail, pour détacher les bras du dos.
  function assombrirCss(hex) {
    var m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    var n = parseInt(m[1], 16);
    var c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(function (v) { return Math.round(v * 0.86); });
    return 'rgb(' + c.join(',') + ')';
  }

  // Remplit la scène avec l'occupant (avatar de dos, bras à ses couleurs, accessoire).
  function habiller(r, devId) {
    var a = LTP.ui.apparence(devId);
    LTP.ui.vider(r.corps); LTP.ui.vider(r.bras); LTP.ui.vider(r.brasPlies); LTP.ui.vider(r.accessoire);
    var manche = assombrirCss(a.chandail);
    r.bras.appendChild(s('path', { d: 'M26.4 99 C21.4 93 23 83 33.8 73', fill: 'none', stroke: manche, 'stroke-width': 5.6, 'stroke-linecap': 'round' }));
    r.bras.appendChild(s('path', { d: 'M53.6 99 C58.6 93 57 83 46.2 73', fill: 'none', stroke: manche, 'stroke-width': 5.6, 'stroke-linecap': 'round' }));
    r.bras.appendChild(s('circle', { cx: 34.4, cy: 71.6, r: 2.7, fill: a.teint }));
    r.bras.appendChild(s('circle', { cx: 45.6, cy: 71.6, r: 2.7, fill: a.teint }));
    r.brasPlies.appendChild(s('rect', { x: 15, y: 79, width: 50, height: 10, rx: 5, fill: manche }));
    r.brasPlies.appendChild(s('path', { d: 'M40 79.6 V88.4', stroke: 'rgba(0,0,0,0.18)', 'stroke-width': 1 }));
    var av = LTP.ui.avatar(devId, { vue: 'dos', taille: 44 });
    av.setAttribute('x', '18'); av.setAttribute('y', '66'); av.setAttribute('width', '44'); av.setAttribute('height', '44');
    av.removeAttribute('class');
    r.corps.appendChild(av);
    var def = LTP.q.defDev(devId);
    if (def && def.accessoire) r.accessoire.appendChild(LTP.ui.accessoire(def.accessoire));
  }

  // --- Construction d'un îlot ---
  function construireIlot(def, vue) {
    var ilot = { id: def.id, def: def, bureaux: [] };
    var idBureaux = 'os-bureaux-' + def.id;
    ilot.voyants = {};
    var voyants = ['prive', 'nonFiable', 'sortie'].map(function (k) {
      var v = h('span', { class: 'os-voyant', 'data-voyant': k }, icone(k));
      ilot.voyants[k] = v;
      return v;
    });
    ilot.triade = h('span', { class: 'os-triade', 'aria-hidden': 'true' }, voyants);
    ilot.compteurVal = h('span', { class: 'os-compteur-val' }, '0');
    ilot.compteurBarre = h('span', { class: 'os-compteur-barre' }, h('span', { class: 'os-compteur-remplie' }));
    ilot.badgeSurcharge = h('span', { class: 'os-badge os-badge--surcharge' }, icone('losange'), h('span', { class: 'os-badge-n' }));
    ilot.badgeCap = h('span', { class: 'os-badge os-badge--cap' }, icone('prise'), h('span', { class: 'os-badge-n' }));
    ilot.badgeInc = h('span', { class: 'os-badge os-badge--inc' }, icone('flamme'), h('span', { class: 'os-badge-n' }));
    ilot.resume = h('span', { class: 'os-resume', 'aria-hidden': 'true' });
    ilot.etiquette = h('button', { type: 'button', class: 'os-etiquette' },
      h('span', { class: 'os-etiquette-haut', 'aria-hidden': 'true' },
        h('span', { class: 'os-postit' }, h('span', { class: 'os-postit-nom' }, def.court)),
        ilot.triade),
      h('span', { class: 'os-etiquette-bas', 'aria-hidden': 'true' },
        h('span', { class: 'os-compteur' }, ilot.compteurVal, h('span', { class: 'os-compteur-u' }, 'jetons'), ilot.compteurBarre),
        h('span', { class: 'os-badges' }, ilot.badgeSurcharge, ilot.badgeCap, ilot.badgeInc)),
      ilot.resume);
    ilot.replier = h('button', { type: 'button', class: 'os-replier', 'aria-expanded': 'true', 'aria-controls': idBureaux,
      'aria-label': 'Replier ' + def.nom }, icone('chevron'));
    ilot.conteneurBureaux = h('div', { class: 'os-bureaux', id: idBureaux });
    var sieges = vue.siegesParIlot[def.id];
    sieges.forEach(function (devId, i) {
      var b = { index: i, occupant: null, cleBulle: null };
      b.scene = construireScene(def.id + ':' + i);
      b.nom = h('span', { class: 'os-nom', 'aria-hidden': 'true' });
      b.etatTexte = h('span', { class: 'os-etat-texte', 'aria-hidden': 'true' });
      b.bulle = h('span', { class: 'os-bulle', 'aria-hidden': 'true' });
      b.el = h('button', { type: 'button', class: 'os-bureau os-bureau--' + i },
        h('span', { class: 'os-scene-cadre' }, b.scene.svg),
        h('span', { class: 'os-plaque' }, b.nom, b.etatTexte),
        b.bulle);
      ilot.bureaux.push(b);
      ilot.conteneurBureaux.appendChild(b.el);
    });
    ilot.el = h('div', { class: 'os-ilot', 'data-ilot': def.id, style: { '--os-teinte': String(def.teinte) } },
      h('div', { class: 'os-ilot-tete' }, ilot.etiquette, ilot.replier),
      ilot.conteneurBureaux);
    return ilot;
  }

  function coinCafe() {
    var c = { fantomes: [] };
    var svg = s('svg', { class: 'os-cafe-scene', viewBox: '0 0 200 170', 'aria-hidden': 'true', focusable: 'false' },
      // comptoir et machine à café
      s('rect', { class: 'c-comptoir', x: 8, y: 6, width: 184, height: 30, rx: 3 }),
      s('rect', { class: 'c-machine', x: 18, y: 9, width: 26, height: 22, rx: 3 }),
      s('circle', { class: 'c-machine-voyant', cx: 38, cy: 14, r: 2 }),
      s('rect', { class: 'c-tasse', x: 26, y: 21, width: 9, height: 7, rx: 2 }),
      s('rect', { class: 'c-boite-dons', x: 52, y: 12, width: 18, height: 16, rx: 2 }),
      s('rect', { class: 'c-micro', x: 120, y: 9, width: 34, height: 22, rx: 2.5 }),
      s('rect', { class: 'c-micro-vitre', x: 124, y: 13, width: 20, height: 14, rx: 1.5 }),
      s('rect', { class: 'c-frigo', x: 164, y: 4, width: 26, height: 34, rx: 3 }),
      s('rect', { class: 'c-postit', x: 168, y: 10, width: 8, height: 8, transform: 'rotate(-6 172 14)' }),
      s('rect', { class: 'c-postit c-postit--2', x: 178, y: 20, width: 8, height: 8, transform: 'rotate(5 182 24)' }),
      // fontaine d'eau
      s('circle', { class: 'c-fontaine', cx: 22, cy: 62, r: 11 }),
      s('circle', { class: 'c-fontaine-eau', cx: 22, cy: 62, r: 7 }),
      // grande plante
      s('g', { class: 'c-plante' },
        s('circle', { cx: 178, cy: 146, r: 13, class: 'c-pot' }),
        [0, 1, 2, 3, 4, 5, 6, 7].map(function (i) {
          return s('ellipse', { cx: 178, cy: 132, rx: 5, ry: 14, class: i % 2 ? 'c-feuille' : 'c-feuille c-feuille--2', transform: 'rotate(' + (i * 45) + ' 178 146)' });
        })),
      // table ronde et chaises
      s('circle', { class: 'c-chaise', cx: 100, cy: 72, r: 10 }),
      s('circle', { class: 'c-chaise', cx: 62, cy: 108, r: 10 }),
      s('circle', { class: 'c-chaise', cx: 138, cy: 108, r: 10 }),
      s('circle', { class: 'c-chaise', cx: 100, cy: 144, r: 10 }),
      s('circle', { class: 'c-table', cx: 100, cy: 108, r: 30 }));
    // Applications hors TI (Kevin) : de petits portables qui s'accumulent sur la table du coin café
    var POS = [[86, 96], [104, 92], [94, 112], [112, 110], [80, 116], [100, 126], [118, 124], [88, 130]];
    var g = s('g', { class: 'c-fantomes' });
    POS.forEach(function (p, i) {
      var f = s('g', { class: 'c-fantome', transform: 'translate(' + p[0] + ' ' + p[1] + ') rotate(' + ((i * 37) % 30 - 15) + ' 7 5)' },
        s('rect', { class: 'c-fantome-base', x: 0, y: 4, width: 14, height: 9, rx: 1 }),
        s('rect', { class: 'c-fantome-ecran', x: 1, y: 0, width: 12, height: 5, rx: 0.8 }),
        s('rect', { class: 'c-fantome-autocollant', x: 9, y: 7, width: 3, height: 3, rx: 1.5 }));
      c.fantomes.push(f);
      g.appendChild(f);
    });
    svg.appendChild(g);
    c.legende = h('span', { class: 'os-cafe-legende' });
    c.el = h('div', { class: 'os-cafe', role: 'img' }, svg, h('span', { class: 'os-cafe-titre', 'aria-hidden': 'true' }, 'Coin café'), c.legende);
    return c;
  }

  function legende() {
    function ligne(cls, txt) { return h('li', null, h('span', { class: 'os-leg-ico os-leg-ico--' + cls, 'aria-hidden': 'true' }), txt); }
    return h('details', { class: 'os-legende' },
      h('summary', null, 'Lire l\'open space'),
      h('ul', null,
        ligne('ecrans', 'Écrans allumés : sessions d\'agent en parallèle'),
        ligne('pile', 'Pile de papiers : PR qui attendent la revue de cette personne'),
        ligne('brouillard', 'Brouillard : dette cognitive (uniforme tant qu\'on ne la mesure pas)'),
        ligne('flamme', 'Flamme : incident en cours'),
        ligne('sueur', 'Goutte de sueur, puis tête sur le bureau : fatigue'),
        ligne('valise', 'Valise : congé · boîte : départ · livre : formation'),
        ligne('compteur', 'Compteur : jetons consommés par l\'îlot cette semaine'),
        ligne('triade', 'Trois voyants : triade létale (données privées, contenu non fiable, sortie externe)'),
        ligne('losange', 'Losange : réglages propres à l\'îlot · prise : skills et MCP équipés')));
  }

  // --- Données dérivées ---
  function siegesParIlot() {
    var m = {};
    (LTP.donnees.ilots || []).forEach(function (il) { m[il.id] = []; });
    (LTP.donnees.devs || []).forEach(function (d) { if (m[d.ilot]) m[d.ilot].push(d.id); });
    return m;
  }

  // Occupants courants : un remplaçant (développeur absent des données) prend le siège d'une personne partie.
  function occupants(etat, ilotId, base) {
    var extras = Object.keys(etat.devs).filter(function (id) {
      var d = etat.devs[id];
      return d.ilot === ilotId && d.present && base.indexOf(id) < 0;
    });
    return base.map(function (id) {
      var d = etat.devs[id];
      if ((!d || !d.present || d.ilot !== ilotId) && extras.length) return extras.shift();
      return id;
    });
  }

  // Répartit la file de revue de l'îlot sur ses réviseurs (revues faites cette semaine, ou trait reviseur).
  function repartirPiles(etat, ids, file) {
    var res = {}, poids = [], total = 0;
    ids.forEach(function (id) {
      res[id] = 0;
      var d = etat.devs[id];
      if (!d || !d.present || d.etat === 'conge' || d.etat === 'absent') return;
      var p = (d.derniere && d.derniere.revues > 0 ? d.derniere.revues : 0) + (LTP.q.aTrait(d, 'reviseur') ? 2 : 0);
      if (p > 0) { poids.push([id, p]); total += p; }
    });
    file = Math.max(0, Math.round(file || 0));
    if (!file) return res;
    if (!total) {
      // Personne n'a relu : la pile atterrit chez la personne la plus expérimentée encore au bureau.
      var rang = { senior: 3, inter: 2, junior: 1, stagiaire: 0 }, meilleur = null;
      ids.forEach(function (id) {
        var d = etat.devs[id];
        if (!d || !d.present || d.etat === 'conge' || d.etat === 'absent') return;
        if (!meilleur || (rang[d.niveau] || 0) > (rang[etat.devs[meilleur].niveau] || 0)) meilleur = id;
      });
      if (meilleur) res[meilleur] = file;
      return res;
    }
    var reste = file, fractions = [];
    poids.forEach(function (p) {
      var exact = file * p[1] / total, ent = Math.floor(exact);
      res[p[0]] = ent; reste -= ent; fractions.push([p[0], exact - ent]);
    });
    fractions.sort(function (a, b) { return b[1] - a[1]; });
    for (var i = 0; i < reste; i++) res[fractions[i % fractions.length][0]] += 1;
    return res;
  }

  function cleBulle(d) {
    if (!d || !d.present) return null;
    if (d.fatigue >= K.seuilBulleFatigue && ETATS_ACTIFS[d.etat] && d.etat !== 'incident') return 'fatigue';
    return d.etat;
  }
  function texteBulle(etat, id) {
    var d = etat.devs[id];
    var P = LTP.systemes.personnes;
    if (P && typeof P.bulle === 'function') {
      try { var t = P.bulle(etat, id); if (t) return LTP.ui.texteContenu(t, etat, { dev: id, ilot: d && d.ilot }); } catch (e) { /* repli */ }
    }
    var cle = cleBulle(d);
    if (!cle) return '';
    var def = LTP.q.defDev(id);
    var liste = (def && def.bulles && def.bulles[cle]) || (LTP.donnees.bulles && LTP.donnees.bulles[cle]);
    if ((!liste || !liste.length) && cle === 'fatigue') liste = (def && def.bulles && def.bulles[d.etat]) || (LTP.donnees.bulles || {})[d.etat];
    if (!liste) return '';
    if (!Array.isArray(liste)) liste = [liste];
    if (!liste.length) return '';
    var val = liste[Math.floor(O.hachage(id + ':' + etat.semaine) * liste.length)];
    return LTP.ui.texteContenu(val, etat, { dev: id, ilot: d.ilot });
  }

  // Une ou deux bulles visibles : celles des personnes dont l'état a changé cette semaine, sinon une rotation.
  function choisirBulles(etat, precedent, ids) {
    var prio = { incident: 5, conge: 4, absent: 0, formation: 3, attente: 2, revue: 1, code: 1 };
    var candidats = [];
    var semaineChangee = precedent && precedent.semaine !== etat.semaine;
    ids.forEach(function (id) {
      var d = etat.devs[id];
      if (!d || !d.present) return;
      var avant = precedent && precedent.devs && precedent.devs[id];
      var change = semaineChangee ? (avant && avant.etat !== d.etat) : (!precedent && d.etat === 'incident');
      if (change) candidats.push([id, (prio[d.etat] || 1) + O.hachage(id + etat.semaine)]);
    });
    candidats.sort(function (a, b) { return b[1] - a[1]; });
    var choisis = [], ilotsPris = {};
    candidats.forEach(function (c) {
      var il = etat.devs[c[0]].ilot;
      if (choisis.length < K.bullesMax && !ilotsPris[il]) { choisis.push(c[0]); ilotsPris[il] = true; }
    });
    if (!choisis.length) {
      var presents = ids.filter(function (id) { var d = etat.devs[id]; return d && d.present && cleBulle(d); });
      if (presents.length) {
        choisis.push(presents[(etat.semaine * 5) % presents.length]);
        var autre = presents[(etat.semaine * 5 + 11) % presents.length];
        if (etat.semaine % 2 === 0 && autre && etat.devs[autre].ilot !== etat.devs[choisis[0]].ilot) choisis.push(autre);
      }
    }
    var m = {};
    choisis.forEach(function (id) { m[id] = true; });
    return m;
  }

  function niveauFatigue(f) {
    if (f >= K.seuilEpuise) return 'épuisement';
    if (f >= K.seuilSueur) return 'fatigue élevée';
    if (f >= 0.35) return 'fatigue modérée';
    return 'fatigue faible';
  }

  // --- Mise à jour d'un bureau ---
  function rendreBureau(b, devId, etat, ui, ctx) {
    var d = etat.devs[devId];
    if (b.occupant !== devId) {
      b.occupant = devId;
      attr(b.el, 'data-dev', devId);
      habiller(b.scene, devId);
      texte(b.nom, (d && d.nom) || LTP.ui.nomDev(devId));
      b.cleBulle = null;
    }
    var present = !!(d && d.present);
    var et = present ? d.etat : 'absent';
    var actif = present && !!ETATS_ACTIFS[et];
    var conge = present && et === 'conge';
    var absent = !present || et === 'absent';
    var formation = present && et === 'formation';
    var incident = present && et === 'incident';
    var fatigue = present ? d.fatigue : 0;
    var epuise = actif && fatigue >= K.seuilEpuise;
    var sueur = actif && !epuise && fatigue >= K.seuilSueur;
    var sessions = present ? Math.max(1, Math.min(6, Math.round(d.sessions || 1))) : 1;
    var pile = ctx.piles[devId] || 0;

    var el = b.el;
    classe(el, 'os-b--actif', actif);
    classe(el, 'os-b--conge', conge);
    classe(el, 'os-b--absent', absent);
    classe(el, 'os-b--vide', conge || absent);
    classe(el, 'os-b--formation', formation);
    classe(el, 'os-b--incident', incident);
    classe(el, 'os-b--sueur', sueur);
    classe(el, 'os-b--epuise', epuise);
    classe(el, 'os-b--pile', pile > 0);
    classe(el, 'os-b--choisi', !!(ui.selection && ui.selection.type === 'dev' && ui.selection.id === devId));
    // Petit rebond quand l'état change d'une semaine à l'autre (deux classes alternées relancent l'animation).
    if (ctx.semaineChangee && ctx.precedent && ctx.precedent.devs && ctx.precedent.devs[devId] && ctx.precedent.devs[devId].etat !== et) {
      var pair = etat.semaine % 2 === 0;
      classe(el, 'os-b--change-a', pair);
      classe(el, 'os-b--change-b', !pair);
    } else if (ctx.semaineChangee) {
      classe(el, 'os-b--change-a', false);
      classe(el, 'os-b--change-b', false);
    }

    // Écrans
    var sc = b.scene;
    if (sc.nEcrans !== sessions) {
      sc.nEcrans = sessions;
      var pos = positionsEcrans(sessions);
      sc.ecrans.forEach(function (e, i) {
        var on = i < sessions;
        classe(e.g, 'on', on);
        classe(e.lueur, 'on', on);
        if (on) {
          var t = 'translate(' + pos[i][0] + 'px, ' + pos[i][1] + 'px)';
          style(e.g, 'transform', t);
          style(e.lueur, 'transform', t);
        }
      });
      style(sc.flaque, 'opacity', String(Math.min(1, 0.35 + sessions * 0.13)));
    }
    // Pile de PR
    if (sc.pile !== pile) {
      sc.pile = pile;
      sc.feuilles.forEach(function (f, i) { classe(f, 'on', i < Math.min(pile, K.feuillesMax)); });
      texte(sc.etiquettePile, pile > 99 ? '99+' : String(pile));
      attr(sc.badgePile, 'transform', 'translate(0 ' + (-Math.min(pile, K.feuillesMax) * 1.7 + 6) + ')');
    }
    // Brouillard
    var brouillard;
    if (!present) brouillard = 0;
    else if (ctx.comprehensionVisible) brouillard = O.borner((0.85 - (d.comprehension == null ? 0.8 : d.comprehension)) / 0.6, 0, 0.85);
    else brouillard = K.brouillardInconnu;
    style(sc.brouillard, 'opacity', brouillard.toFixed(2));
    classe(el, 'os-b--flou-inconnu', present && !ctx.comprehensionVisible);

    // Textes de la plaque (téléphone) et nom accessible
    var etatLib = LIBELLES_ETAT[et] || et;
    var court = absent && !present ? 'parti' : conge ? 'congé' : formation ? 'formation' : incident ? 'incident' : et === 'revue' ? 'revue' : et === 'attente' ? 'attend' : 'code';
    var plaque = actif ? court + ' · ' + pluriel(sessions, 'écran', 'écrans') : court;
    if (pile > 0) plaque += ' · ' + pile + ' PR';
    texte(b.etatTexte, plaque);
    var nom = (d && d.nom) || LTP.ui.nomDev(devId);
    var parts = [nom];
    if (!present) parts.push('a quitté son poste');
    else {
      parts.push(etatLib);
      if (actif) parts.push(pluriel(sessions, 'session d\'agent', 'sessions d\'agent'));
      if (pile > 0) parts.push(pluriel(pile, 'PR à relire', 'PR à relire'));
      if (!conge) parts.push(niveauFatigue(fatigue));
      if (ctx.comprehensionVisible && d.comprehension != null) parts.push('compréhension du code ' + F.pourcent(d.comprehension));
    }
    attr(el, 'aria-label', parts.join(', ') + '. Ouvrir la fiche.');

    // Bulle de pensée
    var cle = etat.semaine + '|' + cleBulle(d);
    if (b.cleBulle !== cle) {
      b.cleBulle = cle;
      texte(b.bulle, present ? texteBulle(etat, devId) : '');
    }
    // La bulle apparaît, reste quelques secondes puis s'efface ; deux classes alternées relancent l'animation chaque semaine.
    var vue = !!ctx.bulles[devId] && !!b.bulle.__osT, pair = etat.semaine % 2 === 0;
    classe(b.bulle, 'os-bulle--vue', vue);
    classe(b.bulle, 'os-bulle--a', vue && pair);
    classe(b.bulle, 'os-bulle--b', vue && !pair);
    classe(b.bulle, 'os-bulle--vide', !b.bulle.__osT);
  }

  // --- Mise à jour d'un îlot ---
  function rendreIlot(il, etat, ui, ctx) {
    var e = etat.ilots[il.id];
    if (!e) return;
    var occ = occupants(etat, il.id, ctx.vue.siegesParIlot[il.id]);
    ctx.piles = repartirPiles(etat, occ, e.fileRevue);
    il.bureaux.forEach(function (b, i) { rendreBureau(b, occ[i], etat, ui, ctx); });

    var jetons = (e.derniere && e.derniere.jetons) || 0;
    texte(il.compteurVal, F.jetons(jetons));
    style(il.compteurBarre, '--os-part', ctx.jetonsMax > 0 ? (jetons / ctx.jetonsMax).toFixed(3) : '0');
    var t = e.triade || {};
    classe(il.voyants.prive, 'allume', !!t.prive);
    classe(il.voyants.nonFiable, 'allume', !!t.nonFiable);
    classe(il.voyants.sortie, 'allume', !!t.sortie);
    var complete = !!(t.prive && t.nonFiable && t.sortie);
    classe(il.el, 'os-ilot--triade', complete);
    var sur = Object.keys((etat.leviers.surcharges || {})[il.id] || {}).length;
    var cap = (e.capacites || []).length;
    var inc = LTP.q.incidentsActifs(etat, il.id).length;
    texte(il.badgeSurcharge.lastChild, sur); classe(il.badgeSurcharge, 'on', sur > 0);
    texte(il.badgeCap.lastChild, cap); classe(il.badgeCap, 'on', cap > 0);
    texte(il.badgeInc.lastChild, inc); classe(il.badgeInc, 'on', inc > 0);
    attr(il.badgeSurcharge, 'title', sur ? pluriel(sur, 'réglage propre à l\'îlot', 'réglages propres à l\'îlot') : null);
    attr(il.badgeCap, 'title', cap ? pluriel(cap, 'capacité équipée', 'capacités équipées') : null);
    attr(il.badgeInc, 'title', inc ? pluriel(inc, 'incident en cours', 'incidents en cours') : null);
    var file = Math.round(e.fileRevue || 0);
    var sessions = O.somme(occ, function (id) { var d = etat.devs[id]; return d && d.present && ETATS_ACTIFS[d.etat] ? Math.round(d.sessions || 1) : 0; });
    texte(il.resume, pluriel(sessions, 'session', 'sessions') + (file ? ' · ' + file + ' PR à relire' : ''));
    var choisi = (ui.selection && ui.selection.type === 'ilot' && ui.selection.id === il.id) || ui.ilotEdition === il.id;
    classe(il.el, 'os-ilot--choisi', !!choisi);

    var triadeTexte = 'données privées ' + (t.prive ? 'oui' : 'non') + ', contenu non fiable ' + (t.nonFiable ? 'oui' : 'non') +
      ', sortie externe ' + (t.sortie ? 'oui' : 'non');
    var lib = il.def.nom + ' : ' + F.jetons(jetons) + ' jetons cette semaine' +
      (file ? ', ' + file + ' PR à relire' : '') +
      (complete ? '. Triade létale complète' : '. Triade létale') + ' (' + triadeTexte + ')' +
      (cap ? ', ' + pluriel(cap, 'capacité équipée', 'capacités équipées') : '') +
      (sur ? ', ' + pluriel(sur, 'réglage propre', 'réglages propres') : '') +
      (inc ? ', ' + pluriel(inc, 'incident en cours', 'incidents en cours') : '') +
      '. Ouvrir les réglages de l\'îlot.';
    // D'où viennent les branches allumées, et quand on pourra agir (infobulle et nom accessible).
    var F_ = LTP.ui.F, origines = null;
    if (F_ && F_.originesTriade && (t.prive || t.nonFiable || t.sortie)) { try { origines = F_.originesTriade(etat, il.id); } catch (err) { origines = null; } }
    if (origines) lib = lib.replace(/\. Ouvrir les réglages de l'îlot\.$/, '. ' + origines.lignes.join(' ') + (origines.quand ? ' ' + origines.quand : '') + ' Ouvrir les réglages de l\'îlot.');
    attr(il.etiquette, 'aria-label', lib);
    attr(il.triade, 'title', origines ? origines.resume : null);
    attr(il.etiquette, 'title', origines ? origines.resume : null);
    attr(il.voyants.prive, 'title', 'Données privées : ' + (t.prive ? 'oui' : 'non'));
    attr(il.voyants.nonFiable, 'title', 'Contenu non fiable : ' + (t.nonFiable ? 'oui' : 'non'));
    attr(il.voyants.sortie, 'title', 'Sortie externe : ' + (t.sortie ? 'oui' : 'non'));
  }

  LTP.ui.regions.openspace = {
    monter: function (racine, app) {
      var vue = { app: app, ilots: [], replies: {}, siegesParIlot: siegesParIlot(), bulles: {}, bullesSemaine: null };
      this._vue = vue;
      vue.sommaire = h('p', { class: 'os-sommaire' });
      vue.toutReplier = h('button', { type: 'button', class: 'os-tout-replier' }, 'Tout replier');
      vue.plancher = h('div', { class: 'os-plancher' });
      (LTP.donnees.ilots || []).slice().sort(function (a, b) { return a.ordre - b.ordre; }).forEach(function (def) {
        var il = construireIlot(def, vue);
        vue.ilots.push(il);
        vue.plancher.appendChild(il.el);
      });
      vue.cafe = coinCafe();
      vue.plancher.appendChild(vue.cafe.el);
      vue.racine = h('div', { class: 'os' },
        definitions(),
        h('div', { class: 'os-entete' },
          h('h2', { class: 'os-titre' }, 'L\'open space'),
          vue.sommaire,
          vue.toutReplier,
          legende()),
        vue.plancher);
      vue.racine.addEventListener('click', function (ev) {
        var cible = ev.target;
        if (!cible || !cible.closest) return;
        var rep = cible.closest('.os-replier');
        if (rep) {
          var idIlot = rep.closest('[data-ilot]').getAttribute('data-ilot');
          basculer(vue, idIlot, !vue.replies[idIlot]);
          return;
        }
        if (cible.closest('.os-tout-replier')) {
          var tous = vue.ilots.every(function (il) { return vue.replies[il.id]; });
          vue.ilots.forEach(function (il) { basculer(vue, il.id, !tous); });
          return;
        }
        var et = cible.closest('.os-etiquette');
        if (et) {
          var id = et.closest('[data-ilot]').getAttribute('data-ilot');
          app.setUI({ selection: { type: 'ilot', id: id }, ilotEdition: id });
          return;
        }
        var bureau = cible.closest('[data-dev]');
        if (bureau) {
          var dev = bureau.getAttribute('data-dev');
          app.setUI({ selection: { type: 'dev', id: dev }, modale: { type: 'fiche', id: dev } });
        }
      });
      LTP.ui.remplacer(racine, vue.racine);
    },

    rendre: function (etat, ui, precedent) {
      var vue = this._vue;
      if (!vue || !etat) return;
      var ids = [];
      vue.ilots.forEach(function (il) { ids = ids.concat(occupants(etat, il.id, vue.siegesParIlot[il.id])); });
      var semaineChangee = !!(precedent && precedent.semaine !== etat.semaine);
      if (vue.bullesSemaine !== etat.semaine || vue.bullesGraine !== etat.graine) {
        vue.bullesSemaine = etat.semaine;
        vue.bullesGraine = etat.graine;
        vue.bulles = choisirBulles(etat, precedent, ids);
      }
      var jetonsMax = 0;
      LTP.q.ilots(etat).forEach(function (il) { jetonsMax = Math.max(jetonsMax, (il.derniere && il.derniere.jetons) || 0); });
      var ctx = {
        vue: vue, precedent: precedent, semaineChangee: semaineChangee, bulles: vue.bulles, jetonsMax: jetonsMax,
        comprehensionVisible: LTP.q.metriqueVisible(etat, 'comprehension'),
      };
      vue.ilots.forEach(function (il) { rendreIlot(il, etat, ui, ctx); });

      // Sommaire de l'étage
      var presents = 0, sessions = 0, file = 0;
      ids.forEach(function (id) {
        var d = etat.devs[id];
        if (!d || !d.present) return;
        if (d.etat !== 'conge' && d.etat !== 'absent') presents++;
        if (ETATS_ACTIFS[d.etat]) sessions += Math.round(d.sessions || 1);
      });
      LTP.q.ilots(etat).forEach(function (il) { file += Math.round(il.fileRevue || 0); });
      var inc = LTP.q.incidentsActifs(etat).length;
      texte(vue.sommaire, presents + ' au bureau · ' + pluriel(sessions, 'session d\'agent', 'sessions d\'agent') + ' · ' +
        pluriel(file, 'PR en attente', 'PR en attente') + (inc ? ' · ' + pluriel(inc, 'incident', 'incidents') : ''));

      // Coin café : applications hors TI
      var fantomes = Math.round(O.somme(LTP.q.ilots(etat), function (il) { return il.applisFantomes || 0; }));
      vue.cafe.fantomes.forEach(function (f, i) { classe(f, 'on', i < fantomes); });
      attr(vue.cafe.el, 'aria-label', 'Coin café, hors de la zone TI' + (fantomes ? ' : ' + pluriel(fantomes, 'application faite hors TI', 'applications faites hors TI') + ' sur la table' : ''));
      texte(vue.cafe.legende, fantomes ? pluriel(fantomes, 'appli hors TI', 'applis hors TI') : '');
    },
  };

  function basculer(vue, idIlot, replie) {
    vue.replies[idIlot] = replie;
    var il = vue.ilots.filter(function (x) { return x.id === idIlot; })[0];
    if (!il) return;
    classe(il.el, 'os-ilot--replie', replie);
    attr(il.replier, 'aria-expanded', replie ? 'false' : 'true');
    attr(il.replier, 'aria-label', (replie ? 'Déplier ' : 'Replier ') + il.def.nom);
    var tous = vue.ilots.every(function (x) { return vue.replies[x.id]; });
    texte(vue.toutReplier, tous ? 'Tout déplier' : 'Tout replier');
  }
})(globalThis.LTP = globalThis.LTP || {});
