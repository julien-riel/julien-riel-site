/* 42-ui-open-space.js */
/* L'open space vu de haut (CONCEPTION §3.3 et §8). Propriétaire : agent E.
   Le DOM est construit une fois au montage : 7 îlots (div[data-ilot]) de 3 bureaux (button[data-dev]),
   chacun avec une petite scène SVG vue par-dessus l'épaule. Chaque rendu ne fait que des différences :
   classes, attributs, transformations et textes. Les écouteurs sont délégués à la racine de la région.

   Signaux par bureau (classes sur le bouton) :
     écrans allumés = dev.sessions · pile de papiers = PR à relire (ilot.fileRevue réparti sur les réviseurs)
     brouillard = 1 − compréhension (uniforme si la métrique n'est pas visible) · flamme = incident
     goutte, puis tête sur le bureau = fatigue · valise = congé · boîte = départ ou absence · livre = formation
     pastille ✦ près du nom = licence Copilot ; sans licence, l'écran n'a pas de lueur d'agent (on code comme avant)
   Bulles : les réflexions du jour (etat.reflexions.liste, 33-reflexions) apparaissent en décalé, étalées sur la
   journée, et restent lisibles environ 8 s quelle que soit la vitesse (au plus 4 à la fois ; les plus importantes
   passent devant). En pause ou en avance manuelle, toutes les réflexions du jour restent affichées.
   Au téléphone (< 700 px), la même structure devient une liste d'îlots repliables (voir 30-open-space.css). */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, s = LTP.ui.svg, O = LTP.outils, F = LTP.format;

  LTP.K.openspace = {
    seuilSueur: 0.55,       // fatigue à partir de laquelle la goutte de sueur apparaît
    seuilEpuise: 0.8,       // fatigue à partir de laquelle la tête tombe sur le bureau
    feuillesMax: 7,         // feuilles dessinées dans une pile (le nombre exact est écrit sur le post-it)
    bullesMax: 4,           // bulles visibles à la fois pendant la lecture
    bulleDureeMs: 8000,     // une bulle reste lisible environ 8 s, quelle que soit la vitesse
    bulleMinMs: 2600,       // une bulle plus importante peut en remplacer une autre affichée depuis au moins ce temps
    bulleAttenteMs: 6000,   // une réflexion qui n'a pas trouvé de place à temps est abandonnée (elle reste dans « Ce qu'on entend »)
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
      var b = { index: i, occupant: null };
      b.scene = construireScene(def.id + ':' + i);
      b.nom = h('span', { class: 'os-nom', 'aria-hidden': 'true' });
      // Repère générique de licence Copilot (pas de logo de marque) : une petite pastille étoilée.
      b.licence = h('span', { class: 'os-licence', 'aria-hidden': 'true' },
        s('svg', { viewBox: '0 0 12 12', focusable: 'false' }, s('path', { d: 'M6 0.6 C6.4 4 8 5.6 11.4 6 C8 6.4 6.4 8 6 11.4 C5.6 8 4 6.4 0.6 6 C4 5.6 5.6 4 6 0.6 Z' })));
      b.etatTexte = h('span', { class: 'os-etat-texte', 'aria-hidden': 'true' });
      b.bulle = h('span', { class: 'os-bulle', 'aria-hidden': 'true' });
      b.el = h('button', { type: 'button', class: 'os-bureau os-bureau--' + i },
        h('span', { class: 'os-scene-cadre' }, b.scene.svg),
        h('span', { class: 'os-plaque' }, h('span', { class: 'os-nom-ligne' }, b.licence, b.nom), b.etatTexte),
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
        ligne('ecrans', 'Écrans allumés : sessions d\'agent en parallèle'),
        ligne('licence', 'Pastille étoilée : licence Copilot. Sans licence, pas de lueur d\'agent : on code comme avant'),
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

  // =====================================================================
  // Bulles de réflexion
  // =====================================================================
  // Une entrée par réflexion à montrer : { cle, r, rang, eligible, debut, fin, statut: 'attente' | 'vue' | 'finie' }
  function cleReflexion(r) { return r.dev + '|' + r.semaine + '|' + r.jour + '|' + (r.immediat ? '!' : '') + r.texte; }
  function lectureEnCours(etat, ui) { return !!(ui.vitesse && etat.phase === 'execution' && !ui.modaleBloquante && !etat.evenementEnAttente); }
  function maintenant() { return Date.now(); }

  // Met à jour la liste des bulles d'après les réflexions du jour (sans toucher au DOM).
  function synchroniserBulles(vue, etat, ui) {
    var R = etat.reflexions || {};
    var liste = (R.liste || []).filter(function (r) { return r && r.dev && r.texte; });
    var jour = etat.graine + ':' + LTP.q.semaineCourante(etat) + ':' + (etat.jour || 0);
    var lecture = lectureEnCours(etat, ui);
    var now = maintenant();
    var b = vue.bulles;
    var connues = {};
    b.items.forEach(function (it) { connues[it.cle] = it; });
    var duJour = {};
    var nouvelles = [];
    liste.forEach(function (r) {
      var cle = cleReflexion(r);
      duJour[cle] = true;
      if (!connues[cle]) nouvelles.push(r);
    });
    var jourChange = b.jour !== jour;
    b.jour = jour;
    // Une réflexion du jour remplacée par une réaction (33-reflexions) disparaît si elle attendait encore.
    b.items = b.items.filter(function (it) {
      if (it.jour === jour && !duJour[it.cle] && it.statut === 'attente') return false;
      return it.statut !== 'finie' || it.jour === jour;
    });
    // Les réflexions du jour s'étalent sur la durée de la journée ; les réactions immédiates passent tout de suite.
    var dureeJour = LTP.K.ui.dureeJourMs / (ui.vitesse || 1);
    var nonImm = nouvelles.filter(function (r) { return !r.immediat; });
    nouvelles.forEach(function (r) {
      var i = nonImm.indexOf(r);
      var decal = r.immediat || !jourChange ? 0 : Math.round(i * dureeJour / Math.max(1, nonImm.length));
      b.items.push({ cle: cleReflexion(r), r: r, jour: jour, rang: LTP.ui.F ? LTP.ui.F.rangReflexion(r) : (r.immediat ? 4 : 1),
        eligible: now + decal, debut: null, fin: null, statut: 'attente', ordre: b.compteur++ });
    });
    // Passage de la lecture à la pause (ou l'inverse).
    var mode = lecture ? 'lecture' : 'tout';
    if (mode !== b.mode) {
      if (mode === 'lecture' && b.mode === 'tout') {
        // On reprend : on garde au plus bullesMax bulles, les plus importantes, encore quelques secondes.
        var visibles = b.items.filter(function (it) { return it.jour === jour; }).sort(function (x, y) { return y.rang - x.rang || x.ordre - y.ordre; });
        visibles.forEach(function (it, k) {
          if (k < K.bullesMax) { it.statut = 'vue'; it.debut = now - K.bulleMinMs; it.fin = now + 2500 + k * 600; }
          else it.statut = 'finie';
        });
      }
      b.mode = mode;
    }
    if (mode === 'tout') {
      // En pause : toutes les réflexions du jour, et rien d'autre.
      b.items.forEach(function (it) {
        if (it.jour === jour && duJour[it.cle]) { if (it.statut !== 'vue') { it.statut = 'vue'; it.debut = now; } it.fin = null; }
        else if (it.statut !== 'finie') it.statut = 'finie';
      });
    }
  }

  // Une étape de l'ordonnanceur pendant la lecture : expire, abandonne, montre. Renvoie le prochain instant utile.
  function ordonnancer(vue, now) {
    var b = vue.bulles;
    if (b.mode !== 'lecture') return null;
    b.items.forEach(function (it) {
      if (it.statut === 'vue' && it.fin != null && it.fin <= now) it.statut = 'finie';
      else if (it.statut === 'attente' && it.eligible + K.bulleAttenteMs <= now) it.statut = 'finie';
    });
    var prets = b.items.filter(function (it) { return it.statut === 'attente' && it.eligible <= now; })
      .sort(function (x, y) { return y.rang - x.rang || x.eligible - y.eligible || x.ordre - y.ordre; });
    prets.forEach(function (it) {
      var vues = b.items.filter(function (x) { return x.statut === 'vue'; });
      // Une seule bulle par bureau : la nouvelle remplace l'ancienne de la même personne.
      var memeDev = vues.filter(function (x) { return x.r.dev === it.r.dev; })[0];
      if (memeDev) { memeDev.statut = 'finie'; vues = vues.filter(function (x) { return x !== memeDev; }); }
      if (vues.length >= K.bullesMax) {
        // Plus de place : on remplace la moins importante, si elle l'est moins et qu'elle a eu le temps d'être lue
        // (une réaction immédiate n'attend pas).
        var faible = vues.filter(function (x) { return x.rang < it.rang && (it.rang >= 4 || now - x.debut >= K.bulleMinMs); })
          .sort(function (x, y) { return x.rang - y.rang || x.debut - y.debut; })[0];
        if (!faible) return;
        faible.statut = 'finie';
      }
      it.statut = 'vue'; it.debut = now; it.fin = now + K.bulleDureeMs;
    });
    var prochain = null;
    b.items.forEach(function (it) {
      var tps = it.statut === 'vue' ? it.fin : it.statut === 'attente' ? (it.eligible > now ? it.eligible : now + 400) : null;
      if (tps != null && (prochain == null || tps < prochain)) prochain = tps;
    });
    return prochain;
  }

  // Réflexion affichée par chaque bureau, et texte de survol (la dernière réflexion déjà entendue).
  function appliquerBulles(vue) {
    var etat = vue.etatCourant;
    if (!etat) return;
    var b = vue.bulles;
    var affichees = {};
    b.items.forEach(function (it) { if (it.statut === 'vue') affichees[it.r.dev] = it; });
    var R = etat.reflexions || {};
    var aVenir = {};
    b.items.forEach(function (it) { if (it.statut === 'attente') aVenir[it.cle] = true; });
    var fil = (R.fil || []).filter(function (r) { return r && !aVenir[cleReflexion(r)]; });
    vue.ilots.forEach(function (il) {
      var visibles = [];
      il.bureaux.forEach(function (bu) {
        var id = bu.occupant;
        var it = id ? affichees[id] : null;
        var r = it ? it.r : null;
        if (!r && id) for (var k = fil.length - 1; k >= 0; k--) if (fil[k].dev === id) { r = fil[k]; break; }
        var txt = r && etat.devs[id] && etat.devs[id].present ? LTP.ui.texteContenu(r.texte, etat, { dev: id, ilot: r.ilot }) : '';
        texte(bu.bulle, txt);
        var vu = !!it && !!txt;
        classe(bu.bulle, 'os-bulle--vue', vu);
        classe(bu.bulle, 'os-bulle--vide', !txt);
        classe(bu.bulle, 'os-bulle--plus', !!r && r.ton === 'plus');
        classe(bu.bulle, 'os-bulle--moins', !!r && r.ton === 'moins');
        classe(bu.bulle, 'os-bulle--immediat', !!r && !!r.immediat && vu);
        if (vu) visibles.push(bu);
      });
      // Deux bulles dans le même îlot : la deuxième monte au-dessus de la première pour ne pas la cacher.
      var decal = 0;
      visibles.forEach(function (bu, k) {
        var haut = k % 2 === 1;
        style(bu.bulle, '--os-bulle-decal', haut ? decal + 'px' : '0px');
        classe(bu.bulle, 'os-bulle--haut', haut);
        if (!haut) decal = (bu.bulle.offsetHeight || 0) + 8;
      });
      il.bureaux.forEach(function (bu) { if (visibles.indexOf(bu) < 0) { classe(bu.bulle, 'os-bulle--haut', false); style(bu.bulle, '--os-bulle-decal', '0px'); } });
    });
  }

  function cadencer(vue) {
    clearTimeout(vue.bulles.minuterie);
    var prochain = ordonnancer(vue, maintenant());
    appliquerBulles(vue);
    if (prochain != null) {
      vue.bulles.minuterie = setTimeout(function () { cadencer(vue); }, Math.max(60, prochain - maintenant() + 20));
    }
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
    // Sans licence Copilot : un seul écran, sans agent (dev.sessions vaut 0).
    var licence = present && LTP.q.aLicence(etat, d);
    var agents = licence ? Math.max(1, Math.min(6, Math.round(d.sessions || 1))) : 0;
    var sessions = Math.max(1, agents);
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
    classe(el, 'os-b--licence', licence);
    classe(el, 'os-b--sans-licence', present && !licence);
    attr(b.licence, 'title', licence ? (d.pilote ? 'Groupe pilote : licence Copilot' : 'Licence Copilot') : null);
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
    var plaque = actif ? court + ' · ' + (licence ? pluriel(agents, 'écran', 'écrans') : 'sans agent') : court;
    if (pile > 0) plaque += ' · ' + pile + ' PR';
    texte(b.etatTexte, plaque);
    var nom = (d && d.nom) || LTP.ui.nomDev(devId);
    var parts = [nom];
    if (!present) parts.push('a quitté son poste');
    else {
      parts.push(etatLib === LIBELLES_ETAT.code && !licence ? 'code' : etatLib);
      parts.push(licence ? 'avec licence Copilot' : 'sans licence Copilot');
      if (actif && licence) parts.push(pluriel(agents, 'session d\'agent', 'sessions d\'agent'));
      if (pile > 0) parts.push(pluriel(pile, 'PR à relire', 'PR à relire'));
      if (!conge) parts.push(niveauFatigue(fatigue));
      if (ctx.comprehensionVisible && d.comprehension != null) parts.push('compréhension du code ' + F.pourcent(d.comprehension));
    }
    attr(el, 'aria-label', parts.join(', ') + '. Ouvrir la fiche.');
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
    var sessions = O.somme(occ, function (id) { var d = etat.devs[id]; return d && d.present && ETATS_ACTIFS[d.etat] && LTP.q.aLicence(etat, d) ? Math.max(1, Math.round(d.sessions || 1)) : 0; });
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
      var vue = { app: app, ilots: [], replies: {}, siegesParIlot: siegesParIlot(),
        bulles: { items: [], jour: null, mode: null, compteur: 0, minuterie: null } };
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
      var jetonsMax = 0;
      LTP.q.ilots(etat).forEach(function (il) { jetonsMax = Math.max(jetonsMax, (il.derniere && il.derniere.jetons) || 0); });
      var ctx = {
        vue: vue, precedent: precedent, semaineChangee: semaineChangee, jetonsMax: jetonsMax,
        comprehensionVisible: LTP.q.metriqueVisible(etat, 'comprehension'),
      };
      vue.ilots.forEach(function (il) { rendreIlot(il, etat, ui, ctx); });

      // Bulles de réflexion : l'ordonnanceur garde sa propre minuterie entre deux rendus.
      vue.etatCourant = etat;
      synchroniserBulles(vue, etat, ui);
      cadencer(vue);

      // Sommaire de l'étage
      var presents = 0, sessions = 0, file = 0, licences = 0;
      ids.forEach(function (id) {
        var d = etat.devs[id];
        if (!d || !d.present) return;
        if (d.etat !== 'conge' && d.etat !== 'absent') presents++;
        var lic = LTP.q.aLicence(etat, d);
        if (lic) licences++;
        if (ETATS_ACTIFS[d.etat] && lic) sessions += Math.max(1, Math.round(d.sessions || 1));
      });
      LTP.q.ilots(etat).forEach(function (il) { file += Math.round(il.fileRevue || 0); });
      var inc = LTP.q.incidentsActifs(etat).length;
      texte(vue.sommaire, presents + ' au bureau · ' + pluriel(licences, 'licence Copilot', 'licences Copilot') + ' · ' + pluriel(sessions, 'session d\'agent', 'sessions d\'agent') + ' · ' +
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
