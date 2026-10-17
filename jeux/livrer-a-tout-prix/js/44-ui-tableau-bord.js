/* 44-ui-tableau-bord.js */
/* Tableau de bord (région « tableau ») : le rapport hebdomadaire, publié le vendredi (les métriques ne changent
   qu'à ce moment-là). Tuiles groupées par LTP.donnees.groupesMetriques, valeur selon l'unité,
   tendance sur 13 semaines (sparkline SVG tirée de etat.historique), couleur de la tendance selon le sens
   (une métrique « neutre » n'a jamais l'air d'une réussite), métriques cachées grisées avec l'indice de Ginette.
   Les métriques que la dernière note de service invite à surveiller sont mises en évidence pendant 3 semaines,
   et chaque courbe porte une petite marque aux semaines où un levier a changé (etat.leviers.changements).
   Sur téléphone : un groupe à la fois. Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, s = LTP.ui.svg, t = LTP.t, F = LTP.ui.F;
  var FENETRE = 13;
  var etat_ = { groupeMobile: null };
  var SURVEILLANCE_SEMAINES = 3;   // une métrique « à surveiller » le reste 3 semaines après sa note de service

  // Dernière note de service encore « fraîche » : { entree, ids: { metrique: true } } ou null.
  function surveillance(etat) {
    if (etat.options && etat.options.toutDebloque) return null;
    var sc = LTP.q.semaineCourante(etat);
    var e = (LTP.donnees.calendrier || []).filter(function (c) { return c.semaine <= sc && sc - c.semaine <= SURVEILLANCE_SEMAINES && (c.surveiller || []).length; })
      .sort(function (a, b) { return b.semaine - a.semaine; })[0];
    if (!e) return null;
    var ids = {};
    e.surveiller.forEach(function (id) { ids[id] = true; });
    return { entree: e, ids: ids };
  }

  // Changements de leviers regroupés par semaine : [{ semaine, textes: [] }].
  function changementsParSemaine(etat) {
    var parSem = {}, ordre = [];
    ((etat.leviers && etat.leviers.changements) || []).forEach(function (c) {
      if (!c || !c.semaine || !LTP.q.defLevier(c.id)) return;
      if (!parSem[c.semaine]) { parSem[c.semaine] = { semaine: c.semaine, textes: [] }; ordre.push(c.semaine); }
      var txt = t('tableau.changement', { semaine: c.semaine, levier: LTP.ui.libelleLevier(c.id), valeur: LTP.ui.libelleValeur(c.id, c.valeur),
        ilot: c.ilot ? ' (' + LTP.ui.nomIlot(c.ilot) + ')' : '' });
      if (parSem[c.semaine].textes.indexOf(txt) < 0) parSem[c.semaine].textes.push(txt);
    });
    return ordre.sort(function (a, b) { return a - b; }).map(function (s) { return parSem[s]; });
  }

  // Les n dernières valeurs numériques de l'historique, jusqu'à la semaine courante : [{ semaine, v }].
  // La semaine 0 (état initial, souvent à zéro pour les flux hebdomadaires) est exclue.
  function serie(etat, id, n) {
    var hist = (etat.historique || {})[id];
    var res = [];
    if (!hist || !hist.length) return res;
    var fin = Math.min(hist.length - 1, etat.semaine);
    for (var i = fin; i >= 1 && res.length < n; i--) {
      var v = hist[i];
      if (typeof v === 'number' && isFinite(v)) res.unshift({ semaine: i, v: v });
    }
    return res;
  }
  LTP.ui.serieMetrique = serie;

  // Variation : moyenne des (au plus) 4 dernières semaines comparée à celle des 4 précédentes.
  // pts vient de serie() : semaine 0 et trous déjà écartés. → { sens: 'mieux' | 'pire' | 'stable' | 'neutre', texte, fleche, n }
  var FENETRE_TENDANCE = 4;
  function moyenne(l) { var t = 0; for (var i = 0; i < l.length; i++) t += l[i].v; return t / l.length; }
  function variation(def, pts) {
    // Au moins deux semaines de chaque côté, sinon la « tendance » n'est que du bruit.
    if (pts.length < 4) return null;
    var k = Math.min(FENETRE_TENDANCE, Math.floor(pts.length / 2));
    var recents = pts.slice(pts.length - k), avant = pts.slice(pts.length - 2 * k, pts.length - k);
    var a = moyenne(avant), b = moyenne(recents), d = b - a;
    var f = LTP.format, texte, relatif;
    var stable = { sens: 'stable', texte: t('tableau.stable'), fleche: '→', n: k };
    if (def.unite === 'ratio') {
      relatif = Math.abs(d);
      texte = (d >= 0 ? '+' : '−') + t('unite.pts', { n: f.nombre(Math.abs(d) * 100, 0) });
      if (relatif < 0.01) return stable;
    } else {
      var base = Math.max(Math.abs(a), 1e-9);
      relatif = Math.abs(d) / base;
      if (def.unite === 'nb' && Math.abs(a) < 5) {
        // Petits comptes (incidents, applications…) : un pourcentage exagérerait ; on montre l'écart moyen par semaine.
        if (Math.abs(d) < 0.1) return stable;
        texte = (d >= 0 ? '+' : '−') + t('tableau.deltaParSemaine', { n: f.nombre(Math.abs(d), 1) });
      } else if (Math.abs(a) < 1e-9) {
        if (Math.abs(d) < 1e-9) return stable;
        texte = (d >= 0 ? '+' : '−') + F.formatMetrique(Math.abs(d), def.unite);
      } else {
        if (relatif < 0.03) return stable;
        texte = (d >= 0 ? '+' : '−') + f.pourcent(Math.min(relatif, 9.99), 0);
      }
    }
    var fleche = d > 0 ? '↗' : '↘';
    var sens;
    if (def.sens === 'neutre' || !def.sens) sens = 'neutre';
    else sens = (def.sens === 'haut') === (d > 0) ? 'mieux' : 'pire';
    return { sens: sens, texte: texte, fleche: fleche, n: k };
  }

  // Sparkline : trait de 1,5 px dans la teinte discrète, dernier point dans l'accent (sauf métrique neutre).
  // marques : [{ semaine }] semaines où un levier a changé ; la marque se place au début de cette semaine
  // (entre le point de la semaine précédente et celui de la semaine du changement).
  function sparkline(def, pts, surScrub, marques) {
    var L = 76, H = 24, P = 3;
    var el = s('svg', { viewBox: '0 0 ' + L + ' ' + H, width: L, height: H, class: 'spark', 'aria-hidden': 'true', focusable: 'false' });
    if (pts.length < 2) {
      el.appendChild(s('line', { x1: P, x2: L - P, y1: H / 2, y2: H / 2, class: 'spark-vide' }));
      return el;
    }
    var min = Infinity, max = -Infinity;
    pts.forEach(function (p) { if (p.v < min) min = p.v; if (p.v > max) max = p.v; });
    if (max - min < 1e-9) { max += 1; min -= 1; }
    var x = function (i) { return P + (L - 2 * P) * (i / (pts.length - 1)); };
    var y = function (v) { return H - P - (H - 2 * P) * ((v - min) / (max - min)); };
    (marques || []).forEach(function (m) {
      var f = (m.semaine - 0.5) - pts[0].semaine;
      if (f < -0.5 || f > pts.length - 0.5) return;
      var xm = Math.max(1, Math.min(L - 1, x(f)));
      el.appendChild(s('line', { x1: xm.toFixed(1), x2: xm.toFixed(1), y1: 1, y2: H - 1, class: 'spark-marque' }));
    });
    var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(p.v).toFixed(1); }).join(' ');
    el.appendChild(s('path', { d: d + ' L' + x(pts.length - 1).toFixed(1) + ' ' + (H - 1) + ' L' + P + ' ' + (H - 1) + 'Z', class: 'spark-aire' }));
    el.appendChild(s('path', { d: d, class: 'spark-ligne' }));
    var dernier = pts[pts.length - 1];
    el.appendChild(s('circle', { cx: x(pts.length - 1), cy: y(dernier.v), r: 2.6, class: 'spark-point' + (def.sens === 'neutre' ? ' is-neutre' : '') }));
    var curseur = s('circle', { cx: 0, cy: 0, r: 2.6, class: 'spark-curseur', style: { display: 'none' } });
    el.appendChild(curseur);
    if (surScrub) {
      el.addEventListener('pointermove', function (ev) {
        var r = el.getBoundingClientRect();
        var i = Math.round(((ev.clientX - r.left) / r.width * L - P) / (L - 2 * P) * (pts.length - 1));
        i = Math.max(0, Math.min(pts.length - 1, i));
        curseur.setAttribute('cx', x(i)); curseur.setAttribute('cy', y(pts[i].v)); curseur.style.display = '';
        surScrub(pts[i]);
      });
      el.addEventListener('pointerleave', function () { curseur.style.display = 'none'; surScrub(null); });
    }
    return el;
  }

  function tuile(def, etat, ctx) {
    var visible = LTP.q.metriqueVisible(etat, def.id);
    var cle = 'met-' + def.id;
    var surv = ctx.surveillance && ctx.surveillance.ids[def.id] ? ctx.surveillance.entree : null;
    var survTitre = surv ? t('tableau.surveillerAide', { titre: surv.titre || '', semaine: surv.semaine }) : null;
    var survEl = surv ? h('span', { class: 'met-surveiller', title: survTitre }, t('tableau.surveiller'), h('span', { class: 'sr-only' }, t('divers.deuxPoints') + survTitre)) : null;
    if (!visible) {
      var idInd = 'ind-' + def.id;
      var indice = def.indice || t('tableau.indiceDefaut');
      return h('li', { class: 'metrique is-cachee' + (surv ? ' is-surveiller' : ''), 'data-metrique': def.id, tabindex: '0', 'data-cle': cle, 'aria-describedby': idInd },
        h('span', { class: 'met-lib' }, def.libelle, survEl),
        h('span', { class: 'met-val met-val--cache' }, t('tableau.nonMesure')),
        h('span', { class: 'spark spark--hachure', 'aria-hidden': 'true' }),
        h('span', { class: 'met-indice', id: idInd, role: 'tooltip' }, F.visage('ginette', 22), h('span', null, h('b', null, F.personne('ginette').nom), ' ', indice)));
    }
    var v = (etat.metriques || {})[def.id];
    var pts = serie(etat, def.id, FENETRE);
    if (v == null && pts.length) v = pts[pts.length - 1].v;
    var varia = variation(def, pts);
    var valeur = F.formatMetrique(v, def.unite);
    var valEl = h('span', { class: 'met-val mono' }, valeur, def.unite === 'score' ? h('small', null, t('unite.score')) : null);
    var scrubEl = h('span', { class: 'met-scrub mono', 'aria-hidden': 'true' });
    var min = null, max = null;
    pts.forEach(function (p) { if (min == null || p.v < min) min = p.v; if (max == null || p.v > max) max = p.v; });
    var resumeTendance = pts.length >= 2 ? t('divers.etiquette', { libelle: t('tableau.tendance', { n: pts.length }),
      valeur: t('tableau.min', { v: F.formatMetrique(min, def.unite) }) + ', ' + t('tableau.max', { v: F.formatMetrique(max, def.unite) }) }) : '';
    // Changements de leviers pendant la fenêtre de la courbe.
    var debut = pts.length ? pts[0].semaine : Infinity;
    var marques = pts.length >= 2 ? ctx.changements.filter(function (c) { return c.semaine >= debut && c.semaine <= pts[pts.length - 1].semaine + 1; }) : [];
    var texteChang = marques.length ? t('tableau.changements', { liste: marques.map(function (c) { return c.textes.join(', '); }).join(t('divers.pointVirgule')) }) : '';
    var titre = [survTitre, resumeTendance, texteChang].filter(Boolean).join('. ');
    var deltaEl = varia ? h('span', { class: 'met-delta met-delta--' + varia.sens, title: (varia.sens === 'neutre' ? t('tableau.neutre') + ' · ' : '') + t('tableau.vs', { n: varia.n }) },
      h('span', { 'aria-hidden': 'true' }, varia.fleche), ' ', varia.texte,
      h('span', { class: 'sr-only' }, ' ' + t('tableau.vs', { n: varia.n }) + (varia.sens === 'mieux' ? ', ' + t('tableau.mieux') : varia.sens === 'pire' ? ', ' + t('tableau.pire') : varia.sens === 'neutre' ? ', ' + t('tableau.neutre') : ''))) : null;
    var li = h('li', { class: 'metrique' + (def.sens === 'neutre' ? ' is-neutre' : '') + (surv ? ' is-surveiller' : ''), 'data-metrique': def.id, title: titre || null },
      h('span', { class: 'met-lib' }, def.libelle, survEl),
      valEl,
      h('span', { class: 'met-spark' }, sparkline(def, pts, function (p) {
        if (!p) { scrubEl.textContent = ''; li.classList.remove('is-scrub'); return; }
        scrubEl.textContent = t('tableau.semaineAbbr', { n: p.semaine }) + ' · ' + F.formatMetrique(p.v, def.unite);
        li.classList.add('is-scrub');
      }, marques), scrubEl),
      deltaEl,
      resumeTendance ? h('span', { class: 'sr-only' }, resumeTendance) : null,
      texteChang ? h('span', { class: 'sr-only' }, '. ' + texteChang) : null);
    return li;
  }

  LTP.ui.regions.tableau = {
    monter: function (racine, app) {
      this._racine = racine;
      this._entete = h('div', { class: 'tableau-entete' });
      this._rapport = h('p', { class: 'tableau-rapport' });
      this._corps = h('div', { class: 'tableau-corps' });
      LTP.ui.remplacer(racine, h('div', { class: 'tableau' },
        h('div', { class: 'tableau-tete' },
          h('div', { class: 'tableau-titres' }, h('h2', { class: 'zone-titre' }, t('tableau.titre')), this._rapport),
          this._entete),
        this._corps));
    },
    rendre: function (etat, ui, precedent, app) {
      var groupes = LTP.donnees.groupesMetriques || [];
      var defs = LTP.donnees.metriques || [];
      if (!etat_.groupeMobile && groupes.length) etat_.groupeMobile = groupes[0].id;
      var gm = etat_.groupeMobile;

      F.reconstruire(this._entete, 'g:' + gm + ':' + groupes.length, function () {
        return F.choix({
          cle: 'groupe-mobile', libelle: t('tableau.groupes'), mode: 'segmente', courant: gm,
          valeurs: groupes.map(function (g) { return { v: g.id, libelle: g.libelle }; }),
          onChoisir: function (v) { etat_.groupeMobile = v; app.notifier(); },
        });
      });

      // Le rapport du vendredi : les chiffres ne bougent qu'une fois par semaine.
      var rapport = etat.semaine > 0 ? t('tableau.rapportSemaine', { semaine: etat.semaine }) : t('tableau.aucuneDonnee');
      if (this._rapport.textContent !== rapport) this._rapport.textContent = rapport;

      var surv = surveillance(etat);
      var chang = changementsParSemaine(etat);
      var ctx = { surveillance: surv, changements: chang };
      var visibles = defs.map(function (d) { return LTP.q.metriqueVisible(etat, d.id) ? 1 : 0; }).join('');
      var sig = JSON.stringify([etat.semaine, visibles, gm, etat.metriques, (etat.historique || {}).prFusionnees && etat.historique.prFusionnees.length,
        surv ? surv.entree.semaine : 0, chang.map(function (c) { return c.semaine + ':' + c.textes.length; }).join(',')]);
      F.reconstruire(this._corps, sig, function () {
        var connus = {};
        groupes.forEach(function (g) { connus[g.id] = true; });
        var listeGroupes = groupes.slice();
        if (defs.some(function (d) { return !connus[d.groupe]; })) listeGroupes.push({ id: '_autres', libelle: '…' });
        return listeGroupes.map(function (g) {
          var membres = defs.filter(function (d) { return g.id === '_autres' ? !connus[d.groupe] : d.groupe === g.id; });
          if (!membres.length) return null;
          var nbCachees = membres.filter(function (d) { return !LTP.q.metriqueVisible(etat, d.id); }).length;
          return h('section', { class: 'groupe' + (g.id === gm ? ' is-mobile-actif' : ''), 'data-groupe': g.id, 'aria-labelledby': 'grp-' + g.id },
            h('h3', { class: 'groupe-titre', id: 'grp-' + g.id }, g.libelle,
              nbCachees ? h('span', { class: 'groupe-cachees mono', title: t('tableau.nonMesure') }, nbCachees + ' ', h('span', { 'aria-hidden': 'true' }, '◌'), h('span', { class: 'sr-only' }, t('tableau.nonMesure'))) : null),
            h('ul', { class: 'metriques' }, membres.map(function (d) { return tuile(d, etat, ctx); })));
        });
      });
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
