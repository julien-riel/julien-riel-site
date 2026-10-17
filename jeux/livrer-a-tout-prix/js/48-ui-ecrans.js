/* 48-ui-ecrans.js */
/* Accueil (écran « accueil »), HUD (région « hud »), « Ce qu'on entend » (région « echos » : les réflexions des
   développeurs et leur cause), journal (région « journal »), visite guidée avec Ginette
   (couche de la région « modale », LTP.ui.tutoriel) et tiroir du mode animateur (dans le HUD).
   Le HUD est construit une fois puis mis à jour par différences, pour ne jamais perdre le focus.
   Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, s = LTP.ui.svg, t = LTP.t, F = LTP.ui.F;

  function texte(el, v) { v = v == null ? '' : String(v); if (el.textContent !== v) el.textContent = v; }
  function cacher(el, oui) { if (el.hidden !== !!oui) el.hidden = !!oui; }

  // =====================================================================
  // Accueil
  // =====================================================================
  var accueil_ = { confirmer: false, efface: false };

  function illustration() {
    // La fenêtre de l'open space, en fin d'après-midi : stores, soleil bas, post-it et tasse.
    var g = s('svg', { viewBox: '0 0 360 300', class: 'accueil-art', role: 'img', 'aria-label': t('accueil.fenetre') },
      s('defs', null,
        s('linearGradient', { id: 'ltp-ciel', x1: 0, y1: 0, x2: 0, y2: 1 },
          s('stop', { offset: '0', 'stop-color': 'var(--ciel-haut)' }), s('stop', { offset: '1', 'stop-color': 'var(--ciel-bas)' })),
        s('linearGradient', { id: 'ltp-rayon', x1: 0, y1: 0, x2: 1, y2: 1 },
          s('stop', { offset: '0', 'stop-color': 'var(--soleil)', 'stop-opacity': '.55' }), s('stop', { offset: '1', 'stop-color': 'var(--soleil)', 'stop-opacity': '0' })),
        s('clipPath', { id: 'ltp-vitre' }, s('rect', { x: 40, y: 22, width: 280, height: 170, rx: 4 }))),
      s('rect', { x: 32, y: 14, width: 296, height: 186, rx: 8, class: 'art-cadre' }),
      s('g', { 'clip-path': 'url(#ltp-vitre)' },
        s('rect', { x: 40, y: 22, width: 280, height: 170, fill: 'url(#ltp-ciel)' }),
        s('circle', { cx: 248, cy: 150, r: 34, class: 'art-soleil' }),
        s('path', { d: 'M40 192 L40 160 L70 160 L70 140 L96 140 L96 166 L128 166 L128 128 L150 128 L150 150 L184 150 L184 172 L214 172 L214 136 L236 136 L236 176 L268 176 L268 156 L300 156 L300 170 L320 170 L320 192 Z', class: 'art-ville' }),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(function (i) { return s('rect', { x: 40, y: 26 + i * 15.4, width: 280, height: 6.5, class: 'art-store' }); })),
      s('line', { x1: 180, y1: 22, x2: 180, y2: 192, class: 'art-meneau' }),
      s('path', { d: 'M150 200 L320 200 L360 300 L120 300 Z', fill: 'url(#ltp-rayon)' }),
      s('rect', { x: 10, y: 228, width: 340, height: 12, rx: 3, class: 'art-bureau' }),
      // écran
      s('rect', { x: 46, y: 162, width: 92, height: 62, rx: 5, class: 'art-ecran' }),
      s('rect', { x: 52, y: 168, width: 80, height: 50, rx: 2, class: 'art-ecran-vitre' }),
      [0, 1, 2, 3, 4].map(function (i) { return s('rect', { x: 58, y: 174 + i * 9, width: 20 + ((i * 37) % 44), height: 3.5, rx: 1.5, class: i === 2 ? 'art-code art-code--accent' : 'art-code' }); }),
      s('rect', { x: 86, y: 224, width: 12, height: 6, class: 'art-pied' }),
      // post-it
      s('g', { transform: 'rotate(-6 196 196)' },
        s('rect', { x: 170, y: 172, width: 50, height: 48, class: 'art-postit' }),
        s('path', { d: 'M178 188h32M178 197h26M178 206h18', class: 'art-postit-ecrit' })),
      // tasse
      s('path', { d: 'M262 204h28v20a6 6 0 0 1-6 6h-16a6 6 0 0 1-6-6z', class: 'art-tasse' }),
      s('path', { d: 'M290 210h5a5 5 0 0 1 0 10h-5', class: 'art-anse' }),
      s('path', { d: 'M270 196c0-6 6-6 6-12M280 196c0-6 6-6 6-12', class: 'art-vapeur' }),
      // pile de PR
      [0, 1, 2, 3].map(function (i) { return s('rect', { x: 306 - i * 1.5, y: 220 - i * 6, width: 36, height: 5, rx: 1, class: 'art-papier' }); }));
    return g;
  }

  function resumeSauvegarde() {
    if (!LTP.sauvegarde || !LTP.sauvegarde.charger) return null;
    var e = LTP.sauvegarde.charger();
    if (!e || e.phase === 'accueil') return null;
    var enCours = e.phase === 'planification' || e.phase === 'execution';
    return { semaine: LTP.q.semaineCourante(e), trimestre: e.trimestre, phase: e.phase, jour: enCours ? LTP.ui.nomJour(e.jour || 0) : '' };
  }

  LTP.ui.ecrans.accueil = {
    monter: function (racine) { this._racine = racine; racine._sig = null; },
    rendre: function (etat, ui, precedent, app) {
      var sauv = resumeSauvegarde();
      var sig = JSON.stringify([sauv, !!etat.options.serveurAchete, accueil_.confirmer, accueil_.efface]);
      F.reconstruire(this._racine, sig, function () {
        var trimestres = [1, 2, 3, 4].map(function (n) {
          return h('li', { class: 'trim-carte' }, h('span', { class: 'trim-num mono' }, 'T' + n),
            h('b', null, t('accueil.t' + n)), h('span', null, t('accueil.t' + n + 'd')));
        });
        var equipe = (LTP.donnees.ilots || []).slice().sort(function (a, b) { return a.ordre - b.ordre; }).map(function (il) {
          var devs = (LTP.donnees.devs || []).filter(function (d) { return d.ilot === il.id; });
          return h('li', { class: 'equipe-ilot', style: { '--teinte': il.teinte } },
            h('span', { class: 'equipe-visages' }, devs.map(function (d) { return h('span', { title: d.nom }, F.visage(d.id, 34)); })),
            h('b', null, il.court || il.nom), h('span', { class: 'equipe-noms' }, devs.map(function (d) { return d.nom; }).join(', ')));
        });
        return h('section', { class: 'accueil' },
          h('div', { class: 'accueil-hero' },
            h('div', { class: 'accueil-texte' },
              h('p', { class: 'eyebrow' }, t('jeu.eyebrow')),
              h('h1', { class: 'titre-jeu' }, t('jeu.titre')),
              h('p', { class: 'lede' }, t('jeu.accroche')),
              h('p', { class: 'univers' }, h('span', { class: 'gpu-badge mono' }, 'GPU'), h('span', null, t('jeu.univers'))),
              h('label', { class: 'case' },
                h('input', { type: 'checkbox', 'data-cle': 'serveur', checked: etat.options.serveurAchete ? true : null,
                  onchange: function (ev) { app.dispatch({ type: 'option', cle: 'serveurAchete', valeur: !!ev.target.checked }); } }),
                h('span', { class: 'case-boite', 'aria-hidden': 'true' }),
                h('span', { class: 'case-texte' }, h('b', null, t('accueil.serveur')), h('small', null, t('accueil.serveurAide')))),
              h('div', { class: 'actions' },
                h('button', { type: 'button', id: 'btn-commencer', class: 'btn btn--primaire btn--grand', 'data-cle': 'commencer',
                  onclick: function () {
                    app.dispatch({ type: 'phase', phase: 'planification' });
                    if (LTP.ui.restaurerFocus) LTP.ui.restaurerFocus(null);
                  } }, t('action.commencer')),
                h('button', { type: 'button', id: 'btn-tutoriel', class: 'btn btn--grand', 'data-cle': 'tutoriel',
                  onclick: function () { app.setUI({ visite: { etape: 0 } }); app.dispatch({ type: 'phase', phase: 'planification' }); } },
                  F.visage('ginette', 24), ' ', t('action.tutoriel')),
                sauv ? h('button', { type: 'button', id: 'btn-reprendre', class: 'btn btn--fantome btn--grand', 'data-cle': 'reprendre',
                  onclick: function () {
                    var e = LTP.sauvegarde.charger();
                    if (e) { app.ui.vitesse = 0; app.demarrer(e); if (LTP.ui.restaurerFocus) LTP.ui.restaurerFocus(null); }
                  } },
                  t('action.reprendre'), h('small', { class: 'mono' }, t('action.reprendreDetail', sauv))) : null),
              h('div', { class: 'accueil-pied' },
                h('a', { href: '?animateur=1', class: 'lien', title: t('accueil.animateurAide') }, t('accueil.animateur')),
                sauv && !accueil_.confirmer ? h('button', { type: 'button', class: 'lien', 'data-cle': 'effacer',
                  onclick: function () { accueil_.confirmer = true; app.notifier(); } }, t('action.effacer')) : null,
                accueil_.confirmer ? h('span', { class: 'confirmer', role: 'alert' }, t('action.effacerConfirmer'), ' ',
                  h('button', { type: 'button', class: 'lien lien--crit', 'data-cle': 'effacer-oui', onclick: function () {
                    if (LTP.sauvegarde) LTP.sauvegarde.effacer();
                    accueil_.confirmer = false; accueil_.efface = true; LTP.ui.annoncer(t('accueil.efface')); app.notifier();
                  } }, t('action.effacerOui')), ' ',
                  h('button', { type: 'button', class: 'lien', 'data-cle': 'effacer-non', onclick: function () { accueil_.confirmer = false; app.notifier(); } }, t('action.annuler'))) : null,
                accueil_.efface && !sauv ? h('span', { class: 'muted' }, t('accueil.efface')) : null)),
            h('div', { class: 'accueil-illu' }, illustration())),
          h('section', { class: 'accueil-annee', 'aria-labelledby': 'acc-annee' },
            h('h2', { id: 'acc-annee', class: 'h-sec' }, t('accueil.trimestres')),
            h('p', { class: 'accueil-rythme' }, t('accueil.rythme')),
            h('ol', { class: 'trims' }, trimestres)),
          h('section', { class: 'accueil-equipe', 'aria-labelledby': 'acc-equipe' },
            h('h2', { id: 'acc-equipe', class: 'h-sec' }, t('jeu.equipe'), ' ', h('span', { class: 'muted' }, t('jeu.equipeNote'))),
            h('ul', { class: 'equipe' }, equipe)));
      });
    },
  };

  // =====================================================================
  // HUD
  // =====================================================================
  var hud_ = { animOuvert: false, derniereVitesse: 1 };

  function stat(cle, aide) {
    var val = h('b', { class: 'mono hud-stat-val' });
    var el = h('div', { class: 'hud-stat hud-stat--' + cle, title: aide }, h('span', { class: 'hud-stat-lib' }, t('hud.' + cle)), val);
    el._val = val;
    return el;
  }

  function tiroirAnimateur(etat, app) {
    var idG = 'anim-graine', idN = 'anim-n';
    return h('div', { class: 'tiroir', role: 'dialog', 'aria-label': t('anim.titre') },
      h('div', { class: 'tiroir-tete' }, h('h2', null, t('anim.titre')),
        h('button', { type: 'button', class: 'modale-fermer', 'aria-label': t('action.fermer'), 'data-cle': 'anim-fermer',
          onclick: function () { hud_.animOuvert = false; app.notifier(); } }, F.icone('fermer', 16))),
      h('div', { class: 'tiroir-ligne' },
        h('label', { for: idG }, t('anim.graine')),
        h('input', { id: idG, type: 'text', inputmode: 'numeric', value: String(etat.graine), class: 'mono', 'data-cle': 'anim-graine' }),
        h('button', { type: 'button', class: 'btn btn--petit', 'data-cle': 'anim-nouvelle', onclick: function () {
          var v = document.getElementById(idG).value.trim();
          var g = /^-?\d+$/.test(v) ? parseInt(v, 10) : v;
          hud_.animOuvert = false;
          app.nouvellePartie(g, { animateur: true, serveurAchete: !!etat.options.serveurAchete });
        } }, t('anim.nouvelle'))),
      h('div', { class: 'tiroir-ligne' },
        h('label', { for: idN }, t('anim.sauter')),
        h('input', { id: idN, type: 'number', min: 1, max: 52, value: '13', class: 'mono', 'data-cle': 'anim-n' }),
        h('span', null, t('anim.semaines')),
        h('button', { type: 'button', class: 'btn btn--petit btn--primaire', 'data-cle': 'anim-simuler', onclick: function () {
          var n = Math.max(1, Math.min(60, parseInt(document.getElementById(idN).value, 10) || 1));
          app.simuler(n);
        } }, t('anim.sauterBouton'))),
      h('p', { class: 'muted petit' }, t('anim.note')),
      h('label', { class: 'case case--petite' },
        h('input', { type: 'checkbox', checked: etat.options.voirCachees ? true : null, 'data-cle': 'anim-cachees',
          onchange: function (ev) { app.dispatch({ type: 'option', cle: 'voirCachees', valeur: !!ev.target.checked }); } }),
        h('span', { class: 'case-boite', 'aria-hidden': 'true' }), h('span', { class: 'case-texte' }, t('anim.cachees'))),
      h('label', { class: 'case case--petite' },
        h('input', { type: 'checkbox', checked: etat.options.toutDebloque ? true : null, 'data-cle': 'anim-tout',
          onchange: function (ev) { app.dispatch({ type: 'option', cle: 'toutDebloque', valeur: !!ev.target.checked }); } }),
        h('span', { class: 'case-boite', 'aria-hidden': 'true' }), h('span', { class: 'case-texte' }, t('anim.tout'))));
  }

  LTP.ui.regions.hud = {
    monter: function (racine, app) {
      var self = this;
      this._racine = racine;
      var el = this._el = {};
      el.retour = h('a', { class: 'tool tool--back', href: '/projets/', title: t('hud.retourAide') }, t('hud.retour'));
      el.titre = h('span', { class: 'hud-titre' }, t('jeu.titre'));
      el.phase = h('span', { class: 'hud-phase' });
      el.tri = h('span', { class: 'hud-tri mono', 'aria-hidden': 'true' });
      // Semaine et jour en toutes lettres (versions courtes au téléphone) ; une phrase complète pour les lecteurs d'écran.
      el.semLong = h('span', { class: 'hud-long' });
      el.semCourt = h('span', { class: 'hud-court' });
      el.jourLong = h('span', { class: 'hud-long' });
      el.jourCourt = h('span', { class: 'hud-court' });
      el.jourSep = h('span', { class: 'hud-sep' }, ' · ');
      el.jour = h('span', { class: 'hud-jour' }, el.jourSep, el.jourLong, el.jourCourt);
      el.semLu = h('span', { class: 'sr-only' });
      el.sem = h('span', { class: 'hud-sem' }, h('span', { 'aria-hidden': 'true' }, el.semLong, el.semCourt, el.jour), el.semLu);
      el.anneeRempli = h('span', { class: 'annee-rempli' });
      el.annee = h('span', { class: 'annee', role: 'img' }, el.anneeRempli,
        h('i', { style: { left: '25%' } }), h('i', { style: { left: '50%' } }), h('i', { style: { left: '75%' } }));
      el.temps = h('div', { class: 'hud-temps' }, h('span', { class: 'hud-quand' }, el.tri, el.sem), el.annee);
      el.budget = stat('budget', t('hud.budgetAide'));
      el.jetons = stat('jetons', t('hud.jetonsAide'));
      el.confiance = stat('confiance', t('hud.confianceAide'));
      el.confMetre = h('span', { class: 'conf-metre', 'aria-hidden': 'true' }, h('i'));
      el.confiance.appendChild(el.confMetre);
      el.stats = h('div', { class: 'hud-stats' }, el.budget, el.jetons, el.confiance);

      el.lecture = h('button', { type: 'button', id: 'btn-lecture', class: 'btn btn--lecture',
        onclick: function () {
          var e = LTP.app.etat, u = LTP.app.ui;
          if (e.phase === 'planification') {
            var r = app.dispatch({ type: 'phase', phase: 'execution' });
            if (!r.erreur) app.setUI({ vitesse: hud_.derniereVitesse || 1 });
            return;
          }
          if (e.phase !== 'execution') return;
          if (u.vitesse) { hud_.derniereVitesse = u.vitesse; app.setUI({ vitesse: 0 }); }
          else app.setUI({ vitesse: hud_.derniereVitesse || 1 });
        } });
      el.vitesses = h('div', { class: 'vitesses', role: 'radiogroup', 'aria-label': t('hud.vitesse') });
      el.vitBoutons = [1, 2, 4].map(function (v, i, liste) {
        var b = h('button', { type: 'button', role: 'radio', class: 'vitesse', 'data-vitesse': String(v),
          onclick: function () { hud_.derniereVitesse = v; app.setUI({ vitesse: v }); },
          onkeydown: function (ev) {
            var j = null;
            if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') j = (i + 1) % liste.length;
            if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') j = (i - 1 + liste.length) % liste.length;
            if (j != null) { ev.preventDefault(); hud_.derniereVitesse = liste[j]; app.setUI({ vitesse: liste[j] }); el.vitBoutons[j].focus(); }
          } }, '×' + v);
        el.vitesses.appendChild(b);
        return b;
      });
      // « +1 jour » : met en pause et avance d'une journée (le vendredi publie le rapport de la semaine).
      el.jourBtn = h('button', { type: 'button', id: 'btn-jour', class: 'btn btn--jour', 'aria-label': t('action.jour'), title: t('action.jourAide'), 'aria-keyshortcuts': 'J',
        onclick: function () {
          var e = LTP.app.etat;
          if (LTP.app.ui.modaleBloquante) return;
          if (e.phase === 'planification') { var r = app.dispatch({ type: 'phase', phase: 'execution' }); if (r.erreur) return; }
          app.setUI({ vitesse: 0 });
          app.avancer();
        } }, F.icone('suivant', 14), h('span', { class: 'btn-lib' }, t('action.jour')), h('span', { class: 'btn-court', 'aria-hidden': 'true' }, t('action.jourCourt')));
      el.attente = h('span', { class: 'hud-attente', role: 'status' }, t('hud.evenementAttend'));
      el.commandes = h('div', { class: 'hud-commandes' }, el.lecture, el.vitesses, el.jourBtn, el.attente);

      el.aide = h('button', { type: 'button', class: 'ibtn', 'aria-label': t('hud.aideLong'), title: t('hud.aideLong'),
        onclick: function () { app.setUI({ visite: { etape: 0 }, vitesse: 0 }); } }, F.icone('aide', 18));
      el.menu = h('button', { type: 'button', class: 'ibtn', title: t('hud.menuAide'), 'aria-label': t('hud.menuAide'),
        onclick: function () { app.nouvellePartie(); } }, F.icone('maison', 18));
      el.anim = h('button', { type: 'button', class: 'ibtn ibtn--anim', 'aria-expanded': 'false', title: t('anim.titre'), 'aria-label': t('anim.titre'),
        onclick: function () { hud_.animOuvert = !hud_.animOuvert; app.notifier(); } }, F.icone('animateur', 18));
      el.outils = h('div', { class: 'hud-outils' }, el.anim, el.menu, el.aide);
      el.tiroir = h('div', { class: 'tiroir-zone' });

      LTP.ui.remplacer(racine, h('div', { class: 'hud-inner' },
        h('div', { class: 'hud-marque' }, el.retour, h('span', { class: 'hud-nom' }, el.titre, el.phase)),
        el.temps, el.stats, el.commandes, el.outils), el.tiroir);

      // Hauteur du HUD, pour coller le panneau juste en dessous.
      try {
        var maj = function () { document.documentElement.style.setProperty('--hud-h', Math.round(racine.getBoundingClientRect().height) + 'px'); };
        if (window.ResizeObserver) new ResizeObserver(maj).observe(racine);
        maj();
      } catch (e) { /* ignoré */ }
      document.addEventListener('keydown', function (ev) {
        if (ev.key === 'Escape' && hud_.animOuvert && self._el.tiroir.contains(document.activeElement)) { hud_.animOuvert = false; self._el.anim.focus(); app.notifier(); }
      });
    },

    rendre: function (etat, ui, precedent, app) {
      var el = this._el, ph = etat.phase;
      var enJeu = ph === 'planification' || ph === 'execution' || ph === 'tutoriel' || ph === 'comite';
      this._racine.setAttribute('data-phase', ph);
      texte(el.phase, t('hud.phase.' + ph));
      cacher(el.temps, !enJeu && ph !== 'fin');
      cacher(el.stats, !enJeu && ph !== 'fin');
      // La semaine que le joueur est en train de jouer, et le jour ouvrable (pas de jour au comité ni au bilan).
      var sc = LTP.q.semaineCourante(etat);
      var avecJour = ph === 'planification' || ph === 'execution';
      var jour = avecJour ? LTP.ui.nomJour(etat.jour || 0) : '';
      texte(el.tri, t('hud.trimestre', etat));
      texte(el.semLong, t('hud.semaine', { semaine: sc }));
      texte(el.semCourt, t('hud.semaineCourt', { semaine: sc }));
      texte(el.jourLong, jour);
      texte(el.jourCourt, avecJour ? LTP.ui.jourCourt(etat.jour || 0) : '');
      cacher(el.jour, !avecJour);
      texte(el.semLu, t('hud.quand', { trimestre: etat.trimestre, semaine: sc, jour: jour }));
      // Progression de l'année, jour par jour (52 × 5 journées).
      var jours = avecJour ? (sc - 1) * 5 + (etat.jour || 0) : etat.semaine * 5;
      el.anneeRempli.style.width = Math.max(0, Math.min(100, jours / 260 * 100)).toFixed(2) + '%';
      el.annee.setAttribute('aria-label', t('hud.annee', { semaine: sc, jour: jour }));

      var org = etat.org || {};
      texte(el.budget._val, LTP.format.argent(org.budget));
      var ratio = org.budgetAnnuel ? org.budget / org.budgetAnnuel : 1;
      el.budget.setAttribute('data-ton', org.budget < 0 ? 'crit' : ratio < 0.2 ? 'warn' : '');
      texte(el.jetons._val, LTP.format.jetons(org.jetonsSemaine || 0) + t('hud.parSemaine'));
      var conf = Math.round(org.confiance || 0);
      texte(el.confiance._val, String(conf));
      el.confiance.setAttribute('data-ton', conf < 25 ? 'crit' : conf < 40 ? 'warn' : '');
      el.confMetre.firstChild.style.width = Math.max(0, Math.min(100, conf)) + '%';

      // Commandes de lecture
      var planif = ph === 'planification', exec = ph === 'execution';
      cacher(el.commandes, !(planif || exec));
      var attente = !!etat.evenementEnAttente;
      cacher(el.vitesses, !exec);
      cacher(el.attente, !(exec && attente));
      el.lecture.classList.toggle('btn--primaire', planif || (exec && !ui.vitesse));
      el.lecture.classList.toggle('btn--grand', planif);
      var libLecture = planif ? t('action.lancer') : ui.vitesse ? t('action.pause') : t('action.jouer');
      var cleL = (planif ? 'p' : ui.vitesse ? 'v' : 'z');
      if (el.lecture._cle !== cleL) {
        el.lecture._cle = cleL;
        LTP.ui.remplacer(el.lecture, F.icone(planif ? 'jouer' : ui.vitesse ? 'pause' : 'jouer', 14), h('span', { class: 'btn-lib' }, libLecture));
      }
      el.lecture.setAttribute('aria-label', libLecture);
      el.lecture.setAttribute('aria-pressed', exec && ui.vitesse ? 'true' : 'false');
      el.lecture.title = planif ? t('hud.planifNote') : '';
      el.lecture.disabled = exec && attente;
      el.jourBtn.disabled = (exec && attente) || !!ui.modaleBloquante;
      var vit = ui.vitesse || 0;
      el.vitBoutons.forEach(function (b) {
        var v = parseInt(b.getAttribute('data-vitesse'), 10);
        var actif = v === (vit || hud_.derniereVitesse);
        b.setAttribute('aria-checked', actif ? 'true' : 'false');
        b.tabIndex = actif ? 0 : -1;
        b.classList.toggle('is-actif', actif);
        b.classList.toggle('is-pause', actif && !vit);
        b.disabled = attente;
      });

      cacher(el.aide, !(planif || exec));
      cacher(el.menu, ph === 'accueil');
      var anim = !!(etat.options && etat.options.animateur);
      cacher(el.anim, !anim);
      el.anim.setAttribute('aria-expanded', anim && hud_.animOuvert ? 'true' : 'false');
      var ouvert = anim && hud_.animOuvert;
      F.reconstruire(el.tiroir, ouvert ? JSON.stringify([etat.graine, etat.options.voirCachees, etat.options.toutDebloque]) : '', function () {
        return ouvert ? tiroirAnimateur(etat, app) : null;
      });
    },
  };

  // =====================================================================
  // Journal
  // =====================================================================
  var ICONE_TYPE = { info: 'info', alerte: 'alerte', incident: 'incident', evenement: 'evenement', levier: 'levier', personne: 'personne', comite: 'comite' };
  var journal_ = { annonce: null };

  LTP.ui.regions.journal = {
    monter: function (racine) {
      this._racine = racine;
      this._liste = h('ol', { class: 'journal-liste' });
      LTP.ui.remplacer(racine, h('div', { class: 'journal' }, h('h2', { class: 'zone-titre' }, t('journal.titre')), this._liste));
    },
    rendre: function (etat, ui, precedent, app) {
      var j = etat.journal || [];
      var dernier = j[j.length - 1];
      var sc = LTP.q.semaineCourante(etat);
      var sig = j.length + ':' + (dernier ? dernier.semaine + ':' + dernier.jour + dernier.cle : '') + ':' + sc;
      F.reconstruire(this._liste, sig, function () {
        var entrees = j.slice(-14).reverse();
        if (!entrees.length) return h('li', { class: 'entree entree--vide' }, t('journal.vide'));
        return entrees.map(function (e) {
          var type = ICONE_TYPE[e.type] ? e.type : 'info';
          var txt;
          try { txt = LTP.ui.texteJournal(e); } catch (err) { txt = e.cle; }
          var aJour = e.jour != null;
          var quand = aJour ? t('journal.quand', { semaine: e.semaine, jour: LTP.ui.jourCourt(e.jour) }) : 'S' + e.semaine;
          // Les entrées qui viennent d'une décision du joueur sont atténuées : le reste, c'est ce qui arrive tout seul.
          return h('li', { class: 'entree entree--' + type + (e.semaine === sc ? ' is-semaine' : '') + (e.joueur ? ' is-joueur' : '') },
            h('span', { class: 'entree-icone', title: e.joueur ? t('journal.joueur') : t('journal.type.' + type) }, F.icone(ICONE_TYPE[type], 15),
              h('span', { class: 'sr-only' }, e.joueur ? t('journal.joueur') : t('journal.type.' + type))),
            h('span', { class: 'entree-sem mono' }, h('span', { 'aria-hidden': 'true' }, quand),
              h('span', { class: 'sr-only' }, aJour ? t('journal.quandLong', { semaine: e.semaine, jour: LTP.ui.nomJour(e.jour) }) : quand)),
            h('span', { class: 'entree-texte' }, txt));
        });
      });
      // Annonce des incidents et alertes du jour (pas chaque journée : seulement ce qui demande l'attention ;
      // les cartes d'événement et les notes de service s'annoncent elles-mêmes).
      var neufs = (app.dernierJournal || []).filter(function (e) { return e.type === 'incident' || e.type === 'alerte'; });
      if (neufs.length) {
        var cle = sc + ':' + (etat.jour || 0) + ':' + neufs.length + ':' + neufs[0].cle;
        if (journal_.annonce !== cle) {
          journal_.annonce = cle;
          try { LTP.ui.annoncer(neufs.map(LTP.ui.texteJournal).slice(0, 2).join(' ')); } catch (e) { /* ignoré */ }
        }
      }
    },
  };

  // =====================================================================
  // Ce qu'on entend : les réflexions des développeurs, avec leur cause
  // =====================================================================
  // Aujourd'hui (etat.reflexions.liste), puis hier, avant-hier et plus tôt (etat.reflexions.fil), environ 8 entrées.
  // Sélecteur stable pour la visite guidée : .echos
  var ECHOS_MAX = 8;
  function echosAMontrer(etat) {
    var R = etat.reflexions || {};
    var F_ = F;
    var auj = F_.aujourdhui(etat);
    var aujourdhui = (R.liste || []).filter(function (r) { return r && r.texte; });
    var vus = {};
    aujourdhui.forEach(function (r) { vus[r.dev + '|' + r.texte + '|' + r.semaine + '|' + r.jour] = true; });
    var avant = (R.fil || []).filter(function (r) {
      return r && r.texte && F_.jourAbsolu(r.semaine, r.jour) < auj && !vus[r.dev + '|' + r.texte + '|' + r.semaine + '|' + r.jour];
    }).slice().reverse();
    var liste = aujourdhui.concat(avant).slice(0, ECHOS_MAX);
    var groupes = [];
    liste.forEach(function (r) {
      var ecart = auj - F_.jourAbsolu(r.semaine, r.jour);
      var cle = ecart <= 0 ? 'aujourdhui' : ecart === 1 ? 'hier' : ecart === 2 ? 'avantHier' : 'plusTot';
      var g = groupes[groupes.length - 1];
      if (!g || g.cle !== cle) groupes.push(g = { cle: cle, items: [] });
      g.items.push(r);
    });
    return groupes;
  }

  LTP.ui.regions.echos = {
    monter: function (racine) {
      this._liste = h('div', { class: 'echos-groupes' });
      LTP.ui.remplacer(racine, h('div', { class: 'echos' },
        h('div', { class: 'echos-tete' }, h('h2', { class: 'zone-titre', id: 'echos-titre' }, t('echos.titre')),
          h('p', { class: 'echos-intro' }, t('echos.intro'))),
        this._liste));
    },
    rendre: function (etat) {
      var R = etat.reflexions || {};
      var fil = R.fil || [];
      var dernier = fil[fil.length - 1];
      var sig = JSON.stringify([etat.graine, LTP.q.semaineCourante(etat), etat.jour || 0, fil.length, dernier ? dernier.dev + dernier.texte : '',
        (R.liste || []).map(function (r) { return r.dev + (r.immediat ? '!' : ''); })]);
      F.reconstruire(this._liste, sig, function () {
        var groupes = echosAMontrer(etat);
        if (!groupes.length) return h('p', { class: 'echos-vide' }, t('echos.vide'));
        return groupes.map(function (g) {
          var idG = 'echos-' + g.cle;
          return h('section', { class: 'echos-groupe echos-groupe--' + g.cle, 'aria-labelledby': idG },
            h('h3', { class: 'echos-quand eyebrow', id: idG }, t('echos.' + g.cle)),
            h('ul', { class: 'echos-liste' }, g.items.map(function (r) {
              return F.ligneReflexion(r, etat, { quand: g.cle === 'plusTot' ? t('journal.quand', { semaine: r.semaine, jour: LTP.ui.jourCourt(r.jour) }) : null });
            })));
        });
      });
    },
  };

  // =====================================================================
  // Visite guidée avec Ginette
  // =====================================================================
  // Chaque étape montre un élément réel de l'écran. Les étapes « attendre » laissent la main au joueur
  // (la bulle n'est alors plus modale) : changer un levier ('levier'), jouer une journée ('jour') ou une semaine ('semaine').
  var tuto_ = { cle: null, ecoute: false, retourFocus: null, etape: null, depart: null, fait: false, minuterie: null, fin: false };

  function etapes() {
    var d = LTP.donnees.tutoriel;
    if (d && d.length) return d;
    return [
      { id: 'd1', region: 'openspace', personnage: 'ginette', texte: t('tuto.defaut.1') },
      { id: 'd2', region: 'openspace', personnage: 'ginette', texte: t('tuto.defaut.2') },
      { id: 'd3', region: 'panneau', personnage: 'ginette', texte: t('tuto.defaut.3') },
      { id: 'd4', region: 'tableau', personnage: 'ginette', texte: t('tuto.defaut.4') },
      { id: 'd5', cibles: ['.echos'], region: 'journal', personnage: 'ginette', texte: t('tuto.defaut.5') },
      { id: 'd6', region: 'hud', personnage: 'ginette', texte: t('tuto.defaut.6') },
    ];
  }

  function visible(el) {
    if (!el || !document.contains(el)) return false;
    var r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden';
  }
  function cibleRegion(region) {
    if (!region) return null;
    if (region === 'hud') return document.getElementById('hud');
    if (region === 'panneau' && F.etroit()) return document.querySelector('.panneau-poignee') || document.querySelector('[data-region="panneau"]');
    return document.querySelector('[data-region="' + region + '"]');
  }
  function cibleDe(et) {
    if (!et) return null;
    var l = et.cibles || (et.cible ? [et.cible] : []);
    for (var i = 0; i < l.length; i++) {
      var el = null;
      try { el = document.querySelector(l[i]); } catch (e) { el = null; }
      if (visible(el)) return el;
    }
    return cibleRegion(et.region);
  }

  function placer(couche) {
    var spot = couche.querySelector('.tuto-spot'), bulle = couche.querySelector('.tuto-bulle');
    if (!spot || !bulle) return;
    var cible = cibleDe(couche._etape);
    var vw = document.documentElement.clientWidth || window.innerWidth, vh = window.innerHeight;
    if (!cible) { spot.style.display = 'none'; bulle.style.left = Math.max(16, (vw - bulle.offsetWidth) / 2) + 'px'; bulle.style.top = Math.max(16, (vh - bulle.offsetHeight) / 2) + 'px'; return; }
    var r = cible.getBoundingClientRect();
    var m = 6;
    var top = Math.max(4, r.top - m), left = Math.max(4, r.left - m);
    var bas = Math.min(vh - 4, r.bottom + m), droite = Math.min(vw - 4, r.right + m);
    spot.style.display = '';
    spot.style.top = top + 'px'; spot.style.left = left + 'px';
    spot.style.width = Math.max(0, droite - left) + 'px'; spot.style.height = Math.max(0, bas - top) + 'px';
    var bw = bulle.offsetWidth, bh = bulle.offsetHeight;
    var bx, by;
    if (vw < 640) {
      bx = Math.max(8, (vw - bw) / 2);
      var hudH = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--hud-h'), 10) || 0;
      // Au-dessus ou au-dessous de la cible, du côté où il reste le plus de place.
      if (vh - bas >= top - hudH) by = Math.min(vh - bh - 8, bas + 10);
      else by = Math.max(8, top - bh - 10);
    } else if (vh - bas > bh + 16) { by = bas + 12; bx = Math.min(Math.max(16, r.left), vw - bw - 16); }
    else if (top > bh + 16) { by = top - bh - 12; bx = Math.min(Math.max(16, r.left), vw - bw - 16); }
    else if (vw - droite > bw + 24) { bx = droite + 12; by = Math.min(Math.max(16, r.top), vh - bh - 16); }
    else if (left > bw + 24) { bx = left - bw - 12; by = Math.min(Math.max(16, r.top), vh - bh - 16); }
    else { bx = Math.max(16, (vw - bw) / 2); by = Math.max(16, vh - bh - 24); }
    bulle.style.left = Math.round(bx) + 'px';
    bulle.style.top = Math.round(Math.max(8, by)) + 'px';
  }

  function focaliser(el) { if (el) { try { el.focus({ preventScroll: true }); } catch (e) { el.focus(); } } }

  function terminer(etat, app) {
    tuto_.fin = true;
    clearTimeout(tuto_.minuterie);
    if (F.etroit() && LTP.ui.panneau) LTP.ui.panneau.fermer();
    app.setUI({ visite: null });
    if (LTP.app.etat.phase === 'tutoriel') app.dispatch({ type: 'phase', phase: 'planification' });
  }

  // Ce que l'étape attend est-il fait ? (comparé à l'instantané pris à l'entrée de l'étape)
  function actionFaite(et, etat) {
    if (!et.attendre || !tuto_.depart) return true;
    if (et.attendre === 'levier') return JSON.stringify(etat.leviers) !== tuto_.depart.leviers;
    if (et.attendre === 'semaine') return etat.semaine > tuto_.depart.semaine && !etat.evenementEnAttente;
    // Une journée de plus (ou une semaine : le vendredi fait passer au lundi suivant).
    if (et.attendre === 'jour') return (etat.semaine !== tuto_.depart.semaine || (etat.jour || 0) !== tuto_.depart.jour) && !etat.evenementEnAttente;
    return true;
  }

  function entrerEtape(et, etat, app) {
    tuto_.depart = { leviers: JSON.stringify(etat.leviers), semaine: etat.semaine, jour: etat.jour || 0 };
    tuto_.fait = false;
    if (et.panneau) {
      var patch = {};
      if (LTP.app.ui.panneau !== et.panneau) patch.panneau = et.panneau;
      if (LTP.app.ui.ilotEdition) patch.ilotEdition = null;
      if (Object.keys(patch).length) app.setUI(patch);
      if (F.etroit() && LTP.ui.panneau) LTP.ui.panneau.ouvrir();
    } else if (F.etroit() && LTP.ui.panneau) LTP.ui.panneau.fermer();
  }

  LTP.ui.tutoriel = {
    actif: function (etat, ui) { return etat.phase === 'tutoriel' || !!(ui.visite && (etat.phase === 'planification' || etat.phase === 'execution')); },
    rendre: function (couche, etat, ui, app) {
      if (!this.actif(etat, ui)) {
        if (tuto_.cle) {
          tuto_.cle = null;
          tuto_.etape = null;
          var r = tuto_.retourFocus; tuto_.retourFocus = null;
          var finie = tuto_.fin; tuto_.fin = false;
          // Fin de visite : retour en haut et focus sur « Lancer le trimestre » (ou « Jouer »), sauf si une note de service s'ouvre.
          if (finie) {
            try { window.scrollTo({ top: 0, behavior: 'auto' }); } catch (e) { window.scrollTo(0, 0); }
            setTimeout(function () {
              // Une note de service (ou une carte) qui s'ouvre à la fin de la visite garde le focus.
              if (document.querySelector('#modale-racine .couche-dialogue [role="dialog"]')) return;
              var b = document.getElementById('btn-lecture');
              if (b && !b.disabled && b.offsetParent !== null) focaliser(b); else if (r && document.contains(r)) focaliser(r);
            }, 40);
          } else if (r && document.contains(r)) focaliser(r);
        }
        F.reconstruire(couche, '', function () { return null; });
        return;
      }
      var liste = etapes();
      var n = Math.max(0, Math.min(liste.length - 1, (ui.visite && ui.visite.etape) || 0));
      var et = liste[n] || {};
      if (!tuto_.cle) { var a = document.activeElement; tuto_.retourFocus = a && a !== document.body ? a : null; }
      if (tuto_.etape !== n) { tuto_.etape = n; entrerEtape(et, etat, app); }
      // Une carte d'événement passe devant : la visite s'efface le temps de la décision.
      if (etat.evenementEnAttente) {
        tuto_.cle = 'tuto:attente';
        F.reconstruire(couche, 'tuto:evenement', function () { return null; });
        return;
      }
      var attend = !!et.attendre;
      var fait = attend && actionFaite(et, etat);
      if (fait && !tuto_.fait) {
        tuto_.fait = true;
        // L'action est faite : on passe à l'étape suivante après un court instant.
        clearTimeout(tuto_.minuterie);
        tuto_.minuterie = setTimeout(function () {
          if (LTP.app.ui.visite && LTP.app.ui.visite.etape === n) app.setUI({ visite: { etape: Math.min(liste.length - 1, n + 1) } });
        }, F.reduit() ? 900 : 1400);
      }
      var cle = 'tuto:' + n + ':' + liste.length + ':' + (fait ? 'fait' : '');
      var ancienne = couche._sig;
      var change = F.reconstruire(couche, cle, function () {
        var p = F.personne(et.personnage || 'ginette');
        var dernier = n === liste.length - 1;
        var action = typeof et.action === 'string' ? et.action : (et.action && et.action.texte) || null;
        // Pendant une étape interactive, la bulle n'est pas modale : le joueur agit sur l'écran.
        var modale = !attend || fait;
        return [
          h('div', { class: 'tuto-spot' + (attend && !fait ? ' is-action' : ''), 'aria-hidden': 'true' }),
          h('div', { class: 'tuto-bulle' + (attend && !fait ? ' is-action' : ''), role: 'dialog', 'aria-modal': modale ? 'true' : 'false', 'aria-labelledby': 'tuto-titre', 'aria-describedby': 'tuto-texte',
            onkeydown: function (ev) { if (ev.key === 'Escape') { ev.preventDefault(); terminer(etat, app); } } },
            h('div', { class: 'tuto-tete' }, F.visage(p.id, 44),
              h('div', null, h('p', { class: 'eyebrow', id: 'tuto-titre', tabindex: '-1' }, t('tuto.titre') + ' · ' + t('tuto.etape', { n: n + 1, total: liste.length })), h('b', null, p.nom))),
            h('p', { class: 'tuto-texte', id: 'tuto-texte' }, F.texte(fait && et.fait ? et.fait : et.texte, etat)),
            attend && !fait && action ? h('p', { class: 'tuto-action', role: 'status' }, h('span', { class: 'tuto-action-puce', 'aria-hidden': 'true' }), F.texte(action, etat)) : null,
            fait ? h('p', { class: 'tuto-action tuto-action--fait', role: 'status' }, h('span', { 'aria-hidden': 'true' }, '✓ '), t('tuto.fait')) : null,
            h('div', { class: 'tuto-points', 'aria-hidden': 'true' }, liste.map(function (x, i) { return h('i', { class: i === n ? 'is-on' : i < n ? 'is-fait' : '' }); })),
            h('div', { class: 'tuto-actions' },
              h('button', { type: 'button', class: 'lien', 'data-cle': 'tuto-passer', onclick: function () { terminer(etat, app); } }, t('action.passerVisite')),
              h('span', { class: 'tuto-nav' },
                n > 0 ? h('button', { type: 'button', class: 'btn btn--petit btn--fantome', 'data-cle': 'tuto-prec', onclick: function () { app.setUI({ visite: { etape: n - 1 } }); } }, t('action.precedent')) : null,
                attend && !fait ? null : h('button', { type: 'button', class: 'btn btn--petit btn--primaire', id: 'btn-tuto-suivant', 'data-cle': 'tuto-suiv', onclick: function () {
                  clearTimeout(tuto_.minuterie);
                  if (dernier) terminer(etat, app); else app.setUI({ visite: { etape: n + 1 } });
                } }, dernier ? t('action.terminerVisite') : t('action.suivant'))))),
        ];
      });
      var nouvelleEtape = !tuto_.cle || tuto_.cle.indexOf('tuto:' + n + ':') !== 0;
      tuto_.cle = cle;
      couche._etape = et;
      if (change) {
        var cible = cibleDe(et);
        if (nouvelleEtape && cible && et.region !== 'hud') {
          try { cible.scrollIntoView({ block: F.etroit() ? 'center' : 'nearest', behavior: F.reduit() ? 'auto' : 'smooth' }); } catch (e) { cible.scrollIntoView(); }
        }
        var boite = couche.querySelector('.tuto-bulle');
        if (attend && !fait) {
          // Étape interactive : le focus va sur l'élément à manipuler.
          if (nouvelleEtape) {
            var f = cible && (cible.matches(F.FOCUSABLES) ? cible : (cible.querySelector('[aria-checked="true"]') || cible.querySelector(F.FOCUSABLES)));
            setTimeout(function () { focaliser(f || boite); }, F.etroit() ? 320 : 30);
          }
        } else if (boite && !boite.contains(document.activeElement)) {
          focaliser(boite.querySelector('#btn-tuto-suivant') || boite.querySelector('#tuto-titre'));
        }
        if (nouvelleEtape || fait) LTP.ui.annoncer(F.texte(fait && et.fait ? et.fait : et.texte, etat) + (attend && !fait && et.action ? ' ' + F.texte(et.action, etat) : ''));
        var debut = Date.now();
        var boucle = function () { placer(couche); if (Date.now() - debut < 900) requestAnimationFrame(boucle); };
        requestAnimationFrame(boucle);
      } else if (ancienne === cle) {
        placer(couche);
      }
      if (!tuto_.ecoute) {
        tuto_.ecoute = true;
        var maj = function () { if (tuto_.cle) placer(couche); };
        window.addEventListener('resize', maj);
        window.addEventListener('scroll', maj, true);
      }
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
