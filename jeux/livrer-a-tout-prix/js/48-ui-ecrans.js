/* 48-ui-ecrans.js */
/* Accueil (écran « accueil »), HUD (région « hud »), journal (région « journal »), visite guidée avec Ginette
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
    return { semaine: e.semaine, trimestre: e.trimestre, phase: e.phase };
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
                  onclick: function () { app.dispatch({ type: 'phase', phase: 'planification' }); } }, t('action.commencer')),
                h('button', { type: 'button', id: 'btn-tutoriel', class: 'btn btn--grand', 'data-cle': 'tutoriel',
                  onclick: function () { app.setUI({ visite: { etape: 0 } }); app.dispatch({ type: 'phase', phase: 'tutoriel' }); } },
                  F.visage('ginette', 24), ' ', t('action.tutoriel')),
                sauv ? h('button', { type: 'button', id: 'btn-reprendre', class: 'btn btn--fantome btn--grand', 'data-cle': 'reprendre',
                  onclick: function () { var e = LTP.sauvegarde.charger(); if (e) { app.ui.vitesse = 0; app.demarrer(e); } } },
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
      el.tri = h('span', { class: 'hud-tri mono' });
      el.sem = h('span', { class: 'hud-sem' });
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
      el.semaine = h('button', { type: 'button', id: 'btn-semaine', class: 'btn btn--semaine',
        onclick: function () {
          var e = LTP.app.etat;
          if (e.phase === 'planification') { var r = app.dispatch({ type: 'phase', phase: 'execution' }); if (r.erreur) return; }
          app.setUI({ vitesse: 0 });
          app.avancer();
        } }, F.icone('suivant', 14), h('span', { class: 'btn-lib' }, t('action.semaine')));
      el.attente = h('span', { class: 'hud-attente', role: 'status' }, t('hud.evenementAttend'));
      el.commandes = h('div', { class: 'hud-commandes' }, el.lecture, el.vitesses, el.semaine, el.attente);

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
      texte(el.tri, t('hud.trimestre', etat));
      texte(el.sem, t('hud.semaine', etat));
      el.anneeRempli.style.width = Math.min(100, etat.semaine / 52 * 100) + '%';
      el.annee.setAttribute('aria-label', t('hud.annee', etat));

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
      el.semaine.disabled = exec && attente;
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
      var sig = j.length + ':' + (dernier ? dernier.semaine + dernier.cle : '') + ':' + etat.semaine;
      F.reconstruire(this._liste, sig, function () {
        var entrees = j.slice(-14).reverse();
        if (!entrees.length) return h('li', { class: 'entree entree--vide' }, t('journal.vide'));
        return entrees.map(function (e) {
          var type = ICONE_TYPE[e.type] ? e.type : 'info';
          var txt;
          try { txt = LTP.ui.texteJournal(e); } catch (err) { txt = e.cle; }
          return h('li', { class: 'entree entree--' + type + (e.semaine === etat.semaine ? ' is-semaine' : '') },
            h('span', { class: 'entree-icone', title: t('journal.type.' + type) }, F.icone(ICONE_TYPE[type], 15), h('span', { class: 'sr-only' }, t('journal.type.' + type))),
            h('span', { class: 'entree-sem mono' }, 'S' + e.semaine),
            h('span', { class: 'entree-texte' }, txt));
        });
      });
      // Annonce des incidents et alertes de la semaine écoulée.
      var neufs = (app.dernierJournal || []).filter(function (e) { return e.type === 'incident' || e.type === 'alerte'; });
      if (neufs.length) {
        var cle = etat.semaine + ':' + neufs.length + ':' + neufs[0].cle;
        if (journal_.annonce !== cle) {
          journal_.annonce = cle;
          try { LTP.ui.annoncer(neufs.map(LTP.ui.texteJournal).slice(0, 2).join(' ')); } catch (e) { /* ignoré */ }
        }
      }
    },
  };

  // =====================================================================
  // Visite guidée avec Ginette
  // =====================================================================
  var tuto_ = { cle: null, ecoute: false, retourFocus: null };

  function etapes() {
    var d = LTP.donnees.tutoriel;
    if (d && d.length) return d;
    return [
      { id: 'd1', region: 'openspace', personnage: 'ginette', texte: t('tuto.defaut.1') },
      { id: 'd2', region: 'openspace', personnage: 'ginette', texte: t('tuto.defaut.2') },
      { id: 'd3', region: 'panneau', personnage: 'ginette', texte: t('tuto.defaut.3') },
      { id: 'd4', region: 'tableau', personnage: 'ginette', texte: t('tuto.defaut.4') },
      { id: 'd5', region: 'journal', personnage: 'ginette', texte: t('tuto.defaut.5') },
      { id: 'd6', region: 'hud', personnage: 'ginette', texte: t('tuto.defaut.6') },
    ];
  }

  function cibleDe(region) {
    if (!region) return null;
    if (region === 'hud') return document.getElementById('hud');
    if (region === 'panneau' && F.etroit()) return document.querySelector('.panneau-poignee') || document.querySelector('[data-region="panneau"]');
    return document.querySelector('[data-region="' + region + '"]');
  }

  function placer(couche) {
    var spot = couche.querySelector('.tuto-spot'), bulle = couche.querySelector('.tuto-bulle');
    if (!spot || !bulle) return;
    var cible = cibleDe(couche._region);
    var vw = window.innerWidth, vh = window.innerHeight;
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
      bx = 12;
      by = (r.top + r.bottom) / 2 > vh / 2 ? 12 + (parseInt(getComputedStyle(document.documentElement).getPropertyValue('--hud-h'), 10) || 0) : vh - bh - 12;
    } else if (vh - bas > bh + 16) { by = bas + 12; bx = Math.min(Math.max(16, r.left), vw - bw - 16); }
    else if (top > bh + 16) { by = top - bh - 12; bx = Math.min(Math.max(16, r.left), vw - bw - 16); }
    else if (vw - droite > bw + 24) { bx = droite + 12; by = Math.min(Math.max(16, r.top), vh - bh - 16); }
    else if (left > bw + 24) { bx = left - bw - 12; by = Math.min(Math.max(16, r.top), vh - bh - 16); }
    else { bx = Math.max(16, (vw - bw) / 2); by = Math.max(16, vh - bh - 24); }
    bulle.style.left = Math.round(bx) + 'px';
    bulle.style.top = Math.round(Math.max(8, by)) + 'px';
  }

  function terminer(etat, app) {
    app.setUI({ visite: null });
    if (LTP.app.etat.phase === 'tutoriel') app.dispatch({ type: 'phase', phase: 'planification' });
  }

  LTP.ui.tutoriel = {
    actif: function (etat, ui) { return etat.phase === 'tutoriel' || !!(ui.visite && (etat.phase === 'planification' || etat.phase === 'execution')); },
    rendre: function (couche, etat, ui, app) {
      if (!this.actif(etat, ui)) {
        if (tuto_.cle) {
          tuto_.cle = null;
          var r = tuto_.retourFocus; tuto_.retourFocus = null;
          if (r && document.contains(r)) r.focus(); else { var b = document.getElementById('btn-lecture'); if (b) b.focus(); }
        }
        F.reconstruire(couche, '', function () { return null; });
        return;
      }
      var liste = etapes();
      var n = Math.max(0, Math.min(liste.length - 1, (ui.visite && ui.visite.etape) || 0));
      var et = liste[n] || {};
      if (!tuto_.cle) { var a = document.activeElement; tuto_.retourFocus = a && a !== document.body ? a : null; }
      var cle = 'tuto:' + n + ':' + liste.length;
      var change = F.reconstruire(couche, cle, function () {
        var p = F.personne(et.personnage || 'ginette');
        var dernier = n === liste.length - 1;
        var action = typeof et.action === 'string' ? et.action : (et.action && et.action.texte) || null;
        return [
          h('div', { class: 'tuto-spot', 'aria-hidden': 'true' }),
          h('div', { class: 'tuto-bulle', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'tuto-titre', 'aria-describedby': 'tuto-texte',
            onkeydown: function (ev) { if (ev.key === 'Escape') { ev.preventDefault(); terminer(etat, app); } } },
            h('div', { class: 'tuto-tete' }, F.visage(p.id, 44),
              h('div', null, h('p', { class: 'eyebrow', id: 'tuto-titre' }, t('tuto.titre') + ' · ' + t('tuto.etape', { n: n + 1, total: liste.length })), h('b', null, p.nom))),
            h('p', { class: 'tuto-texte', id: 'tuto-texte' }, F.texte(et.texte, etat)),
            action ? h('p', { class: 'tuto-action' }, F.texte(action, etat)) : null,
            h('div', { class: 'tuto-points', 'aria-hidden': 'true' }, liste.map(function (x, i) { return h('i', { class: i === n ? 'is-on' : '' }); })),
            h('div', { class: 'tuto-actions' },
              h('button', { type: 'button', class: 'lien', 'data-cle': 'tuto-passer', onclick: function () { terminer(etat, app); } }, t('action.passerVisite')),
              h('span', { class: 'tuto-nav' },
                n > 0 ? h('button', { type: 'button', class: 'btn btn--petit btn--fantome', 'data-cle': 'tuto-prec', onclick: function () { app.setUI({ visite: { etape: n - 1 } }); } }, t('action.precedent')) : null,
                h('button', { type: 'button', class: 'btn btn--petit btn--primaire', id: 'btn-tuto-suivant', 'data-cle': 'tuto-suiv', onclick: function () {
                  if (dernier) terminer(etat, app); else app.setUI({ visite: { etape: n + 1 } });
                } }, dernier ? t('action.terminerVisite') : t('action.suivant'))))),
        ];
      });
      tuto_.cle = cle;
      couche._region = et.region;
      if (change) {
        var cible = cibleDe(et.region);
        if (cible && et.region !== 'hud') {
          try { cible.scrollIntoView({ block: 'nearest', behavior: F.reduit() ? 'auto' : 'smooth' }); } catch (e) { cible.scrollIntoView(); }
        }
        var boite = couche.querySelector('.tuto-bulle');
        if (boite && !boite.contains(document.activeElement)) F.focusPremier(boite, '#btn-tuto-suivant');
        LTP.ui.annoncer(F.texte(et.texte, etat));
        var debut = Date.now();
        var boucle = function () { placer(couche); if (Date.now() - debut < 900) requestAnimationFrame(boucle); };
        requestAnimationFrame(boucle);
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
