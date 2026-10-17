/* 40-ui-dom.js */
/* Fondations de l'interface : fabrication du DOM et du SVG, état de l'application (LTP.app),
   routage des écrans selon la phase, régions et planification du rendu.
   Chaque module d'interface enregistre une région (LTP.ui.regions[nom]) ou un écran (LTP.ui.ecrans[nom])
   et ne touche qu'à sa racine DOM. */
(function (LTP) {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';

  function ajouterEnfants(el, enfants) {
    for (var i = 0; i < enfants.length; i++) {
      var c = enfants[i];
      if (c == null || c === false || c === true) continue;
      if (Array.isArray(c)) { ajouterEnfants(el, c); continue; }
      el.appendChild(typeof c === 'object' && c.nodeType ? c : document.createTextNode(String(c)));
    }
  }

  function appliquerAttributs(el, attrs, estSvg) {
    if (!attrs) return;
    for (var k in attrs) {
      if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
      var v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'class' || k === 'className') { if (estSvg) el.setAttribute('class', v); else el.className = v; }
      else if (k === 'style' && typeof v === 'object') { for (var s in v) if (v[s] != null) el.style.setProperty(s, v[s]); }
      else if (k === 'html') el.innerHTML = v;
      else if (k.slice(0, 2) === 'on' && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    }
  }

  // h('div', { class: 'x', onclick: fn, 'data-dev': 'mathieu' }, 'texte', autreNoeud, [liste])
  function h(tag, attrs) {
    var el = document.createElement(tag);
    appliquerAttributs(el, attrs, false);
    ajouterEnfants(el, Array.prototype.slice.call(arguments, 2));
    return el;
  }
  function svg(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    appliquerAttributs(el, attrs, true);
    ajouterEnfants(el, Array.prototype.slice.call(arguments, 2));
    return el;
  }
  function vider(el) { while (el && el.firstChild) el.removeChild(el.firstChild); return el; }
  function remplacer(el) { vider(el); ajouterEnfants(el, Array.prototype.slice.call(arguments, 1)); return el; }

  function annoncer(texte) {
    var live = document.getElementById('live');
    if (!live) return;
    live.textContent = '';
    setTimeout(function () { live.textContent = texte; }, 30);
  }

  // Libellés utiles à plusieurs régions.
  function libelleLevier(id) { var d = LTP.q.defLevier(id); return d ? d.libelle : id; }
  function libelleValeur(id, v) {
    var d = LTP.q.defLevier(id);
    if (!d) return String(v);
    for (var i = 0; i < d.valeurs.length; i++) if (d.valeurs[i].v === v) return d.valeurs[i].libelle;
    return String(v);
  }
  function nomDev(id) { var d = LTP.q.defDev(id); return d ? d.nom : id; }
  function nomIlot(id) { var d = LTP.q.defIlot(id); return d ? d.court : id; }
  function texteJournal(e) {
    var vars = {};
    for (var k in e.vars) vars[k] = e.vars[k];
    if (vars.levier) { vars.valeurLibelle = libelleValeur(vars.levier, vars.valeur); vars.levier = libelleLevier(vars.levier) + (vars.valeur !== undefined ? ' → ' + vars.valeurLibelle : ''); }
    if (vars.ilot && typeof vars.ilot === 'string') vars.ilot = nomIlot(vars.ilot);
    if (e.ilot && vars.ilot == null) vars.ilot = nomIlot(e.ilot);
    if (vars.dev && typeof vars.dev === 'string') vars.dev = nomDev(vars.dev);
    if (e.dev && vars.dev == null) vars.dev = nomDev(e.dev);
    if (typeof vars.evenement === 'string' && vars.titre == null && LTP.q.defEvenement) {
      var defEv = LTP.q.defEvenement(vars.evenement);
      if (defEv) {
        var ref = {};
        if (e.ilot) ref.ilot = e.ilot;
        if (e.dev) ref.dev = e.dev;
        try { vars.titre = texteContenu(defEv.titre, LTP.app && LTP.app.etat, ref) || null; } catch (err) { vars.titre = null; }
      }
    }
    return LTP.t(e.cle, vars);
  }
  // Texte d'un champ de contenu qui peut être une chaîne ou une fonction (etat, q, ref).
  function texteContenu(val, etat, ref) {
    var s = typeof val === 'function' ? val(etat, LTP.q, ref || {}) : val;
    if (s == null) return '';
    var vars = {};
    if (ref && ref.ilot) vars.ilot = nomIlot(ref.ilot);
    if (ref && ref.dev) vars.dev = nomDev(ref.dev);
    return LTP.gabarit(String(s), vars);
  }

  LTP.ui = {
    h: h, svg: svg, vider: vider, remplacer: remplacer, annoncer: annoncer,
    libelleLevier: libelleLevier, libelleValeur: libelleValeur, nomDev: nomDev, nomIlot: nomIlot,
    texteJournal: texteJournal, texteContenu: texteContenu,
    regions: {},  // nom → { monter(racine, app), rendre(etat, ui, precedent, app) }
    ecrans: {},   // nom → { monter(racine, app), rendre(etat, ui, precedent, app) }
  };

  // Textes propres à l'interface (ajoutés ici pour ne pas toucher aux modules du moteur).
  //   : espace insécable (avant « : » et dans les guillemets) ;   : espace fine insécable.
  LTP.enregistrerTextes({
    'jeu.allerLeviers': 'Aller aux leviers',
    // Le journal nomme l'événement (29-evenements note { evenement: id } ; texteJournal ajoute { titre }).
    'evenements.tire': function (v) {
      return v && v.titre ? 'Événement : « ' + v.titre + ' »' : 'Un événement demande votre attention.';
    },
    'tableau.vs': function (v) {
      return v && v.n > 1 ? 'moyenne des ' + v.n + ' dernières semaines comparée aux ' + v.n + ' précédentes' : 'cette semaine comparée à la précédente';
    },
    'tuto.fait': 'C\'est fait.',
    'fin.totalAnnee': 'total de l\'année',
    'fin.finAnnee': 'en fin d\'année',
  });

  // --- Application ---
  LTP.K.ui = { dureeSemaineMs: 3000 };

  var montees = {};      // région → racine montée
  var ecranCourant = null;
  var rendreDemande = false;

  function ecranPour(phase) {
    if (phase === 'accueil') return 'accueil';
    if (phase === 'fin') return 'fin';
    return 'jeu';
  }

  function monterEcranJeu(racine) {
    var h_ = LTP.ui.h;
    // Lien d'évitement : l'open space compte une trentaine d'arrêts de tabulation avant les leviers.
    var evitement = h_('a', { class: 'lien-evitement', href: '#panneau-feuille',
      onclick: function (ev) {
        ev.preventDefault();
        if (LTP.ui.panneau && LTP.ui.panneau.ouvrir) LTP.ui.panneau.ouvrir();
        setTimeout(function () {
          var cible = document.querySelector('#panneau-feuille .onglet[aria-selected="true"]') || document.querySelector('#panneau-feuille .onglet');
          if (cible) { cible.focus(); try { cible.scrollIntoView({ block: 'nearest' }); } catch (e) { /* ignoré */ } }
        }, 40);
      } }, LTP.t('jeu.allerLeviers'));
    LTP.ui.remplacer(racine,
      evitement,
      h_('div', { class: 'jeu' },
        h_('section', { class: 'zone zone-openspace', 'data-region': 'openspace', 'aria-label': 'Open space' }),
        h_('aside', { class: 'zone zone-panneau', 'data-region': 'panneau', 'aria-label': 'Leviers et registre' }),
        h_('section', { class: 'zone zone-tableau', 'data-region': 'tableau', 'aria-label': 'Tableau de bord' }),
        h_('section', { class: 'zone zone-journal', 'data-region': 'journal', 'aria-label': 'Journal' })));
  }

  function rendreTout() {
    rendreDemande = false;
    var app = LTP.app, etat = app.etat;
    if (!etat) return;
    var racineApp = document.getElementById('app');
    var nomEcran = ecranPour(etat.phase);
    if (nomEcran !== ecranCourant) {
      ['openspace', 'panneau', 'tableau', 'journal'].forEach(function (r) { delete montees[r]; });
      ecranCourant = nomEcran;
      if (nomEcran === 'jeu') monterEcranJeu(racineApp);
      else {
        LTP.ui.vider(racineApp);
        var ec = LTP.ui.ecrans[nomEcran];
        if (ec && ec.monter) ec.monter(racineApp, app);
      }
    }
    if (nomEcran !== 'jeu') {
      var e2 = LTP.ui.ecrans[nomEcran];
      if (e2 && e2.rendre) e2.rendre(etat, app.ui, app.precedent, app);
    }
    // Régions fixes (hors écran) et régions de l'écran de jeu.
    var fixes = { hud: document.getElementById('hud'), modale: document.getElementById('modale-racine') };
    var noms = Object.keys(LTP.ui.regions);
    for (var i = 0; i < noms.length; i++) {
      var nom = noms[i], reg = LTP.ui.regions[nom];
      var racine = fixes[nom] || (nomEcran === 'jeu' ? racineApp.querySelector('[data-region="' + nom + '"]') : null);
      if (!racine) continue;
      if (montees[nom] !== racine) {
        montees[nom] = racine;
        if (reg.monter) reg.monter(racine, app);
      }
      if (reg.rendre) {
        try { reg.rendre(etat, app.ui, app.precedent, app); }
        catch (e) { if (typeof console !== 'undefined') console.error('Région ' + nom, e); }
      }
    }
  }

  function notifier() {
    if (rendreDemande) return;
    rendreDemande = true;
    (typeof requestAnimationFrame === 'function' ? requestAnimationFrame : function (f) { return setTimeout(f, 16); })(rendreTout);
  }

  function sauvegarder() {
    if (LTP.sauvegarde && LTP.sauvegarde.enregistrer) {
      try { LTP.sauvegarde.enregistrer(LTP.app.etat); } catch (e) { /* stockage indisponible */ }
    }
  }

  function remplacerEtat(nouvel, journal) {
    var app = LTP.app;
    var anciennePhase = app.etat && app.etat.phase;
    app.precedent = app.etat;
    app.etat = nouvel;
    app.dernierJournal = journal || [];
    if (nouvel.phase !== 'execution' && anciennePhase === 'execution') app.ui.vitesse = 0;
    sauvegarder();
    notifier();
  }

  LTP.app = {
    etat: null,
    precedent: null,
    dernierJournal: [],
    // vitesse : 0 (pause), 1, 2, 4 ; selection : { type: 'dev' | 'ilot', id } ; panneau : onglet ouvert
    // ilotEdition : îlot dont on édite les surcharges (null = réglages globaux)
    // modale : null | { type: 'fiche', id } | { type: 'ilot', id } | { type: 'aide', … }
    ui: { vitesse: 0, selection: null, panneau: 'pratiques', ilotEdition: null, modale: null, vue: 'normale' },
    demarrer: function (etat) { remplacerEtat(etat, []); },
    dispatch: function (action) {
      var r = LTP.moteur.action(LTP.app.etat, action);
      if (r.erreur) { notifier(); return r; }
      remplacerEtat(r.etat, r.journal);
      return r;
    },
    avancer: function () {
      var r = LTP.moteur.semaine(LTP.app.etat);
      if (!r.avance) return false;
      remplacerEtat(r.etat, r.journal);
      return true;
    },
    // Pour les tests et le mode animateur : joue n semaines en prenant le premier choix partout.
    simuler: function (n, strategie) {
      var r = LTP.moteur.simuler(LTP.app.etat, n, strategie);
      remplacerEtat(r.etat, r.journal);
      return r.etat;
    },
    nouvellePartie: function (graine, options) {
      LTP.app.ui.vitesse = 0;
      LTP.app.ui.selection = null;
      LTP.app.ui.modale = null;
      remplacerEtat(LTP.moteur.creer(graine == null ? Date.now() : graine, options), []);
    },
    setUI: function (patch) {
      for (var k in patch) LTP.app.ui[k] = patch[k];
      notifier();
    },
    notifier: notifier,
  };
})(globalThis.LTP = globalThis.LTP || {});
