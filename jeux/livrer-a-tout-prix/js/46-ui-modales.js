/* 46-ui-modales.js */
/* Région « modale » (#modale-racine) : superpose plusieurs couches, chacune reconstruite seulement si elle change.
     - comité trimestriel (rendu par 47-ui-comite.js, LTP.ui.comite.rendre)
     - visite guidée (rendue par 48-ui-ecrans.js, LTP.ui.tutoriel.rendre)
     - carte d'événement (etat.evenementEnAttente) et fiche de développeur (ui.modale.type === 'fiche')
     - bandeau de conséquence après une décision (etat.derniereDecision)
   Échap et clic hors de la fiche la ferment ; la carte d'événement exige une décision. Piège du focus.
   Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, t = LTP.t, F = LTP.ui.F;

  var etat_ = {
    cleDialogue: null, retourFocus: null,
    decisionsVues: {}, decisionJoueur: null, minuterieToast: null, toastSurvol: false,
  };

  function nomTrait(id) { var tr = (LTP.donnees.traits || {})[id]; return tr && tr.libelle ? tr.libelle : id; }

  // --- Carte d'événement ---
  function carteEvenement(etat, app) {
    var att = etat.evenementEnAttente;
    var def = LTP.q.defEvenement(att.id) || { titre: t('evenement.inconnu'), texte: '', choix: [] };
    var ref = att.ref || {};
    var perso = F.personne(def.personnage || 'ginette', ref);
    var titre = F.texte(def.titre, etat, ref) || t('evenement.inconnu');
    var lieu = ref.ilot ? LTP.ui.nomIlot(ref.ilot) : null;
    var lettres = 'ABCDEF';
    return h('div', { class: 'modale carte-evenement', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'ev-titre', 'aria-describedby': 'ev-texte', 'data-evenement': att.id },
      h('div', { class: 'ev-bande', 'aria-hidden': 'true' }),
      h('header', { class: 'ev-tete' },
        h('p', { class: 'eyebrow' }, t('evenement.eyebrow', { semaine: etat.semaine }), lieu ? ' · ' + lieu : '', def.theme ? h('span', { class: 'ev-theme mono' }, def.theme) : null),
        h('h2', { id: 'ev-titre', class: 'ev-titre', tabindex: '-1' }, titre)),
      h('div', { class: 'ev-corps' },
        h('figure', { class: 'ev-perso' }, F.visage(perso.id, 64), h('figcaption', null, h('b', null, perso.nom), perso.role ? h('span', null, perso.role) : null)),
        h('p', { class: 'ev-texte bulle', id: 'ev-texte' }, F.texte(def.texte, etat, ref))),
      h('div', { class: 'ev-choix', role: 'group', 'aria-label': t('evenement.choix') },
        h('p', { class: 'eyebrow' }, t('evenement.choix')),
        (def.choix || []).map(function (c, i) {
          return h('button', { type: 'button', class: 'choix-ev', 'data-choix': c.id,
            onclick: function () {
              etat_.decisionJoueur = att.id + ':' + c.id + ':' + etat.semaine;
              var r = app.dispatch({ type: 'choix', evenement: att.id, choix: c.id });
              if (r && r.erreur) LTP.ui.annoncer(F.erreur(r.erreur));
            } },
            h('span', { class: 'choix-ev-cle mono', 'aria-hidden': 'true' }, lettres.charAt(i)),
            h('span', { class: 'choix-ev-lib' }, F.texte(c.libelle, etat, ref)));
        })));
  }

  // --- Fiche de développeur ---
  function fiche(etat, id, app) {
    var d = etat.devs[id], def = LTP.q.defDev(id) || { nom: id };
    if (!d) return null;
    var fermer = function () { app.setUI({ modale: null }); };
    var compVisible = LTP.q.metriqueVisible(etat, 'comprehension');
    var reactions = (def.reactions || []).filter(function (r) { return r && r.levier && LTP.q.defLevier(r.levier); });
    var der = d.derniere || {};
    return h('div', { class: 'modale fiche', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'fiche-nom' },
      h('button', { type: 'button', class: 'modale-fermer', 'aria-label': t('action.fermer'), onclick: fermer }, F.icone('fermer', 18)),
      h('header', { class: 'fiche-tete' },
        F.visage(id, 96, 'fiche-visage'),
        h('div', { class: 'fiche-id' },
          h('p', { class: 'eyebrow' }, t('fiche.eyebrow', { ilot: LTP.ui.nomIlot(d.ilot) })),
          h('h2', { id: 'fiche-nom' }, def.nom || id),
          h('p', { class: 'fiche-niveau' }, t('fiche.niveau.' + d.niveau), ' · ',
            h('span', { class: 'fiche-etat etat--' + d.etat }, LTP.aTexte('etat.' + d.etat) ? t('etat.' + d.etat) : d.etat)),
          h('ul', { class: 'traits', 'aria-label': t('fiche.traits') }, (d.traits || []).map(function (tr) { return h('li', { class: 'trait' }, nomTrait(tr)); })))),
      !d.present ? h('p', { class: 'fiche-parti' }, t('fiche.parti')) : null,
      def.tagline ? h('p', { class: 'fiche-tagline' }, def.tagline) : null,
      def.replique ? h('blockquote', { class: 'fiche-replique bulle' }, F.texte(def.replique, etat, { dev: id, ilot: d.ilot })) : null,
      h('section', { class: 'fiche-jauges', 'aria-labelledby': 'fiche-jauges-t' },
        h('h3', { id: 'fiche-jauges-t', class: 'eyebrow' }, t('fiche.jauges')),
        F.jauge(d.competence, { libelle: t('fiche.competence') }),
        F.jauge(d.competenceIA, { libelle: t('fiche.competenceIA') }),
        F.jauge(d.adhesion, { libelle: t('fiche.adhesion') }),
        F.jauge(d.rigueur, { libelle: t('fiche.rigueur') }),
        compVisible ? F.jauge(d.comprehension, { libelle: t('fiche.comprehension'), sens: 'haut' })
          : h('div', { class: 'jauge jauge--cachee' }, h('span', { class: 'jauge-lib' }, t('fiche.comprehension')), h('span', { class: 'jauge-piste jauge-piste--hachure', 'aria-hidden': 'true' }), h('span', { class: 'jauge-val' }, t('tableau.nonMesure'))),
        F.jauge(d.fatigue, { libelle: t('fiche.fatigue'), sens: 'bas' }),
        F.jauge(d.moral, { libelle: t('fiche.moral'), sens: 'haut' }),
        h('div', { class: 'fiche-sessions' }, h('span', { class: 'jauge-lib' }, t('fiche.sessions')),
          h('span', { class: 'ecrans', 'aria-label': d.sessions + ' ' + t('fiche.sessions') },
            [0, 1, 2, 3, 4, 5].map(function (i) { return h('i', { class: i < d.sessions ? 'is-on' : '', 'aria-hidden': 'true' }); })),
          h('span', { class: 'jauge-val mono' }, String(d.sessions)))),
      der.pr != null ? h('p', { class: 'fiche-semaine mono' }, t('fiche.semaine', { pr: LTP.format.nombre(der.pr || 0, 0), revues: LTP.format.nombre(der.revues || 0, 0), jetons: LTP.format.jetons(der.jetons || 0) })) : null,
      reactions.length ? h('section', { class: 'fiche-reactions' },
        h('h3', { class: 'eyebrow' }, t('fiche.reactions')),
        h('ul', null, reactions.map(function (r) {
          var aime = (r.moral || 0) >= 0;
          return h('li', { class: aime ? 'is-aime' : 'is-naime' },
            h('span', { class: 'reaction-signe', 'aria-hidden': 'true' }, aime ? '+' : '−'),
            h('span', null, h('b', null, aime ? t('fiche.aime') : t('fiche.naimePas')), ' : ', LTP.ui.libelleLevier(r.levier), ' (',
              (r.valeurs || []).map(function (v) { return LTP.ui.libelleValeur(r.levier, v); }).join(', '), ')',
              r.bulle ? h('em', null, ' « ' + F.texte(r.bulle, etat, { dev: id }) + ' »') : null));
        }))) : null);
  }

  // --- Bandeau de conséquence ---
  function toastDecision(etat, dec, app, fermer) {
    var def = LTP.q.defEvenement(dec.evenement) || {};
    var ref = dec.ref || {};
    var choix = (def.choix || []).filter(function (c) { return c.id === dec.choix; })[0] || {};
    var consequence = F.texte(choix.consequence, etat, ref);
    var perso = F.personne(def.personnage || 'ginette', ref);
    return h('div', { class: 'toast', role: 'status', 'data-decision': dec.evenement,
      onmouseenter: function () { etat_.toastSurvol = true; }, onmouseleave: function () { etat_.toastSurvol = false; },
      onfocusin: function () { etat_.toastSurvol = true; }, onfocusout: function () { etat_.toastSurvol = false; } },
      F.visage(perso.id, 36),
      h('div', { class: 'toast-corps' },
        h('p', { class: 'eyebrow' }, t('evenement.decision', { semaine: dec.semaine })),
        h('p', { class: 'toast-titre' }, F.texte(def.titre, etat, ref)),
        choix.libelle ? h('p', { class: 'toast-choix' }, t('evenement.vousAvez', { choix: F.texte(choix.libelle, etat, ref) })) : null,
        consequence ? h('p', { class: 'toast-consequence' }, consequence) : null),
      h('button', { type: 'button', class: 'toast-fermer', 'aria-label': t('action.fermer'), onclick: fermer }, F.icone('fermer', 16)));
  }

  // Rend le focus à l'élément déclencheur, sinon au bouton de lecture (ou « +1 semaine »).
  // Différé : le HUD est rendu après cette région, et le bouton de lecture n'est réactivé qu'à ce moment-là.
  function focusable(el) { return !!(el && document.contains(el) && !el.disabled && el.offsetParent !== null); }
  function focaliser(el) { try { el.focus({ preventScroll: true }); } catch (e) { el.focus(); } }
  function restaurerFocus() {
    var r = etat_.retourFocus;
    etat_.retourFocus = null;
    var essayer = function () {
      var a = document.activeElement;
      if (a && a !== document.body && document.contains(a)) return true;
      if (focusable(r)) { focaliser(r); if (document.activeElement === r) return true; }
      var b = document.getElementById('btn-lecture');
      if (!focusable(b)) b = document.getElementById('btn-semaine');
      if (focusable(b)) { focaliser(b); return true; }
      return false;
    };
    setTimeout(function () { if (!essayer()) setTimeout(essayer, 120); }, 30);
  }
  LTP.ui.restaurerFocus = function (el) { etat_.retourFocus = el || null; restaurerFocus(); };

  LTP.ui.regions.modale = {
    monter: function (racine, app) {
      var self = this;
      this._racine = racine;
      this._comite = h('div', { class: 'couche couche-comite' });
      this._tuto = h('div', { class: 'couche couche-tuto' });
      this._dialogue = h('div', { class: 'couche couche-dialogue' });
      this._toast = h('div', { class: 'couche couche-toast' });
      LTP.ui.remplacer(racine, this._comite, this._tuto, this._dialogue, this._toast);
      racine.addEventListener('keydown', function (ev) {
        var actif = self._dialogue.querySelector('[role="dialog"]') || self._comite.querySelector('[role="dialog"]') || self._tuto.querySelector('[role="dialog"]');
        if (ev.key === 'Escape') {
          if (self._dialogue.querySelector('.fiche')) { ev.preventDefault(); ev.stopPropagation(); LTP.app.setUI({ modale: null }); }
          return;
        }
        if (actif && actif.getAttribute('aria-modal') === 'true') F.pieger(actif, ev);
      });
    },

    rendre: function (etat, ui, precedent, app) {
      var self = this;
      // Une demande de modale « îlot » ouvre les réglages de l'îlot dans le panneau.
      if (ui.modale && ui.modale.type === 'ilot') {
        var idIlot = ui.modale.id;
        app.setUI({ modale: null, ilotEdition: etat.ilots[idIlot] ? idIlot : null });
        if (LTP.ui.panneau) LTP.ui.panneau.ouvrir();
        return;
      }

      // Couches déléguées
      if (LTP.ui.comite && LTP.ui.comite.rendre) {
        try { LTP.ui.comite.rendre(this._comite, etat, ui, app); } catch (e) { if (typeof console !== 'undefined') console.error('Comité', e); }
      }
      if (LTP.ui.tutoriel && LTP.ui.tutoriel.rendre) {
        try { LTP.ui.tutoriel.rendre(this._tuto, etat, ui, app); } catch (e) { if (typeof console !== 'undefined') console.error('Tutoriel', e); }
      }

      // Dialogue : événement prioritaire, sinon fiche.
      var att = etat.evenementEnAttente;
      var cle = null;
      if (att && etat.phase === 'execution') cle = 'ev:' + att.id + ':' + etat.semaine;
      else if (ui.modale && ui.modale.type === 'fiche' && etat.devs[ui.modale.id]) cle = 'fiche:' + ui.modale.id + ':' + etat.semaine;
      var avant = etat_.cleDialogue;
      if (cle !== avant) {
        if (cle && !avant) {
          var a = document.activeElement;
          etat_.retourFocus = a && a !== document.body && !this._racine.contains(a) ? a : null;
        }
        etat_.cleDialogue = cle;
        if (!cle) {
          LTP.ui.vider(this._dialogue);
          restaurerFocus();
        } else {
          var estEv = cle.indexOf('ev:') === 0;
          var contenu = estEv ? carteEvenement(etat, app) : fiche(etat, ui.modale.id, app);
          LTP.ui.remplacer(this._dialogue,
            h('div', { class: 'voile' + (estEv ? ' voile--evenement' : ''), 'aria-hidden': 'true',
              onclick: estEv ? null : function () { app.setUI({ modale: null }); } }),
            contenu);
          if (estEv) {
            var defEv = LTP.q.defEvenement(att.id);
            LTP.ui.annoncer(t('evenement.annonce', { titre: defEv ? F.texte(defEv.titre, etat, att.ref) : t('evenement.inconnu') }));
          }
          var boite = this._dialogue.querySelector('[role="dialog"]');
          // La carte d'événement focalise son titre : une Espace ou une Entrée ne choisit rien par accident.
          if (boite) F.focusPremier(boite, estEv ? '#ev-titre' : '.modale-fermer');
        }
      }

      // Bandeau de conséquence
      var dec = etat.derniereDecision;
      var cleDec = dec ? dec.evenement + ':' + dec.choix + ':' + dec.semaine : null;
      // Seules les décisions prises à la main ont un bandeau (pas celles des simulations du mode animateur).
      var montrer = dec && cleDec === etat_.decisionJoueur && !etat_.decisionsVues[cleDec] && !att && etat.phase !== 'fin';
      F.reconstruire(this._toast, montrer ? cleDec : '', function () {
        if (!montrer) return null;
        var fermer = function () { etat_.decisionsVues[cleDec] = true; clearTimeout(etat_.minuterieToast); app.notifier(); };
        var def = LTP.q.defEvenement(dec.evenement) || {};
        var choix = (def.choix || []).filter(function (c) { return c.id === dec.choix; })[0] || {};
        var texte = F.texte(choix.consequence, etat, dec.ref || {});
        if (texte) LTP.ui.annoncer(texte);
        clearTimeout(etat_.minuterieToast);
        var expirer = function () {
          if (etat_.toastSurvol) { etat_.minuterieToast = setTimeout(expirer, 2000); return; }
          fermer();
        };
        etat_.minuterieToast = setTimeout(expirer, 10000);
        return toastDecision(etat, dec, app, fermer);
      });

      // Le jeu reste en pause tant qu'une modale le bloque.
      var tutoActif = !!(LTP.ui.tutoriel && LTP.ui.tutoriel.actif && LTP.ui.tutoriel.actif(etat, ui));
      var bloquante = !!(cle || tutoActif || etat.phase === 'comite');
      if (!!ui.modaleBloquante !== bloquante) app.setUI({ modaleBloquante: bloquante });
      self._racine.classList.toggle('a-dialogue', !!cle);
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
