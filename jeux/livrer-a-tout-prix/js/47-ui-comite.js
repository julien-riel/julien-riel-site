/* 47-ui-comite.js */
/* Comité trimestriel (couche de la région « modale », LTP.ui.comite.rendre) et bilan annuel (écran « fin »).
   Comité : questions de etat.comite.questions (contenu dans LTP.donnees.comites), une à la fois, réponse par
   app.dispatch({ type: 'comite', question, reponse }), réaction du personnage, confiance avant et après,
   puis passage au trimestre suivant. Bilan : profil de fin, étoiles, critères, courbes de l'année.
   Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, s = LTP.ui.svg, t = LTP.t, F = LTP.ui.F;
  var LETTRES = 'ABCDEF';

  var etat_ = { debut: {}, avant: {}, vue: {}, erreur: null, cle: null, retourFocus: null };

  function defQuestion(q) {
    var def = (LTP.donnees.comites || []).filter(function (c) { return c.id === q.id; })[0] || {};
    var res = {};
    var k;
    for (k in def) res[k] = def[k];
    for (k in q) if (q[k] != null) res[k] = q[k];
    return res;
  }

  function evaluer(fn, etat) {
    try { return typeof fn === 'function' ? fn(etat, LTP.q) : (typeof fn === 'number' ? fn : null); } catch (e) { return null; }
  }

  function reponseDonnee(etat, qid) {
    var r = etat.comite && etat.comite.reponses ? etat.comite.reponses[qid] : null;
    if (r == null) return null;
    if (typeof r === 'object') return { id: r.reponse || r.id, score: typeof r.score === 'number' ? r.score : null };
    return { id: r, score: null };
  }

  function scoreDe(etat, q, def, rep) {
    if (rep.score != null) return rep.score;
    var scores = etat.comite && etat.comite.scores;
    if (scores && typeof scores[q.id] === 'number') return scores[q.id];
    var r = (def.reponses || []).filter(function (x) { return x.id === rep.id; })[0];
    return r ? evaluer(r.evaluer, etat) : null;
  }

  function verdict(score) {
    if (score == null) return null;
    if (score > 0.3) return { cls: 'ok', texte: t('comite.bien') };
    if (score < -0.3) return { cls: 'crit', texte: t('comite.mal') };
    return { cls: 'warn', texte: t('comite.moyen') };
  }

  function texteReaction(r, etat, score, ref) {
    if (!r) return '';
    var v = r.reaction;
    try { if (typeof v === 'function') v = v(etat, LTP.q, score); } catch (e) { v = ''; }
    return F.texte(v, etat, ref);
  }

  function chiffreConfiance(avant, apres) {
    var d = Math.round(apres - avant);
    return h('span', { class: 'confiance-delta mono delta--' + (d > 0 ? 'ok' : d < 0 ? 'crit' : 'neutre') },
      t('comite.avantApres', { avant: Math.round(avant), apres: Math.round(apres) }),
      d ? ' (' + (d > 0 ? '+' : '−') + Math.abs(d) + ')' : '');
  }

  function blocQuestion(etat, q, i, total, app) {
    var def = defQuestion(q);
    var ref = q.ref || (def.ilot ? { ilot: def.ilot } : {});
    var perso = F.personne(def.personnage || 'lavigueur', ref);
    var rep = reponseDonnee(etat, q.id);
    var score = rep ? scoreDe(etat, q, def, rep) : null;
    var v = verdict(score);
    var repDef = rep ? (def.reponses || []).filter(function (x) { return x.id === rep.id; })[0] : null;
    var avant = etat_.avant[q.id];

    return h('div', { class: 'salle' + (rep ? ' is-repondue' : '') },
      h('p', { class: 'salle-mur mono', 'aria-hidden': 'true' }, t('comite.mur')),
      h('p', { class: 'eyebrow' }, t('comite.question', { n: i + 1, total: total })),
      h('div', { class: 'salle-question' },
        h('figure', { class: 'fauteuil' }, F.visage(perso.id, 72), h('figcaption', null, h('b', null, perso.nom), perso.role ? h('span', null, perso.role) : null)),
        h('p', { class: 'bulle bulle--grande', id: 'cq-' + q.id }, F.texte(def.question, etat, ref))),
      h('div', { class: 'reponses', role: 'group', 'aria-labelledby': 'cq-' + q.id },
        (def.reponses || []).map(function (r, j) {
          var choisie = rep && rep.id === r.id;
          return h('button', { type: 'button', class: 'reponse' + (choisie ? ' is-choisie' + (v ? ' is-' + v.cls : '') : ''), 'data-reponse': r.id, 'data-cle': 'rep-' + q.id + '-' + j,
            disabled: rep ? true : null, 'aria-pressed': choisie ? 'true' : null,
            onclick: function () {
              etat_.avant[q.id] = etat.org.confiance;
              var res = app.dispatch({ type: 'comite', question: q.id, reponse: r.id });
              if (res && res.erreur) { etat_.erreur = res.erreur; app.notifier(); return; }
              etat_.erreur = null;
              var ap = LTP.app.etat;
              var rr = reponseDonnee(ap, q.id);
              var sc = rr ? scoreDe(ap, q, def, rr) : null;
              var texte = texteReaction(r, ap, sc, ref);
              LTP.ui.annoncer(perso.nom + ' : ' + texte);
            } },
            h('span', { class: 'reponse-cle mono', 'aria-hidden': 'true' }, LETTRES.charAt(j)),
            h('span', null, F.texte(r.texte, etat, ref)));
        })),
      rep ? h('div', { class: 'reaction', role: 'status' },
        F.visage(perso.id, 40),
        h('div', null,
          h('p', { class: 'reaction-tete' }, h('b', null, perso.nom), v ? h('span', { class: 'verdict verdict--' + v.cls }, v.texte) : null),
          h('p', { class: 'bulle' }, texteReaction(repDef, etat, score, ref) || '…'),
          avant != null ? h('p', { class: 'reaction-confiance' }, t('comite.confiance'), ' : ', chiffreConfiance(avant, etat.org.confiance)) : null)) : null);
  }

  LTP.ui.comite = {
    rendre: function (couche, etat, ui, app) {
      if (etat.phase !== 'comite') {
        if (etat_.cle) {
          etat_.cle = null;
          var r = etat_.retourFocus; etat_.retourFocus = null;
          if (r && document.contains(r)) r.focus();
        }
        F.reconstruire(couche, 'aucun', function () { return null; });
        return;
      }
      var com = etat.comite || { trimestre: etat.trimestre, questions: [], reponses: {} };
      var trim = com.trimestre || etat.trimestre;
      if (etat_.debut[trim] == null) etat_.debut[trim] = etat.org.confiance;
      var questions = com.questions || [];
      if (etat_.vue[trim] == null) {
        var premiere = 0;
        while (premiere < questions.length && reponseDonnee(etat, questions[premiere].id)) premiere++;
        etat_.vue[trim] = premiere;
      }
      var idx = etat_.vue[trim];
      var toutes = questions.every(function (q) { return reponseDonnee(etat, q.id); });
      var sig = JSON.stringify([trim, com.reponses, com.scores, etat.org.confiance, idx, etat_.erreur, questions.map(function (q) { return q.id; })]);
      var nouveau = etat_.cle !== 'comite:' + trim;
      if (nouveau) {
        etat_.cle = 'comite:' + trim;
        var a = document.activeElement;
        etat_.retourFocus = a && a !== document.body ? a : null;
        LTP.ui.annoncer(t('comite.annonce'));
      }
      var change = F.reconstruire(couche, sig, function () {
        var corps;
        if (idx < questions.length) {
          var q = questions[idx];
          var repondue = !!reponseDonnee(etat, q.id);
          corps = [blocQuestion(etat, q, idx, questions.length, app),
            repondue ? h('div', { class: 'comite-actions' },
              h('button', { type: 'button', class: 'btn btn--primaire', 'data-cle': 'comite-suivante', id: 'btn-comite-suivante',
                onclick: function () { etat_.vue[trim] = idx + 1; app.notifier(); } },
                idx + 1 < questions.length ? t('comite.suivante') : t('comite.bilan'))) : null];
        } else {
          corps = [
            questions.length ? h('ol', { class: 'comite-resume' }, questions.map(function (q) {
              var def = defQuestion(q), rep = reponseDonnee(etat, q.id);
              var v = rep ? verdict(scoreDe(etat, q, def, rep)) : null;
              var p = F.personne(def.personnage || 'lavigueur', q.ref || {});
              return h('li', null, F.visage(p.id, 28), h('span', { class: 'resume-q' }, h('b', null, p.nom), ' · ', F.texte(def.question, etat, q.ref || {})),
                v ? h('span', { class: 'verdict verdict--' + v.cls }, v.texte) : null);
            })) : h('p', { class: 'comite-aucune' }, t('comite.aucune')),
            h('p', { class: 'comite-total' }, t('comite.confiance'), ' : ', chiffreConfiance(etat_.debut[trim], etat.org.confiance)),
            etat_.erreur ? h('p', { class: 'erreur', role: 'alert' }, F.erreur(etat_.erreur)) : null,
            h('div', { class: 'comite-actions' },
              h('button', { type: 'button', class: 'btn btn--primaire btn--grand', id: 'btn-trimestre-suivant', 'data-cle': 'comite-passer', disabled: toutes ? null : true,
                onclick: function () {
                  var r = app.dispatch({ type: 'phase', phase: 'planification' });
                  if (r && r.erreur) { etat_.erreur = r.erreur; app.notifier(); }
                  else etat_.erreur = null;
                } }, t('comite.passer', { trimestre: etat.trimestre + 1 }))),
          ];
        }
        return [
          h('div', { class: 'voile voile--comite', 'aria-hidden': 'true' }),
          h('div', { class: 'modale comite', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'comite-titre' },
            h('header', { class: 'comite-tete' },
              h('div', null,
                h('p', { class: 'eyebrow' }, t('comite.eyebrow', { trimestre: trim })),
                h('h2', { id: 'comite-titre' }, t('comite.titre'))),
              h('p', { class: 'comite-jauge' }, h('span', { class: 'eyebrow' }, t('comite.confiance')), h('b', { class: 'mono' }, String(Math.round(etat.org.confiance))))),
            idx === 0 && !reponseDonnee(etat, (questions[0] || {}).id) ? h('p', { class: 'comite-intro' }, t('comite.intro')) : null,
            questions.length > 1 ? h('ol', { class: 'comite-points', 'aria-hidden': 'true' }, questions.map(function (q, i) {
              return h('li', { class: (i === idx ? 'is-courante ' : '') + (reponseDonnee(etat, q.id) ? 'is-faite' : '') });
            })) : null,
            corps),
        ];
      });
      if (change) {
        var boite = couche.querySelector('.comite');
        var deja = boite && boite.contains(document.activeElement);
        if (boite && !deja) F.focusPremier(boite, '#btn-comite-suivante, .reponse:not([disabled]), #btn-trimestre-suivant');
      }
    },
  };

  // =====================================================================
  // Bilan annuel (écran « fin »)
  // =====================================================================
  function etoiles(n) {
    n = Math.max(0, Math.min(5, Math.round(n || 0)));
    var el = h('p', { class: 'etoiles', role: 'img', 'aria-label': t('fin.etoiles', { n: n }) });
    for (var i = 0; i < 5; i++) el.appendChild(h('span', { class: i < n ? 'is-on' : '', 'aria-hidden': 'true' }, '★'));
    return el;
  }

  // Courbe de l'année : une série, un axe, repères de trimestre, survol avec curseur.
  function courbe(etat, def) {
    var hist = (etat.historique || {})[def.id] || [];
    var pts = [];
    for (var i = 1; i < hist.length; i++) if (typeof hist[i] === 'number' && isFinite(hist[i])) pts.push({ semaine: i, v: hist[i] });
    if (pts.length < 2) return null;
    var L = 320, H = 132, G = 40, D = 10, HT = 12, B = 22;
    var min = Infinity, max = -Infinity;
    pts.forEach(function (p) { if (p.v < min) min = p.v; if (p.v > max) max = p.v; });
    if (def.unite === 'ratio') { min = Math.min(min, 0); max = Math.max(max, Math.min(1, max * 1.1)); }
    else min = Math.min(0, min);
    if (max - min < 1e-9) max = min + 1;
    var sMax = Math.max(52, pts[pts.length - 1].semaine);
    var x = function (w) { return G + (L - G - D) * (w / sMax); };
    var y = function (v) { return HT + (H - HT - B) * (1 - (v - min) / (max - min)); };
    var el = s('svg', { viewBox: '0 0 ' + L + ' ' + H, class: 'courbe-svg', role: 'img', 'aria-label': def.libelle });
    // grille recessive
    [min, max].forEach(function (v) {
      el.appendChild(s('line', { x1: G, x2: L - D, y1: y(v), y2: y(v), class: 'courbe-grille' }));
      el.appendChild(s('text', { x: G - 6, y: y(v) + 3.5, class: 'courbe-axe', 'text-anchor': 'end' }, F.formatMetrique(v, def.unite)));
    });
    [13, 26, 39, 52].forEach(function (w, k) {
      if (w > sMax) return;
      el.appendChild(s('line', { x1: x(w), x2: x(w), y1: HT, y2: H - B, class: 'courbe-trim' }));
      el.appendChild(s('text', { x: x(w - 6.5), y: H - 6, class: 'courbe-axe', 'text-anchor': 'middle' }, 'T' + (k + 1)));
    });
    var d = pts.map(function (p, j) { return (j ? 'L' : 'M') + x(p.semaine).toFixed(1) + ' ' + y(p.v).toFixed(1); }).join(' ');
    el.appendChild(s('path', { d: d, class: 'courbe-ligne' }));
    var dernier = pts[pts.length - 1];
    el.appendChild(s('circle', { cx: x(dernier.semaine), cy: y(dernier.v), r: 4, class: 'courbe-point' }));
    var curseurL = s('line', { y1: HT, y2: H - B, class: 'courbe-curseur', style: { display: 'none' } });
    var curseurP = s('circle', { r: 4, class: 'courbe-point courbe-point--survol', style: { display: 'none' } });
    el.appendChild(curseurL); el.appendChild(curseurP);
    var bulle = h('span', { class: 'courbe-bulle mono', 'aria-hidden': 'true', hidden: true });
    var zone = s('rect', { x: G, y: 0, width: L - G - D, height: H, fill: 'transparent' });
    el.appendChild(zone);
    el.addEventListener('pointermove', function (ev) {
      var r = el.getBoundingClientRect();
      var w = ((ev.clientX - r.left) / r.width * L - G) / (L - G - D) * sMax;
      var best = pts[0];
      pts.forEach(function (p) { if (Math.abs(p.semaine - w) < Math.abs(best.semaine - w)) best = p; });
      var cx = x(best.semaine), cy = y(best.v);
      curseurL.setAttribute('x1', cx); curseurL.setAttribute('x2', cx); curseurL.style.display = '';
      curseurP.setAttribute('cx', cx); curseurP.setAttribute('cy', cy); curseurP.style.display = '';
      bulle.hidden = false;
      bulle.textContent = 'S' + best.semaine + ' · ' + F.formatMetrique(best.v, def.unite);
      bulle.style.left = (cx / L * 100) + '%';
      bulle.style.top = (cy / H * 100) + '%';
    });
    el.addEventListener('pointerleave', function () { curseurL.style.display = 'none'; curseurP.style.display = 'none'; bulle.hidden = true; });
    var cache = !!def.cachee;
    return h('figure', { class: 'courbe' },
      h('figcaption', null, h('b', null, def.libelle), cache ? h('span', { class: 'badge badge--accent' }, t('fin.revele')) : null,
        h('span', { class: 'mono courbe-fin' }, F.formatMetrique(dernier.v, def.unite))),
      h('div', { class: 'courbe-cadre' }, el, bulle));
  }

  var COURBES = ['prFusionnees', 'incidents', 'comprehension', 'coutSemaine', 'couvertureReelle', 'satisfaction'];

  LTP.ui.ecrans.fin = {
    monter: function (racine) { this._racine = racine; racine._sig = null; },
    rendre: function (etat, ui, precedent, app) {
      var res = etat.resultat || { raison: 'annee', profil: null, etoiles: 0, score: 0, criteres: {} };
      var sig = JSON.stringify([etat.graine, etat.semaine, res]);
      var change = F.reconstruire(this._racine, sig, function () {
        var fins = LTP.donnees.fins || [];
        var profil = res.profil || (res.raison === 'confiance' ? 'congedie' : null);
        var fin = fins.filter(function (f) { return f.id === profil; })[0] || {};
        var texte = F.texte(fin.texte, etat);
        var replique = F.texte(fin.replique, etat);
        var lecon = F.texte(fin.lecon, etat);
        var criteres = res.criteres || {};
        var defs = LTP.donnees.metriquesParId || {};
        var courbes = COURBES.map(function (id) { return defs[id] ? courbe(etat, defs[id]) : null; }).filter(Boolean).slice(0, 4);
        return h('section', { class: 'bilan' },
          h('header', { class: 'bilan-tete' },
            h('p', { class: 'eyebrow' }, t('fin.eyebrow')),
            h('p', { class: 'bilan-raison' }, res.raison === 'confiance' ? t('fin.raison.confiance', { trimestre: etat.trimestre }) : t('fin.raison.annee'))),
          h('article', { class: 'profil-fin', 'aria-labelledby': 'profil-titre' },
            h('div', { class: 'profil-tampon', 'aria-hidden': 'true' }, 'GPU'),
            h('h1', { id: 'profil-titre', class: 'profil-titre' }, fin.titre || t('fin.profilInconnu')),
            fin.sousTitre ? h('p', { class: 'profil-sous' }, F.texte(fin.sousTitre, etat)) : null,
            etoiles(res.etoiles),
            res.score != null ? h('p', { class: 'profil-score mono' }, t('fin.score', { score: Math.round(res.score) })) : null,
            texte ? h('p', { class: 'profil-texte' }, texte) : null,
            replique ? h('figure', { class: 'profil-replique' }, F.visage('lavigueur', 56),
              h('blockquote', { class: 'bulle' }, h('p', null, replique), h('footer', null, (F.personne('lavigueur').nom)))) : null,
            lecon ? h('div', { class: 'profil-lecon' }, h('p', { class: 'eyebrow' }, t('fin.lecon')), F.postit(lecon)) : null),
          Object.keys(criteres).length ? h('section', { class: 'bilan-criteres', 'aria-labelledby': 'bc-t' },
            h('h2', { id: 'bc-t' }, t('fin.criteres')),
            Object.keys(criteres).map(function (k) {
              var v = criteres[k];
              return F.jauge(typeof v === 'number' ? v : 0, { libelle: LTP.aTexte('critere.' + k) ? t('critere.' + k) : k, sens: 'haut' });
            })) : null,
          courbes.length ? h('section', { class: 'bilan-courbes', 'aria-labelledby': 'bk-t' },
            h('h2', { id: 'bk-t' }, t('fin.courbes')), h('p', { class: 'muted' }, t('fin.courbesNote')),
            h('div', { class: 'courbes' }, courbes)) : null,
          h('div', { class: 'bilan-actions' },
            h('button', { type: 'button', class: 'btn btn--primaire btn--grand', id: 'btn-nouvelle', onclick: function () { app.nouvellePartie(); } }, t('action.recommencer')),
            h('button', { type: 'button', class: 'btn btn--fantome', onclick: function () {
              app.nouvellePartie(etat.graine, { serveurAchete: !!etat.options.serveurAchete, animateur: !!etat.options.animateur });
            } }, t('action.rejouerGraine'))));
      });
      if (change) {
        var titre = this._racine.querySelector('.profil-titre');
        if (titre) LTP.ui.annoncer(t('jeu.titre') + ' · ' + titre.textContent);
      }
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
