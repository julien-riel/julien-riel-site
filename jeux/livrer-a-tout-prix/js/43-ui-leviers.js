/* 43-ui-leviers.js */
/* Outils partagés par les modules d'interface de pilotage (LTP.ui.F) et panneau des leviers (région « panneau »).
   Le panneau génère ses onglets depuis LTP.donnees.panneaux et ses leviers depuis LTP.donnees.leviers ;
   l'onglet Registre est délégué à 45-ui-registre.js. Sur téléphone, le panneau devient une feuille du bas.
   Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, t = LTP.t;

  // =====================================================================
  // Outils partagés
  // =====================================================================
  var F = LTP.ui.F = LTP.ui.F || {};
  var FOCUSABLES = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
  F.FOCUSABLES = FOCUSABLES;

  F.reduit = function () {
    try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; }
  };
  F.etroit = function () {
    try { return window.matchMedia('(max-width: 900px)').matches; } catch (e) { return false; }
  };
  F.texte = function (val, etat, ref) {
    try { return LTP.ui.texteContenu(val, etat, ref); } catch (e) { return ''; }
  };
  F.echapper = function (s) { return (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/["\\]/g, '\\$&'); };

  // Reconstruit le contenu d'un conteneur seulement si la signature a changé,
  // en conservant l'élément qui a le focus (repéré par data-cle) et la position de défilement.
  F.reconstruire = function (conteneur, signature, fabrique) {
    if (!conteneur || conteneur._sig === signature) return false;
    conteneur._sig = signature;
    var actif = document.activeElement, cle = null;
    if (actif && actif !== conteneur && conteneur.contains(actif)) {
      var porteur = actif.closest('[data-cle]');
      if (porteur && conteneur.contains(porteur)) cle = porteur.getAttribute('data-cle');
    }
    var defil = [{ el: conteneur, y: conteneur.scrollTop }];
    var p = conteneur.parentElement;
    while (p && p !== document.body) { if (p.scrollTop) defil.push({ el: p, y: p.scrollTop }); p = p.parentElement; }
    var internes = {};
    conteneur.querySelectorAll('[data-defile]').forEach(function (el) { internes[el.getAttribute('data-defile')] = el.scrollTop; });
    LTP.ui.remplacer(conteneur, fabrique());
    conteneur.querySelectorAll('[data-defile]').forEach(function (el) {
      var y = internes[el.getAttribute('data-defile')];
      if (y) el.scrollTop = y;
    });
    defil.forEach(function (d) { if (d.el.scrollTop !== d.y) d.el.scrollTop = d.y; });
    if (cle != null) {
      var cible = conteneur.querySelector('[data-cle="' + F.echapper(cle) + '"]');
      if (cible) {
        var f = cible.matches(FOCUSABLES) ? cible : cible.querySelector(FOCUSABLES);
        if (f) { try { f.focus({ preventScroll: true }); } catch (e) { f.focus(); } }
      }
    }
    return true;
  };

  // Piège du focus dans un élément (à brancher sur keydown).
  F.pieger = function (el, ev) {
    if (ev.key !== 'Tab') return;
    var liste = Array.prototype.filter.call(el.querySelectorAll(FOCUSABLES), function (x) { return x.offsetParent !== null || x === document.activeElement; });
    if (!liste.length) { ev.preventDefault(); return; }
    var premier = liste[0], dernier = liste[liste.length - 1];
    if (ev.shiftKey && (document.activeElement === premier || !el.contains(document.activeElement))) { ev.preventDefault(); dernier.focus(); }
    else if (!ev.shiftKey && (document.activeElement === dernier || !el.contains(document.activeElement))) { ev.preventDefault(); premier.focus(); }
  };
  F.focusPremier = function (el, selecteur) {
    var c = (selecteur && el.querySelector(selecteur)) || el.querySelector(FOCUSABLES);
    if (c) { try { c.focus({ preventScroll: true }); } catch (e) { c.focus(); } }
  };

  // Visage d'un développeur ou d'un personnage (fourni par 41-ui-avatars.js).
  F.visage = function (id, taille, classe) {
    var el;
    try { el = LTP.ui.avatar ? LTP.ui.avatar(id, { taille: taille || 40 }) : null; } catch (e) { el = null; }
    if (!el) {
      var p = F.personne(id);
      el = h('span', { class: 'visage-secours', 'aria-hidden': 'true', style: { width: (taille || 40) + 'px', height: (taille || 40) + 'px' } }, (p.nom || '?').charAt(0));
    }
    return h('span', { class: 'visage' + (classe ? ' ' + classe : ''), 'aria-hidden': 'true' }, el);
  };

  // { id, nom, role } pour un développeur, un personnage secondaire ou une direction d'affaires.
  F.personne = function (id, ref) {
    if (!id) return { id: 'ginette', nom: 'Ginette', role: '' };
    var sec = (LTP.donnees.personnages || {})[id];
    if (sec) return { id: id, nom: sec.nom, role: sec.role || '' };
    var dev = LTP.q.defDev && LTP.q.defDev(id);
    if (dev) return { id: id, nom: dev.nom, role: LTP.ui.nomIlot(dev.ilot) };
    if (id === 'direction' || String(id).indexOf('direction') === 0) {
      var ilotId = (ref && ref.ilot) || String(id).split(':')[1];
      var il = ilotId && LTP.q.defIlot(ilotId);
      if (il && il.direction) return { id: 'direction-' + il.id, nom: il.direction.nom, role: il.direction.titre };
      return { id: 'direction', nom: t('comite.direction'), role: '' };
    }
    return { id: id, nom: String(id), role: '' };
  };

  // Mot de Ginette (ou d'un autre personnage) sur un post-it.
  F.postit = function (texte, opts) {
    opts = opts || {};
    var p = F.personne(opts.personnage || 'ginette');
    return h('div', { class: 'postit' + (opts.classe ? ' ' + opts.classe : ''), role: opts.role || null },
      F.visage(p.id, opts.taille || 30),
      h('p', null, h('b', null, p.nom), ' ', texte),
      opts.fermer ? h('button', { type: 'button', class: 'postit-fermer', 'aria-label': opts.fermerLibelle || t('action.fermer'), onclick: opts.fermer, 'data-cle': opts.cleFermer || null }, '×') : null);
  };

  // Jauge horizontale 0–1. opts : { libelle, sens: 'haut' | 'bas' | 'neutre', texte }
  F.jauge = function (valeur, opts) {
    opts = opts || {};
    var v = LTP.outils.borner(valeur);
    var ton = 'neutre';
    if (opts.sens === 'haut') ton = v < 0.3 ? 'crit' : v < 0.5 ? 'warn' : 'ok';
    else if (opts.sens === 'bas') ton = v > 0.7 ? 'crit' : v > 0.5 ? 'warn' : 'ok';
    var texte = opts.texte || LTP.format.pourcent(v);
    return h('div', { class: 'jauge jauge--' + ton },
      opts.libelle ? h('span', { class: 'jauge-lib' }, opts.libelle) : null,
      h('span', { class: 'jauge-piste', role: 'img', 'aria-label': (opts.libelle ? opts.libelle + ' : ' : '') + texte },
        h('span', { class: 'jauge-rempli', style: { width: Math.round(v * 100) + '%' } })),
      h('span', { class: 'jauge-val mono' }, texte));
  };

  // Trois voyants de la triade létale.
  F.voyants = function (triade, opts) {
    opts = opts || {};
    triade = triade || {};
    var branches = ['prive', 'nonFiable', 'sortie'];
    var n = branches.filter(function (b) { return triade[b]; }).length;
    var resume = t('triade.titre') + ' : ' + branches.map(function (b) { return t('triade.' + b) + ' ' + (triade[b] ? t('triade.allume') : t('triade.eteint')); }).join(', ');
    return h('div', { class: 'voyants' + (n === 3 ? ' is-complete' : '') + (opts.etiquettes ? ' voyants--etiquettes' : ''), role: 'img', 'aria-label': resume, title: opts.etiquettes ? null : resume },
      branches.map(function (b) {
        return h('span', { class: 'voyant' + (triade[b] ? ' is-on' : ''), 'data-branche': b },
          h('i', { 'aria-hidden': 'true' }), opts.etiquettes ? h('span', null, t('triade.' + b)) : null);
      }));
  };

  // D'où vient chaque branche allumée de la triade d'un îlot, et quand le joueur pourra agir.
  // Reprend la règle de 26-registre : exposition de l'îlot (nature du travail), branches des cartes équipées,
  // accès des agents à la production (données privées), approbation humaine des sorties (coupe la sortie).
  // → { branches: [{ b, allume, sources: [texte] }], lignes: [texte], quand: texte, resume: texte }
  var NATURE = {
    prive: 'le travail touche des données personnelles ou financières',
    nonFiable: 'les agents lisent du contenu venu de l\'extérieur',
    sortie: 'le travail publie ou envoie vers l\'extérieur',
  };
  F.originesTriade = function (etat, ilotId) {
    var il = etat.ilots[ilotId] || {}, def = LTP.q.defIlot(ilotId) || {};
    var expo = def.exposition || {}, triade = il.triade || {};
    var N = LTP.format.NBSP;
    var res = { branches: [], lignes: [] };
    var accesProd = LTP.q.levier(etat, 'accesProd', ilotId) === 'oui';
    var approbation = LTP.q.levier(etat, 'approbationSorties', ilotId) === true;
    ['prive', 'nonFiable', 'sortie'].forEach(function (b) {
      var sources = [];
      if (expo[b]) sources.push('nature du travail (' + NATURE[b] + ')');
      (il.capacites || []).forEach(function (cid) {
        var c = LTP.q.defCapacite ? LTP.q.defCapacite(cid) : null;
        if (c && (c.branches || []).indexOf(b) >= 0) sources.push('carte «' + N + (c.nom || cid) + N + '»');
      });
      if (b === 'prive' && accesProd) sources.push('accès des agents à la production');
      res.branches.push({ b: b, allume: !!triade[b], sources: sources });
      if (triade[b]) res.lignes.push(t('triade.' + b) + N + ': ' + (sources.length ? sources.join(', ') : 'réglages de l\'îlot') + '.');
      else if (b === 'sortie' && approbation && sources.length) res.lignes.push(t('triade.sortie') + N + ': coupée par l\'approbation humaine des sorties.');
    });
    var dAcces = LTP.q.defLevier('accesProd'), dSorties = LTP.q.defLevier('approbationSorties');
    var ouvert = LTP.q.levierDebloque(etat, 'approbationSorties');
    var nb = res.branches.filter(function (x) { return x.allume; }).length;
    if (!nb) res.quand = '';
    else if (!ouvert) {
      res.quand = 'Vous pourrez couper une branche à la semaine ' + ((dSorties && dSorties.semaine) || 30) + ' (onglet Données et sécurité)' + N + ': «' + N + (dSorties ? dSorties.libelle : 'approbation des sorties') + N + '» et «' + N + (dAcces ? dAcces.libelle : 'accès à la production') + N + '».' + (nb < 3 ? ' D\'ici là, évitez d\'équiper des cartes qui allument la branche manquante.' : '');
    } else {
      res.quand = 'Pour couper une branche' + N + ': l\'approbation humaine des sorties, l\'accès à la production en lecture seule, ou retirer une carte.';
    }
    res.resume = (nb === 3 ? t('triade.complete') : nb ? nb + ' branche' + (nb > 1 ? 's' : '') + ' sur 3 allumée' + (nb > 1 ? 's' : '') + '.' : t('triade.eteinte') + '.') +
      (res.lignes.length ? ' ' + res.lignes.join(' ') : '') + (res.quand ? ' ' + res.quand : '');
    return res;
  };

  // Valeur d'une métrique selon son unité.
  F.formatMetrique = function (v, unite) {
    var f = LTP.format, N = f.NBSP;
    if (v == null || typeof v !== 'number' || isNaN(v)) return '—';
    switch (unite) {
      case 'ratio': return f.pourcent(v);
      case '$': return f.argent(v);
      case 'jetons': return f.jetons(v);
      case 'semaines': return f.nombre(v, 1) + N + t('unite.semaines');
      case 'h': return f.nombre(v, v < 10 ? 1 : 0) + N + t('unite.h');
      case 'kLOC': return f.nombre(v, 0) + N + t('unite.kLOC');
      case 'score': return f.nombre(v, 0);
      default: return f.nombre(v, Math.abs(v) >= 10 || Math.round(v) === v ? 0 : 1);
    }
  };

  F.erreur = function (code) {
    var c = String(code || '');
    if (LTP.aTexte('erreur.' + c)) return t('erreur.' + c);
    return t('erreur.generique', { code: c.split(':')[0] });
  };

  // Groupe de choix accessible (radiogroup). opts :
  //   { cle, libelle, valeurs: [{ v, libelle, court?, marque?, marques?: [], marqueTexte?, titre?, desactive?, nomAccessible? }],
  //     courant, mode: 'segmente' | 'liste', desactive, onChoisir(v), idLibelle }
  // Une valeur desactive (pas encore permise) reste visible mais ne se choisit pas ; les flèches la sautent.
  F.choix = function (opts) {
    var boutons = [];
    function permise(i) { var x = opts.valeurs[i]; return !!x && !x.desactive; }
    function choisir(i, focus) {
      if (opts.desactive || !permise(i)) return;
      var x = opts.valeurs[i];
      if (focus && boutons[i]) boutons[i].focus();
      if (x && x.v !== opts.courant) opts.onChoisir(x.v);
    }
    // Prochaine valeur permise dans la direction pas (±1), à partir de i.
    function voisine(i, pas) {
      var n = opts.valeurs.length;
      for (var k = 1; k <= n; k++) { var j = ((i + pas * k) % n + n) % n; if (permise(j)) return j; }
      return null;
    }
    var groupe = h('div', {
      class: 'choix choix--' + (opts.mode || 'segmente') + (opts.desactive ? ' is-desactive' : ''),
      role: 'radiogroup', 'aria-labelledby': opts.idLibelle || null, 'aria-label': opts.idLibelle ? null : opts.libelle,
      'aria-disabled': opts.desactive ? 'true' : null,
    });
    var indexCourant = -1;
    opts.valeurs.forEach(function (x, i) { if (x.v === opts.courant) indexCourant = i; });
    var premierPermis = 0;
    for (var p = 0; p < opts.valeurs.length; p++) if (permise(p)) { premierPermis = p; break; }
    opts.valeurs.forEach(function (x, i) {
      var actif = i === indexCourant;
      var marques = (x.marques || []).concat(x.marque ? [x.marque] : []);
      var nomAcc = x.nomAccessible || (x.court && x.court !== x.libelle ? x.libelle : null);
      var b = h('button', {
        type: 'button', class: 'choix-opt' + (actif ? ' is-actif' : '') + marques.map(function (m) { return ' is-' + m; }).join(''),
        role: 'radio', 'aria-checked': actif ? 'true' : 'false',
        tabindex: (actif || (indexCourant < 0 && i === premierPermis)) ? '0' : '-1',
        'data-cle': opts.cle + ':' + i, 'data-valeur': String(x.v),
        'aria-label': nomAcc,
        'aria-disabled': x.desactive && !opts.desactive ? 'true' : null,
        disabled: opts.desactive ? true : null,
        title: x.titre || (x.court && x.court !== x.libelle ? x.libelle : null),
        onclick: function () { choisir(i, false); },
        onkeydown: function (ev) {
          var n = opts.valeurs.length, j = null;
          if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') j = voisine(i, 1);
          else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') j = voisine(i, -1);
          else if (ev.key === 'Home') j = permise(0) ? 0 : voisine(0, 1);
          else if (ev.key === 'End') j = permise(n - 1) ? n - 1 : voisine(n - 1, -1);
          if (j != null) { ev.preventDefault(); choisir(j, true); }
        },
      },
        opts.mode === 'liste' ? h('span', { class: 'choix-puce', 'aria-hidden': 'true' }) : null,
        h('span', { class: 'choix-lib' }, x.court || x.libelle),
        x.marqueTexte ? h('span', { class: 'choix-marque' }, x.marqueTexte) : null);
      boutons.push(b);
      groupe.appendChild(b);
    });
    return groupe;
  };

  // --- Réflexions des développeurs (33-reflexions) : texte, cause, moment ---
  F.texteReflexion = function (r, etat) {
    return F.texte(r && r.texte, etat, { dev: r && r.dev, ilot: r && r.ilot });
  };
  // Situation du contenu (09-donnees-reflexions) d'après le motif 'situation:<id>'.
  function situationDe(motif) {
    var id = String(motif || '').slice('situation:'.length);
    var l = (LTP.donnees.reflexions && LTP.donnees.reflexions.situations) || [];
    for (var i = 0; i < l.length; i++) if (l[i] && l[i].id === id) return l[i];
    return null;
  }
  // Ce qui a fait parler la personne, en une étiquette courte (null : l'activité du moment, rien à relier).
  //   levier ou réaction → « ↳ Réviseurs par PR : Deux » ; note de service → « ↳ Note de service » ;
  //   situation → son libellé (repli : rien).
  F.causeReflexion = function (r) {
    if (!r || !r.motif) return null;
    var N = LTP.format.NBSP;
    if ((r.motif === 'levier' || r.motif === 'reaction') && r.levier && LTP.q.defLevier(r.levier)) {
      return { type: 'levier', levier: r.levier, texte: '↳ ' + LTP.ui.libelleLevier(r.levier) + (r.valeur !== undefined ? N + ': ' + LTP.ui.libelleValeur(r.levier, r.valeur) : '') };
    }
    if (r.motif === 'nouveaute') return { type: 'nouveaute', levier: r.levier || null, texte: '↳ ' + t('echos.note') };
    if (String(r.motif).indexOf('situation:') === 0) {
      var s = situationDe(r.motif);
      var lib = s && s.libelle ? F.texte(s.libelle, LTP.app && LTP.app.etat) : '';
      return lib ? { type: 'situation', levier: s.levier || null, texte: '↳ ' + lib } : null;
    }
    return null;
  };
  // Rang d'importance d'une réflexion (réaction immédiate > levier, nouveauté, situation > activité).
  F.rangReflexion = function (r) {
    if (!r) return 0;
    if (r.immediat) return 4;
    if (r.motif === 'reaction' || r.motif === 'levier' || r.motif === 'nouveaute') return 3;
    if (String(r.motif || '').indexOf('situation:') === 0) return 2;
    return 1;
  };
  // Jour absolu (0 = lundi de la semaine 1) d'une semaine et d'un jour ouvrable.
  F.jourAbsolu = function (semaine, jour) { return (Math.max(1, semaine || 1) - 1) * 5 + (jour || 0); };
  F.aujourdhui = function (etat) { return F.jourAbsolu(LTP.q.semaineCourante(etat), etat.jour || 0); };
  // « Aujourd'hui », « Hier », « Avant-hier » ou « S5 · mar. ».
  F.quandReflexion = function (r, etat) {
    var ecart = F.aujourdhui(etat) - F.jourAbsolu(r.semaine, r.jour);
    if (ecart <= 0) return t('echos.aujourdhui');
    if (ecart === 1) return t('echos.hier');
    if (ecart === 2) return t('echos.avantHier');
    return t('journal.quand', { semaine: r.semaine, jour: LTP.ui.jourCourt(r.jour) });
  };
  // Une réflexion en ligne : visage, nom, îlot, texte entre guillemets et cause. opts : { taille, quand, classe }
  F.ligneReflexion = function (r, etat, opts) {
    opts = opts || {};
    var N = LTP.format.NBSP;
    var cause = F.causeReflexion(r);
    var nom = LTP.ui.nomDev(r.dev);
    var texte = F.texteReflexion(r, etat);
    return h('li', { class: 'echo echo--' + (r.ton || 'neutre') + (r.immediat ? ' is-immediat' : '') + (opts.classe ? ' ' + opts.classe : ''), 'data-echo-dev': r.dev },
      F.visage(r.dev, opts.taille || 32, 'echo-visage'),
      h('div', { class: 'echo-corps' },
        h('p', { class: 'echo-qui' }, h('b', null, nom), h('span', { class: 'echo-ilot' }, ' · ' + LTP.ui.nomIlot(r.ilot)),
          opts.quand ? h('span', { class: 'echo-quand mono' }, opts.quand) : null),
        h('p', { class: 'echo-texte' }, '«' + N + texte + N + '»'),
        cause ? h('p', { class: 'echo-cause echo-cause--' + cause.type }, cause.texte,
          r.immediat ? h('span', { class: 'echo-immediat' }, ' · ' + t('echos.immediat')) : null) : null));
  };

  // Petites icônes en SVG (traits de 1,6 px, héritent de currentColor).
  var ICONES = {
    info: 'M12 8v.01M11 11h1v6h1',
    alerte: 'M12 4 3 20h18L12 4zM12 10v4M12 17v.01',
    incident: 'M12 3c1 4 5 5 5 10a5 5 0 0 1-10 0c0-2 1-3 2-4 0 2 1 3 2 3 0-3-1-5 1-9z',
    evenement: 'M5 4h14v16H5zM8 8h8M8 12h8M8 16h5',
    levier: 'M5 7h14M5 17h14M9 4v6M15 14v6',
    personne: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21c1-4 4-6 8-6s7 2 8 6',
    comite: 'M4 20h16M6 20V10M18 20V10M12 20V10M3 10l9-6 9 6z',
    cadenas: 'M7 11V8a5 5 0 0 1 10 0v3M5 11h14v10H5z',
    aide: 'M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14M12 17v.01',
    jouer: 'M8 5v14l11-7z',
    pause: 'M8 5h3v14H8zM13 5h3v14h-3z',
    suivant: 'M6 5v14l9-7zM17 5v14',
    fermer: 'M6 6l12 12M18 6 6 18',
    fleche: 'M6 9l6 6 6-6',
    maison: 'M4 11 12 4l8 7M6 9.5V20h12V9.5M10 20v-5h4v5',
    animateur: 'M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3M7.5 14h9',
    reglage: 'M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M16 4v4M10 10v4M18 16v4',
  };
  F.icone = function (nom, taille) {
    var s = LTP.ui.svg, d = ICONES[nom] || ICONES.info;
    var el = s('svg', { viewBox: '0 0 24 24', width: taille || 16, height: taille || 16, class: 'icone icone--' + nom, 'aria-hidden': 'true', focusable: 'false' },
      s('path', { d: d, fill: nom === 'jouer' || nom === 'pause' || nom === 'suivant' ? 'currentColor' : 'none', stroke: 'currentColor', 'stroke-width': 1.7, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
    if (nom === 'info' || nom === 'aide') el.insertBefore(s('circle', { cx: 12, cy: 12, r: 9, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.7 }), el.firstChild);
    return el;
  };

  // =====================================================================
  // Panneau des leviers
  // =====================================================================
  var etat_ = {
    feuilleOuverte: false,
    indice: null,          // { id, valeur, texte } : dernier commentaire de Ginette
    erreur: null,          // { id, code }
    ouvertureVue: {},      // trimestre → true quand le mot d'ouverture a été masqué
    derniereSelection: null,
    eclaire: null,         // { id, jusqua, aFaire } : levier à montrer (bouton « Voir le levier » de la note de service)
  };
  // Une nouveauté reste « nouvelle » pendant la semaine où elle arrive et la suivante.
  var FENETRE_NOUVEAU = 2;
  function estNouveau(etat, semaine) {
    if (etat.options && etat.options.toutDebloque) return false;
    var sc = LTP.q.semaineCourante(etat);
    return semaine != null && semaine <= sc && sc - semaine < FENETRE_NOUVEAU;
  }
  function panneauxDebloques(etat) { return panneaux().filter(function (p) { return LTP.q.panneauDebloque(etat, p.id); }); }

  function panneaux() { return LTP.donnees.panneaux || []; }
  function leviersDu(panneauId) { return (LTP.donnees.leviers || []).filter(function (l) { return l.panneau === panneauId; }); }

  function modeDe(def) {
    var vals = def.valeurs || [];
    var total = vals.reduce(function (s, x) { return s + String(x.libelle || x.v).length; }, 0);
    if (vals.length <= 4 && total <= 40) return { mode: 'segmente', court: false };
    var numeriques = vals.length <= 6 && vals.every(function (x) { return typeof x.v === 'number' && String(x.libelle).indexOf(String(x.v)) >= 0; });
    if (numeriques && total > 40) return { mode: 'segmente', court: true };
    return { mode: 'liste', court: false };
  }

  function indiceDe(id, valeur) {
    var ind = LTP.donnees.indices && LTP.donnees.indices.leviers && LTP.donnees.indices.leviers[id];
    if (!ind) return null;
    var s = ind[String(valeur)];
    return s ? String(s) : null;
  }

  function surchargesDe(etat, id) {
    var res = [];
    var s = etat.leviers.surcharges || {};
    (etat.ordreIlots || []).forEach(function (ilotId) {
      if (s[ilotId] && Object.prototype.hasOwnProperty.call(s[ilotId], id)) res.push({ ilot: ilotId, valeur: s[ilotId][id] });
    });
    return res;
  }

  function estSurcharge(etat, id, ilotId) {
    var s = etat.leviers.surcharges && etat.leviers.surcharges[ilotId];
    return !!(s && Object.prototype.hasOwnProperty.call(s, id));
  }

  function fabriquerLevier(def, etat, ui, app) {
    var ilotEd = ui.ilotEdition || null;
    var debloque = LTP.q.levierDebloque(etat, def.id);
    var parIlot = !!(ilotEd && def.surchargeable);
    var ilotLecture = parIlot ? ilotEd : null;
    var courant = LTP.q.levier(etat, def.id, ilotLecture);
    var attente = LTP.q.enAttente ? LTP.q.enAttente(etat, def.id, ilotLecture) : null;
    var surcharge = parIlot && estSurcharge(etat, def.id, ilotEd);
    var desactive = !debloque || etat.phase === 'fin' || (ilotEd && !def.surchargeable);
    var m = modeDe(def);
    var idLib = 'lev-' + def.id;

    var N = LTP.format.NBSP;
    // Valeurs pas encore permises : visibles, désactivées, avec la semaine où elles arrivent.
    var nouvellesValeurs = [];
    var valeurs = (def.valeurs || []).map(function (x) {
      var o = { v: x.v, libelle: x.libelle || String(x.v), marques: [] };
      if (m.court) o.court = String(x.v);
      if (attente && attente.valeur === x.v) { o.marques.push('programme'); o.marqueTexte = m.mode === 'liste' ? '⏱' : null; }
      var sv = x.semaine || def.semaine || 1;
      if (!LTP.q.valeurDebloquee(etat, def.id, x.v)) {
        o.desactive = true;
        o.marques.push('verrou');
        o.marqueTexte = t('panneau.valeurVerrou', { semaine: sv });
        o.titre = t('panneau.valeurVerrouAide', { semaine: sv });
        o.nomAccessible = o.libelle + ' (' + o.titre.toLowerCase() + ')';
      } else if (sv > (def.semaine || 1) && estNouveau(etat, sv)) {
        o.marques.push('nouveau');
        o.nomAccessible = o.libelle + ' (' + t('panneau.nouveau').toLowerCase() + ')';
        nouvellesValeurs.push({ semaine: sv, libelle: o.libelle });
      }
      return o;
    });

    var badges = [];
    if (debloque && estNouveau(etat, def.semaine || 1)) {
      badges.push(h('span', { class: 'badge badge--nouveau', title: t('panneau.nouveauAide', { semaine: def.semaine || 1 }) }, t('panneau.nouveau')));
    } else if (nouvellesValeurs.length) {
      badges.push(h('span', { class: 'badge badge--nouveau', title: t('panneau.nouveauValeurs', { semaine: nouvellesValeurs[0].semaine,
        valeurs: nouvellesValeurs.map(function (x) { return x.libelle; }).join(', ') }) }, t('panneau.nouveau')));
    }
    if (def.surchargeable) badges.push(h('span', { class: 'badge badge--losange', title: t('panneau.porteeAide') }, '◆'));
    if (surcharge) badges.push(h('span', { class: 'badge badge--accent', title: t('panneau.surchargeAide') }, t('panneau.surcharge')));

    var pied = [];
    if (debloque && def.delai > 0 && !attente) pied.push(h('p', { class: 'levier-note mono' }, t('panneau.delai', { n: def.delai })));
    if (attente) {
      var n = Math.max(0, (attente.semaine || 0) - etat.semaine);
      pied.push(h('p', { class: 'levier-attente' }, F.icone('levier', 14), ' ',
        t('panneau.programmeDans', { valeur: LTP.ui.libelleValeur(def.id, attente.valeur), n: n })));
    }
    if (ilotEd && !def.surchargeable && debloque) {
      pied.push(h('p', { class: 'levier-note' }, t('panneau.globalSeulement'), ' · ',
        h('button', { type: 'button', class: 'lien', 'data-cle': 'glob-' + def.id, onclick: function () { app.setUI({ ilotEdition: null }); } }, t('panneau.global'))));
    }
    if (!ilotEd && def.surchargeable) {
      var sur = surchargesDe(etat, def.id);
      if (sur.length) {
        pied.push(h('p', { class: 'levier-note' }, t('panneau.surchargesDans', {
          liste: sur.map(function (x) { return LTP.ui.nomIlot(x.ilot) + ' (' + LTP.ui.libelleValeur(def.id, x.valeur) + ')'; }).join(', '),
        })));
      }
    }
    if (surcharge) {
      pied.push(h('button', { type: 'button', class: 'btn btn--petit btn--fantome', 'data-cle': 'retour-' + def.id,
        onclick: function () {
          var r = app.dispatch({ type: 'retirerSurcharge', id: def.id, ilot: ilotEd });
          if (r && r.erreur) etat_.erreur = { id: def.id, code: r.erreur };
        } }, t('panneau.retourGlobal', { valeur: LTP.ui.libelleValeur(def.id, LTP.q.levier(etat, def.id, null)) })));
    }
    if (etat_.erreur && etat_.erreur.id === def.id) pied.push(h('p', { class: 'erreur', role: 'alert' }, F.erreur(etat_.erreur.code)));
    if (etat_.indice && etat_.indice.id === def.id) {
      pied.push(F.postit(etat_.indice.texte, { classe: 'postit--levier', fermer: function () { etat_.indice = null; app.notifier(); }, cleFermer: 'indice-' + def.id }));
    }

    var eclaire = etat_.eclaire && etat_.eclaire.id === def.id && Date.now() < etat_.eclaire.jusqua;
    return h('div', { class: 'levier' + (surcharge ? ' is-surcharge' : '') + (!debloque ? ' is-verrou' : '') + (desactive ? ' is-inactif' : '') + (eclaire ? ' is-eclaire' : ''),
      'data-levier': def.id, role: 'group', 'aria-labelledby': idLib },
      h('div', { class: 'levier-tete' }, h('span', { class: 'levier-nom', id: idLib }, def.libelle || def.id), badges.length ? h('span', { class: 'levier-badges' }, badges) : null),
      def.aide ? h('p', { class: 'levier-aide' }, def.aide) : null,
      F.choix({
        cle: 'lev-' + def.id, idLibelle: idLib, valeurs: valeurs, courant: courant, mode: m.mode, desactive: desactive,
        onChoisir: function (v) {
          var action = { type: 'levier', id: def.id, valeur: v };
          if (parIlot) action.ilot = ilotEd;
          etat_.erreur = null;
          var texte = indiceDe(def.id, v);
          etat_.indice = texte ? { id: def.id, valeur: v, texte: texte } : null;
          var r = app.dispatch(action);
          if (r && r.erreur) { etat_.erreur = { id: def.id, code: r.erreur }; etat_.indice = null; }
          else if (texte) LTP.ui.annoncer('Ginette : ' + texte);
        },
      }),
      m.court ? h('p', { class: 'levier-courant' }, LTP.ui.libelleValeur(def.id, courant)) : null,
      pied);
  }

  // Repli : un panneau pas encore ouvert (son onglet n'est normalement pas affiché).
  function fabriquerVerrou(p, etat) {
    var prochains = leviersDu(p.id);
    return h('div', { class: 'panneau-verrou' },
      h('div', { class: 'verrou-icone' }, F.icone('cadenas', 28)),
      h('h3', null, t('panneau.verrouTitre', { semaine: p.semaine || 1 })),
      p.id === 'registre' ? h('p', null, t('registre.verrou')) : h('p', null, t('panneau.verrouTexte')),
      prochains.length ? h('ul', { class: 'verrou-liste' }, prochains.map(function (l) { return h('li', null, l.libelle); })) : null);
  }

  function fabriquerOuverture(etat, app) {
    var n = etat.trimestre;
    if (etat_.ouvertureVue[n]) return null;
    if (etat.phase !== 'planification' && etat.phase !== 'execution') return null;
    // Les nouveautés arrivent maintenant une à une par notes de service : le mot d'ouverture ne les énumère plus.
    var texte = LTP.donnees.indices && LTP.donnees.indices.trimestres && LTP.donnees.indices.trimestres[n];
    if (!texte) return null;
    return h('div', { class: 'ouverture', role: 'note' },
      h('p', { class: 'eyebrow' }, t('panneau.ouverture', { trimestre: n })),
      F.postit(F.texte(texte, etat), {
        fermer: function () { etat_.ouvertureVue[n] = true; app.notifier(); }, fermerLibelle: t('panneau.ouvertureFermer'), cleFermer: 'ouverture',
      }));
  }

  // Ligne discrète au bas du panneau : la prochaine note de service du calendrier.
  function fabriquerProchaine(etat) {
    if (etat.options && etat.options.toutDebloque) return h('p', { class: 'prochaine' }, t('panneau.toutDebloque'));
    if (etat.phase === 'fin') return null;
    var p = LTP.q.prochainDeblocage(etat);
    return h('p', { class: 'prochaine' }, F.icone('evenement', 14), ' ',
      p ? t('panneau.prochaine', { titre: p.titre || '', semaine: p.semaine }) : t('panneau.prochaineAucune'));
  }

  function signatureLeviers(etat, ui, pid) {
    var ilotEd = ui.ilotEdition || '';
    var vals = leviersDu(pid).map(function (l) {
      var a = LTP.q.enAttente ? LTP.q.enAttente(etat, l.id, l.surchargeable && ilotEd ? ilotEd : null) : null;
      return [LTP.q.levier(etat, l.id, l.surchargeable && ilotEd ? ilotEd : null), a ? a.valeur + '@' + (a.semaine - etat.semaine) : '',
        JSON.stringify(surchargesDe(etat, l.id)), LTP.q.levierDebloque(etat, l.id)];
    });
    var ecl = etat_.eclaire && Date.now() < etat_.eclaire.jusqua ? etat_.eclaire.id : '';
    return JSON.stringify([pid, ilotEd, LTP.q.semaineCourante(etat), !!etat.options.toutDebloque, etat.phase === 'fin', vals, etat_.indice, etat_.erreur, ecl]);
  }

  LTP.ui.panneau = {
    ouvrir: function () { etat_.feuilleOuverte = true; LTP.app.notifier(); },
    fermer: function () { etat_.feuilleOuverte = false; LTP.app.notifier(); },
    // Ouvre le panneau et l'onglet d'un levier, le fait défiler jusqu'à lui et le met en évidence un instant.
    montrerLevier: function (id) {
      var def = LTP.q.defLevier(id);
      if (!def) return;
      etat_.eclaire = { id: id, jusqua: Date.now() + 2800, aFaire: true };
      etat_.feuilleOuverte = true;
      LTP.app.setUI({ panneau: def.panneau, ilotEdition: null });
    },
    // Ouvre un onglet (le registre, par exemple) sans viser de levier.
    montrerPanneau: function (pid) {
      etat_.feuilleOuverte = true;
      LTP.app.setUI({ panneau: pid, ilotEdition: null });
      setTimeout(function () {
        var o = document.querySelector('#panneau-feuille .onglet[data-panneau="' + F.echapper(pid) + '"]');
        if (o) { try { o.focus({ preventScroll: true }); } catch (e) { o.focus(); } }
      }, F.etroit() ? 320 : 60);
    },
  };

  // Après le rendu : défile jusqu'au levier demandé, le met en évidence et y met le focus.
  function eclairer(conteneur) {
    var e = etat_.eclaire;
    if (!e || !e.aFaire) return;
    var el = conteneur.querySelector('[data-levier="' + F.echapper(e.id) + '"]');
    if (!el) return;
    e.aFaire = false;
    el.classList.add('is-eclaire');
    setTimeout(function () {
      try { el.scrollIntoView({ block: F.etroit() ? 'center' : 'nearest', behavior: F.reduit() ? 'auto' : 'smooth' }); } catch (err) { el.scrollIntoView(); }
      var f = el.querySelector('[aria-checked="true"]') || el.querySelector(FOCUSABLES);
      if (f) { try { f.focus({ preventScroll: true }); } catch (err) { f.focus(); } }
    }, F.etroit() ? 320 : 60);
    var duree = Math.max(0, e.jusqua - Date.now());
    setTimeout(function () {
      if (etat_.eclaire === e) etat_.eclaire = null;
      var cible = document.querySelector('#panneau-feuille [data-levier="' + F.echapper(e.id) + '"]');
      if (cible) cible.classList.remove('is-eclaire');
    }, duree);
  }

  LTP.ui.regions.panneau = {
    monter: function (racine, app) {
      var self = this;
      this._racine = racine;
      this._poignee = h('button', { type: 'button', class: 'panneau-poignee', 'aria-expanded': 'false', 'aria-controls': 'panneau-feuille',
        onclick: function () { etat_.feuilleOuverte = !etat_.feuilleOuverte; app.notifier(); } },
        h('span', { class: 'poignee-barre', 'aria-hidden': 'true' }),
        h('span', { class: 'poignee-lib' }, F.icone('reglage', 18), ' ', h('span', { class: 'poignee-texte' }, t('panneau.poignee'))),
        h('span', { class: 'poignee-fleche', 'aria-hidden': 'true' }, F.icone('fleche', 18)));
      this._onglets = h('div', { class: 'onglets', role: 'tablist', 'aria-label': t('panneau.onglets') });
      // Au téléphone, les onglets défilent horizontalement : un fondu signale ce qui dépasse de chaque côté.
      var onglets = this._onglets;
      this._majDebord = function () {
        var max = onglets.scrollWidth - onglets.clientWidth;
        onglets.classList.toggle('deborde-g', max > 2 && onglets.scrollLeft > 2);
        onglets.classList.toggle('deborde-d', max > 2 && onglets.scrollLeft < max - 2);
      };
      onglets.addEventListener('scroll', this._majDebord, { passive: true });
      window.addEventListener('resize', this._majDebord);
      this._select = h('select', { class: 'portee-select', id: 'portee-ilot', onchange: function (ev) {
        app.setUI({ ilotEdition: ev.target.value || null });
      } });
      this._porteeNote = h('p', { class: 'portee-note' });
      this._portee = h('div', { class: 'portee' },
        h('label', { for: 'portee-ilot', class: 'portee-lib' }, t('panneau.portee')),
        this._select, this._porteeNote);
      this._triade = h('div', { class: 'triade-explic-zone' });
      this._ouverture = h('div', { class: 'ouverture-zone' });
      this._leviers = h('div', { class: 'panneau-leviers', id: 'panneau-contenu', role: 'tabpanel' });
      this._registre = h('div', { class: 'panneau-registre', role: 'tabpanel', hidden: true });
      this._prochaine = h('div', { class: 'prochaine-zone' });
      this._corps = h('div', { class: 'panneau-corps' }, this._triade, this._ouverture, this._leviers, this._registre, this._prochaine);
      this._feuille = h('div', { class: 'panneau', id: 'panneau-feuille', 'data-ouvert': 'false' },
        this._poignee,
        h('div', { class: 'panneau-tete' }, h('h2', { class: 'panneau-titre' }, t('panneau.titre')), this._onglets, this._portee),
        this._corps);
      LTP.ui.remplacer(racine, this._feuille);
      document.addEventListener('keydown', function (ev) {
        if (ev.key === 'Escape' && etat_.feuilleOuverte && F.etroit() && !LTP.app.ui.modale && self._feuille.contains(document.activeElement)) {
          etat_.feuilleOuverte = false; self._poignee.focus(); LTP.app.notifier();
        }
      });
    },

    rendre: function (etat, ui, precedent, app) {
      var self = this;
      // Un clic sur un îlot de l'open space ouvre ses réglages.
      var sel = ui.selection;
      var cleSel = sel ? sel.type + ':' + sel.id : null;
      if (cleSel !== etat_.derniereSelection) {
        etat_.derniereSelection = cleSel;
        if (sel && sel.type === 'ilot' && etat.ilots[sel.id] && ui.ilotEdition !== sel.id) {
          etat_.feuilleOuverte = true;
          app.setUI({ ilotEdition: sel.id });
          return;
        }
      }

      // Seuls les panneaux débloqués ont un onglet ; un onglet pas encore ouvert bascule sur le premier ouvert.
      var ouverts = panneauxDebloques(etat);
      if (!ouverts.length) ouverts = panneaux().slice(0, 1);
      var pid = ui.panneau || 'ia';
      var p = ouverts.filter(function (x) { return x.id === pid; })[0] || ouverts[0];
      if (!p) return;
      pid = p.id;
      var debloque = LTP.q.panneauDebloque(etat, pid);

      // Feuille du bas (téléphone)
      this._feuille.setAttribute('data-ouvert', etat_.feuilleOuverte ? 'true' : 'false');
      this._poignee.setAttribute('aria-expanded', etat_.feuilleOuverte ? 'true' : 'false');
      this._poignee.setAttribute('aria-label', etat_.feuilleOuverte ? t('panneau.replier') : t('panneau.ouvrir'));
      var texteP = this._poignee.querySelector('.poignee-texte');
      var resumeP = p.libelle + (ui.ilotEdition ? ' · ' + LTP.ui.nomIlot(ui.ilotEdition) : '');
      if (texteP.textContent !== resumeP) texteP.textContent = resumeP;

      // Onglets
      F.reconstruire(this._onglets, JSON.stringify([pid, ouverts.map(function (x) { return x.id; }), LTP.q.semaineCourante(etat), etat.options.toutDebloque]), function () {
        var liste = ouverts;
        return liste.map(function (x, i) {
          var actif = x.id === pid;
          var nouveau = estNouveau(etat, x.semaine || 1) && (x.semaine || 1) > 1;
          return h('button', {
            type: 'button', role: 'tab', class: 'onglet' + (actif ? ' is-actif' : '') + (nouveau ? ' is-nouveau' : ''),
            'aria-selected': actif ? 'true' : 'false', tabindex: actif ? '0' : '-1', 'aria-controls': x.id === 'registre' ? null : 'panneau-contenu',
            'data-cle': 'onglet-' + x.id, 'data-panneau': x.id,
            title: nouveau ? t('panneau.nouveauAide', { semaine: x.semaine }) : null,
            onclick: function () { app.setUI({ panneau: x.id }); },
            onkeydown: function (ev) {
              var j = null;
              if (ev.key === 'ArrowRight') j = (i + 1) % liste.length;
              else if (ev.key === 'ArrowLeft') j = (i - 1 + liste.length) % liste.length;
              if (j != null) {
                ev.preventDefault();
                app.setUI({ panneau: liste[j].id });
                var b = self._onglets.querySelector('[data-panneau="' + liste[j].id + '"]');
                if (b) b.focus();
              }
            },
          }, x.libelle, nouveau ? h('span', { class: 'onglet-nouveau', 'aria-hidden': 'true' }) : null, nouveau ? h('span', { class: 'sr-only' }, ' (' + t('panneau.nouveau').toLowerCase() + ')') : null);
        });
      });

      var actifO = this._onglets.querySelector('.onglet.is-actif');
      if (actifO && this._ongletVu !== pid && F.etroit()) {
        this._ongletVu = pid;
        var o = this._onglets, g = actifO.offsetLeft, d = g + actifO.offsetWidth;
        if (g < o.scrollLeft) o.scrollLeft = g - 24;
        else if (d > o.scrollLeft + o.clientWidth) o.scrollLeft = d - o.clientWidth + 24;
      }
      this._majDebord();

      // Portée : globale ou par îlot
      var options = [''].concat(etat.ordreIlots || []);
      if (this._select.options.length !== options.length) {
        LTP.ui.remplacer(this._select, options.map(function (id) {
          return h('option', { value: id }, id ? LTP.ui.nomIlot(id) : t('panneau.global'));
        }));
      }
      var valSel = ui.ilotEdition || '';
      if (this._select.value !== valSel) this._select.value = valSel;
      var note = pid === 'registre' ? (ui.ilotEdition ? '' : t('registre.choisirIlot'))
        : (ui.ilotEdition ? t('panneau.porteeIlot', { ilot: LTP.ui.nomIlot(ui.ilotEdition) }) : t('panneau.porteeAide'));
      if (this._porteeNote.textContent !== note) this._porteeNote.textContent = note;
      this._portee.classList.toggle('is-ilot', !!ui.ilotEdition);
      this._portee.hidden = !debloque;

      // Triade de l'îlot réglé : d'où viennent les branches allumées, et quand on pourra agir.
      var ilotT = ui.ilotEdition && etat.ilots[ui.ilotEdition] && pid !== 'registre' ? ui.ilotEdition : null;
      F.reconstruire(this._triade, ilotT ? JSON.stringify([ilotT, etat.ilots[ilotT].triade, etat.ilots[ilotT].capacites, etat.trimestre,
        LTP.q.levier(etat, 'accesProd', ilotT), LTP.q.levier(etat, 'approbationSorties', ilotT)]) : '', function () {
        if (!ilotT) return null;
        var o = F.originesTriade(etat, ilotT);
        var nb = o.branches.filter(function (x) { return x.allume; }).length;
        return h('section', { class: 'triade-explic' + (nb === 3 ? ' is-complete' : ''), 'aria-labelledby': 'triade-explic-t' },
          h('div', { class: 'triade-explic-tete' },
            h('p', { class: 'eyebrow', id: 'triade-explic-t' }, t('triade.titre') + ' · ' + LTP.ui.nomIlot(ilotT)),
            F.voyants(etat.ilots[ilotT].triade)),
          o.lignes.length ? h('ul', { class: 'triade-explic-lignes' }, o.lignes.map(function (l) { return h('li', null, l); }))
            : h('p', { class: 'muted petit' }, t('triade.eteinte') + '.'),
          o.quand ? h('p', { class: 'triade-explic-quand' }, o.quand) : null);
      });

      // Mot d'ouverture du trimestre
      F.reconstruire(this._ouverture, JSON.stringify([etat.trimestre, etat.phase, !!etat_.ouvertureVue[etat.trimestre]]), function () {
        return fabriquerOuverture(etat, app);
      });

      // Prochaine nouveauté du calendrier
      var proch = LTP.q.prochainDeblocage(etat);
      F.reconstruire(this._prochaine, JSON.stringify([proch && proch.semaine, !!etat.options.toutDebloque, etat.phase === 'fin']), function () {
        return fabriquerProchaine(etat);
      });

      // Contenu
      var estRegistre = pid === 'registre' && debloque;
      this._registre.hidden = !estRegistre;
      this._leviers.hidden = estRegistre;
      if (estRegistre) {
        if (LTP.ui.registre && LTP.ui.registre.rendre) {
          try { LTP.ui.registre.rendre(this._registre, etat, ui, app); }
          catch (e) { if (typeof console !== 'undefined') console.error('Registre', e); }
        }
        return;
      }
      if (!debloque) {
        F.reconstruire(this._leviers, 'verrou:' + pid + ':' + LTP.q.semaineCourante(etat), function () { return fabriquerVerrou(p, etat); });
        return;
      }
      F.reconstruire(this._leviers, signatureLeviers(etat, ui, pid), function () {
        // Seulement les leviers déjà arrivés (tous en mode animateur « tout débloquer ») ;
        // en mode îlot, les leviers surchargeables d'abord.
        var liste = leviersDu(pid).filter(function (l) { return LTP.q.levierDebloque(etat, l.id); });
        if (ui.ilotEdition) liste = liste.slice().sort(function (a, b) { return (a.surchargeable ? 0 : 1) - (b.surchargeable ? 0 : 1); });
        return liste.map(function (def) { return fabriquerLevier(def, etat, ui, app); });
      });
      eclairer(this._leviers);
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
