/* 00-noyau.js */
/* Espace de noms, hasard déterministe, outils numériques, formatage (fr-CA ou en-CA selon LTP.langue),
   textes et surcouches de traduction.
   Aucun accès au DOM : ce fichier est aussi chargé dans Node pour les tests. */
(function (LTP) {
  'use strict';

  LTP.K = LTP.K || {};               // constantes d'équilibrage, un bloc par système
  LTP.systemes = LTP.systemes || {}; // systèmes du moteur (voir 31-boucle.js)
  LTP.donnees = LTP.donnees || {};   // contenu : îlots, personnages, leviers, cartes, événements…
  LTP.textes = LTP.textes || { fr: {} };
  LTP.langue = LTP.langue || 'fr';
  LTP.textesManquants = LTP.textesManquants || {};
  LTP.q = LTP.q || {};               // requêtes sur l'état (voir 20-etat.js)

  var NBSP = ' ';
  var NNBSP = ' '; // espace fine insécable, avant « ; ! ? » et le symbole %

  // --- Hasard déterministe (mulberry32). L'état est un entier sérialisable. ---
  function rngSuivant(s) {
    s = (s + 0x6D2B79F5) | 0;
    var t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return { etat: s, valeur: ((t ^ (t >>> 14)) >>> 0) / 4294967296 };
  }

  // Crée un générateur lié à un porteur { rngEtat } (l'état du jeu) : chaque tirage avance porteur.rngEtat.
  function creerRng(porteur) {
    var r = {
      suivant: function () {
        var x = rngSuivant(porteur.rngEtat | 0);
        porteur.rngEtat = x.etat;
        return x.valeur;
      },
      entre: function (a, b) { return a + (b - a) * r.suivant(); },
      entier: function (a, b) { return a + Math.floor(r.suivant() * (b - a + 1)); },
      chance: function (p) { return r.suivant() < p; },
      choisir: function (liste) { return liste.length ? liste[Math.floor(r.suivant() * liste.length)] : undefined; },
      // poids : fonction(element) → nombre ≥ 0, ou nom de propriété
      choisirPondere: function (liste, poids) {
        var f = typeof poids === 'function' ? poids : function (e) { return e[poids || 'poids'] || 0; };
        var total = 0, i;
        for (i = 0; i < liste.length; i++) total += Math.max(0, f(liste[i]) || 0);
        if (total <= 0) return undefined;
        var x = r.suivant() * total;
        for (i = 0; i < liste.length; i++) {
          x -= Math.max(0, f(liste[i]) || 0);
          if (x < 0) return liste[i];
        }
        return liste[liste.length - 1];
      },
      // Bruit centré : moyenne 0, écart-type ≈ e (somme de trois uniformes).
      bruit: function (e) { return ((r.suivant() + r.suivant() + r.suivant()) / 3 - 0.5) * 2 * 1.732 * (e || 1); },
      melanger: function (liste) {
        var a = liste.slice();
        for (var i = a.length - 1; i > 0; i--) {
          var j = Math.floor(r.suivant() * (i + 1));
          var tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
        return a;
      },
    };
    return r;
  }

  // Graine à partir d'un texte ou d'un nombre.
  function graineDe(x) {
    if (typeof x === 'number' && isFinite(x)) return x | 0;
    var s = String(x == null ? Date.now() : x), h = 2166136261;
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h | 0;
  }

  // Hachage stable d'un texte en [0, 1) : sert aux variations visuelles déterministes.
  function hachage(s) {
    var h = 2166136261;
    s = String(s);
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return ((h >>> 0) % 100000) / 100000;
  }

  // --- Outils numériques ---
  function borner(v, min, max) {
    if (min == null) min = 0;
    if (max == null) max = 1;
    if (typeof v !== 'number' || isNaN(v)) return min;
    return v < min ? min : v > max ? max : v;
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function somme(liste, f) {
    var s = 0;
    for (var i = 0; i < liste.length; i++) s += f ? (typeof f === 'function' ? f(liste[i]) : liste[i][f]) : liste[i];
    return s;
  }
  function moyenne(liste, f) { return liste.length ? somme(liste, f) / liste.length : 0; }
  function arrondir(v, d) { var m = Math.pow(10, d || 0); return Math.round(v * m) / m; }
  // L'état du jeu est du JSON pur : la copie profonde passe par JSON (disponible aussi dans un contexte vm).
  function cloner(o) { return o === undefined ? undefined : JSON.parse(JSON.stringify(o)); }

  function lireChemin(obj, chemin) {
    var parts = String(chemin).split('.');
    for (var i = 0; i < parts.length; i++) {
      if (obj == null) return undefined;
      obj = obj[parts[i]];
    }
    return obj;
  }
  function ecrireChemin(obj, chemin, valeur) {
    var parts = String(chemin).split('.');
    for (var i = 0; i < parts.length - 1; i++) {
      if (obj[parts[i]] == null || typeof obj[parts[i]] !== 'object') obj[parts[i]] = {};
      obj = obj[parts[i]];
    }
    obj[parts[parts.length - 1]] = valeur;
  }

  // --- Formatage fr-CA (1 234,5 $ · 12 %) ou en-CA ($1,234.5 · 12%) selon LTP.langue, lue à chaque appel ---
  function anglais() { return LTP.langue === 'en'; }
  function nombre(n, d) {
    if (n === Infinity) return '∞';
    if (n == null || isNaN(n)) return '—';
    var dd = d == null ? (Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2) : d;
    return Number(n).toLocaleString(anglais() ? 'en-CA' : 'fr-CA', { maximumFractionDigits: dd, minimumFractionDigits: 0 }).replace(/^-/, '−');
  }
  function argent(n) {
    if (n == null || isNaN(n)) return '—';
    var a = Math.abs(n);
    if (anglais()) {
      var signe = n < 0 ? '−' : '';
      if (a >= 1e6) return signe + '$' + nombre(a / 1e6, 2) + 'M';
      if (a >= 1e4) return signe + '$' + nombre(a / 1e3, 0) + 'k';
      return signe + '$' + nombre(Math.round(a), 0);
    }
    if (a >= 1e6) return nombre(n / 1e6, 2) + NBSP + 'M$';
    if (a >= 1e4) return nombre(n / 1e3, 0) + NBSP + 'k$';
    return nombre(Math.round(n), 0) + NBSP + '$';
  }
  function pourcent(v, d) { return nombre(v * 100, d == null ? 0 : d) + (anglais() ? '' : NNBSP) + '%'; }
  function jetons(n) {
    if (n == null || isNaN(n)) return '—';
    var sep = anglais() ? '' : NBSP;
    if (n >= 1e9) return nombre(n / 1e9, 1) + sep + (anglais() ? 'B' : 'G');
    if (n >= 1e6) return nombre(n / 1e6, 1) + sep + 'M';
    if (n >= 1e3) return nombre(n / 1e3, 0) + sep + 'k';
    return nombre(n, 0);
  }

  // --- Textes : t('cle', { nom: 'Mathieu' }) → gabarit avec {nom} remplacé.
  // Chaque module peut enregistrer ses propres clés sous son préfixe avec LTP.enregistrerTextes().
  function t(cle, vars) {
    var dict = LTP.textes[LTP.langue] || {};
    var s = dict[cle];
    if (s == null && LTP.langue !== 'fr') {
      s = (LTP.textes.fr || {})[cle];
      if (s != null) LTP.textesManquants[cle] = true; // repli sur le français : relevé par les tests de l'édition anglaise
    }
    if (s == null) return '⟦' + cle + '⟧';
    if (typeof s === 'function') return s(vars || {});
    if (Array.isArray(s)) s = s[0];
    return gabarit(s, vars);
  }
  function gabarit(s, vars) {
    if (!vars) return s;
    return String(s).replace(/\{([a-zA-Z0-9_.]+)\}/g, function (m, k) {
      var v = lireChemin(vars, k);
      return v == null ? m : String(v);
    });
  }
  function aTexte(cle) { return (LTP.textes[LTP.langue] || {})[cle] != null || (LTP.textes.fr || {})[cle] != null; }
  function enregistrerTextes(dict, langue) {
    var l = langue || 'fr';
    LTP.textes[l] = LTP.textes[l] || {};
    for (var k in dict) if (Object.prototype.hasOwnProperty.call(dict, k)) LTP.textes[l][k] = dict[k];
  }

  // --- Surcouches de traduction (dossier en/ : voir en/LISEZMOI.md) ---
  // LTP.traduire('ilots', { appro: { nom: 'Procurement' } }) fusionne le patch dans LTP.donnees.ilots, en place :
  // - une chaîne ou une fonction remplace la valeur ; posée sur un objet qui a un champ texte, elle remplace ce texte ;
  // - un tableau se fusionne élément par élément (même longueur exigée ; null garde l'élément) ;
  // - un objet posé sur un tableau désigne ses éléments par id (ou par v, pour les valeurs d'un levier) ;
  // - une clé absente de la cible est une erreur : le français a changé, la traduction doit suivre.
  // Les erreurs sont gardées dans LTP.erreursTraduction (vérifiées par tests/traduction.test.js).
  LTP.erreursTraduction = LTP.erreursTraduction || [];
  function estObjet(x) { return x !== null && typeof x === 'object' && !Array.isArray(x); }
  function trouver(liste, cle) {
    for (var i = 0; i < liste.length; i++) {
      var e = liste[i];
      if (e && typeof e === 'object' && (e.id === cle || (e.id === undefined && e.v !== undefined && String(e.v) === cle))) return i;
    }
    return -1;
  }
  function fusionner(cible, patch, chemin) {
    if (typeof patch === 'string' || typeof patch === 'function') {
      if (estObjet(cible) && cible.texte !== undefined) { cible.texte = patch; return cible; }
      if (cible !== undefined && typeof cible !== 'string' && typeof cible !== 'function') {
        LTP.erreursTraduction.push(chemin + ' : un texte remplacerait une valeur de type ' + (Array.isArray(cible) ? 'tableau' : typeof cible));
        return cible;
      }
      return patch;
    }
    if (Array.isArray(patch)) {
      if (!Array.isArray(cible)) { LTP.erreursTraduction.push(chemin + ' : tableau attendu dans la cible'); return cible; }
      if (patch.length !== cible.length) LTP.erreursTraduction.push(chemin + ' : ' + patch.length + ' éléments traduits pour ' + cible.length + ' en français');
      for (var i = 0; i < Math.min(patch.length, cible.length); i++) {
        if (patch[i] !== null && patch[i] !== undefined) cible[i] = fusionner(cible[i], patch[i], chemin + '[' + i + ']');
      }
      return cible;
    }
    if (estObjet(patch)) {
      if (Array.isArray(cible)) {
        for (var cle in patch) {
          if (!Object.prototype.hasOwnProperty.call(patch, cle)) continue;
          var j = trouver(cible, cle);
          if (j < 0) { LTP.erreursTraduction.push(chemin + '.' + cle + ' : élément introuvable'); continue; }
          cible[j] = fusionner(cible[j], patch[cle], chemin + '.' + cle);
        }
        return cible;
      }
      if (!estObjet(cible)) { LTP.erreursTraduction.push(chemin + ' : objet attendu dans la cible'); return cible; }
      for (var k in patch) {
        if (!Object.prototype.hasOwnProperty.call(patch, k)) continue;
        if (cible[k] === undefined) { LTP.erreursTraduction.push(chemin + '.' + k + ' : clé absente du français'); continue; }
        cible[k] = fusionner(cible[k], patch[k], chemin + '.' + k);
      }
      return cible;
    }
    LTP.erreursTraduction.push(chemin + ' : valeur de traduction non prise en charge');
    return cible;
  }
  function traduire(chemin, patch) {
    var cible = lireChemin(LTP.donnees, chemin);
    if (cible === undefined) { LTP.erreursTraduction.push(chemin + ' : absent de LTP.donnees'); return; }
    var r = fusionner(cible, patch, chemin);
    if (r !== cible) ecrireChemin(LTP.donnees, chemin, r);
  }

  LTP.rng = { creer: creerRng, graineDe: graineDe };
  LTP.outils = {
    borner: borner, lerp: lerp, somme: somme, moyenne: moyenne, arrondir: arrondir, cloner: cloner,
    lireChemin: lireChemin, ecrireChemin: ecrireChemin, hachage: hachage,
  };
  LTP.format = { nombre: nombre, argent: argent, pourcent: pourcent, jetons: jetons, NBSP: NBSP, NNBSP: NNBSP };
  LTP.t = t;
  LTP.gabarit = gabarit;
  LTP.aTexte = aTexte;
  LTP.enregistrerTextes = enregistrerTextes;
  LTP.traduire = traduire;
})(globalThis.LTP = globalThis.LTP || {});
