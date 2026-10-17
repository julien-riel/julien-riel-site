/* 00-noyau.js */
/* Espace de noms, hasard déterministe, outils numériques, formatage fr-CA et textes.
   Aucun accès au DOM : ce fichier est aussi chargé dans Node pour les tests. */
(function (LTP) {
  'use strict';

  LTP.K = LTP.K || {};               // constantes d'équilibrage, un bloc par système
  LTP.systemes = LTP.systemes || {}; // systèmes du moteur (voir 31-boucle.js)
  LTP.donnees = LTP.donnees || {};   // contenu : îlots, personnages, leviers, cartes, événements…
  LTP.textes = LTP.textes || { fr: {} };
  LTP.langue = LTP.langue || 'fr';
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

  // --- Formatage fr-CA ---
  function nombre(n, d) {
    if (n === Infinity) return '∞';
    if (n == null || isNaN(n)) return '—';
    var dd = d == null ? (Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2) : d;
    return Number(n).toLocaleString('fr-CA', { maximumFractionDigits: dd, minimumFractionDigits: 0 }).replace(/^-/, '−');
  }
  function argent(n) {
    if (n == null || isNaN(n)) return '—';
    var a = Math.abs(n);
    if (a >= 1e6) return nombre(n / 1e6, 2) + NBSP + 'M$';
    if (a >= 1e4) return nombre(n / 1e3, 0) + NBSP + 'k$';
    return nombre(Math.round(n), 0) + NBSP + '$';
  }
  function pourcent(v, d) { return nombre(v * 100, d == null ? 0 : d) + NNBSP + '%'; }
  function jetons(n) {
    if (n == null || isNaN(n)) return '—';
    if (n >= 1e9) return nombre(n / 1e9, 1) + NBSP + 'G';
    if (n >= 1e6) return nombre(n / 1e6, 1) + NBSP + 'M';
    if (n >= 1e3) return nombre(n / 1e3, 0) + NBSP + 'k';
    return nombre(n, 0);
  }

  // --- Textes : t('cle', { nom: 'Mathieu' }) → gabarit avec {nom} remplacé.
  // Chaque module peut enregistrer ses propres clés sous son préfixe avec LTP.enregistrerTextes().
  function t(cle, vars) {
    var dict = LTP.textes[LTP.langue] || {};
    var s = dict[cle];
    if (s == null) s = (LTP.textes.fr || {})[cle];
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
})(globalThis.LTP = globalThis.LTP || {});
