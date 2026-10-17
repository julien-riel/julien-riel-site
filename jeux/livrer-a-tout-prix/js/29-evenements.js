/* 29-evenements.js */
/* Tirage des cartes d'événement et résolution des choix (CONCEPTION §6).
   Au plus un événement par semaine : d'abord les suites planifiées (etat.evenementsForces) échues,
   sinon un tirage pondéré parmi les cartes admissibles. Les effets sont appliqués par LTP.effets (19-effets.js).
   Le contenu peut être vide ou fautif : une condition ou un choisirCible qui lève une exception est ignoré. */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.evenements = {
    probabilite: 0.35,            // chance hebdomadaire qu'une carte soit tirée
    probabiliteDebut: 0.1,        // pendant les premières semaines (le joueur découvre l'open space)
    semainesDebut: 1,             // nombre de semaines « calmes » en début de partie
    poidsDefaut: 1,               // poids d'une carte qui n'en précise pas
    delaiRetirageDefaut: 8,       // semaines avant qu'une carte puisse revenir
    bonusTheme: 2,                // multiplicateur de poids pour les thèmes dominants du trimestre
    // Thèmes dominants par trimestre (CONCEPTION §2.1). Les thèmes des données sont normalisés
    // (minuscules, sans accents ni tirets) : « non-TI », « nonTI » et « nonti » sont équivalents.
    themesTrimestre: {
      1: ['revue', 'ia', 'qualite'],
      2: ['jetons', 'couts', 'skills', 'produit'],
      3: ['securite', 'donnees', 'nonti'],
      4: ['personnes', 'direction', 'dette'],
    },
    pasEnFinDeTrimestre: true,    // pas de tirage la semaine du comité : la carte arriverait après la réunion
    essaisCible: 6,               // cartes essayées au plus si choisirCible ne trouve pas de cible
  };
  var K = LTP.K.evenements;

  function normaliserTheme(t) {
    return String(t == null ? '' : t).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9]/g, '');
  }

  function initialiser(etat) {
    if (!etat.evenementsForces) etat.evenementsForces = [];
    if (!etat.evenementsVus) etat.evenementsVus = {};
    if (!etat.evenementsCompte) etat.evenementsCompte = { tires: 0, parTrimestre: {} };
  }

  function sur(f, args, defaut) {
    try { return f.apply(null, args); } catch (e) { return defaut; }
  }

  // Une carte est admissible si son trimestre, son délai de retirage, son unicité et sa condition le permettent.
  function admissible(etat, def) {
    if (!def || !def.id || !Array.isArray(def.choix) || !def.choix.length) return false;
    if (Array.isArray(def.trimestres) && def.trimestres.indexOf(etat.trimestre) < 0) return false;
    var vu = etat.evenementsVus[def.id];
    if (vu != null) {
      if (def.unique) return false;
      var delai = def.delaiRetirage == null ? K.delaiRetirageDefaut : def.delaiRetirage;
      if (etat.semaine - vu < delai) return false;
    }
    if (etat.evenementEnAttente && etat.evenementEnAttente.id === def.id) return false;
    if (typeof def.condition === 'function' && !sur(def.condition, [etat, LTP.q], false)) return false;
    return poids(etat, def) > 0;
  }

  function poids(etat, def) {
    var p = def.poids == null ? K.poidsDefaut : Number(def.poids);
    if (!(p > 0) || !isFinite(p)) return 0;
    var dominants = (K.themesTrimestre[etat.trimestre] || []).map(normaliserTheme);
    if (def.theme && dominants.indexOf(normaliserTheme(def.theme)) >= 0) p *= K.bonusTheme;
    return p;
  }

  // Cible par défaut : selon def.cible ('org', 'ilot', 'dev'), au hasard parmi les présents.
  function cibleParDefaut(etat, def, rng) {
    var q = LTP.q;
    if (def.cible === 'org') return {};
    if (def.cible === 'dev') {
      var dispo = q.devsPresents(etat).filter(function (d) { return q.disponible(etat, d); });
      var d = rng.choisir(dispo.length ? dispo : q.devsPresents(etat));
      return d ? { dev: d.id, ilot: d.ilot } : {};
    }
    var ilotId = rng.choisir(etat.ordreIlots);
    var ref = ilotId ? { ilot: ilotId } : {};
    var devs = ilotId ? q.devsDe(etat, ilotId) : [];
    if (devs.length) ref.dev = rng.choisir(devs).id;
    return ref;
  }

  // Normalise une référence : objet { ilot?, dev? } avec des identifiants qui existent. null si invalide.
  function normaliserRef(etat, ref) {
    if (!ref || typeof ref !== 'object') return null;
    var r = {};
    if (ref.dev != null) {
      if (!etat.devs[ref.dev]) return null;
      r.dev = String(ref.dev);
    }
    if (ref.ilot != null) {
      if (!etat.ilots[ref.ilot]) return null;
      r.ilot = String(ref.ilot);
    } else if (r.dev) r.ilot = etat.devs[r.dev].ilot;
    return r;
  }

  function cibler(etat, def, rng) {
    if (typeof def.choisirCible === 'function') {
      var brute;
      try { brute = def.choisirCible(etat, LTP.q, rng); } catch (e) { return null; }
      return normaliserRef(etat, brute);
    }
    return normaliserRef(etat, cibleParDefaut(etat, def, rng)) || {};
  }

  function poser(etat, def, ref, ctx, force) {
    var r = ref || {};
    var vars = {};
    if (r.ilot) vars.ilot = r.ilot;
    if (r.dev) vars.dev = r.dev;
    etat.evenementEnAttente = { id: def.id, ref: O.cloner(r), vars: vars };
    var c = etat.evenementsCompte || (etat.evenementsCompte = { tires: 0, parTrimestre: {} });
    c.tires += 1;
    c.parTrimestre[etat.trimestre] = (c.parTrimestre[etat.trimestre] || 0) + 1;
    if (ctx && ctx.noter) ctx.noter('evenement', 'evenements.tire', { evenement: def.id, force: !!force }, r);
  }

  // Étape 12.
  function tirage(etat, ctx) {
    if (etat.evenementEnAttente) return;
    var q = LTP.q, rng = ctx.rng;
    var spt = LTP.K.etat.semainesParTrimestre;
    if (K.pasEnFinDeTrimestre && etat.semaine % spt === 0) return;

    // 1. Suites planifiées échues (dans l'ordre de planification).
    var forces = etat.evenementsForces || [];
    for (var i = 0; i < forces.length; i++) {
      var f = forces[i];
      if (!(f && f.semaine <= etat.semaine)) continue;
      forces.splice(i, 1);
      i -= 1;
      var defF = q.defEvenement(f.id);
      if (!defF || !Array.isArray(defF.choix) || !defF.choix.length) continue;
      var refF = normaliserRef(etat, f.ref) || cibler(etat, defF, rng) || {};
      poser(etat, defF, refF, ctx, true);
      return;
    }

    // 2. Tirage au hasard.
    var p = etat.semaine <= K.semainesDebut ? K.probabiliteDebut : K.probabilite;
    if (!rng.chance(p)) return;
    var candidats = (LTP.donnees.evenements || []).filter(function (d) { return admissible(etat, d); });
    for (var essai = 0; essai < K.essaisCible && candidats.length; essai++) {
      var def = rng.choisirPondere(candidats, function (d) { return poids(etat, d); });
      if (!def) return;
      var ref = cibler(etat, def, rng);
      if (ref) { poser(etat, def, ref, ctx, false); return; }
      candidats = candidats.filter(function (d) { return d !== def; });
    }
  }

  function texteDe(x, etat, ref, score) {
    if (typeof x === 'function') {
      try { return String(x(etat, LTP.q, ref, score)); } catch (e) { return ''; }
    }
    return x == null ? '' : String(x);
  }

  // Résout le choix du joueur. Renvoie un code d'erreur ou null.
  function resoudre(etat, choixId, ctx) {
    var att = etat.evenementEnAttente;
    if (!att) return 'aucun-evenement';
    var def = LTP.q.defEvenement(att.id);
    if (!def) {
      // Carte disparue du contenu (vieille sauvegarde) : on la retire plutôt que de bloquer la partie.
      etat.evenementEnAttente = null;
      return null;
    }
    var choix = (def.choix || []).filter(function (c) { return c && c.id === choixId; })[0];
    if (!choix) return 'choix-inconnu';
    var ref = att.ref || {};
    etat.evenementEnAttente = null;
    LTP.effets.appliquer(etat, choix.effets || [], ref, ctx);
    etat.evenementsVus[att.id] = etat.semaine;
    etat.derniereDecision = { evenement: att.id, choix: choixId, semaine: etat.semaine, ref: O.cloner(ref) };
    var consequence = texteDe(choix.consequence, etat, ref);
    if (ctx && ctx.noter) {
      ctx.noter('evenement', consequence ? 'evenements.resolu' : 'evenements.resoluSansTexte',
        { evenement: att.id, choix: choixId, consequence: consequence }, ref);
    }
    return null;
  }

  LTP.systemes.evenements = {
    initialiser: initialiser,
    tirage: tirage,
    resoudre: resoudre,
    // Exposés pour les tests et le mode animateur.
    admissible: admissible,
    poids: poids,
    normaliserTheme: normaliserTheme,
  };

  LTP.enregistrerTextes({
    'evenements.tire': 'Un événement demande votre attention.',
    'evenements.resolu': '{consequence}',
    'evenements.resoluSansTexte': 'Décision prise.',
  });
})(globalThis.LTP = globalThis.LTP || {});
