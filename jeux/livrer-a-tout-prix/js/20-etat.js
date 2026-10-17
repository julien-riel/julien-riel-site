/* 20-etat.js */
/* Schéma de l'état du jeu, création d'une partie et requêtes (LTP.q).
   L'état est du JSON pur : pas de fonctions, pas de références circulaires, pas de Date. */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.etat = {
    semainesParTrimestre: 13,
    semainesTotal: 52,
    niveaux: {
      stagiaire: { competence: 0.3,  competenceIA: 0.35, adhesion: 0.6,  rigueur: 0.4,  comprehension: 0.5 },
      junior:    { competence: 0.45, competenceIA: 0.3,  adhesion: 0.55, rigueur: 0.45, comprehension: 0.6 },
      inter:     { competence: 0.62, competenceIA: 0.3,  adhesion: 0.5,  rigueur: 0.55, comprehension: 0.72 },
      senior:    { competence: 0.8,  competenceIA: 0.25, adhesion: 0.45, rigueur: 0.65, comprehension: 0.8 },
    },
    devDepart: { fatigue: 0.2, moral: 0.65 },
    org: {
      budgetAnnuel: 650000,   // $ pour l'IA, les outils et la formation
      confiance: 60,          // 0–100
      donnees: { qualite: 0.6, duplication: 0.4, fraicheur: 0.6, classification: 0.1 },
      menace: 0.2,
      releve: 0.5,
    },
    couvertureDepart: 0.35,
    conventionsDepart: 0.3,
    journalMax: 200,
  };

  // Bornes des champs normalisés : les effets et les systèmes doivent les respecter.
  // Les chemins sont relatifs à l'entité (dev, îlot, org). Tout autre nombre doit rester fini et ≥ 0,
  // sauf org.budget, qui peut devenir négatif (dépassement).
  LTP.etat = LTP.etat || {};
  LTP.etat.BORNES = {
    dev: { competence: [0, 1], competenceIA: [0, 1], adhesion: [0, 1], rigueur: [0, 1], comprehension: [0, 1], fatigue: [0, 1], moral: [0, 1] },
    ilot: {
      criticite: [0, 1], sensibilite: [0, 1], volatilite: [0, 1], adequationPile: [0, 1], satisfaction: [0, 1],
      'codebase.couvertureApparente': [0, 1], 'codebase.couvertureReelle': [0, 1], 'codebase.conventions': [0, 1],
    },
    org: {
      confiance: [0, 100], menace: [0, 1], observabilite: [0, 1], releve: [0, 1],
      'donnees.qualite': [0, 1], 'donnees.duplication': [0, 1], 'donnees.fraicheur': [0, 1], 'donnees.classification': [0, 1],
    },
  };
  LTP.etat.NEGATIFS_PERMIS = { org: { budget: true } };

  // Statistiques de la dernière semaine, remises à zéro par la boucle au début de chaque semaine.
  LTP.etat.derniereIlotVide = function () {
    return {
      points: 0, pointsLivres: 0, code: 0, pr: 0, prFusionnees: 0, deploiements: 0,
      defautsInjectes: 0, defautsEchappes: 0, bogues: 0, incidents: 0,
      jetons: 0, cout: 0, delai: 0, attention: 1, lgtm: 0, demandes: 0, demandesRefusees: 0,
    };
  };
  // Fractions du temps d'un développeur réservées par d'autres systèmes, remises à zéro chaque semaine par la boucle :
  // registre (26, développement d'une carte maison ou entretien), formation (27). La production (21) les soustrait.
  LTP.etat.occupationVide = function () { return { registre: 0, formation: 0 }; };
  LTP.etat.derniereDevVide = function () {
    return { points: 0, code: 0, pr: 0, revues: 0, lgtm: 0, jetons: 0, ecritSoiMeme: 0, luEnRevue: 0 };
  };
  // Modificateurs agrégés d'un îlot, recalculés chaque semaine par 26-registre.js (étape 3) et lus par les autres systèmes.
  LTP.etat.modsIlotNeutres = function () {
    return {
      debit: 0,          // fraction additive sur le gain de l'IA (0,1 = +10 %)
      tauxDefaut: 1,     // multiplicatif
      jetons: 1,         // multiplicatif
      mttr: 1,           // multiplicatif
      comprehension: 0,  // additif par semaine
      adequationPile: 0, // additif
      couverture: 0,     // additif sur la couverture réelle produite
      verbosite: 1,      // multiplicatif sur le volume de code
      poidsContexte: 0,  // somme des poids des cartes équipées
      incoherence: 0,    // 0–1 : conflits de skills
      taillePR: 1,       // multiplicatif sur la taille des PR (Goodhart « nombre de PR » : < 1)
      attention: 1,      // multiplicatif sur l'attention en revue (Goodhart « délai » : < 1)
    };
  };

  function statsDepart(def) {
    var K = LTP.K.etat;
    var base = O.cloner(K.niveaux[def.niveau] || K.niveaux.inter);
    base.fatigue = K.devDepart.fatigue;
    base.moral = K.devDepart.moral;
    (def.traits || []).forEach(function (tid) {
      var tr = LTP.donnees.traits[tid];
      if (!tr || !tr.init) return;
      for (var k in tr.init) base[k] = (base[k] || 0) + tr.init[k];
    });
    for (var k in base) base[k] = O.borner(base[k]);
    return base;
  }

  function creerEtat(graine, options) {
    var K = LTP.K.etat, opts = options || {};
    var g = LTP.rng.graineDe(graine);
    var etat = {
      version: 1,
      graine: g,
      rngEtat: g,
      semaine: 0,
      trimestre: 1,
      phase: 'accueil',
      options: { animateur: !!opts.animateur, toutDebloque: !!opts.toutDebloque, serveurAchete: !!opts.serveurAchete },
      leviers: { global: {}, surcharges: {}, enAttente: [] },
      org: {
        budgetAnnuel: K.org.budgetAnnuel,
        budget: K.org.budgetAnnuel,
        jetonsSemaine: 0, jetonsCumul: 0, coutSemaine: 0, coutCumul: 0,
        confiance: K.org.confiance,
        donnees: O.cloner(K.org.donnees),
        menace: K.org.menace,
        observabilite: 0,
        releve: K.org.releve,
        objectif: 'aucun',
        drapeaux: {},
      },
      ilots: {},
      ordreIlots: [],
      devs: {},
      capacites: {},
      incidents: [],
      differes: [],
      evenementEnAttente: null,
      evenementsForces: [],
      evenementsVus: {},
      derniereDecision: null, // { evenement, choix, semaine } : posé par 29-evenements.resoudre, affiché par l'interface
      comite: null,
      resultat: null,
      metriques: {},
      historique: {},
      journal: [],
      compteurs: { incidents: 0 },
    };

    LTP.donnees.leviers.forEach(function (l) { etat.leviers.global[l.id] = l.defaut; });

    LTP.donnees.ilots.slice().sort(function (a, b) { return a.ordre - b.ordre; }).forEach(function (d) {
      etat.ordreIlots.push(d.id);
      etat.ilots[d.id] = {
        id: d.id,
        criticite: d.criticite, sensibilite: d.sensibilite, volatilite: d.volatilite, adequationPile: d.adequationPile,
        backlog: d.backlogInitial, satisfaction: d.satisfactionInitiale, applisFantomes: 0,
        codebase: {
          volume: d.volumeInitial, defautsLatents: Math.round(d.volumeInitial * 0.05),
          couvertureApparente: K.couvertureDepart, couvertureReelle: K.couvertureDepart * 0.8,
          conventions: K.conventionsDepart,
        },
        fileRevue: 0,
        capacites: [],
        triade: { prive: !!d.exposition.prive, nonFiable: !!d.exposition.nonFiable, sortie: !!d.exposition.sortie },
        mods: LTP.etat.modsIlotNeutres(),
        derniere: LTP.etat.derniereIlotVide(),
      };
    });

    LTP.donnees.devs.forEach(function (d) {
      var s = statsDepart(d);
      etat.devs[d.id] = {
        id: d.id, ilot: d.ilot, niveau: d.niveau, traits: d.traits.slice(),
        competence: s.competence, competenceIA: s.competenceIA, adhesion: s.adhesion, rigueur: s.rigueur,
        comprehension: s.comprehension, fatigue: s.fatigue, moral: s.moral,
        sessions: 1, etat: 'code', indisponibleJusqua: 0, present: true,
        occupation: LTP.etat.occupationVide(),
        derniere: LTP.etat.derniereDevVide(),
      };
    });

    (LTP.donnees.capacites || []).forEach(function (c) {
      etat.capacites[c.id] = {
        equipeeSur: [], proprietaire: c.auteur || null, fraicheur: 1,
        disponibleSemaine: null, enDeveloppementPar: null,
        decouverte: c.disponible !== 'evenement',
      };
    });

    // Chaque système peut ajouter ses champs propres.
    LTP.moteur && LTP.moteur.ORDRE_INIT && LTP.moteur.ORDRE_INIT.forEach(function (nom) {
      var s = LTP.systemes[nom];
      if (s && s.initialiser) s.initialiser(etat);
    });
    return etat;
  }

  // --- Requêtes ---
  var q = LTP.q;

  q.defLevier = function (id) { return LTP.donnees.leviersParId[id]; };

  // Valeur effective d'un levier, avec la surcharge de l'îlot si elle existe.
  q.levier = function (etat, id, ilotId) {
    if (ilotId && etat.leviers.surcharges[ilotId] && Object.prototype.hasOwnProperty.call(etat.leviers.surcharges[ilotId], id)) {
      return etat.leviers.surcharges[ilotId][id];
    }
    if (Object.prototype.hasOwnProperty.call(etat.leviers.global, id)) return etat.leviers.global[id];
    var d = q.defLevier(id);
    return d ? d.defaut : undefined;
  };
  // Rang d'une valeur dans la liste du levier (0 = première) : pratique pour les leviers ordonnés.
  q.rangLevier = function (etat, id, ilotId) {
    var d = q.defLevier(id), v = q.levier(etat, id, ilotId);
    for (var i = 0; i < d.valeurs.length; i++) if (d.valeurs[i].v === v) return i;
    return 0;
  };
  q.levierDebloque = function (etat, id) {
    var d = q.defLevier(id);
    return !!d && (etat.options.toutDebloque || d.trimestre <= etat.trimestre);
  };
  q.panneauDebloque = function (etat, id) {
    var p = (LTP.donnees.panneaux || []).filter(function (x) { return x.id === id; })[0];
    return !!p && (etat.options.toutDebloque || p.trimestre <= etat.trimestre);
  };
  q.enAttente = function (etat, id, ilotId) {
    return etat.leviers.enAttente.filter(function (a) { return a.id === id && (a.ilot || null) === (ilotId || null); })[0] || null;
  };

  q.ilots = function (etat) { return etat.ordreIlots.map(function (id) { return etat.ilots[id]; }); };
  q.devs = function (etat) { return Object.keys(etat.devs).map(function (id) { return etat.devs[id]; }); };
  q.devsPresents = function (etat) { return q.devs(etat).filter(function (d) { return d.present; }); };
  q.devsDe = function (etat, ilotId) { return q.devsPresents(etat).filter(function (d) { return d.ilot === ilotId; }); };
  q.disponible = function (etat, dev) { return dev.present && dev.indisponibleJusqua <= etat.semaine && dev.etat !== 'conge'; };
  q.ilotDe = function (etat, devId) { var d = etat.devs[devId]; return d ? etat.ilots[d.ilot] : null; };
  q.aTrait = function (dev, trait) { return dev.traits.indexOf(trait) >= 0; };
  // Produit (mult) ou somme (add) d'un modificateur de trait sur un développeur.
  q.modTrait = function (dev, cle, neutre) {
    var mult = neutre === undefined ? 1 : neutre, v = mult;
    dev.traits.forEach(function (t) {
      var tr = LTP.donnees.traits[t];
      if (tr && tr.mods && tr.mods[cle] != null) v = (mult === 1) ? v * tr.mods[cle] : v + tr.mods[cle];
    });
    return v;
  };
  q.defDev = function (id) { return (LTP.donnees.devs || []).filter(function (d) { return d.id === id; })[0]; };
  q.defIlot = function (id) { return (LTP.donnees.ilots || []).filter(function (d) { return d.id === id; })[0]; };
  q.defCapacite = function (id) { return (LTP.donnees.capacites || []).filter(function (d) { return d.id === id; })[0]; };
  q.defEvenement = function (id) { return (LTP.donnees.evenements || []).filter(function (d) { return d.id === id; })[0]; };
  q.triadeComplete = function (etat, ilotId) { var t = etat.ilots[ilotId].triade; return !!(t.prive && t.nonFiable && t.sortie); };
  q.moyenne = function (liste, champ) { return O.moyenne(liste, champ); };
  q.drapeau = function (etat, nom) { return !!etat.org.drapeaux[nom]; };
  q.metriqueVisible = function (etat, id) {
    if (etat.options.animateur && etat.options.voirCachees) return true;
    var s = LTP.systemes.metriques;
    if (s && s.visible) return s.visible(etat, id);
    var m = LTP.donnees.metriquesParId[id];
    return !!m && !m.cachee;
  };

  // Crée un incident ; la durée vient du système de dette (MTTR) s'il est présent.
  q.creerIncident = function (etat, spec, ctx) {
    var ilot = etat.ilots[spec.ilot];
    if (!ilot) return null;
    etat.compteurs.incidents += 1;
    var gravite = O.borner(spec.gravite == null ? 0.5 : spec.gravite);
    var heures = LTP.systemes.dette && LTP.systemes.dette.mttr ? LTP.systemes.dette.mttr(etat, spec.ilot, gravite) : 8;
    var inc = {
      id: 'inc' + etat.compteurs.incidents, ilot: spec.ilot, gravite: gravite, cause: spec.cause || 'bogue',
      debut: etat.semaine, mttrHeures: heures, finSemaine: etat.semaine + Math.max(0, Math.ceil(heures / 40) - 1), resolu: false,
    };
    etat.incidents.push(inc);
    ilot.derniere.incidents += 1;
    if (ctx && ctx.noter) ctx.noter('incident', 'incident.' + inc.cause, { ilot: spec.ilot, gravite: gravite }, { ilot: spec.ilot });
    return inc;
  };
  q.incidentsActifs = function (etat, ilotId) {
    return etat.incidents.filter(function (i) { return !i.resolu && (!ilotId || i.ilot === ilotId); });
  };

  LTP.etat.creer = creerEtat;
  LTP.etat.statsDepart = statsDepart;
})(globalThis.LTP = globalThis.LTP || {});
