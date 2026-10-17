/* 26-registre.js */
/* Le Registre des capacités, skills et MCP (CONCEPTION §4.9, §4.12) : étape 3 de la semaine.
   Recalcule ilot.mods (repart de LTP.etat.modsIlotNeutres()) et ilot.triade à partir des cartes équipées
   et actives ; gère la péremption, l'entretien par les propriétaires, le développement maison,
   l'installation en libre-service et les actions du joueur (equiper, desequiper, developper, proprietaire).

   Une carte équipée est ACTIVE si : elle est découverte, elle est disponible (une carte « developpement »
   doit avoir été développée : capacites[id].disponibleSemaine ≤ semaine) et son prérequis de levier est rempli
   dans l'îlot. Sinon elle reste équipée mais n'a aucun effet (apercu() donne la raison).

   Champs propres ajoutés à l'état (initialiser) — JSON pur :
     etat.registre = {
       cartes: { [carteId]: {
         progression,        // semaines de développement déjà faites (cartes maison)
         auteur,             // développeur qui l'a développée (ou def.auteur) : connaissance capturée
         abandonnee,         // carte communautaire que plus personne ne maintient (risque « abandon »)
         perimee,            // fraîcheur sous K.registre.seuilPeremption (pour ne noter qu'une fois)
       } },
       parIlot: { [ilotId]: {
         actives: [carteId], inactives: [{ id, raison }],
         poids, exces,       // somme des poids de contexte ; dépassement du seuil
         facteurSaturation,  // multiplicateur des gains (1 sous le seuil)
         mauvaisOutil,       // probabilité de « mauvais outil choisi » cette semaine
         mauvaisOutilSemaine,// vrai si c'est arrivé cette semaine
         conflits: [{ a, b, declencheur }], incoherence,
         perimees: [carteId], catalogue,  // îlot couvert par une carte MCP de catalogue active
         facteurDonnees,     // multiplicateur des défauts dû aux données (déjà inclus dans mods.tauxDefaut)
         attenuee,           // bac à sable filtré : la triade est atténuée (lu par 25)
       } },
       orphelines,           // cartes équipées sans propriétaire présent (métrique cartesOrphelines)
       perimees,             // cartes équipées périmées
       connaissance: { [devId]: 0–K.capture.max },  // compréhension capturée dans les cartes d'un développeur
       connaissanceCapturee, // 0–1 : moyenne sur les développeurs présents, rapportée au maximum (métrique)
       installations,        // cumul des installations en libre-service
     }

   Contrat avec les autres systèmes :
     - mods.tauxDefaut inclut déjà le facteur données (CONCEPTION §4.7), les cartes périmées et le « mauvais
       outil » ; il n'inclut PAS l'incohérence, que 22-qualite applique (× (1 + K × incoherence)) et que
       27-personnes traduit en baisse de moral.
     - Goodhart (levier objectif) : « pr » → mods.taillePR < 1 ; « delai » → mods.attention < 1.
       « satisfaction » et « incidents » sont traités par 28-demandes.
     - dev.occupation.registre : part du temps prise par le développement maison et l'entretien des cartes. */
(function (LTP) {
  'use strict';
  var O = LTP.outils, q = LTP.q;

  LTP.K.registre = {
    // --- Saturation du contexte ---
    seuilContexte: 0.25,          // somme des poids de contexte au-delà de laquelle les gains s'érodent
    penteSaturation: 2.5,         // perte de gain par unité de poids au-delà du seuil (0,1 de trop → −25 %)
    facteurSaturationMin: 0.3,    // plancher du multiplicateur de gain
    incoherenceSaturation: 1.5,   // incohérence par unité de poids au-delà du seuil
    mauvaisOutilPente: 1.5,       // probabilité de « mauvais outil choisi » par unité de poids au-delà du seuil
    mauvaisOutilMax: 0.5,
    mauvaisOutilDefaut: 1.15,     // tauxDefaut × la semaine où l'agent choisit le mauvais outil
    mauvaisOutilJetons: 1.1,      // jetons × la même semaine (il tourne en rond)
    // --- Conflits de déclencheurs ---
    incoherenceConflit: 0.35,     // incohérence par paire de skills au même déclencheur dans un îlot
                                  // (effets : défauts dans 22-qualite, moral dans 27-personnes)
    // --- Péremption et entretien ---
    perteFraicheur: 0.02,         // par semaine, carte maison ou personnelle sans propriétaire présent
    multPerime: 1.5,              // multiplicateur de perte pour les cartes au risque « perime »
    perteAbandon: 0.06,           // par semaine, carte communautaire abandonnée
    entretienFraicheur: 0.03,     // regain par semaine quand le propriétaire est disponible
    tempsEntretien: 0.03,         // part du temps du propriétaire, par carte maison ou personnelle équipée
    tempsEntretienExterne: 0.01,  // idem pour une carte de fournisseur ou communautaire (veille)
    seuilPeremption: 0.45,        // sous ce seuil la carte est périmée
    inversion: 0.5,               // à fraîcheur nulle, le gain vaut −50 % du gain d'origine
    defautPerimee: 1.08,          // tauxDefaut × par carte périmée (elle suit l'ancienne API)
    // --- Développement maison ---
    partDeveloppement: 0.5,       // part du temps du développeur réservée pendant coutDev semaines
    occupationMax: 0.8,           // plafond de dev.occupation.registre
    // --- Libre-service ---
    facteurLibreService: 1,       // × mod de trait installeCommunaute (probabilité par semaine)
    installationsMaxSemaine: 1,
    // --- Données (CONCEPTION §4.7) ---
    donneesQualite: 0.5,          // tauxDefaut + 0,5 × (1 − qualité)
    donneesDuplication: 0.4,      // + 0,4 × duplication (l'agent prend la mauvaise des trois listes)
    donneesFraicheur: 0.25,       // + 0,25 × (1 − fraîcheur)
    amplificationAutonomie: [0.3, 0.5, 0.75, 1, 1.15], // part du facteur données selon l'autonomie (l'agent amplifie)
    neutralisationCatalogue: 0.8, // part du facteur données neutralisée par une carte MCP de catalogue active
    // --- Goodhart (CONCEPTION §4.12) ---
    goodhartTaillePR: 0.5,        // objectif « nombre de PR » : les PR rapetissent artificiellement
    goodhartAttention: 0.75,      // objectif « délai » : la revue se relâche
    // --- Bornes des modificateurs ---
    multMin: 0.05, multMax: 5,    // modificateurs multiplicatifs
    additifMin: -0.5, additifMax: 1.5,
    poidsDefaut: 0.05,            // poids de contexte d'une carte qui n'en déclare pas
    // --- Connaissance capturée ---
    capture: { parCarte: 0.15, nonSenior: 0.6, max: 0.5 },
  };

  // Index des cartes par id, reconstruit si la liste du contenu change (les tests y poussent des cartes).
  var index = { liste: null, taille: -1, parId: {} };
  function defCarte(id) {
    var l = LTP.donnees.capacites || [];
    if (index.liste !== l || index.taille !== l.length) {
      index = { liste: l, taille: l.length, parId: {} };
      l.forEach(function (d) { index.parId[d.id] = d; });
    }
    return index.parId[id];
  }

  var ADDITIFS = { debit: true, comprehension: true, adequationPile: true, couverture: true, poidsContexte: true, incoherence: true };

  function etatRegistreVide() {
    return { cartes: {}, parIlot: {}, orphelines: 0, perimees: 0, connaissance: {}, connaissanceCapturee: 0, installations: 0 };
  }
  function infoCarte(etat, id) {
    if (!etat.registre) etat.registre = etatRegistreVide();
    var c = etat.registre.cartes[id];
    if (!c) c = etat.registre.cartes[id] = { progression: 0, auteur: null, abandonnee: false, perimee: false };
    return c;
  }
  function lireInfoCarte(etat, id) { return (etat.registre && etat.registre.cartes[id]) || {}; }

  function debloquee(etat, def) { return etat.options.toutDebloque || (def.trimestre || 1) <= etat.trimestre; }
  function developpee(etat, def, etc) {
    if (def.disponible !== 'developpement') return true;
    return etc.disponibleSemaine != null && etc.disponibleSemaine <= etat.semaine;
  }
  function prerequisRempli(etat, def, ilotId) {
    var p = def.prerequis;
    if (!p || !p.levier) return true;
    return (p.valeurs || []).indexOf(q.levier(etat, p.levier, ilotId)) >= 0;
  }
  function estCatalogue(def) {
    return def.type === 'mcp' && (String(def.id).indexOf('catalogue') >= 0 || !!(def.gains && def.gains.catalogue) || !!def.catalogue);
  }
  function proprietairePresent(etat, etc) {
    var d = etc.proprietaire && etat.devs[etc.proprietaire];
    return !!(d && d.present);
  }
  function poidsDe(def) { return typeof def.poidsContexte === 'number' ? Math.max(0, def.poidsContexte) : LTP.K.registre.poidsDefaut; }

  // Raison pour laquelle une carte équipée n'agit pas (null : active).
  function raisonInactive(etat, def, etc, ilotId) {
    if (!etc.decouverte) return 'non-decouverte';
    if (!developpee(etat, def, etc)) return etc.enDeveloppementPar ? 'en-developpement' : 'a-developper';
    if (!prerequisRempli(etat, def, ilotId)) return 'prerequis';
    return null;
  }

  // Gains de la carte pour un îlot : les clés de ilotsGains[ilot] remplacent celles de gains.
  function gainsPour(def, ilotId) {
    var g = {}, k;
    for (k in def.gains || {}) g[k] = def.gains[k];
    var p = def.ilotsGains && def.ilotsGains[ilotId];
    for (k in p || {}) g[k] = p[k];
    return g;
  }

  // Multiplicateur dû à la fraîcheur : 1 au-dessus du seuil, puis jusqu'à −K.inversion à fraîcheur nulle.
  function facteurFraicheur(fraicheur) {
    var K = LTP.K.registre, f = O.borner(fraicheur == null ? 1 : fraicheur);
    if (f >= K.seuilPeremption) return 1;
    return O.lerp(-K.inversion, 1, f / K.seuilPeremption);
  }

  // Cartes actives d'un îlot : [{ id, def, etc }]. Fonction pure, utilisée aussi par 25-donnees-secu.
  function cartesActives(etat, ilotId) {
    var il = etat.ilots[ilotId], res = [];
    if (!il) return res;
    il.capacites.forEach(function (id) {
      var def = defCarte(id), etc = etat.capacites[id];
      if (def && etc && !raisonInactive(etat, def, etc, ilotId)) res.push({ id: id, def: def, etc: etc });
    });
    return res;
  }

  // Facteur des données sur le taux de défauts (CONCEPTION §4.7).
  function facteurDonnees(etat, ilotId, catalogue) {
    var K = LTP.K.registre, d = etat.org.donnees;
    var brut = K.donneesQualite * (1 - d.qualite) + K.donneesDuplication * d.duplication + K.donneesFraicheur * (1 - d.fraicheur);
    var a = Math.max(0, Math.min(4, q.levier(etat, 'autonomie', ilotId) | 0));
    brut *= K.amplificationAutonomie[a];
    if (catalogue) brut *= 1 - K.neutralisationCatalogue;
    return 1 + Math.max(0, brut);
  }

  // Calcul complet d'un îlot, sans hasard ni mutation : base de semaine() et de apercu().
  function calculerIlot(etat, ilotId) {
    var K = LTP.K.registre, il = etat.ilots[ilotId];
    var mods = LTP.etat.modsIlotNeutres();
    var actives = [], inactives = [];
    il.capacites.forEach(function (id) {
      var def = defCarte(id), etc = etat.capacites[id];
      if (!def || !etc) return;
      var r = raisonInactive(etat, def, etc, ilotId);
      if (r) inactives.push({ id: id, raison: r });
      else actives.push({ id: id, def: def, etc: etc });
    });

    // Saturation du contexte.
    var poids = O.somme(actives, function (c) { return poidsDe(c.def); });
    var exces = Math.max(0, poids - K.seuilContexte);
    var facteurSat = Math.max(K.facteurSaturationMin, 1 - K.penteSaturation * exces);
    var mauvaisOutil = Math.min(K.mauvaisOutilMax, K.mauvaisOutilPente * exces);

    // Conflits : deux skills actives au même déclencheur.
    var conflits = [], parDeclencheur = {};
    actives.forEach(function (c) {
      if (c.def.type !== 'skill' || !c.def.declencheur) return;
      var liste = parDeclencheur[c.def.declencheur] = parDeclencheur[c.def.declencheur] || [];
      liste.forEach(function (autre) { conflits.push({ a: autre, b: c.id, declencheur: c.def.declencheur }); });
      liste.push(c.id);
    });
    var incoherence = O.borner(K.incoherenceConflit * conflits.length + K.incoherenceSaturation * exces);

    // Gains, atténués par la saturation et inversés en partie par la péremption.
    var perimees = [];
    actives.forEach(function (c) {
      var ff = facteurFraicheur(c.etc.fraicheur);
      if (ff < 1) perimees.push(c.id);
      var f = facteurSat * ff, g = gainsPour(c.def, ilotId);
      for (var k in g) {
        if (!Object.prototype.hasOwnProperty.call(mods, k) || typeof g[k] !== 'number' || !isFinite(g[k])) continue;
        if (ADDITIFS[k]) mods[k] += g[k] * f;
        else if (g[k] > 0) mods[k] *= 1 + (g[k] - 1) * f;
      }
    });
    mods.poidsContexte = poids;
    mods.incoherence = incoherence;

    // Défauts : données et cartes périmées (l'incohérence est appliquée par 22-qualite).
    var catalogue = actives.some(function (c) { return estCatalogue(c.def); });
    var fd = facteurDonnees(etat, ilotId, catalogue);
    mods.tauxDefaut *= fd * Math.pow(K.defautPerimee, perimees.length);

    // Goodhart.
    var objectif = q.levier(etat, 'objectif');
    if (objectif === 'pr') mods.taillePR *= K.goodhartTaillePR;
    if (objectif === 'delai') mods.attention *= K.goodhartAttention;

    bornerMods(mods);

    // Triade létale : exposition de l'îlot, branches des cartes, accès à la prod, approbation des sorties.
    var expo = (q.defIlot(ilotId) || {}).exposition || {};
    var triade = { prive: !!expo.prive, nonFiable: !!expo.nonFiable, sortie: !!expo.sortie };
    actives.forEach(function (c) {
      (c.def.branches || []).forEach(function (b) { if (b in triade) triade[b] = true; });
    });
    if (q.levier(etat, 'accesProd', ilotId) === 'oui') triade.prive = true;
    if (q.levier(etat, 'approbationSorties', ilotId) === true) triade.sortie = false;

    return {
      mods: mods, triade: triade, actives: actives, inactives: inactives,
      poids: poids, exces: exces, facteurSaturation: facteurSat, mauvaisOutil: mauvaisOutil,
      conflits: conflits, incoherence: incoherence, perimees: perimees, catalogue: catalogue, facteurDonnees: fd,
      attenuee: q.levier(etat, 'bacASable', ilotId) === 'filtre',
    };
  }

  function bornerMods(mods) {
    var K = LTP.K.registre;
    for (var k in mods) {
      if (k === 'poidsContexte') mods[k] = Math.max(0, isFinite(mods[k]) ? mods[k] : 0);
      else if (k === 'incoherence') mods[k] = O.borner(mods[k]);
      else if (ADDITIFS[k]) mods[k] = O.borner(mods[k], K.additifMin, K.additifMax);
      else mods[k] = O.borner(mods[k], K.multMin, K.multMax);
    }
  }

  // --- Étapes hebdomadaires ---

  // Développement maison : le développeur réservé avance d'une semaine s'il est disponible.
  function developpement(etat, ctx) {
    var K = LTP.K.registre;
    Object.keys(etat.capacites).forEach(function (id) {
      var etc = etat.capacites[id], def = defCarte(id);
      if (!def || !etc.enDeveloppementPar) return;
      var dev = etat.devs[etc.enDeveloppementPar], info = infoCarte(etat, id);
      if (!dev || !dev.present) {
        ctx.noter('alerte', 'registre.developpementInterrompu', { carte: id, nom: def.nom || id });
        etc.enDeveloppementPar = null;
        return;
      }
      if (!q.disponible(etat, dev)) return;
      dev.occupation.registre = Math.min(K.occupationMax, dev.occupation.registre + K.partDeveloppement);
      info.progression += 1;
      if (info.progression >= (def.coutDev || 0)) {
        etc.disponibleSemaine = etat.semaine;
        etc.enDeveloppementPar = null;
        etc.fraicheur = 1;
        info.auteur = def.auteur || dev.id;
        if (!etc.proprietaire) etc.proprietaire = dev.id;
        ctx.noter('info', 'registre.carteDisponible', { carte: id, nom: def.nom || id, dev: dev.id }, { dev: dev.id, ilot: dev.ilot });
      }
    });
  }

  // Péremption et entretien des cartes équipées.
  function peremption(etat, ctx) {
    var K = LTP.K.registre;
    Object.keys(etat.capacites).forEach(function (id) {
      var etc = etat.capacites[id], def = defCarte(id);
      if (!def || !etc.equipeeSur.length || !developpee(etat, def, etc)) return;
      var info = infoCarte(etat, id), interne = def.provenance === 'maison' || def.provenance === 'personnelle';
      var proprio = etc.proprietaire && etat.devs[etc.proprietaire];
      // Abandon d'une carte communautaire par ses mainteneurs.
      if (def.provenance === 'communaute' && !info.abandonnee && def.risque && def.risque.type === 'abandon' && ctx.rng.chance(def.risque.proba || 0)) {
        info.abandonnee = true;
        ctx.noter('alerte', 'registre.abandonnee', { carte: id, nom: def.nom || id });
      }
      if (proprio && proprio.present) {
        if (q.disponible(etat, proprio)) {
          proprio.occupation.registre = Math.min(K.occupationMax, proprio.occupation.registre + (interne ? K.tempsEntretien : K.tempsEntretienExterne));
          if (interne || info.abandonnee) etc.fraicheur = O.borner(etc.fraicheur + K.entretienFraicheur);
        }
      } else if (interne) {
        var mult = def.risque && def.risque.type === 'perime' ? K.multPerime : 1;
        etc.fraicheur = O.borner(etc.fraicheur - K.perteFraicheur * mult);
      }
      if (info.abandonnee && !(proprio && proprio.present)) etc.fraicheur = O.borner(etc.fraicheur - K.perteAbandon);
      var perimee = etc.fraicheur < K.seuilPeremption;
      if (perimee && !info.perimee) ctx.noter('alerte', 'registre.perimee', { carte: id, nom: def.nom || id });
      info.perimee = perimee;
    });
  }

  // Libre-service : des développeurs curieux installent eux-mêmes des cartes communautaires découvertes.
  function libreService(etat, ctx) {
    var K = LTP.K.registre;
    if (q.levier(etat, 'politiqueRegistre') !== 'libre') return;
    var candidates = (LTP.donnees.capacites || []).filter(function (def) {
      var etc = etat.capacites[def.id];
      return etc && def.provenance === 'communaute' && etc.decouverte && debloquee(etat, def) && developpee(etat, def, etc);
    });
    if (!candidates.length) return;
    var faites = 0;
    q.devsPresents(etat).forEach(function (dev) {
      if (faites >= K.installationsMaxSemaine || !q.disponible(etat, dev)) return;
      var p = q.modTrait(dev, 'installeCommunaute', 0) * K.facteurLibreService;
      if (p <= 0 || !ctx.rng.chance(p)) return;
      var choix = candidates.filter(function (def) {
        return etat.ilots[dev.ilot].capacites.indexOf(def.id) < 0 && prerequisRempli(etat, def, dev.ilot);
      });
      var def = ctx.rng.choisir(choix);
      if (!def) return;
      equiperCarte(etat, def.id, dev.ilot);
      faites += 1;
      etat.registre.installations += 1;
      ctx.noter('alerte', 'registre.installationLibre', { carte: def.id, nom: def.nom || def.id, dev: dev.id, ilot: dev.ilot }, { dev: dev.id, ilot: dev.ilot });
    });
  }

  function equiperCarte(etat, carteId, ilotId) {
    var etc = etat.capacites[carteId], il = etat.ilots[ilotId];
    if (etc.equipeeSur.indexOf(ilotId) < 0) etc.equipeeSur.push(ilotId);
    if (il.capacites.indexOf(carteId) < 0) il.capacites.push(carteId);
  }

  // Applique le calcul d'un îlot à l'état. avecHasard : tirage du « mauvais outil » et notes au journal.
  function appliquerIlot(etat, ilotId, ctx, avecHasard) {
    var K = LTP.K.registre, il = etat.ilots[ilotId];
    var r = calculerIlot(etat, ilotId);
    var avant = etat.registre.parIlot[ilotId] || { exces: 0, conflits: [] };
    var mauvais = false;
    if (avecHasard && r.mauvaisOutil > 0 && ctx.rng.chance(r.mauvaisOutil)) {
      mauvais = true;
      r.mods.tauxDefaut = O.borner(r.mods.tauxDefaut * K.mauvaisOutilDefaut, K.multMin, K.multMax);
      r.mods.jetons = O.borner(r.mods.jetons * K.mauvaisOutilJetons, K.multMin, K.multMax);
    }
    il.mods = r.mods;
    il.triade = r.triade;
    if (avecHasard) {
      // Nouveaux conflits et début de saturation : une note au journal.
      var cleAvant = (avant.conflits || []).map(function (c) { return c.a + '|' + c.b; });
      r.conflits.forEach(function (c) {
        if (cleAvant.indexOf(c.a + '|' + c.b) < 0) {
          ctx.noter('alerte', 'registre.conflit', { a: c.a, b: c.b, declencheur: c.declencheur, ilot: ilotId }, { ilot: ilotId });
        }
      });
      if (r.exces > 0 && !(avant.exces > 0)) ctx.noter('alerte', 'registre.saturation', { ilot: ilotId }, { ilot: ilotId });
    }
    etat.registre.parIlot[ilotId] = {
      actives: r.actives.map(function (c) { return c.id; }), inactives: r.inactives,
      poids: r.poids, exces: r.exces, facteurSaturation: r.facteurSaturation,
      mauvaisOutil: r.mauvaisOutil, mauvaisOutilSemaine: mauvais,
      conflits: r.conflits, incoherence: r.incoherence, perimees: r.perimees,
      catalogue: r.catalogue, facteurDonnees: r.facteurDonnees, attenuee: r.attenuee,
    };
  }

  // Cartes orphelines, périmées et connaissance capturée.
  function bilan(etat) {
    var K = LTP.K.registre, reg = etat.registre;
    var orphelines = 0, perimees = 0, connaissance = {};
    Object.keys(etat.capacites).forEach(function (id) {
      var etc = etat.capacites[id], def = defCarte(id);
      if (!def || !etc.equipeeSur.length || !developpee(etat, def, etc)) return;
      if (!proprietairePresent(etat, etc)) orphelines += 1;
      if (etc.fraicheur < K.seuilPeremption) perimees += 1;
      if (def.provenance !== 'maison' && def.provenance !== 'personnelle') return;
      var auteurId = def.auteur || lireInfoCarte(etat, id).auteur || etc.proprietaire, auteur = etat.devs[auteurId];
      if (!auteur) return;
      var part = K.capture.parCarte * (auteur.niveau === 'senior' ? 1 : K.capture.nonSenior) * O.borner(etc.fraicheur);
      connaissance[auteurId] = Math.min(K.capture.max, (connaissance[auteurId] || 0) + part);
    });
    reg.orphelines = orphelines;
    reg.perimees = perimees;
    reg.connaissance = connaissance;
    var presents = q.devsPresents(etat);
    reg.connaissanceCapturee = presents.length
      ? O.borner(O.moyenne(presents, function (d) { return (connaissance[d.id] || 0) / K.capture.max; })) : 0;
  }

  // --- Actions du joueur ---
  function action(etat, a, ctx) {
    if (etat.phase === 'fin') return 'partie-terminee';
    var def = defCarte(a.carte), etc = etat.capacites[a.carte];
    if (!def || !etc) return 'carte-inconnue';
    var pol = q.levier(etat, 'politiqueRegistre');
    switch (a.type) {
      case 'equiper': {
        if (!etat.ilots[a.ilot]) return 'ilot-inconnu';
        if (!etc.decouverte) return 'carte-non-decouverte';
        if (!debloquee(etat, def)) return 'carte-verrouillee';
        if (!prerequisRempli(etat, def, a.ilot)) return 'prerequis-absent';
        if (def.provenance === 'communaute' && (pol === 'listeBlanche' || pol === 'proprietaires')) return 'politique-liste-blanche';
        if (pol === 'proprietaires' && !proprietairePresent(etat, etc)) return 'proprietaire-requis';
        if (etat.ilots[a.ilot].capacites.indexOf(a.carte) >= 0) return 'deja-equipee';
        equiperCarte(etat, a.carte, a.ilot);
        ctx.noter('info', 'registre.equipee', { carte: a.carte, nom: def.nom || a.carte, ilot: a.ilot }, { ilot: a.ilot });
        rafraichir(etat, a.ilot);
        return null;
      }
      case 'desequiper': {
        if (!etat.ilots[a.ilot]) return 'ilot-inconnu';
        if (etat.ilots[a.ilot].capacites.indexOf(a.carte) < 0) return 'carte-non-equipee';
        LTP.effets.retirerCarte(etat, a.carte, a.ilot);
        ctx.noter('info', 'registre.desequipee', { carte: a.carte, nom: def.nom || a.carte, ilot: a.ilot }, { ilot: a.ilot });
        rafraichir(etat, a.ilot);
        return null;
      }
      case 'developper': {
        if (def.disponible !== 'developpement') return 'carte-non-developpable';
        if (developpee(etat, def, etc)) return 'deja-developpee';
        if (etc.enDeveloppementPar) return 'deja-en-developpement';
        if (!etc.decouverte) return 'carte-non-decouverte';
        if (!debloquee(etat, def)) return 'carte-verrouillee';
        var dev = etat.devs[a.dev];
        if (!dev || !dev.present) return 'dev-indisponible';
        var occupe = Object.keys(etat.capacites).some(function (id) { return etat.capacites[id].enDeveloppementPar === a.dev; });
        if (occupe) return 'dev-occupe';
        etc.enDeveloppementPar = a.dev;
        infoCarte(etat, a.carte);
        ctx.noter('personne', 'registre.developpement', { carte: a.carte, nom: def.nom || a.carte, dev: a.dev, semaines: def.coutDev || 0 }, { dev: a.dev, ilot: dev.ilot });
        return null;
      }
      case 'proprietaire': {
        if (a.dev == null) { etc.proprietaire = null; return null; }
        var d = etat.devs[a.dev];
        if (!d || !d.present) return 'dev-indisponible';
        etc.proprietaire = a.dev;
        ctx.noter('info', 'registre.proprietaire', { carte: a.carte, nom: def.nom || a.carte, dev: a.dev }, { dev: a.dev });
        return null;
      }
      default:
        return 'action-inconnue';
    }
  }

  // Après une action, l'interface voit tout de suite les voyants et les gains à jour (sans hasard).
  function rafraichir(etat, ilotId) {
    if (!etat.registre) etat.registre = etatRegistreVide();
    appliquerIlot(etat, ilotId, null, false);
    bilan(etat);
  }

  // Aperçu pour l'interface : fonction pure, ne modifie pas l'état.
  function apercu(etat, ilotId) {
    var K = LTP.K.registre;
    if (!etat.ilots[ilotId]) return null;
    var r = calculerIlot(etat, ilotId);
    var cartes = r.actives.map(function (c) { return { id: c.id, active: true, raison: null }; })
      .concat(r.inactives.map(function (c) { return { id: c.id, active: false, raison: c.raison }; }))
      .map(function (c) {
        var etc = etat.capacites[c.id], def = defCarte(c.id);
        c.nom = def.nom || c.id;
        c.type = def.type;
        c.provenance = def.provenance;
        c.poids = poidsDe(def);
        c.fraicheur = etc.fraicheur;
        c.perimee = etc.fraicheur < K.seuilPeremption;
        c.proprietaire = etc.proprietaire || null;
        c.orpheline = !proprietairePresent(etat, etc);
        c.branches = (def.branches || []).slice();
        return c;
      });
    return {
      ilot: ilotId, cartes: cartes,
      gains: r.mods, poids: r.poids, seuil: K.seuilContexte, exces: r.exces,
      facteurSaturation: r.facteurSaturation, mauvaisOutil: r.mauvaisOutil,
      conflits: r.conflits, incoherence: r.incoherence,
      triade: r.triade, triadeComplete: !!(r.triade.prive && r.triade.nonFiable && r.triade.sortie), attenuee: r.attenuee,
      catalogue: r.catalogue, facteurDonnees: r.facteurDonnees,
    };
  }

  LTP.systemes.registre = {
    initialiser: function (etat) {
      etat.registre = etatRegistreVide();
      (LTP.donnees.capacites || []).forEach(function (def) {
        var info = infoCarte(etat, def.id);
        info.auteur = def.auteur || null;
      });
      q.ilots(etat).forEach(function (il) { appliquerIlot(etat, il.id, null, false); });
      bilan(etat);
    },
    // Étape 3 : développement, péremption, libre-service, puis ilot.mods et ilot.triade.
    semaine: function (etat, ctx) {
      if (!etat.registre) etat.registre = etatRegistreVide();
      developpement(etat, ctx);
      peremption(etat, ctx);
      libreService(etat, ctx);
      q.ilots(etat).forEach(function (il) { appliquerIlot(etat, il.id, ctx, true); });
      bilan(etat);
    },
    action: action,
    apercu: apercu,
    cartesActives: cartesActives,
    facteurFraicheur: facteurFraicheur,
  };

  LTP.enregistrerTextes({
    'registre.equipee': 'La carte {nom} est équipée sur l\'îlot {ilot}.',
    'registre.desequipee': 'La carte {nom} est retirée de l\'îlot {ilot}.',
    'registre.developpement': '{dev} commence le développement de {nom} ({semaines} sem.).',
    'registre.developpementInterrompu': 'Le développement de {nom} est interrompu : la personne responsable n\'est plus là.',
    'registre.carteDisponible': 'La carte {nom} est prête : {dev} vient de la terminer.',
    'registre.proprietaire': '{dev} devient responsable de {nom}.',
    'registre.installationLibre': '{dev} a installé {nom} de son propre chef. « Ça avait l\'air pratique. »',
    'registre.perimee': 'La carte {nom} est périmée : elle décrit encore l\'ancienne API.',
    'registre.abandonnee': 'La carte {nom} n\'est plus maintenue par sa communauté. Dernier commit : il y a longtemps.',
    'registre.conflit': 'Conflit à l\'îlot {ilot} : {a} et {b} se disputent le déclencheur « {declencheur} ».',
    'registre.saturation': 'Contexte saturé à l\'îlot {ilot} : trop de skills et de MCP, l\'agent s\'y perd.',
  });
})(globalThis.LTP = globalThis.LTP || {});
