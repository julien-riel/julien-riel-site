/* 28-demandes.js */
/* Demandes des directions d'affaires (CONCEPTION §4.11, §4.12) : étape 10 de la semaine.
   Nouvelles demandes (appétit qui monte avec le gain de l'IA perçu),
   filtre de la discipline du « non », changements de portée, gonflement fonctionnel, satisfaction,
   retards, applications hors TI et deux effets de Goodhart (« satisfaction », « incidents »).

   Backlog : 22-qualite le décrémente à la livraison (ilot.derniere.pointsLivres) ; ce système y ajoute
   les demandes acceptées, la reprise des changements de portée et les incidents reclassés.

   Champs propres ajoutés à l'état (initialiser) :
     ilot.retard            0–1 : part de retard (semaines de backlog au-delà de la tolérance) ; métrique « retard »
     ilot.applisEncadrees   applications hors TI encadrées (registre approuvé), moins risquées que ilot.applisFantomes
     ilot.produit = {
       emaLivres,           // moyenne lissée des points livrés par semaine
       emaValeur,           // moyenne lissée de la valeur utile livrée (points livrés moins le gonflement)
       emaDemandes,         // moyenne lissée des demandes acceptées
       gainPercu,           // gain de l'IA perçu par la direction (0–2)
       gonflement,          // cumul du code sans valeur (kLOC)
       reprise,             // cumul des points de reprise dus aux changements de portée
     }
     etat.produit = { applisCreees, incidentsReclasses }
       incidentsReclasses : Goodhart « incidents » ; les incidents reclassés portent inc.reclasse = true
       (30-metriques peut les retirer du compte visible tant que l'observabilité n'est pas complète). */
(function (LTP) {
  'use strict';
  var O = LTP.outils, q = LTP.q;

  LTP.K.demandes = {
    // --- Nouvelles demandes ---
    alea: 0.6,                    // demande × (1 + volatilité × alea), alea uniforme dans [−0,6 ; 0,6]
    appetit: 0.6,                 // demande × (1 + 0,6 × gain de l'IA perçu)
    perception: [0, 0.15, 0.35, 0.6, 0.85],  // gain perçu selon l'autonomie
    perceptionDebit: 0.5,         // + 0,5 × mods.debit (les skills qui accélèrent se voient)
    // --- Discipline du « non » ---
    refus: { laxiste: 0, normale: 0.15, ferme: 0.35 },          // part des demandes refusées (faible valeur)
    sansValeur: { laxiste: 0.3, normale: 0.12, ferme: 0.04 },   // part des points livrés qui n'apportent rien
    gonflement: { laxiste: 0.25, normale: 0.08, ferme: 0 },     // code sans valeur ajouté : × code de la semaine
    defautsParKloc: 0.05,         // défauts latents par kLOC de gonflement
    // --- Changements de portée ---
    portee: {
      proba: 0.05,                // par semaine × (0,5 + volatilité) × atténuation des specs
      specs: { non: 1, legeres: 0.6, obligatoires: 0.35 },
      part: 0.3,                  // reprise : 0,3 × (points livrés + 20 % du backlog) × atténuation
      seuilNote: 4,               // reprise minimale (points) pour une note au journal
    },
    // --- Satisfaction ---
    lissage: 0.25,                // poids de la semaine dans les moyennes lissées
    sat: {
      base: 0.15, valeur: 0.55, ratioMax: 1.3,  // cible = base + valeur × min(ratioMax, valeur livrée / demandes acceptées)
      bogue: 0.02, boguesMax: 5,  // par bogue de la semaine
      incident: 0.06, incidentActif: 0.03,
      retard: 0.2, delai: 0.02, delaiTolere: 2,  // délai de livraison toléré (semaines)
      refus: 0.08,                // × part des demandes refusées (le « non » déçoit à court terme)
      vitesse: 0.12,              // rapprochement hebdomadaire de la cible
    },
    retard: { tolere: 3, echelle: 6 }, // semaines de backlog tolérées, puis 0 → 1 sur 6 semaines
    // --- Applications hors TI ---
    nonTI: {
      trimestreMin: 2,            // le développement citoyen commence au T2
      base: 0.04,                 // probabilité par semaine et par îlot × insatisfaction
      seuilSat: 0.55,             // sous cette satisfaction, la direction se débrouille
      echelleSat: 0.35,
      retard: 0.5,                // + 0,5 × retard dans l'insatisfaction
      gainPercu: 0.5,             // × (1 + 0,5 × gain perçu) : « c'est facile maintenant »
      politique: { ignorer: 1, interdire: 0.55, encadrer: 1 },   // × probabilité d'apparition
      encadreePart: 0.85,         // encadrer : part des nouvelles applications qui naissent encadrées
      regularisation: 0.06,       // encadrer : probabilité par semaine de régulariser une application fantôme
      absorbe: 0.3,               // encadrer : points de backlog absorbés par semaine et par application encadrée
      satEncadrer: 0.03,          // + cible de satisfaction (accompagnement)
      satInterdire: 0.05,         // − cible de satisfaction (ça fâche)
      ilotsKevin: ['bureautique', 'web'],
    },
    // --- Goodhart « incidents » : reclassement en demandes ---
    goodhart: { reclassement: 0.5, pointsParIncident: 3 },
  };

  function initialiserIlot(il) {
    var def = q.defIlot(il.id) || {}, base = def.demandeBase || 10;
    if (il.retard == null) il.retard = 0;
    if (il.applisEncadrees == null) il.applisEncadrees = 0;
    if (!il.produit) il.produit = { emaLivres: base * 0.9, emaValeur: base * 0.8, emaDemandes: base, gainPercu: 0, gonflement: 0, reprise: 0 };
  }

  function discipline(etat, ilotId) {
    // Goodhart « satisfaction » : on dit oui à tout.
    if (q.levier(etat, 'objectif') === 'satisfaction') return 'laxiste';
    return q.levier(etat, 'disciplineNon', ilotId);
  }

  function gainPercu(etat, il) {
    var K = LTP.K.demandes, a = Math.max(0, Math.min(4, q.levier(etat, 'autonomie', il.id) | 0));
    var devs = q.devsDe(etat, il.id);
    var adhesion = devs.length ? O.moyenne(devs, 'adhesion') : 0.5;
    return O.borner(K.perception[a] * (0.6 + 0.4 * adhesion) + K.perceptionDebit * Math.max(0, il.mods.debit || 0), 0, 2);
  }

  // Goodhart « incidents » : une part des incidents de la semaine est reclassée en demande.
  function reclasserIncidents(etat, ctx) {
    var K = LTP.K.demandes.goodhart;
    if (q.levier(etat, 'objectif') !== 'incidents') return;
    etat.incidents.forEach(function (inc) {
      if (inc.debut !== etat.semaine || inc.reclasse || !ctx.rng.chance(K.reclassement)) return;
      var il = etat.ilots[inc.ilot];
      if (!il) return;
      inc.reclasse = true;
      il.derniere.incidents = Math.max(0, il.derniere.incidents - 1);
      il.backlog += K.pointsParIncident;
      etat.produit.incidentsReclasses += 1;
    });
  }

  function semaineIlot(etat, ctx, il) {
    var K = LTP.K.demandes, rng = ctx.rng, def = q.defIlot(il.id) || {};
    initialiserIlot(il);
    var P = il.produit;
    var d = discipline(etat, il.id);
    var livres = Math.max(0, il.derniere.pointsLivres || 0);

    // Nouvelles demandes, filtrées par la discipline du « non ».
    P.gainPercu = gainPercu(etat, il);
    var demandes = Math.max(0, (def.demandeBase || 10) * (1 + il.volatilite * rng.entre(-K.alea, K.alea)) * (1 + K.appetit * P.gainPercu));
    var refusees = demandes * (K.refus[d] || 0);
    var acceptees = demandes - refusees;
    il.backlog += acceptees;
    il.derniere.demandes += demandes;
    il.derniere.demandesRefusees += refusees;

    // Changements de portée : une part du travail en cours est à refaire.
    var Kp = K.portee, att = Kp.specs[q.levier(etat, 'specs')] || 1;
    if (rng.chance(Kp.proba * (0.5 + il.volatilite) * att)) {
      var reprise = Kp.part * (livres + 0.2 * il.backlog) * att;
      il.backlog += reprise;
      P.reprise += reprise;
      if (reprise >= Kp.seuilNote) ctx.noter('info', 'demandes.portee', { ilot: il.id, points: Math.round(reprise) }, { ilot: il.id });
    }

    // Gonflement fonctionnel : du code sans valeur, et la valeur livrée qui ne suit pas.
    var gonfle = Math.max(0, il.derniere.code || 0) * (K.gonflement[d] || 0);
    if (gonfle > 0) {
      il.codebase.volume += gonfle;
      il.codebase.defautsLatents += gonfle * K.defautsParKloc;
      P.gonflement += gonfle;
    }
    var valeur = livres * (1 - (K.sansValeur[d] || 0));

    // Applications encadrées : elles absorbent une partie de la demande.
    if (q.levier(etat, 'politiqueNonTI') === 'encadrer' && il.applisEncadrees > 0) {
      il.backlog = Math.max(0, il.backlog - K.nonTI.absorbe * il.applisEncadrees);
    }

    // Moyennes lissées et retard.
    var l = K.lissage;
    P.emaLivres = O.lerp(P.emaLivres, livres, l);
    P.emaValeur = O.lerp(P.emaValeur, valeur, l);
    P.emaDemandes = O.lerp(P.emaDemandes, acceptees, l);
    var semainesBacklog = il.backlog / Math.max(1, P.emaLivres);
    il.retard = O.borner((semainesBacklog - K.retard.tolere) / K.retard.echelle);

    satisfaction(etat, il, demandes ? refusees / demandes : 0);
    nonTI(etat, ctx, il);
  }

  function satisfaction(etat, il, partRefus) {
    var K = LTP.K.demandes, S = K.sat, P = il.produit, pol = q.levier(etat, 'politiqueNonTI');
    var ratio = P.emaValeur / Math.max(1, P.emaDemandes);
    var actifs = q.incidentsActifs(etat, il.id).length;
    var cible = S.base + S.valeur * Math.min(S.ratioMax, ratio)
      - S.bogue * Math.min(S.boguesMax, il.derniere.bogues || 0)
      - S.incident * (il.derniere.incidents || 0) - S.incidentActif * actifs
      - S.retard * il.retard
      - S.delai * Math.max(0, (il.derniere.delai || 0) - S.delaiTolere)
      - S.refus * partRefus;
    if (pol === 'encadrer') cible += K.nonTI.satEncadrer;
    if (pol === 'interdire') cible -= K.nonTI.satInterdire;
    cible = O.borner(cible);
    il.satisfaction = O.borner(il.satisfaction + S.vitesse * (cible - il.satisfaction));
  }

  // Applications hors TI : la demande insatisfaite trouve son chemin.
  function nonTI(etat, ctx, il) {
    var K = LTP.K.demandes.nonTI, rng = ctx.rng, pol = q.levier(etat, 'politiqueNonTI');
    // Encadrer : on régularise peu à peu les applications fantômes connues.
    if (pol === 'encadrer' && il.applisFantomes > 0 && rng.chance(K.regularisation)) {
      il.applisFantomes -= 1;
      il.applisEncadrees += 1;
    }
    if (etat.trimestre < K.trimestreMin && !etat.options.toutDebloque) return;
    var insat = O.borner((K.seuilSat - il.satisfaction) / K.echelleSat + K.retard * il.retard);
    var p = K.base * insat * (1 + K.gainPercu * il.produit.gainPercu) * (K.politique[pol] == null ? 1 : K.politique[pol]);
    if (p <= 0 || !rng.chance(p)) return;
    var encadree = pol === 'encadrer' && rng.chance(K.encadreePart);
    if (encadree) il.applisEncadrees += 1;
    else il.applisFantomes += 1;
    etat.produit.applisCreees += 1;
    if (!etat.org.drapeaux.kevinAppli && K.ilotsKevin.indexOf(il.id) >= 0) {
      etat.org.drapeaux.kevinAppli = true;
      ctx.noter('alerte', encadree ? 'demandes.kevinAppliEncadree' : 'demandes.kevinAppli', { ilot: il.id }, { ilot: il.id });
    } else {
      ctx.noter(encadree ? 'info' : 'alerte', encadree ? 'demandes.appliEncadree' : 'demandes.appliFantome', { ilot: il.id }, { ilot: il.id });
    }
  }

  LTP.systemes.demandes = {
    initialiser: function (etat) {
      if (!etat.produit) etat.produit = { applisCreees: 0, incidentsReclasses: 0 };
      q.ilots(etat).forEach(initialiserIlot);
    },
    // Étape 10 : après les incidents (9), avant les personnes (11).
    semaine: function (etat, ctx) {
      if (!etat.produit) etat.produit = { applisCreees: 0, incidentsReclasses: 0 };
      reclasserIncidents(etat, ctx);
      q.ilots(etat).forEach(function (il) { semaineIlot(etat, ctx, il); });
    },
  };

  LTP.enregistrerTextes({
    'demandes.portee': 'Changement de portée à l\'îlot {ilot} : « finalement, on le voudrait autrement ». {points} points à refaire.',
    'demandes.kevinAppli': 'Kevin, du marketing, a mis en ligne sa propre application. Elle marche. Personne ne sait où sont les données.',
    'demandes.kevinAppliEncadree': 'Kevin, du marketing, a construit sa propre application, dans le registre approuvé et avec un coup de main de l\'équipe.',
    'demandes.appliFantome': 'Une nouvelle application hors TI est apparue du côté de l\'îlot {ilot}.',
    'demandes.appliEncadree': 'Une direction a construit une application encadrée avec l\'îlot {ilot}.',
  });
})(globalThis.LTP = globalThis.LTP || {});
