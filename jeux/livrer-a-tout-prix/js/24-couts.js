/* 24-couts.js */
/* Étape 7 de la semaine : jetons, coûts et budget (CONCEPTION §4.6).
   Jetons par développeur : sessions × intensité(autonomie) × (1 + poids du contexte) × ilot.mods.jetons,
   plus les boucles d'agent (rares, sans plafond) et l'attente d'une CI lente. Prix selon le modèle
   (modèle par tâche : environ −35 %). Coûts fixes hebdomadaires des leviers. Le budget baisse.
   Plafond de jetons : souple (alerte, boucles écourtées) ou strict (coupure). La coupure stricte est
   anticipée par 21 (estimerJetons, plafondIlot) : la part IA des points est perdue la semaine même ;
   la perte est gardée dans etat.travail.ilots[id].pointsPerdusPlafond et notée au journal ici.
   Unités : jetons bruts (org.jetonsSemaine), dollars (org.coutSemaine, org.budget). */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.couts = {
    // Millions de jetons par session et par semaine, selon l'autonomie (0 à 4).
    intensite: [0.5, 3, 10, 15, 17],
    // Un développeur habile gaspille moins : × (plancher − pente × competenceIA).
    competenceIA: [1.2, 0.4],
    // Une pile mal connue de l'agent le fait tourner plus longtemps : × (plancher − pente × adéquation).
    pile: [1.3, 0.3],
    // Prix moyen en dollars par million de jetons (entrée et sortie confondues, avec cache).
    prix: { economique: 2.5, intermediaire: 4, haut: 10, local: 0.1 },
    reductionParTache: 0.35,
    // Hausse de prix du fournisseur : tirée une fois au début du T2 (modèles infonuagiques seulement).
    hausseFournisseur: { trimestre: 2, proba: 0.5, facteur: 1.3 },
    // Boucle d'agent : probabilité par semaine et par session (autonomie ≥ 2), en millions de jetons.
    boucle: { autonomie: 2, proba: 0.004, jetons: [150, 450], souple: 0.3 },
    // Attente d'une CI lente : l'agent relance et interroge en boucle. Fraction de jetons en plus (autonomie ≥ 2),
    // multipliée si un événement a posé org.drapeaux.ciLente.
    attenteCI: { aucune: 0, build: 0.02, tests: 0.05, complet: 0.08 },
    facteurCILente: 4,
    // Plafond par développeur présent (millions de jetons par semaine) ; souple : alerte au-delà.
    plafondParDev: 60,
    // Coûts fixes hebdomadaires ($).
    licenceParDev: 100,        // licence Copilot, postes d'agent et outils, par développeur présent qui a une licence
    outilsParDev: 20,          // IDE et outils, par développeur présent sans licence
    formation: { aucune: 0, atelier: 400, programme: 1200, mentorat: 2000 },
    ciParIlot: { aucune: 0, build: 40, tests: 100, complet: 200 },
    mutationParIlot: 40,
    observabilite: { aucune: 0, base: 500, complete: 1200 },
    passerelleMCP: 350,
    bacASableParIlot: { aucun: 0, conteneur: 60, filtre: 120 },
    classification: { non: 0, manuelle: 300, auto: 600 },
    catalogue: { aucun: 0, inventaire: 300, sources: 700 },
    modeleLocal: 600,          // location d'un GPU modeste
    modeleLocalServeur: 900,   // électricité et entretien du serveur acheté
    // Confiance perdue chaque semaine où le budget est dépassé : base + pente × dépassement relatif, plafonnée.
    confianceDepassement: [0.1, 0.7, 0.8],
    seuilsBudget: [50, 75, 90],
    delaiJournal: 13,          // semaines minimales entre deux avis de plafond pour un îlot
  };

  var M = 1e6;
  function P() { return LTP.systemes.production; }

  // Prix effectif par million de jetons.
  function prixModele(etat) {
    var K = LTP.K.couts, q = LTP.q;
    var cle = P().cleModele(etat);
    var prix = K.prix[cle] != null ? K.prix[cle] : K.prix.intermediaire;
    if (q.levier(etat, 'modeleParTache')) prix *= 1 - K.reductionParTache;
    if (cle !== 'local' && etat.org.drapeaux && etat.org.drapeaux.hausseFournisseur && K.hausseFournisseur) prix *= K.hausseFournisseur.facteur;
    return prix;
  }

  // Jetons prévisibles d'un développeur pour la semaine (sans boucle aléatoire), en jetons bruts.
  function estimerJetons(etat, dev, ilot) {
    var K = LTP.K.couts, p = P();
    var s = dev.sessions || 0;
    if (s <= 0) return 0;
    var a = p.autonomie(etat, ilot.id);
    var adequation = O.borner(ilot.adequationPile + p.modIlot(ilot, 'adequationPile'));
    var j = s * K.intensite[a] * M * Math.max(0.2, K.competenceIA[0] - K.competenceIA[1] * dev.competenceIA) *
      Math.max(0.2, K.pile[0] - K.pile[1] * adequation) *
      (1 + Math.max(0, p.modIlot(ilot, 'poidsContexte'))) * Math.max(0, p.modIlot(ilot, 'jetons'));
    if (a >= K.boucle.autonomie) {
      var ci = LTP.q.levier(etat, 'ci', ilot.id);
      j *= 1 + (K.attenteCI[ci] || 0) * (etat.org.drapeaux.ciLente ? K.facteurCILente : 1);
    }
    return isFinite(j) ? j : 0;
  }

  // Plafond hebdomadaire de l'îlot, en jetons bruts.
  function plafondIlot(etat, ilot, nbDevs) {
    var n = nbDevs != null ? nbDevs : LTP.q.devsDe(etat, ilot.id).length;
    return LTP.K.couts.plafondParDev * M * Math.max(1, n);
  }

  // Valeur effective d'un levier, ou sa valeur programmée si un changement est en attente (on paie dès la décision).
  function levierPaye(etat, id, ilotId) {
    var att = LTP.q.enAttente(etat, id, ilotId);
    return att ? att.valeur : LTP.q.levier(etat, id, ilotId);
  }

  // Coûts fixes de la semaine : { org, ilots: { [id]: $ } }.
  function coutsFixes(etat) {
    var K = LTP.K.couts, q = LTP.q;
    var org = 0, parIlot = {};
    q.devsPresents(etat).forEach(function (d) { org += q.aLicence(etat, d) ? K.licenceParDev : K.outilsParDev; });
    org += K.formation[levierPaye(etat, 'formation')] || 0;
    org += K.observabilite[levierPaye(etat, 'observabilite')] || 0;
    if (levierPaye(etat, 'passerelleMCP')) org += K.passerelleMCP;
    org += K.classification[levierPaye(etat, 'classification')] || 0;
    org += K.catalogue[levierPaye(etat, 'catalogue')] || 0;
    if (P().cleModele(etat) === 'local') org += etat.options && etat.options.serveurAchete ? K.modeleLocalServeur : K.modeleLocal;
    q.ilots(etat).forEach(function (il) {
      var c = (K.ciParIlot[q.levier(etat, 'ci', il.id)] || 0) + (K.bacASableParIlot[q.levier(etat, 'bacASable', il.id)] || 0);
      if (q.levier(etat, 'tests', il.id) === 'mutation') c += K.mutationParIlot;
      parIlot[il.id] = c;
    });
    return { org: org, ilots: parIlot };
  }

  function semaine(etat, ctx) {
    var K = LTP.K.couts, q = LTP.q, p = P(), rng = ctx.rng;
    var T = p.travail(etat);
    // Hausse de prix du fournisseur, tirée une seule fois au trimestre prévu.
    var KH = K.hausseFournisseur;
    if (KH && !T.org.hausseTiree && etat.trimestre >= KH.trimestre) {
      T.org.hausseTiree = true;
      if (rng.chance(KH.proba)) {
        etat.org.drapeaux.hausseFournisseur = true;
        ctx.noter('alerte', 'couts.hausseFournisseur', { pourcent: Math.round((KH.facteur - 1) * 100) }, null);
      }
    }
    var prix = prixModele(etat);
    var fixes = coutsFixes(etat);
    var jetonsTotal = 0, coutTotal = fixes.org;

    q.ilots(etat).forEach(function (ilot) {
      var ti = T.ilots[ilot.id];
      var a = p.autonomie(etat, ilot.id);
      var plafond = q.levier(etat, 'plafondJetons', ilot.id);
      var devs = q.devsDe(etat, ilot.id);
      var jetonsIlot = 0, boucles = [];

      devs.forEach(function (d) {
        var td = T.devs[d.id] || {};
        var j = estimerJetons(etat, d, ilot) * (plafond === 'strict' ? O.borner(td.facteurPlafond == null ? 1 : td.facteurPlafond) : 1);
        // Boucle d'agent : rare, coûteuse, seulement si l'agent travaille seul.
        if (a >= K.boucle.autonomie && d.sessions > 0 && rng.chance(K.boucle.proba * d.sessions)) {
          var extra = rng.entre(K.boucle.jetons[0], K.boucle.jetons[1]) * M;
          if (plafond === 'souple') extra *= K.boucle.souple;
          boucles.push({ dev: d, jetons: extra });
          j += extra;
        }
        d.derniere.jetons = Math.round(j);
        jetonsIlot += j;
      });

      // Plafond strict : rien ne dépasse, les boucles sont coupées net.
      var cap = plafondIlot(etat, ilot, devs.length);
      if (plafond === 'strict' && jetonsIlot > cap) {
        var r = cap / jetonsIlot;
        devs.forEach(function (d) { d.derniere.jetons = Math.round(d.derniere.jetons * r); });
        jetonsIlot = cap;
        boucles = [];
      }
      boucles.forEach(function (b) {
        ctx.noter('alerte', 'couts.boucle', {
          dev: b.dev.id, ilot: ilot.id, jetons: LTP.format.jetons(b.jetons), cout: LTP.format.argent(b.jetons / M * prix),
        }, { ilot: ilot.id, dev: b.dev.id });
      });
      if (plafond === 'souple' && jetonsIlot > cap && p.semainesDepuis(etat, ti, 'plafond') >= K.delaiJournal) {
        ti.avis.plafond = etat.semaine;
        ctx.noter('alerte', 'couts.plafondSouple', { ilot: ilot.id, jetons: LTP.format.jetons(jetonsIlot) }, { ilot: ilot.id });
      }
      if (plafond === 'strict' && ti.pointsPerdusPlafond > 0.5 && p.semainesDepuis(etat, ti, 'plafond') >= K.delaiJournal) {
        ti.avis.plafond = etat.semaine;
        ctx.noter('alerte', 'couts.plafondCoupe', { ilot: ilot.id, points: Math.round(ti.pointsPerdusPlafond) }, { ilot: ilot.id });
      }

      var coutIlot = jetonsIlot / M * prix + (fixes.ilots[ilot.id] || 0);
      ilot.derniere.jetons = Math.round(jetonsIlot);
      ilot.derniere.cout = O.arrondir(coutIlot, 2);
      jetonsTotal += jetonsIlot;
      coutTotal += coutIlot;
    });

    var org = etat.org;
    var avant = org.budget;
    org.jetonsSemaine = Math.round(jetonsTotal);
    org.jetonsCumul = (org.jetonsCumul || 0) + org.jetonsSemaine;
    org.coutSemaine = O.arrondir(coutTotal, 2);
    org.coutCumul = O.arrondir((org.coutCumul || 0) + org.coutSemaine, 2);
    org.budget = O.arrondir(org.budget - org.coutSemaine, 2);

    // Seuils de dépense annoncés une seule fois ; dépassement : la confiance s'effrite chaque semaine.
    var annuel = org.budgetAnnuel || 1;
    var depense = (annuel - org.budget) / annuel * 100;
    K.seuilsBudget.forEach(function (s) {
      if (depense >= s && !T.org.seuilsBudget[s]) {
        T.org.seuilsBudget[s] = etat.semaine;
        ctx.noter('alerte', 'couts.budget', { pourcent: s }, null);
      }
    });
    if (org.budget < 0) {
      if (avant >= 0 && !T.org.budgetDepasse) {
        T.org.budgetDepasse = true;
        ctx.noter('alerte', 'couts.budgetDepasse', {}, null);
      }
      var C = K.confianceDepassement;
      org.confiance = O.borner(org.confiance - Math.min(C[2], C[0] + C[1] * (-org.budget / annuel)), 0, 100);
    }
  }

  LTP.systemes.couts = {
    initialiser: function (etat) { P().travail(etat); },
    semaine: semaine,
    estimerJetons: estimerJetons,
    plafondIlot: plafondIlot,
    prixModele: prixModele,
    coutsFixes: coutsFixes,
  };

  LTP.enregistrerTextes({
    'couts.boucle': '{dev} : un agent a tourné en boucle et brûlé {jetons} jetons ({cout}).',
    'couts.plafondSouple': '{ilot} : alerte de plafond, {jetons} jetons cette semaine.',
    'couts.plafondCoupe': '{ilot} : le plafond strict a coupé les agents en pleine tâche. Environ {points} points de travail perdus.',
    'couts.budget': '{pourcent} % du budget annuel de l\'IA et des outils est dépensé.',
    'couts.budgetDepasse': 'Le budget annuel est dépassé. Régis a réservé une salle pour en parler.',
    'couts.hausseFournisseur': 'Le fournisseur de modèles augmente ses tarifs de {pourcent} %. Chaque jeton coûte désormais plus cher.',
  });
})(globalThis.LTP = globalThis.LTP || {});
