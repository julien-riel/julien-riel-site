/* 35-demo.js */
/* États de démonstration pour développer et tester l'interface sans dépendre de l'équilibre du moteur.
   LTP.demo.etat(nom) → état complet et plausible. Chargé dans la page : ?demo=<nom>.
   Les valeurs sont injectées à la main ; elles ne prétendent pas sortir de la simulation. */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  var NOMS = ['debut', 't2-tendu', 'triade-appro', 'evenement', 'comite', 'fin-usine'];

  // Contenu de secours, ajouté seulement s'il n'existe pas déjà dans les données.
  function assurerContenuDemo() {
    if (!LTP.q.defEvenement('pr-geante')) {
      LTP.donnees.evenements.push({
        id: 'pr-geante', theme: 'revue', trimestres: [1, 2], poids: 0, personnage: 'nadia',
        titre: 'La PR de 4 000 lignes', texte: 'Mathieu a ouvert une PR de 4 000 lignes.',
        choix: [
          { id: 'approuver', libelle: 'On fusionne', consequence: 'Personne ne l\'a lue.', effets: [] },
          { id: 'decouper', libelle: 'On découpe', consequence: 'Nadia respire.', effets: [] },
        ],
      });
    }
    if (!(LTP.donnees.comites || []).some(function (c) { return c.id === 'demo-regis'; })) {
      LTP.donnees.comites = LTP.donnees.comites || [];
      LTP.donnees.comites.push({
        id: 'demo-regis', personnage: 'regis', trimestres: [], poids: 0,
        question: 'La facture de jetons a doublé ce trimestre. Je résume : on paie deux fois plus pour la même chose ?',
        reponses: [
          { id: 'a', texte: 'Non : le débit a augmenté de 60 % et voici le coût par point livré.', evaluer: function () { return 1; }, reaction: 'Un chiffre. Enfin.' },
          { id: 'b', texte: 'C\'est un investissement stratégique.', evaluer: function () { return -0.5; }, reaction: 'Tout le monde dit ça.' },
          { id: 'c', texte: 'Ce n\'est pas vraiment le double.', evaluer: function () { return -1; }, reaction: 'J\'ai la facture sous les yeux.' },
        ],
      });
    }
  }

  // Historique synthétique : n semaines autour d'une tendance, avec un peu de bruit.
  function serie(rng, n, depart, arrivee, bruit, entier) {
    var s = [];
    for (var i = 0; i <= n; i++) {
      var v = O.lerp(depart, arrivee, n ? i / n : 1) + rng.bruit(bruit);
      v = Math.max(0, v);
      s.push(entier ? Math.round(v) : v);
    }
    return s;
  }

  function remplir(etat, semaine, p) {
    var rng = LTP.rng.creer({ rngEtat: 777 + semaine });
    etat.semaine = semaine;
    etat.trimestre = Math.min(4, Math.floor((semaine - 1) / 13) + 1 || 1);
    var etats = ['code', 'code', 'code', 'revue', 'code', 'attente', 'code'];
    LTP.q.devs(etat).forEach(function (d, i) {
      d.sessions = Math.max(1, Math.min(6, Math.round(1 + p.sessions * rng.suivant() + (LTP.q.aTrait(d, 'multiClauding') || LTP.q.aTrait(d, 'dixX') ? 2 : 0))));
      d.fatigue = O.borner(0.2 + p.fatigue * rng.suivant() + (d.sessions > 3 ? 0.2 : 0));
      d.moral = O.borner(0.7 - p.fatigue * 0.4 * rng.suivant());
      d.comprehension = O.borner(d.comprehension - p.dette * rng.suivant());
      d.competenceIA = O.borner(d.competenceIA + 0.2 * p.sessions / 4);
      d.etat = etats[i % etats.length];
      if (d.fatigue > 0.85) d.etat = 'conge';
      d.derniere = { points: 4 + rng.entier(0, 6), code: 0.5 + rng.suivant() * 3 * p.sessions, pr: rng.entier(1, 3 + p.sessions * 2),
        revues: rng.entier(0, 4), lgtm: rng.entier(0, p.lgtm), jetons: Math.round(2e5 * d.sessions * (1 + rng.suivant())), ecritSoiMeme: 0.4, luEnRevue: 0.5 };
    });
    LTP.q.ilots(etat).forEach(function (il) {
      il.fileRevue = Math.round(p.file * (0.5 + rng.suivant()) * (il.id === 'portail' ? 2 : 1));
      il.backlog = Math.round(il.backlog * (0.7 + rng.suivant() * 0.8));
      il.satisfaction = O.borner(il.satisfaction + rng.bruit(0.1));
      il.codebase.volume = Math.round(il.codebase.volume * (1 + p.croissance * rng.suivant()));
      il.codebase.couvertureApparente = O.borner(il.codebase.couvertureApparente + p.croissance * 0.5);
      il.codebase.couvertureReelle = O.borner(il.codebase.couvertureReelle + p.croissance * 0.15);
      var pts = Math.round(15 + 20 * p.sessions * rng.suivant());
      il.derniere = { points: pts, pointsLivres: Math.round(pts * 0.8), code: pts * 0.3, pr: rng.entier(3, 6 + p.sessions * 5), prFusionnees: rng.entier(3, 5 + p.sessions * 4),
        deploiements: rng.entier(1, 6), defautsInjectes: rng.entier(2, 12), defautsEchappes: rng.entier(0, 5), bogues: rng.entier(0, 4), incidents: 0,
        jetons: Math.round(8e5 * (1 + p.sessions) * rng.suivant()), cout: Math.round(800 * (1 + p.sessions) * rng.suivant()),
        delai: O.arrondir(1 + p.file * 0.2 * rng.suivant(), 1), attention: O.borner(1 - p.lgtm * 0.1), lgtm: rng.entier(0, p.lgtm), demandes: 15, demandesRefusees: 2 };
    });
    for (var k = 0; k < p.incidents; k++) {
      var il = etat.ordreIlots[(k * 3 + 1) % etat.ordreIlots.length];
      etat.incidents.push({ id: 'demo' + k, ilot: il, gravite: 0.4 + 0.2 * k, cause: k ? 'injection' : 'bogue', debut: semaine, mttrHeures: 14, finSemaine: semaine + 1, resolu: false });
      var dev = LTP.q.devsDe(etat, il)[0];
      if (dev) dev.etat = 'incident';
    }
    etat.org.jetonsSemaine = O.somme(LTP.q.ilots(etat), function (i) { return i.derniere.jetons; });
    etat.org.coutSemaine = O.somme(LTP.q.ilots(etat), function (i) { return i.derniere.cout; });
    etat.org.budget = etat.org.budgetAnnuel * (1 - semaine / 52 * p.depense);
    etat.org.confiance = p.confiance;

    // Métriques et historique synthétiques
    var n = semaine;
    var h = {
      prFusionnees: serie(rng, n, 20, 20 + 60 * p.sessions / 4, 4, true),
      pointsLivres: serie(rng, n, 95, 95 + 50 * p.sessions / 4, 6, true),
      delai: serie(rng, n, 1.5, 1.5 + p.file * 0.25, 0.2),
      frequenceDeploiement: serie(rng, n, 4, 12, 1, true),
      backlog: serie(rng, n, 215, 215 + 40 * p.dette, 8, true),
      satisfaction: serie(rng, n, 0.55, 0.55 + 0.1 - p.dette * 0.3, 0.03),
      bogues: serie(rng, n, 4, 4 + 10 * p.dette, 1.5, true),
      incidents: serie(rng, n, 0, p.incidents, 0.6, true),
      tauxEchec: serie(rng, n, 0.12, 0.12 + 0.2 * p.dette, 0.02),
      mttr: serie(rng, n, 6, 6 + 14 * p.dette, 1),
      volumeCode: serie(rng, n, 700, 700 * (1 + p.croissance), 5, true),
      couvertureApparente: serie(rng, n, 0.35, 0.35 + p.croissance * 0.5, 0.01),
      couvertureReelle: serie(rng, n, 0.28, 0.28 + p.croissance * 0.15, 0.01),
      comprehension: serie(rng, n, 0.75, 0.75 - p.dette * 0.5, 0.01),
      jetonsSemaine: serie(rng, n, 1e6, etat.org.jetonsSemaine, 2e5, true),
      coutSemaine: serie(rng, n, 1200, etat.org.coutSemaine, 200, true),
      budgetRestant: serie(rng, n, etat.org.budgetAnnuel, etat.org.budget, 0, true),
      retard: serie(rng, n, 0.2, 0.2 + p.dette * 0.3, 0.02),
      surfaceAttaque: serie(rng, n, 15, 15 + 60 * p.risque, 2, true),
      expositionLoi25: serie(rng, n, 20, 20 + 50 * p.risque, 2, true),
      applisFantomes: serie(rng, n, 0, 6 * p.risque, 0.3, true),
      cartesOrphelines: serie(rng, n, 0, 5 * p.risque, 0.3, true),
      moral: serie(rng, n, 0.65, 0.65 - p.fatigue * 0.3, 0.02),
      fatigue: serie(rng, n, 0.2, 0.2 + p.fatigue * 0.5, 0.02),
      releve: serie(rng, n, 0.5, 0.5 - p.dette * 0.2, 0.01),
      connaissanceCapturee: serie(rng, n, 0.1, 0.1 + 0.3 * (1 - p.risque), 0.01),
      confiance: serie(rng, n, 60, p.confiance, 2, true),
    };
    etat.historique = h;
    etat.metriques = {};
    for (var key in h) etat.metriques[key] = h[key][h[key].length - 1];
    etat.metriques.incidents = LTP.q.incidentsActifs(etat).length;
  }

  function equiper(etat, carteId, ilotId) {
    if (!etat.capacites[carteId]) return;
    etat.capacites[carteId].decouverte = true;
    etat.capacites[carteId].equipeeSur.push(ilotId);
    etat.ilots[ilotId].capacites.push(carteId);
  }

  function etatDemo(nom) {
    assurerContenuDemo();
    var etat = LTP.moteur.creer(42, { animateur: true });
    etat.phase = 'planification';
    var L = etat.leviers.global;
    switch (nom) {
      case 't2-tendu':
        L.autonomie = 3; L.sessionsMax = 4; L.modele = 'haut';
        etat.leviers.surcharges.iot = { autonomie: 1 };
        etat.leviers.enAttente.push({ id: 'formation', valeur: 'programme', ilot: null, semaine: 23 });
        remplir(etat, 20, { sessions: 3, fatigue: 0.4, dette: 0.35, lgtm: 6, file: 8, croissance: 0.3, incidents: 2, risque: 0.4, depense: 1.3, confiance: 48 });
        equiper(etat, 'conventions-gpu', 'portail');
        etat.phase = 'execution';
        break;
      case 'triade-appro':
        L.autonomie = 3; L.sessionsMax = 3;
        remplir(etat, 30, { sessions: 2, fatigue: 0.3, dette: 0.25, lgtm: 3, file: 5, croissance: 0.25, incidents: 1, risque: 0.7, depense: 1.1, confiance: 55 });
        equiper(etat, 'mcp-courriel-fournisseurs', 'appro');
        etat.ilots.appro.triade = { prive: true, nonFiable: true, sortie: true };
        etat.ilots.web.triade = { prive: false, nonFiable: true, sortie: true };
        etat.phase = 'execution';
        break;
      case 'evenement':
        L.autonomie = 2; L.sessionsMax = 2;
        remplir(etat, 8, { sessions: 1, fatigue: 0.2, dette: 0.1, lgtm: 2, file: 4, croissance: 0.1, incidents: 0, risque: 0.2, depense: 1, confiance: 60 });
        etat.phase = 'execution';
        etat.evenementEnAttente = { id: 'pr-geante', ref: { ilot: 'portail', dev: 'mathieu' }, vars: {} };
        break;
      case 'comite':
        L.autonomie = 3; L.sessionsMax = 3;
        remplir(etat, 26, { sessions: 2, fatigue: 0.35, dette: 0.3, lgtm: 4, file: 6, croissance: 0.25, incidents: 1, risque: 0.4, depense: 1.4, confiance: 52 });
        etat.trimestre = 2;
        etat.phase = 'comite';
        etat.comite = { trimestre: 2, questions: [{ id: 'demo-regis' }], reponses: {} };
        break;
      case 'fin-usine':
        L.autonomie = 4; L.sessionsMax = 6; L.reviseurs = 0;
        remplir(etat, 52, { sessions: 4, fatigue: 0.6, dette: 0.6, lgtm: 10, file: 12, croissance: 0.8, incidents: 3, risque: 0.8, depense: 1.6, confiance: 34 });
        etat.trimestre = 4;
        etat.phase = 'fin';
        etat.resultat = { raison: 'annee', profil: 'usine', etoiles: 2, score: 41,
          criteres: { livraison: 0.85, qualite: 0.3, couts: 0.35, risques: 0.4, personnes: 0.35 } };
        break;
      default: // debut
        break;
    }
    return etat;
  }

  LTP.demo = { noms: NOMS, etat: etatDemo };
})(globalThis.LTP = globalThis.LTP || {});
