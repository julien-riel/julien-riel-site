/* 32-fin.js */
/* Comités trimestriels (CONCEPTION §2.2, §7), bilans de trimestre, score et profil de fin (§2.3).
   Contenu : 15-donnees-comites.js (questions), 17-donnees-fins.js (profils et textes).
   Champs propres : etat.fin = { reference, bilans, questionsPosees, cumulPrecedent }. */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.fin = {
    seuilConfiance: 15,           // sous ce seuil, la partie se termine (profil « congedie »)
    questionsParComite: 3,
    poidsDefaut: 1,
    penaliteMemePersonnage: 0.25, // poids d'une question dont le personnage a déjà une question au même comité
    penaliteDejaPosee: 0.2,       // poids d'une question déjà posée à un comité précédent
    confianceParPoint: 6,         // confiance gagnée ou perdue par point de score (−1..1) d'une réponse

    // Débit de référence sans IA (points par semaine et par niveau × compétence) si 21-production n'en fournit pas.
    baseProduction: { stagiaire: 2, junior: 3, inter: 5, senior: 6 },
    facteurReference: 0.8,        // part du temps de code d'une semaine sans IA (calée sur une partie sans IA : ratio ≈ 1)

    // Critères (0–1) : bornes des normalisations.
    criteres: {
      debit: [0.7, 1.4],          // ratio des points livrés / référence sans IA → 0..1
      poidsDebitLivraison: 0.6,   // le reste : satisfaction finale
      incidentsParSemaine: 1.5,   // au-delà, la note d'incidents tombe à 0
      comprehension: [0.3, 0.8],
      couverture: [0.2, 0.7],
      poidsQualite: { incidents: 0.4, comprehension: 0.35, couverture: 0.25 },
      // Ratio dépense / budget au prorata : 1 jusqu'à 0,6 ; 0,8 au budget ; 0 à 1,6.
      budget: { pleinAvant: 0.6, auBudget: 0.8, nulApres: 1.6 },
      risques: { surface: 0.3, loi25: 0.15, fuites: 0.25, securite: 0.1, fantomes: 0.2, fuitesRef: 2, securiteRef: 4, fantomesRef: 5 },
      personnes: { moral: 0.35, fatigue: 0.2, releve: 0.25, departs: 0.2, departsRef: 4 },
    },
    poidsScore: { livraison: 0.25, qualite: 0.2, couts: 0.15, risques: 0.2, personnes: 0.2 },
    malusCongedie: 0.6,           // le score est multiplié par ce facteur si la partie se termine tôt
    etoilesCongedieMax: 1,

    // Règles de profil. Les règles fermes passent en premier ; sinon le profil le plus proche l'emporte.
    profils: {
      farwest: { fuitesMajeures: 1, surface: 0.45 },      // règle ferme : fuite majeure et surface d'attaque élevée
      facture: { depassement: 1.5, criteresMin: 0.45 },   // règle ferme : budget dépassé de 50 % et le reste correct
      // Affinités : [min, max] des normalisations utilisées par le profil le plus proche.
      // lgtm : part des PR fusionnées en « LGTM » (celles que l'agent fusionne seul n'y sont pas : bornes basses).
      usine: { debit: [0.95, 1.5], lgtm: [0.05, 0.3], incidents: [0.3, 1.5], dette: [0.3, 0.6] },
      // planDebit : part de l'affinité « musée » qui subsiste quand le débit a nettement monté (sinon ce n'est pas un musée).
      musee: { planDebit: 0.4, debit: [0.75, 1.15], satisfaction: [0.3, 0.55], fantomes: [1, 6] },
      chateau: { volume: [1.1, 1.8], dette: [0.4, 0.7], releve: [0.2, 0.45] },
      // planDebit : part de l'affinité « réussie » acquise même sans hausse de débit (le reste dépend du débit).
      // exigence : poids du pire terme (0 : simple moyenne) ; satisfaction : les directions doivent suivre.
      // « Transition » réussie : il faut qu'il y ait eu transition (autonomie moyenne ou sessions parallèles au-dessus du départ).
      reussie: { autonomieMin: 1.6, sessionsMin: 1.6, planDebit: 0.4, exigence: 0.5, debit: [0.95, 1.25], incidents: [0.15, 0.6], comprehension: [0.5, 0.8], surface: [0.25, 0.5],
        budget: [1.1, 1.5], satisfaction: [0.25, 0.5] },
      farwestAffinite: { surface: [0.3, 0.7], fuites: [0, 2], loi25: [0.2, 0.6] },
    },
  };
  var K = LTP.K.fin;

  function num(x, d) { return typeof x === 'number' && isFinite(x) ? x : (d == null ? 0 : d); }
  // Normalise v de [a, b] vers [0, 1].
  function norm(v, ab) { return ab[1] === ab[0] ? (v >= ab[1] ? 1 : 0) : O.borner((v - ab[0]) / (ab[1] - ab[0])); }

  function debitReference(etat) {
    // Même facteur de compétence que 21-production, × la part du temps réellement passée à coder sans IA
    // (fatigue, revue, réunions) : un ratio de 1 veut dire « le débit de l'équipe sans IA ».
    var KP = LTP.K.production || {};
    var base = KP.base || K.baseProduction;
    var plancher = KP.competencePlancher != null ? KP.competencePlancher : 0, pente = KP.competencePente != null ? KP.competencePente : 1;
    var s = 0;
    LTP.q.devs(etat).forEach(function (d) {
      s += num(base[d.niveau], K.baseProduction[d.niveau] || 4) * (plancher + pente * num(d.competence, 0.5));
    });
    return Math.max(1, s * K.facteurReference);
  }

  function initialiser(etat) {
    var ilots = LTP.q.ilots(etat);
    etat.fin = {
      reference: {
        debit: debitReference(etat),
        volume: O.somme(ilots, function (il) { return num(il.codebase && il.codebase.volume); }),
        comprehension: O.moyenne(LTP.q.devs(etat), 'comprehension'),
        satisfaction: O.moyenne(ilots, 'satisfaction'),
        budget: num(etat.org.budgetAnnuel),
        devs: Object.keys(etat.devs).length,
      },
      bilans: [],
      questionsPosees: [],
      cumulPrecedent: {},
    };
  }
  function propre(etat) { if (!etat.fin) initialiser(etat); return etat.fin; }

  function sur(f, args, defaut) { try { return f.apply(null, args); } catch (e) { return defaut; } }
  function texteDe(x, args) {
    if (typeof x === 'function') { var r = sur(x, args, ''); return r == null ? '' : String(r); }
    return x == null ? '' : String(x);
  }
  function defQuestion(id) { return (LTP.donnees.comites || []).filter(function (c) { return c && c.id === id; })[0]; }

  // --- Comités ---
  function admissible(etat, def) {
    if (!def || !def.id || !Array.isArray(def.reponses) || !def.reponses.length) return false;
    if (Array.isArray(def.trimestres) && def.trimestres.indexOf(etat.trimestre) < 0) return false;
    var p = def.poids == null ? K.poidsDefaut : Number(def.poids);
    if (!(p > 0)) return false;
    if (typeof def.condition === 'function' && !sur(def.condition, [etat, LTP.q], false)) return false;
    return true;
  }

  function preparerComite(etat, ctx) {
    var f = propre(etat);
    var candidats = (LTP.donnees.comites || []).filter(function (d) { return admissible(etat, d); });
    var choisis = [], personnages = {};
    var rng = ctx && ctx.rng ? ctx.rng : LTP.rng.creer(etat);
    while (choisis.length < K.questionsParComite && candidats.length) {
      var def = rng.choisirPondere(candidats, function (d) {
        var p = d.poids == null ? K.poidsDefaut : Number(d.poids);
        if (personnages[d.personnage]) p *= K.penaliteMemePersonnage;
        if (f.questionsPosees.indexOf(d.id) >= 0) p *= K.penaliteDejaPosee;
        return p;
      });
      if (!def) break;
      choisis.push(def);
      personnages[def.personnage] = true;
      candidats = candidats.filter(function (d) { return d !== def; });
    }
    etat.comite = {
      trimestre: etat.trimestre,
      // reponses : identifiants seulement (l'état reste du JSON), pour qu'une stratégie par défaut puisse répondre.
      questions: choisis.map(function (d) {
        return { id: d.id, personnage: d.personnage || null, reponses: d.reponses.map(function (r) { return { id: r.id }; }) };
      }),
      reponses: {},
      scores: {},
      reactions: {},
    };
    choisis.forEach(function (d) { if (f.questionsPosees.indexOf(d.id) < 0) f.questionsPosees.push(d.id); });
  }

  function repondreComite(etat, questionId, reponseId, ctx) {
    var c = etat.comite;
    if (!c) return 'pas-de-comite';
    var qq = (c.questions || []).filter(function (x) { return x.id === questionId; })[0];
    if (!qq) return 'question-inconnue';
    if (c.reponses[questionId] != null) return 'deja-repondu';
    var def = defQuestion(questionId);
    if (!def) return 'question-inconnue';
    var rep = (def.reponses || []).filter(function (r) { return r && r.id === reponseId; })[0];
    if (!rep) return 'reponse-inconnue';
    var score = typeof rep.evaluer === 'function' ? num(sur(rep.evaluer, [etat, LTP.q], 0)) : 0;
    score = Math.max(-1, Math.min(1, score));
    var avant = etat.org.confiance;
    etat.org.confiance = O.borner(avant + score * K.confianceParPoint, 0, 100);
    c.reponses[questionId] = reponseId;
    if (!c.scores) c.scores = {};
    if (!c.reactions) c.reactions = {};
    c.scores[questionId] = score;
    c.reactions[questionId] = texteDe(rep.reaction, [etat, LTP.q, score]);
    if (ctx && ctx.noter) {
      ctx.noter('comite', score >= 0 ? 'fin.reponseBonne' : 'fin.reponseMauvaise',
        { personnage: def.personnage, delta: O.arrondir(etat.org.confiance - avant, 1) });
    }
    return null;
  }

  function comiteTermine(etat) {
    var c = etat.comite;
    if (!c) return true;
    return (c.questions || []).every(function (x) { return (c.reponses && c.reponses[x.id] != null) || !defQuestion(x.id); });
  }

  // --- Bilan du trimestre qui vient de se terminer (appelé après trimestre += 1) ---
  // Différence entre les cumuls de la partie (30-metriques) maintenant et à la fin du trimestre précédent.
  var CHAMPS_BILAN = ['pointsLivres', 'prFusionnees', 'incidents', 'fuites', 'bogues'];
  function debutTrimestre(etat, ctx) {
    var f = propre(etat);
    var t = etat.trimestre - 1;
    if (t < 1) return;
    var c = etat.cumuls || {}, avant = f.cumulPrecedent || {};
    var bilan = { trimestre: t };
    CHAMPS_BILAN.forEach(function (k) { bilan[k] = Math.round(num(c[k]) - num(avant[k])); });
    bilan.cout = Math.round(num(etat.org.coutCumul) - num(avant.coutCumul));
    bilan.confiance = Math.round(num(etat.org.confiance));
    f.bilans.push(bilan);
    f.cumulPrecedent = { coutCumul: num(etat.org.coutCumul) };
    CHAMPS_BILAN.forEach(function (k) { f.cumulPrecedent[k] = num(c[k]); });
    if (ctx && ctx.noter) ctx.noter('info', 'fin.bilanTrimestre', bilan);
  }

  // --- Résultat de la partie ---
  function faitsDe(etat, raison) {
    var f = propre(etat), ref = f.reference, q = LTP.q;
    var c = etat.cumuls || {};
    var m = etat.metriques || {};
    var semaines = Math.max(1, num(c.semaines, etat.semaine) || etat.semaine || 1);
    var ilots = q.ilots(etat), devs = q.devsPresents(etat);
    var met = LTP.systemes.metriques;
    var surface = num(m.surfaceAttaque) / 100, loi25 = num(m.expositionLoi25) / 100;
    var budgetDepense = Math.max(0, num(ref.budget) - num(etat.org.budget));
    var prorata = num(etat.org.budgetAnnuel, ref.budget) * Math.min(1, semaines / LTP.K.etat.semainesTotal);
    var departs = (etat.personnes && etat.personnes.departs) || [];
    var volume = O.somme(ilots, function (il) { return num(il.codebase && il.codebase.volume); });
    return {
      raison: raison,
      semaines: semaines,
      pointsLivres: Math.round(num(c.pointsLivres)),
      pointsReference: Math.round(ref.debit * semaines),
      ratioDebit: O.arrondir(num(c.pointsLivres) / Math.max(1, ref.debit * semaines), 3),
      prFusionnees: Math.round(num(c.prFusionnees)),
      lgtm: Math.round(num(c.lgtm)),
      autonomieMoyenne: O.arrondir(num(c.autonomie) / Math.max(1, num(c.semaines)), 2),
      sessionsMoyennes: O.arrondir(num(c.sessions) / Math.max(1, num(c.semaines)), 2),
      partLgtm: O.arrondir(num(c.lgtm) / Math.max(1, num(c.prFusionnees) || num(c.pr)), 3),
      incidents: num(c.incidents),
      incidentsParSemaine: O.arrondir(num(c.incidents) / semaines, 3),
      incidentsGraves: num(c.incidentsGraves),
      incidentsSecurite: num(c.incidentsSecurite),
      fuites: num(c.fuites),
      // 25-donnees-secu pose aussi le drapeau « fuiteMajeure » (une fuite créée par un événement compte aussi).
      fuitesMajeures: Math.max(num(c.fuitesMajeures), etat.org.drapeaux && etat.org.drapeaux.fuiteMajeure ? 1 : 0),
      bogues: Math.round(num(c.bogues)),
      budgetAnnuel: num(etat.org.budgetAnnuel),
      budgetDepense: Math.round(budgetDepense),
      budgetRestant: Math.round(num(etat.org.budget)),
      ratioBudget: O.arrondir(budgetDepense / Math.max(1, prorata), 3),
      jetonsCumul: num(etat.org.jetonsCumul),
      comprehension: O.arrondir(O.moyenne(devs, 'comprehension'), 3),
      comprehensionDepart: O.arrondir(num(ref.comprehension), 3),
      couvertureReelle: O.arrondir(num(m.couvertureReelle), 3),
      couvertureApparente: O.arrondir(num(m.couvertureApparente), 3),
      volumeCode: Math.round(volume),
      volumeDepart: Math.round(num(ref.volume)),
      ratioVolume: O.arrondir(volume / Math.max(1, num(ref.volume)), 3),
      satisfaction: O.arrondir(O.moyenne(ilots, 'satisfaction'), 3),
      satisfactionDepart: O.arrondir(num(ref.satisfaction), 3),
      applisFantomes: Math.round(O.somme(ilots, function (il) { return num(il.applisFantomes); })),
      surfaceAttaque: O.arrondir(surface, 3),
      expositionLoi25: O.arrondir(loi25, 3),
      moral: O.arrondir(O.moyenne(devs, 'moral'), 3),
      fatigue: O.arrondir(O.moyenne(devs, 'fatigue'), 3),
      releve: O.arrondir(num(etat.org.releve), 3),
      departs: departs.filter(function (d) { return d.motif === 'demission'; }).length,
      finsContrat: departs.filter(function (d) { return d.motif !== 'demission'; }).length,
      partis: departs.map(function (d) { return d.dev; }),
      confiance: Math.round(num(etat.org.confiance)),
      mttr: O.arrondir(met && met.mttrExact ? num(met.mttrExact(etat), 8) : num(m.mttr, 8), 1),
      evenements: (etat.evenementsCompte && etat.evenementsCompte.tires) || 0,
    };
  }

  function criteresDe(faits) {
    var KC = K.criteres;
    var livraison = KC.poidsDebitLivraison * norm(faits.ratioDebit, KC.debit) + (1 - KC.poidsDebitLivraison) * O.borner(faits.satisfaction);
    var pq = KC.poidsQualite;
    var qualite = pq.incidents * (1 - O.borner(faits.incidentsParSemaine / KC.incidentsParSemaine))
      + pq.comprehension * norm(faits.comprehension, KC.comprehension)
      + pq.couverture * norm(faits.couvertureReelle, KC.couverture);
    // Coûts : 1 jusqu'à pleinAvant, auBudget à 100 % du budget au prorata, 0 à nulApres (le dépassement coûte vite).
    var B = KC.budget, rb = faits.ratioBudget;
    var couts = B.auBudget == null ? 1 - norm(rb, [B.pleinAvant, B.nulApres])
      : rb <= 1 ? 1 - (1 - B.auBudget) * norm(rb, [B.pleinAvant, 1]) : B.auBudget * (1 - norm(rb, [1, B.nulApres]));
    var r = KC.risques;
    var risques = 1 - r.surface * O.borner(faits.surfaceAttaque) - r.loi25 * O.borner(faits.expositionLoi25)
      - r.fuites * O.borner(faits.fuites / r.fuitesRef) - r.securite * O.borner(faits.incidentsSecurite / r.securiteRef)
      - r.fantomes * O.borner(faits.applisFantomes / r.fantomesRef);
    var p = KC.personnes;
    var personnes = p.moral * O.borner(faits.moral) + p.fatigue * (1 - O.borner(faits.fatigue)) + p.releve * O.borner(faits.releve)
      + p.departs * (1 - O.borner(faits.departs / p.departsRef));
    return {
      livraison: O.arrondir(O.borner(livraison), 3),
      qualite: O.arrondir(O.borner(qualite), 3),
      couts: O.arrondir(O.borner(couts), 3),
      risques: O.arrondir(O.borner(risques), 3),
      personnes: O.arrondir(O.borner(personnes), 3),
    };
  }

  function moyenneListe(l) { return l.length ? l.reduce(function (a, b) { return a + b; }, 0) / l.length : 0; }
  // Moyenne tirée vers le pire terme : (1 − e) × moyenne + e × minimum.
  function exigeant(l, e) {
    var x = num(e, 0);
    return l.length ? (1 - x) * moyenneListe(l) + x * Math.min.apply(null, l) : 0;
  }

  // Affinité (0–1) de chaque profil, pour le cas où aucune règle ferme ne s'applique.
  function affinites(faits) {
    var P = K.profils;
    var dette = 1 - faits.comprehension;
    return {
      usine: moyenneListe([norm(faits.ratioDebit, P.usine.debit), norm(faits.partLgtm, P.usine.lgtm),
        norm(faits.incidentsParSemaine, P.usine.incidents), norm(dette, P.usine.dette)]),
      musee: moyenneListe([1 - norm(faits.ratioDebit, P.musee.debit), 1 - norm(faits.satisfaction, P.musee.satisfaction),
        Math.max(norm(faits.applisFantomes, P.musee.fantomes), 1 - O.borner(faits.surfaceAttaque * 2))])
        * (P.musee.planDebit == null ? 1 : P.musee.planDebit + (1 - P.musee.planDebit) * (1 - norm(faits.ratioDebit, P.musee.debit))),
      chateau: moyenneListe([norm(faits.ratioVolume, P.chateau.volume), norm(dette, P.chateau.dette), 1 - norm(faits.releve, P.chateau.releve)]),
      // La transition réussie exige un débit en hausse : sans lui, la prudence ressemble au musée.
      // Elle exige aussi toutes ses conditions à la fois : la pire compte pour une part « exigence ».
      reussie: exigeant([1 - norm(faits.incidentsParSemaine, P.reussie.incidents),
        norm(faits.comprehension, P.reussie.comprehension), 1 - norm(faits.surfaceAttaque, P.reussie.surface),
        1 - norm(faits.ratioBudget, P.reussie.budget), norm(faits.satisfaction, P.reussie.satisfaction)], P.reussie.exigence)
        * (P.reussie.planDebit + (1 - P.reussie.planDebit) * norm(faits.ratioDebit, P.reussie.debit)),
      farwest: moyenneListe([norm(faits.surfaceAttaque, P.farwestAffinite.surface), norm(faits.fuites, P.farwestAffinite.fuites),
        norm(faits.expositionLoi25, P.farwestAffinite.loi25)]) * (faits.fuitesMajeures > 0 ? 1 : 0.6),
    };
  }

  function profilDe(raison, faits, criteres) {
    var P = K.profils;
    if (raison === 'confiance') return 'congedie';
    if (faits.fuitesMajeures >= P.farwest.fuitesMajeures && faits.surfaceAttaque >= P.farwest.surface) return 'farwest';
    var autres = moyenneListe([criteres.livraison, criteres.qualite, criteres.risques, criteres.personnes]);
    if (faits.ratioBudget >= P.facture.depassement && autres >= P.facture.criteresMin) return 'facture';
    var a = affinites(faits), meilleur = null;
    var transition = num(faits.autonomieMoyenne) >= P.reussie.autonomieMin || num(faits.sessionsMoyennes) >= P.reussie.sessionsMin;
    ['reussie', 'usine', 'musee', 'chateau', 'farwest'].forEach(function (id) {
      if (id === 'reussie' && !transition) return;
      if (meilleur == null || a[id] > a[meilleur] + 1e-9) meilleur = id;
    });
    return meilleur;
  }

  function calculerResultat(etat, ctx, raison) {
    var faits = faitsDe(etat, raison);
    var criteres = criteresDe(faits);
    var ps = K.poidsScore, total = 0, poids = 0;
    for (var k in ps) { total += ps[k] * num(criteres[k]); poids += ps[k]; }
    var score = 100 * (poids > 0 ? total / poids : 0);
    if (raison === 'confiance') score *= K.malusCongedie;
    score = Math.round(Math.max(0, Math.min(100, score)));
    var etoiles = Math.max(0, Math.min(5, Math.round(score / 20)));
    if (raison === 'confiance') etoiles = Math.min(etoiles, K.etoilesCongedieMax);
    var profil = profilDe(raison, faits, criteres);
    var ids = (LTP.donnees.fins || []).map(function (x) { return x.id; });
    if (ids.length && ids.indexOf(profil) < 0) profil = ids.indexOf('reussie') >= 0 ? 'reussie' : ids[0];
    if (ctx && ctx.noter) ctx.noter('info', 'fin.resultat', { profil: profil, etoiles: etoiles, score: score });
    return { raison: raison, profil: profil, etoiles: etoiles, score: score, criteres: criteres, faits: faits, affinites: arrondirTout(affinites(faits)) };
  }
  function arrondirTout(o) { var r = {}; for (var k in o) r[k] = O.arrondir(num(o[k]), 3); return r; }

  LTP.systemes.fin = {
    seuilConfiance: K.seuilConfiance,
    initialiser: initialiser,
    preparerComite: preparerComite,
    repondreComite: repondreComite,
    comiteTermine: comiteTermine,
    debutTrimestre: debutTrimestre,
    calculerResultat: calculerResultat,
    // Exposés pour les tests et l'équilibrage.
    faits: faitsDe,
    criteres: criteresDe,
    affinites: affinites,
    profil: profilDe,
  };
  // La boucle lit seuilConfiance sur le système : on le garde synchronisé avec la constante (équilibrage).
  Object.defineProperty(LTP.systemes.fin, 'seuilConfiance', { get: function () { return K.seuilConfiance; }, enumerable: true });

  LTP.enregistrerTextes({
    'fin.reponseBonne': 'Le comité hoche la tête. Confiance : +{delta}.',
    'fin.reponseMauvaise': 'Le comité fronce les sourcils. Confiance : {delta}.',
    'fin.bilanTrimestre': 'Bilan du trimestre {trimestre} : {pointsLivres} points livrés et {prFusionnees} PR fusionnées. Incidents : {incidents}.',
    'fin.resultat': 'Bilan annuel : {etoiles} sur 5 étoiles.',
  });
})(globalThis.LTP = globalThis.LTP || {});
