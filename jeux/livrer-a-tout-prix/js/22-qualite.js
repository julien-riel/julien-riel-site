/* 22-qualite.js */
/* Étape 5 de la semaine : revue, cascade de défauts, couverture de tests et agent qui triche (CONCEPTION §4.4).
   La file de revue détaillée vit dans etat.travail.ilots[id].file = { pr, points, lignes, defauts } ;
   ilot.fileRevue (nombre de PR, arrondi) en est le reflet public. Un effet qui modifie ilot.fileRevue
   est réconcilié au début de l'étape : on ajoute ou on retire des PR de composition moyenne.
   Écrit ilot.derniere.{prFusionnees, pointsLivres, deploiements, defautsInjectes, defautsEchappes, delai,
   attention, lgtm}, dev.derniere.{revues, lgtm, luEnRevue}, ilot.backlog (baisse à la livraison) et ilot.codebase. */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.qualite = {
    // --- Défauts injectés (par millier de lignes) ---
    defautsParKloc: 2.5,
    // Code écrit à la main : × (plancher − pente × competence).
    humainCompetence: [1.4, 0.8],
    // Code écrit par l'agent : × modèle × (plancher − pente × adéquation de la pile) × (plancher − pente × competenceIA).
    modele: { economique: 1.5, intermediaire: 1, haut: 0.7, local: 1.3, localServeur: 1.1 },
    agentPile: [1.5, 0.6],
    agentCompetenceIA: [1.2, 0.4],
    // Rigueur de l'auteur, qui relit et teste : × (plancher − pente × rigueur).
    rigueur: [1.3, 0.6],
    specs: { non: 1, legeres: 0.85, obligatoires: 0.7 },
    // Conventions du code : × (plancher − pente × conventions).
    conventions: [1.2, 0.5],
    // Conflits de skills (ilot.mods.incoherence, 0–1) : × (1 + incoherence × ce facteur).
    incoherence: 0.6,

    // --- Filtres avant la revue ---
    lint: { aucun: 0, formatage: 0.03, strict: 0.12, verrouille: 0.15 },
    // Typage implicite de la pile de chaque îlot (part des défauts attrapés par le compilateur ou le typeur).
    typage: { appro: 0.05, operations: 0.08, bureautique: 0, inspections: 0.3, iot: 0.15, web: 0.22, portail: 0.28 },
    typageDefaut: 0.1,
    // Tests : part attrapée = couvertureReelle × efficacite × fiabilité selon la CI (des tests qui ne tournent pas en CI
    // tournent moins souvent).
    efficaciteTests: 0.6,
    testsSelonCI: { aucune: 0.6, build: 0.7, tests: 1, complet: 1 },
    ci: { aucune: 0, build: 0.08, tests: 0.12, complet: 0.22 },

    // --- Revue humaine ---
    heuresPar100Lignes: 1.0,     // heures de lecture attentive pour 100 lignes
    bruitLint: { aucun: 0, formatage: 0.05, strict: 0.1, verrouille: 0.1 }, // temps de revue épargné
    partRevueMax: 0.25,          // part maximale du temps disponible d'un développeur consacrée à la revue
    partRevueMaxReviseur: 0.4,   // trait « reviseur » : tout le monde veut sa revue
    occupationSessions: 0.15,    // chaque session au-delà de la première gruge le temps libre pour la revue
    tempsLibreMin: 0.4,
    efficaciteReviseur: 1.2,     // trait « reviseur » : lit mieux
    proprietaire: { capacite: 0.75, attention: 1.15 }, // approbation par le propriétaire du code
    compromis: 0.55,           // file saturée : part traitée = (capacité / demande) ^ compromis ; le reste attend
    distractionSessions: 0.04,   // attention × (1 − ce facteur × (sessions moyennes − 1))
    distractionMin: 0.6,
    taillePetite: 300,           // en deçà, la PR est lue en entier ; au-delà de tailleGrosse, attention × 0,6
    tailleGrosse: 900,
    facteurGrossePR: 0.6,
    efficaciteRevue: 0.6,        // défauts attrapés par un réviseur = attention × efficacité
    seuilLGTM: 0.35,             // sous ce seuil d'attention, la PR passe en « LGTM » (non lue)
    largeurLGTM: 0.1,            // transition graduelle autour du seuil
    attentionLGTM: 0.15,         // une PR en LGTM est quand même survolée
    fusionAgent: 0.4,            // autonomie 4 : part des PR fusionnées par l'agent sans regard humain

    // --- Revue par l'IA (active avec l'approbation automatique ou une autonomie ≥ 3) ---
    revueIA: 0.4,                // part des défauts que l'IA attraperait si ses ratés étaient indépendants
    correlationMemeModele: 0.65, // ratés corrélés avec l'agent auteur (même modèle)
    correlationAutreModele: 0.35,// modèle choisi selon la tâche : un autre modèle relit
    autonomieRevueIA: 3,

    // --- Délai (semaines) ---
    delaiBase: 0.4,
    delaiCI: { aucune: 1, build: 0.5, tests: 0.3, complet: 0.1 },
    delaiSpecs: { non: 0, legeres: 0.1, obligatoires: 0.25 },
    latenceReviseur: 0.25,       // attente par réviseur requis
    delaiMax: 26,

    // --- Couverture de tests ---
    cibleApparente: { aucun: 0.1, unitaires: 0.5, integration: 0.65, mutation: 0.7 },
    boostAgent: 0.2,             // l'agent écrit beaucoup de tests : + boostAgent × part écrite par l'agent
    ratioHumain: 0.85,           // couverture réelle / apparente des tests écrits à la main
    ratioIA: 0.5,                // tests de l'agent sans mutation : beaucoup sont tautologiques
    ratioIAMutation: 0.9,
    vitesseBase: 0.03,           // vitesse de convergence de la couverture vers la cible, par semaine
    vitesseCode: 2,              // + vitesseCode × (code neuf / volume)
    vitesseMax: 0.4,
    apparenteMax: 0.95,

    // --- Conventions ---
    cibleConventions: { aucun: 0.2, formatage: 0.35, strict: 0.6, verrouille: 0.7 },
    vitesseConventions: 0.03,
    erosionSessions: 0.004,      // sessions moyennes au-delà de 2 : les agents écrivent chacun à leur façon

    // --- L'agent qui triche (lint non verrouillé, autonomie ≥ 3, CI rouge) ---
    triche: { autonomie: 3, proba: 0.25, rougeEchelle: 3, baisseCouverture: 0.04, fuite: 0.3, delaiJournal: 13 },

    // --- Journal ---
    seuilSaturation: 0.6,        // capacité / demande sous ce seuil : la revue est saturée
    seuilLGTMJournal: 3,
    delaiJournal: 10,          // semaines minimales entre deux avis du même genre pour un îlot
  };

  function P() { return LTP.systemes.production; }

  function lin(param, x) { return Math.max(0.05, param[0] - param[1] * x); }

  // Taux de défauts par kLOC d'un développeur, pour le code qu'il a produit cette semaine.
  function tauxDefaut(etat, dev, ilot) {
    var K = LTP.K.qualite, q = LTP.q, p = P();
    var cle = p.cleModele(etat);
    var km = cle === 'local' && etat.options && etat.options.serveurAchete ? K.modele.localServeur : (K.modele[cle] || 1);
    var adequation = O.borner(ilot.adequationPile + p.modIlot(ilot, 'adequationPile'));
    var humain = lin(K.humainCompetence, dev.competence);
    var agent = km * lin(K.agentPile, adequation) * lin(K.agentCompetenceIA, dev.competenceIA);
    var ecrit = O.borner(dev.derniere.ecritSoiMeme);
    var t = K.defautsParKloc * (ecrit * humain + (1 - ecrit) * agent) * lin(K.rigueur, dev.rigueur);
    t *= K.specs[q.levier(etat, 'specs')] || 1;
    t *= lin(K.conventions, ilot.codebase.conventions);
    t *= p.modIlot(ilot, 'tauxDefaut') * (1 + K.incoherence * O.borner(p.modIlot(ilot, 'incoherence')));
    t *= q.modTrait(dev, 'tauxDefaut');
    return Math.max(0, t);
  }

  // Réconcilie la file détaillée avec ilot.fileRevue si un effet l'a modifiée.
  function reconcilierFile(ilot, ti) {
    var f = ti.file, externe = Math.max(0, ilot.fileRevue || 0), delta = externe - (ti.prEcrit || 0);
    if (Math.abs(delta) < 1e-9) return;
    if (delta > 0) {
      var lignes = f.pr > 0 ? f.lignes / f.pr : 350;
      var points = f.pr > 0 ? f.points / f.pr : lignes / LTP.K.production.lignesParPoint;
      var defauts = f.pr > 0 ? f.defauts / f.pr : lignes / 1000 * LTP.K.qualite.defautsParKloc * 0.7;
      f.pr += delta; f.lignes += delta * lignes; f.points += delta * points; f.defauts += delta * defauts;
    } else if (f.pr > 0) {
      var r = Math.max(0, (f.pr + delta) / f.pr);
      f.pr *= r; f.lignes *= r; f.points *= r; f.defauts *= r;
    }
  }

  function vider(f) { f.pr = 0; f.points = 0; f.lignes = 0; f.defauts = 0; }

  function semaine(etat, ctx) {
    var K = LTP.K.qualite, q = LTP.q, p = P();
    var T = p.travail(etat);
    var lint = q.levier(etat, 'lint');
    var approbation = q.levier(etat, 'approbation');
    var parTache = !!q.levier(etat, 'modeleParTache');
    var cleMod = p.cleModele(etat);
    var qm = p.qualiteModele(etat);

    q.ilots(etat).forEach(function (ilot) {
      var ti = T.ilots[ilot.id], f = ti.file, d = ilot.derniere, cb = ilot.codebase;
      var a = p.autonomie(etat, ilot.id);
      var ci = q.levier(etat, 'ci', ilot.id);
      var tests = q.levier(etat, 'tests', ilot.id);
      var r = q.levier(etat, 'reviseurs', ilot.id);
      r = typeof r === 'number' ? r : 1;
      if (approbation === 'auto') r = 0;
      reconcilierFile(ilot, ti);

      // --- 1. Défauts injectés et filtres avant la revue ---
      var auteurs = q.devsDe(etat, ilot.id).filter(function (x) { return x.derniere.code > 0; });
      var injectes = 0, lignesNeuves = 0, ecritPondere = 0, couvTrait = 0;
      auteurs.forEach(function (x) {
        injectes += x.derniere.code * tauxDefaut(etat, x, ilot);
        lignesNeuves += x.derniere.code * 1000;
        ecritPondere += x.derniere.code * x.derniere.ecritSoiMeme;
        couvTrait += x.derniere.code * q.modTrait(x, 'couverture', 0);
      });
      var codeNeuf = lignesNeuves / 1000;
      var ecritMoy = codeNeuf > 0 ? ecritPondere / codeNeuf : 0.7;
      couvTrait = codeNeuf > 0 ? couvTrait / codeNeuf : 0;

      var passeLint = 1 - (K.lint[lint] || 0);
      var passeType = 1 - (K.typage[ilot.id] != null ? K.typage[ilot.id] : K.typageDefaut);
      var attrapeTests = O.borner(cb.couvertureReelle * K.efficaciteTests * (K.testsSelonCI[ci] || 0.6));
      var attrapeCI = K.ci[ci] || 0;
      var avantTests = injectes * passeLint * passeType;
      var attrapesCI = avantTests * (1 - (1 - attrapeTests) * (1 - attrapeCI));
      var versRevue = avantTests - attrapesCI;

      // --- 2. L'agent qui triche : la CI est rouge, l'agent désactive une règle ou un test ---
      var K3 = K.triche;
      if (lint !== 'verrouille' && a >= K3.autonomie && (ci === 'tests' || ci === 'complet') && attrapesCI > 0) {
        var rouge = 1 - Math.exp(-attrapesCI / K3.rougeEchelle);
        if (ctx.rng.chance(K3.proba * rouge)) {
          cb.couvertureReelle = O.borner(cb.couvertureReelle - K3.baisseCouverture);
          versRevue += attrapesCI * K3.fuite;
          ti.triches = (ti.triches || 0) + 1;
          etat.org.drapeaux.agentTriche = true;
          etat.org.drapeaux['agentTriche_' + ilot.id] = true;
          if (p.semainesDepuis(etat, ti, 'triche') >= K3.delaiJournal) {
            ti.avis.triche = etat.semaine;
            ctx.noter('info', 'qualite.triche', { ilot: ilot.id }, { ilot: ilot.id });
          }
        }
      }

      // --- 3. Répartition : fusion sans regard humain ou file de revue ---
      var bypass = r === 0 ? 1 : (a >= 4 ? K.fusionAgent : 0);
      var prNeuves = d.pr, ptsNeufs = d.points;
      var direct = { pr: prNeuves * bypass, points: ptsNeufs * bypass, lignes: lignesNeuves * bypass, defauts: versRevue * bypass };
      f.pr += prNeuves * (1 - bypass);
      f.points += ptsNeufs * (1 - bypass);
      f.lignes += lignesNeuves * (1 - bypass);
      f.defauts += versRevue * (1 - bypass);

      // --- 4. Capacité et attention de la revue humaine ---
      var tailleMoy = f.pr > 0 ? f.lignes / f.pr : 0;
      var heuresPR = tailleMoy / 100 * K.heuresPar100Lignes * (1 - (K.bruitLint[lint] || 0));
      var demande = f.pr * r * heuresPR; // heures de lecture attentive nécessaires
      var revs = q.devsDe(etat, ilot.id).map(function (x) {
        var td = T.devs[x.id] || { dispo: 0 };
        var dispo = q.disponible(etat, x) && x.etat !== 'absent' ? td.dispo : 0;
        var part = q.aTrait(x, 'reviseur') ? K.partRevueMaxReviseur : K.partRevueMax;
        var libre = Math.max(K.tempsLibreMin, 1 - K.occupationSessions * Math.max(0, (x.sessions || 1) - 1));
        var capReel = part * LTP.K.production.heuresSemaine * dispo * libre;
        if (approbation === 'proprietaire') capReel *= K.proprietaire.capacite;
        var eff = (0.5 + 0.5 * x.competence) * (1 - 0.5 * x.fatigue) * (q.aTrait(x, 'reviseur') ? K.efficaciteReviseur : 1);
        return { dev: x, capReel: capReel, eff: eff, capEff: capReel * eff };
      });
      var capEff = O.somme(revs, 'capEff');
      var ratio = demande > 0 ? (capEff > 0 ? capEff / demande : 0) : 1;
      var traite = r > 0 && f.pr > 0 ? Math.min(1, Math.pow(Math.min(1, ratio), K.compromis)) : 0;
      var heuresUtilesEff = Math.min(capEff, demande * traite);
      var facteurTemps = demande * traite > 0 ? O.borner(heuresUtilesEff / (demande * traite)) : 1;

      var rigueurRev = capEff > 0 ? O.somme(revs, function (x) { return x.capEff * x.dev.rigueur; }) / capEff : 0.5;
      var sessionsMoy = revs.length ? O.moyenne(revs, function (x) { return Math.max(1, x.dev.sessions || 1); }) : 1;
      var distraction = Math.max(K.distractionMin, 1 - K.distractionSessions * (sessionsMoy - 1));
      var fTaille = tailleMoy <= K.taillePetite ? 1 : tailleMoy >= K.tailleGrosse ? K.facteurGrossePR :
        O.lerp(1, K.facteurGrossePR, (tailleMoy - K.taillePetite) / (K.tailleGrosse - K.taillePetite));
      var attention = O.borner(facteurTemps * (0.4 + 0.6 * rigueurRev) * fTaille * distraction *
        p.modIlot(ilot, 'attention') * (approbation === 'proprietaire' ? K.proprietaire.attention : 1));
      var partLGTM = O.borner((K.seuilLGTM + K.largeurLGTM - attention) / (2 * K.largeurLGTM));
      var attentionLue = attention * (1 - partLGTM) + Math.min(attention, K.attentionLGTM) * partLGTM;

      // --- 5. Revue par l'IA, corrélée à l'agent auteur ---
      var revueIA = approbation === 'auto' || a >= K.autonomieRevueIA;
      var passeIA = revueIA ? 1 - K.revueIA * Math.min(1.2, qm) * (1 - (parTache ? K.correlationAutreModele : K.correlationMemeModele)) : 1;

      // --- 6. Fusion ---
      var fus = { pr: f.pr * traite, points: f.points * traite, lignes: f.lignes * traite, defauts: f.defauts * traite };
      var passeHumain = Math.pow(1 - attentionLue * K.efficaciteRevue, r);
      var echappes = fus.defauts * passeHumain * passeIA + direct.defauts * passeIA;
      f.pr -= fus.pr; f.points -= fus.points; f.lignes -= fus.lignes; f.defauts -= fus.defauts;
      if (f.pr < 1e-6) vider(f);

      var prFusionnees = fus.pr + direct.pr;
      var lignesFusionnees = fus.lignes + direct.lignes;
      var pointsLivres = Math.min(ilot.backlog, fus.points + direct.points);
      ilot.backlog = Math.max(0, ilot.backlog - pointsLivres);
      cb.volume = Math.max(0, cb.volume + lignesFusionnees / 1000);
      cb.defautsLatents = Math.max(0, cb.defautsLatents + echappes);

      // --- 7. Répartition des revues entre les développeurs (temps passé, lectures, LGTM) ---
      var heuresReelles = capEff > 0 ? heuresUtilesEff / capEff : 0; // fraction de la capacité utilisée
      var revuesTotal = fus.pr * r;
      var lignesRevues = fus.lignes * r;
      ti.heuresRevue = 0;
      revs.forEach(function (x) {
        var part = capEff > 0 ? x.capEff / capEff : 0;
        var td = T.devs[x.dev.id];
        if (td) td.heuresRevue = x.capReel * heuresReelles;
        ti.heuresRevue += x.capReel * heuresReelles;
        var dd = x.dev.derniere;
        dd.revues = O.arrondir(revuesTotal * part, 2);
        dd.lgtm = O.arrondir(revuesTotal * part * partLGTM, 2);
        var propre = d.code > 0 ? O.borner(dd.code / d.code) : 0;
        var autres = lignesFusionnees * (1 - propre);
        dd.luEnRevue = autres > 0 ? O.borner(lignesRevues * part * (1 - partLGTM) / autres) : 0;
      });

      // Statistiques transmises à la dette cognitive (23).
      ti.semaine = {
        lignesFusionnees: lignesFusionnees,
        lignesBypass: direct.lignes,
        lignesLGTM: fus.lignes * partLGTM,
        lignesLues: fus.lignes * (1 - partLGTM),
        attention: r > 0 && fus.pr > 0 ? attentionLue : 0,
        prRevues: fus.pr,
      };

      // --- 8. Couverture de tests et conventions ---
      var cibleA = K.cibleApparente[tests] != null ? K.cibleApparente[tests] : 0.5;
      if (tests !== 'aucun') cibleA += K.boostAgent * (1 - ecritMoy);
      cibleA = Math.min(K.apparenteMax, cibleA);
      var ratioIA = tests === 'mutation' ? K.ratioIAMutation : K.ratioIA;
      var cibleR = O.borner(cibleA * (ecritMoy * K.ratioHumain + (1 - ecritMoy) * ratioIA) +
        p.modIlot(ilot, 'couverture') + couvTrait, 0, cibleA);
      var vitesse = Math.min(K.vitesseMax, K.vitesseBase + K.vitesseCode * codeNeuf / Math.max(1, cb.volume));
      cb.couvertureApparente = O.borner(cb.couvertureApparente + vitesse * (cibleA - cb.couvertureApparente));
      cb.couvertureReelle = O.borner(cb.couvertureReelle + vitesse * (cibleR - cb.couvertureReelle), 0, cb.couvertureApparente);
      var cibleC = (K.cibleConventions[lint] != null ? K.cibleConventions[lint] : 0.35) -
        K.erosionSessions * 10 * Math.max(0, sessionsMoy - 2) * (lint === 'verrouille' ? 0.3 : 1);
      cb.conventions = O.borner(cb.conventions + K.vitesseConventions * (O.borner(cibleC) - cb.conventions));

      // --- 9. Statistiques de la semaine ---
      var delai = K.delaiBase + (K.delaiCI[ci] != null ? K.delaiCI[ci] : 0.5) + (K.delaiSpecs[q.levier(etat, 'specs')] || 0) +
        r * K.latenceReviseur + (f.pr > 0 ? f.pr / Math.max(0.5, prFusionnees) : 0);
      var deploiements = 0;
      if (prFusionnees > 0) {
        if (ci === 'complet') deploiements = Math.round(prFusionnees);
        else if (ci === 'tests') deploiements = Math.min(2, Math.ceil(prFusionnees));
        else if (ci === 'build') deploiements = 1;
        else deploiements = etat.semaine % 2 === 0 ? 1 : 0;
      }
      d.prFusionnees = O.arrondir(prFusionnees, 2);
      d.pointsLivres = O.arrondir(pointsLivres, 3);
      d.deploiements = deploiements;
      d.defautsInjectes = O.arrondir(injectes, 3);
      d.defautsEchappes = O.arrondir(echappes, 3);
      d.delai = O.arrondir(Math.min(K.delaiMax, delai), 2);
      d.attention = O.arrondir(ti.semaine.attention, 3);
      d.lgtm = O.arrondir(fus.pr * partLGTM, 2);
      ilot.fileRevue = Math.round(f.pr);
      ti.prEcrit = ilot.fileRevue;

      // --- 10. Journal, avec parcimonie ---
      if (r > 0 && ratio < K.seuilSaturation && f.pr >= 4 && p.semainesDepuis(etat, ti, 'saturation') >= K.delaiJournal) {
        ti.avis.saturation = etat.semaine;
        ctx.noter('alerte', 'qualite.revueSaturee', { ilot: ilot.id, pr: ilot.fileRevue }, { ilot: ilot.id });
      } else if (d.lgtm >= K.seuilLGTMJournal && p.semainesDepuis(etat, ti, 'lgtm') >= K.delaiJournal) {
        ti.avis.lgtm = etat.semaine;
        ctx.noter('alerte', 'qualite.lgtm', { ilot: ilot.id, lgtm: Math.round(d.lgtm) }, { ilot: ilot.id });
      }
    });
  }

  LTP.systemes.qualite = {
    initialiser: function (etat) { P().travail(etat); },
    semaine: semaine,
    tauxDefaut: tauxDefaut,
  };

  LTP.enregistrerTextes({
    'qualite.revueSaturee': '{ilot} : {pr} PR attendent une revue. Les réviseurs lisent en diagonale pour suivre.',
    'qualite.lgtm': '{ilot} : {lgtm} PR approuvées en « LGTM » cette semaine, sans vraie lecture.',
    'qualite.triche': '{ilot} : un commit d\'agent désactive une règle de vérification. La CI est redevenue verte.',
  });
})(globalThis.LTP = globalThis.LTP || {});
