/* 21-production.js */
/* Étape 4 de la semaine : production des points, du code et des PR, par développeur (CONCEPTION §4.4).
   Lit les leviers, les mods d'îlot (26) et l'occupation (26, 27) ; écrit dev.sessions, dev.etat,
   dev.derniere.{points, code, pr, ecritSoiMeme} et ilot.derniere.{points, code, pr}.
   Le code est compté en milliers de lignes (kLOC), comme codebase.volume ; une PR, en lignes.
   Ce fichier crée aussi etat.travail, l'état interne partagé par 21 à 24 (voir assurerTravail). */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.production = {
    // Points par semaine sans IA, à compétence de référence (un inter ≈ 5 points, un îlot ≈ 15).
    base: { stagiaire: 2.2, junior: 3.5, inter: 5, senior: 5.8 },
    // Facteur de compétence : plancher + pente × competence (inter à 0,62 → ≈ 1).
    competencePlancher: 0.5,
    competencePente: 0.8,
    // niveauAutonomie[a] du gain de l'IA, a = 0 (autocomplétion) à 4 (agent qui fusionne seul).
    autonomie: [0.08, 0.3, 1.4, 2.1, 2.6],
    // Qualité du modèle dans le gain de l'IA. Le modèle local est meilleur sur le serveur acheté.
    qualiteModele: { economique: 0.6, intermediaire: 1, haut: 1.2, local: 0.65, localServeur: 0.85 },
    // Rendement décroissant des sessions parallèles : sessions ^ exposant.
    exposantSessions: 0.6,
    // Envie de sessions : 1 + envieSessions × adhésion × envieAutonomie[a] + mod de trait, plafonnée par le levier.
    envieSessions: 5,
    envieAutonomie: [0, 0.4, 0.8, 1, 1.1],
    // Perte de débit due à la fatigue : × (1 − fatigue × ce facteur).
    fatigue: 0.5,
    // Pair programming : part du temps prise par le pair humain, gain additionnel de l'IA en pair humain-IA.
    pair: {
      aucun:    { temps: 0,    gainIA: 0 },
      humain:   { temps: 0.2,  gainIA: 0 },
      ia:       { temps: 0,    gainIA: 0.1 },
      rotation: { temps: 0.06, gainIA: 0.05 },
    },
    // Temps hebdomadaire moyen des revues d'explication et de la rédaction des specs.
    tempsExplication: { non: 0, mensuelles: 0.025, majeures: 0.06 },
    tempsSpecs: { non: 0, legeres: 0.04, obligatoires: 0.08 },
    // Incident en cours sur l'îlot : heures de l'incident × mobilisation, réparties sur les développeurs.
    heuresSemaine: 40,
    mobilisationIncident: 1.5,
    tempsIncidentMax: 0.6,
    // Volume : lignes par point, puis verbosité (modèle, autonomie, discipline du « non »).
    lignesParPoint: 60,
    verbositeModele: { economique: 1.25, intermediaire: 1, haut: 0.9, local: 1.15 },
    verbositeAutonomie: [1, 1.05, 1.3, 1.7, 2.2],
    verbositeDiscipline: { laxiste: 1.2, normale: 1, ferme: 0.92 },
    // Taille naturelle d'une PR (lignes), qui grossit avec l'autonomie : × (1 + tailleAutonomie × a).
    tailleNaturelle: 250,
    tailleAutonomie: 0.4,
    // Part du code écrite à la main avant l'effet du débit de l'IA, selon l'autonomie ; bonus du pair humain.
    ecritMain: [0.95, 0.8, 0.5, 0.25, 0.1],
    ecritPairHumain: 0.1,
    // État affiché : « revue » au-delà de cette part du temps passée en revue ; « attente » si moins de
    // cette part du travail possible a pu être faite (carnet vide, plafond strict).
    seuilEtatRevue: 0.22,
    seuilAttente: 0.5,
    // Semaines minimales entre deux avis « carnet vide » pour un îlot.
    delaiJournal: 26,
  };

  var ETATS_EXTERNES = { conge: true, formation: true, absent: true };

  // --- État interne partagé par les systèmes 21 à 24 (JSON pur) ---
  //   etat.travail.ilots[id] : file de revue détaillée et statistiques de passage entre étapes
  //   etat.travail.devs[id]  : temps de revue de la semaine précédente, facteur de plafond, temps disponible
  //   etat.travail.org       : seuils de budget déjà annoncés
  function assurerTravail(etat) {
    var t = etat.travail;
    if (!t || typeof t !== 'object') t = etat.travail = { ilots: {}, devs: {}, org: {} };
    t.ilots = t.ilots || {};
    t.devs = t.devs || {};
    t.org = t.org || {};
    if (!t.org.seuilsBudget) t.org.seuilsBudget = {};
    if (t.org.budgetDepasse == null) t.org.budgetDepasse = false;
    etat.ordreIlots.forEach(function (id) {
      var ti = t.ilots[id];
      if (!ti) {
        ti = t.ilots[id] = {
          file: { pr: 0, points: 0, lignes: 0, defauts: 0 },
          prEcrit: etat.ilots[id].fileRevue || 0,
          heuresRevue: 0,
          semaine: { lignesFusionnees: 0, lignesBypass: 0, lignesLGTM: 0, lignesLues: 0, attention: 1, prRevues: 0 },
          pointsPerdusPlafond: 0,
          comprehension: null,
          triches: 0,
          avis: {},
        };
      }
      ti.avis = ti.avis || {};
    });
    Object.keys(etat.devs).forEach(function (id) {
      if (!t.devs[id]) t.devs[id] = { heuresRevue: 0, dispo: 1, facteurPlafond: 1 };
    });
    return t;
  }

  // Modificateur d'îlot, avec sa valeur neutre si 26 ne l'a pas fourni.
  var NEUTRES = LTP.etat.modsIlotNeutres();
  function modIlot(ilot, cle) {
    var m = ilot.mods || {};
    return typeof m[cle] === 'number' && isFinite(m[cle]) ? m[cle] : NEUTRES[cle];
  }

  function cleModele(etat) {
    var m = LTP.q.levier(etat, 'modele');
    return LTP.K.production.qualiteModele[m] != null ? m : 'intermediaire';
  }
  // Qualité du modèle par défaut, avec le bonus du serveur acheté pour le modèle local.
  function qualiteModele(etat) {
    var K = LTP.K.production, m = cleModele(etat);
    if (m === 'local' && etat.options && etat.options.serveurAchete) return K.qualiteModele.localServeur;
    return K.qualiteModele[m];
  }
  function autonomie(etat, ilotId) {
    var a = LTP.q.levier(etat, 'autonomie', ilotId);
    return O.borner(typeof a === 'number' ? Math.round(a) : 1, 0, 4);
  }

  // Taille d'une PR en lignes, selon le levier de taille maximale, l'autonomie, le trait et l'îlot.
  function taillePR(etat, dev, ilot) {
    var K = LTP.K.production, a = autonomie(etat, ilot.id);
    var naturelle = K.tailleNaturelle * (1 + K.tailleAutonomie * a) * LTP.q.modTrait(dev, 'taillePR');
    var plafond = LTP.q.levier(etat, 'taillePR');
    var t = typeof plafond === 'number' ? Math.min(plafond, naturelle) : naturelle;
    return Math.max(20, t * modIlot(ilot, 'taillePR'));
  }

  // Heures d'incident à absorber cette semaine par l'îlot (incidents ouverts les semaines précédentes).
  function heuresIncident(etat, ilotId) {
    var h = 0;
    LTP.q.incidentsActifs(etat, ilotId).forEach(function (inc) {
      if (inc.debut >= etat.semaine) return;
      var nb = Math.max(1, (inc.finSemaine - inc.debut) + 1);
      h += (inc.mttrHeures || 0) / nb;
    });
    return h;
  }

  // Envie de sessions d'un développeur, plafonnée par le levier. Sans licence Copilot : aucune session,
  // donc aucun gain de l'IA ni aucun jeton (le développeur code comme avant).
  function sessionsVoulues(etat, dev, a) {
    var K = LTP.K.production;
    if (!LTP.q.aLicence(etat, dev)) return 0;
    var max = LTP.q.levier(etat, 'sessionsMax');
    max = typeof max === 'number' ? max : 1;
    var envie = 1 + K.envieSessions * dev.adhesion * K.envieAutonomie[a] + LTP.q.modTrait(dev, 'sessions', 0);
    return Math.max(1, Math.min(max, Math.round(envie)));
  }

  function semaine(etat, ctx) {
    var K = LTP.K.production, q = LTP.q;
    var T = assurerTravail(etat);
    var mod = cleModele(etat), qm = qualiteModele(etat);
    var pair = K.pair[q.levier(etat, 'pair')] || K.pair.aucun;
    var tExpl = K.tempsExplication[q.levier(etat, 'revuesExplication')] || 0;
    var tSpecs = K.tempsSpecs[q.levier(etat, 'specs')] || 0;
    var couts = LTP.systemes.couts;

    q.ilots(etat).forEach(function (ilot) {
      var ti = T.ilots[ilot.id];
      var a = autonomie(etat, ilot.id);
      var adequation = O.borner(ilot.adequationPile + modIlot(ilot, 'adequationPile'));
      var verbIlot = K.verbositeModele[mod] * K.verbositeAutonomie[a] *
        (K.verbositeDiscipline[q.levier(etat, 'disciplineNon', ilot.id)] || 1) * modIlot(ilot, 'verbosite');
      var devsIlot = q.devsDe(etat, ilot.id);
      var actifs = [];

      // Développeurs absents ou partis : aucune session, rien produit.
      q.devs(etat).forEach(function (d) {
        if (d.ilot !== ilot.id) return;
        if (!d.present || !q.disponible(etat, d) || d.etat === 'absent') {
          d.sessions = 0;
          T.devs[d.id] = T.devs[d.id] || { heuresRevue: 0, dispo: 0, facteurPlafond: 1 };
          T.devs[d.id].dispo = 0;
          T.devs[d.id].facteurPlafond = 1;
        } else actifs.push(d);
      });

      var nbPresents = Math.max(1, actifs.length || devsIlot.length);
      var tIncident = Math.min(K.tempsIncidentMax, heuresIncident(etat, ilot.id) * K.mobilisationIncident / (K.heuresSemaine * nbPresents));

      // Premier passage : temps disponible, sessions, points bruts (part humaine et part IA séparées).
      var calc = actifs.map(function (d) {
        var td = T.devs[d.id] = T.devs[d.id] || { heuresRevue: 0, dispo: 1, facteurPlafond: 1 };
        d.sessions = sessionsVoulues(etat, d, a);
        var occ = d.occupation || {};
        var dispo = O.borner(1 - (occ.registre || 0) - (occ.formation || 0) - tIncident);
        var tRevue = Math.min(dispo, (td.heuresRevue || 0) / K.heuresSemaine);
        var tCode = Math.max(0, dispo - tRevue - pair.temps - tExpl - tSpecs);
        td.dispo = dispo;

        var gainIA = K.autonomie[a] * qm * adequation * (0.5 + d.competenceIA) * d.adhesion *
          Math.max(0, 1 + modIlot(ilot, 'debit') + pair.gainIA);
        var facteurSess = Math.pow(d.sessions, K.exposantSessions);
        var fComp = K.competencePlancher + K.competencePente * d.competence;
        var socle = (K.base[d.niveau] || K.base.inter) * fComp * (1 - K.fatigue * d.fatigue) *
          Math.max(0, 1 + q.modTrait(d, 'debit', 0)) * tCode;
        return {
          dev: d, tRevue: tRevue, tCode: tCode, socle: socle, gainIA: gainIA, facteurSess: facteurSess,
          humain: socle, ia: socle * gainIA * facteurSess,
        };
      });

      // Plafond strict de jetons : l'agent est coupé quand l'îlot atteint son plafond ; la part IA est perdue.
      var facteurPlafond = 1;
      ti.pointsPerdusPlafond = 0;
      if (couts && couts.plafondIlot && q.levier(etat, 'plafondJetons', ilot.id) === 'strict') {
        var estime = O.somme(calc, function (c) { return couts.estimerJetons(etat, c.dev, ilot); });
        var plafond = couts.plafondIlot(etat, ilot, actifs.length);
        if (estime > plafond && estime > 0) {
          facteurPlafond = plafond / estime;
          ti.pointsPerdusPlafond = O.somme(calc, function (c) { return c.ia; }) * (1 - facteurPlafond);
        }
      }
      calc.forEach(function (c) {
        T.devs[c.dev.id].facteurPlafond = facteurPlafond;
        c.possible = c.humain + c.ia;
        c.ia *= facteurPlafond;
      });

      // Les points produits ne dépassent pas le carnet disponible (demandes pas déjà en revue).
      var total = O.somme(calc, function (c) { return c.humain + c.ia; });
      var disponible = Math.max(0, ilot.backlog - (ti.file ? ti.file.points : 0));
      var echelle = total > disponible && total > 0 ? disponible / total : 1;

      var dIlot = ilot.derniere;
      calc.forEach(function (c) {
        var d = c.dev, dd = d.derniere;
        var points = (c.humain + c.ia) * echelle;
        // Sans licence (aucune session), on code comme avant : verbosité et part écrite à la main du niveau 0.
        var sansIA = !(d.sessions > 0);
        var verbDev = sansIA ? verbIlot * K.verbositeAutonomie[0] / K.verbositeAutonomie[a] : verbIlot;
        var code = points * K.lignesParPoint * verbDev * q.modTrait(d, 'verbosite') / 1000;
        var taille = taillePR(etat, d, ilot);
        var pr = code > 0.001 ? Math.ceil(code * 1000 / taille) : 0;
        // Part écrite à la main : la base de l'autonomie, diluée par tout ce que l'agent ajoute.
        var partHumaine = c.humain + c.ia > 0 ? c.humain / (c.humain + c.ia) : 1;
        var ecrit = K.ecritMain[sansIA ? 0 : a] * partHumaine;
        if (q.levier(etat, 'pair') === 'humain') ecrit += K.ecritPairHumain * (1 - ecrit);
        else if (q.levier(etat, 'pair') === 'rotation') ecrit += 0.5 * K.ecritPairHumain * (1 - ecrit);

        dd.points = O.arrondir(points, 3);
        dd.code = O.arrondir(code, 4);
        dd.pr = pr;
        dd.ecritSoiMeme = O.borner(points > 0 ? ecrit : 0);
        dIlot.points += points;
        dIlot.code += code;
        dIlot.pr += pr;

        // État affiché, sauf ceux posés par 27 (congé, formation, absence).
        if (!ETATS_EXTERNES[d.etat]) {
          var fait = c.possible > 0 ? points / c.possible : 0;
          if (c.tRevue >= K.seuilEtatRevue) d.etat = 'revue';
          else if (c.tCode < 0.1 || fait < K.seuilAttente) d.etat = 'attente';
          else d.etat = 'code';
        }
      });
      dIlot.points = O.arrondir(dIlot.points, 3);
      dIlot.code = O.arrondir(dIlot.code, 4);

      // Carnet épuisé : les agents tournent à vide (noté avec parcimonie).
      if (echelle < 0.6 && total > 1 && semainesDepuis(etat, ti, 'attente') >= K.delaiJournal) {
        ti.avis.attente = etat.semaine;
        ctx.noter('info', 'production.carnetVide', { ilot: ilot.id }, { ilot: ilot.id });
      }
    });
  }

  function semainesDepuis(etat, ti, cle) {
    var s = ti.avis && ti.avis[cle];
    return typeof s === 'number' ? etat.semaine - s : Infinity;
  }

  LTP.systemes.production = {
    initialiser: function (etat) { assurerTravail(etat); },
    semaine: semaine,
    // Outils partagés avec 22, 23 et 24.
    travail: assurerTravail,
    modIlot: modIlot,
    cleModele: cleModele,
    qualiteModele: qualiteModele,
    autonomie: autonomie,
    sessionsVoulues: sessionsVoulues,
    taillePR: taillePR,
    semainesDepuis: semainesDepuis,
  };

  LTP.enregistrerTextes({
    'production.carnetVide': '{ilot} : le carnet de demandes est presque vide. Les agents tournent à vide en attendant du travail.',
  });
})(globalThis.LTP = globalThis.LTP || {});
