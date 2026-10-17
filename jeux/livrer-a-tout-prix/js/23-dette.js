/* 23-dette.js */
/* Dette cognitive (CONCEPTION §4.5) en deux étapes de la semaine :
   - étape 6, semaine() : évolution de dev.comprehension (code écrit soi-même, lu en revue, pair, revues
     d'explication, contre le code fusionné sans lecture, dilué par le volume) ; drapeau « personne ne comprend » ;
   - étape 9, incidents() : défauts latents → bogues → incidents selon la criticité, résolution des incidents échus,
     développeurs mobilisés en état « incident ».
   mttr(etat, ilotId, gravite) donne la durée prévue d'un incident, en heures (appelée par LTP.q.creerIncident). */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.dette = {
    // --- Compréhension ---
    // comprehension += apprentissage × gagne × (1 − c) × mod de trait − dilution × perdu × c
    apprentissage: 0.04,
    dilution: 2.5,
    poidsEcrit: 0.5,          // gagne : part du code écrite soi-même
    poidsLu: 0.3,             // gagne : part du code des autres lue attentivement en revue
    pair: { aucun: 0, humain: 0.4, ia: 0.1, rotation: 0.25 },
    explication: { non: 0, mensuelles: 0.25, majeures: 0.5 },
    // Part du code de son propre agent que l'auteur comprend vraiment, selon l'autonomie (0 à 4).
    lectureAuteur: [1, 0.9, 0.75, 0.45, 0.2],
    // Volume relatif = code fusionné / (volume + tampon) : un petit code se dilue vite, mais pas à l'infini.
    tamponVolume: 100,
    // Pair en rotation : la compréhension de chacun se rapproche de la moyenne de l'îlot.
    partageRotation: 0.05,
    // Drapeau org.drapeaux.personneNeComprend_<ilot> : posé sous seuilPersonne, retiré au-dessus de seuilRetabli.
    seuilPersonne: 0.3,
    seuilRetabli: 0.4,
    comprehensionSansDev: 0.2,

    // --- Bogues et incidents ---
    eclosion: 0.12,           // probabilité par semaine qu'un défaut latent se manifeste en production
    // Probabilité qu'un bogue devienne un incident :
    //   incidentParBogue × criticité × (1 + incidentComprehension × (1 − compréhension de l'îlot))
    //   × (1 + incidentDensite × défauts latents par kLOC) / (1 + bogues / saturationBogues).
    // Un code que personne ne comprend et truffé de défauts latents casse plus souvent et plus fort ;
    // beaucoup de bogues la même semaine partagent souvent une même cause (saturation).
    incidentParBogue: 0.032,
    incidentComprehension: 1.5,
    incidentDensite: 3,
    saturationBogues: 3,
    maxIncidentsSemaine: 2,   // par îlot, pour éviter les avalanches absurdes
    gravite: [0.2, 0.6],      // gravité tirée dans cet intervalle, + graviteCriticite × criticité
    graviteCriticite: 0.3,
    graviteComprehension: 0.25, // + 0,25 × (1 − compréhension de l'îlot)
    // Incident grave (gravité ≥ seuilGrave) : la direction le voit. Confiance −confianceGrave × (1 à 2) selon la gravité.
    seuilGrave: 0.87,
    confianceGrave: 0.8,
    seuilDetteGrave: 0.5,     // compréhension sous ce seuil : le journal dit que personne ne comprenait le code
    mobilises: 2,             // développeurs de l'îlot en état « incident » pendant un incident
    seuilBoguesJournal: 8,
    delaiJournal: 10,

    // --- MTTR (heures) ---
    // base × (plancher + pente × gravité) × (1 + mttr × (1 − compréhension)) × (1 + volume / volumeRef)
    //   × ilot.mods.mttr × (1 − observabilite × reductionObservabilite) × accesProd
    mttrBase: 4.5,
    graviteMTTR: [0.6, 0.8],
    mttr: 2.5,
    volumeRef: 400,
    reductionObservabilite: 0.35,
    observabiliteLevier: { aucune: 0, base: 0.5, complete: 1 },
    accesProd: { oui: 0.85, lecture: 0.85, non: 1 },
    mttrMin: 1,
    seuilMTTRJournal: 24,
  };

  function P() { return LTP.systemes.production; }

  // Compréhension moyenne d'un îlot (développeurs présents).
  function comprehensionIlot(etat, ilotId) {
    var devs = LTP.q.devsDe(etat, ilotId);
    return devs.length ? O.moyenne(devs, 'comprehension') : LTP.K.dette.comprehensionSansDev;
  }

  // Durée de rétablissement prévue, en heures.
  function mttr(etat, ilotId, gravite) {
    var K = LTP.K.dette, q = LTP.q, ilot = etat.ilots[ilotId];
    if (!ilot) return 8;
    var g = O.borner(gravite == null ? 0.5 : gravite);
    var c = comprehensionIlot(etat, ilotId);
    var obs = Math.max(O.borner(etat.org.observabilite || 0), K.observabiliteLevier[q.levier(etat, 'observabilite')] || 0);
    var acces = K.accesProd[q.levier(etat, 'accesProd', ilotId)];
    var modMttr = P() ? P().modIlot(ilot, 'mttr') : 1;
    var h = K.mttrBase * (K.graviteMTTR[0] + K.graviteMTTR[1] * g) *
      (1 + K.mttr * (1 - c)) * (1 + Math.max(0, ilot.codebase.volume) / K.volumeRef) *
      modMttr * (1 - obs * K.reductionObservabilite) * (acces != null ? acces : 1);
    return O.arrondir(Math.max(K.mttrMin, isFinite(h) ? h : 8), 1);
  }

  // Étape 6 : compréhension de chaque développeur présent.
  function semaine(etat, ctx) {
    var K = LTP.K.dette, q = LTP.q, p = P();
    var T = p.travail(etat);
    var pair = q.levier(etat, 'pair');
    var gainPair = K.pair[pair] || 0;
    var gainExpl = K.explication[q.levier(etat, 'revuesExplication')] || 0;

    q.ilots(etat).forEach(function (ilot) {
      var ti = T.ilots[ilot.id], s = ti.semaine || {};
      var a = p.autonomie(etat, ilot.id);
      var devs = q.devsDe(etat, ilot.id);
      var codeIlot = ilot.derniere.code;
      var fusion = (s.lignesFusionnees || 0) / 1000;
      var volumeRelatif = fusion / (Math.max(0, ilot.codebase.volume) + K.tamponVolume);
      var partBypass = s.lignesFusionnees > 0 ? s.lignesBypass / s.lignesFusionnees : 0;
      var attention = s.attention || 0;

      devs.forEach(function (d) {
        var dd = d.derniere;
        var actif = q.disponible(etat, d) && d.etat !== 'absent';
        var ecrit = actif ? O.borner(dd.ecritSoiMeme) : 0;
        var propre = codeIlot > 0 ? O.borner(dd.code / codeIlot) : 0;
        // Ce que la personne comprend du code neuf de l'îlot : le sien (à la main, ou relu de son agent)
        // et celui des autres lu attentivement. Le code fusionné seul par l'agent est moins relu par son auteur.
        var lectureAgent = K.lectureAuteur[a] * (1 - 0.5 * partBypass);
        var comprisPropre = propre * (ecrit + (1 - ecrit) * lectureAgent);
        var comprisAutres = (1 - propre) * O.borner(dd.luEnRevue) * Math.max(attention, 0.3);
        var partComprise = O.borner(comprisPropre + comprisAutres);

        var gagne = (actif ? K.poidsEcrit * ecrit + K.poidsLu * O.borner(dd.luEnRevue) + gainPair + gainExpl : 0);
        var perdu = (1 - partComprise) * volumeRelatif;
        var c = d.comprehension;
        c += K.apprentissage * gagne * (1 - c) * q.modTrait(d, 'comprehension') - K.dilution * perdu * c;
        c += p.modIlot(ilot, 'comprehension');
        d.comprehension = O.borner(c);
      });

      // Pair en rotation : la connaissance circule dans l'îlot.
      if (pair === 'rotation' && devs.length > 1) {
        var moy = O.moyenne(devs, 'comprehension');
        devs.forEach(function (d) { d.comprehension = O.borner(d.comprehension + K.partageRotation * (moy - d.comprehension)); });
      }

      // Drapeau « personne ne comprend » (les événements s'en servent), avec hystérésis.
      var ci = comprehensionIlot(etat, ilot.id), nom = 'personneNeComprend_' + ilot.id;
      ti.comprehension = O.arrondir(ci, 4);
      if (ci < K.seuilPersonne && !etat.org.drapeaux[nom]) {
        etat.org.drapeaux[nom] = true;
        ctx.noter('alerte', 'dette.personneNeComprend', { ilot: ilot.id }, { ilot: ilot.id });
      } else if (ci >= K.seuilRetabli && etat.org.drapeaux[nom]) {
        delete etat.org.drapeaux[nom];
      }
    });
  }

  // Étape 9 : bogues, incidents, résolution, mobilisation.
  function incidents(etat, ctx) {
    var K = LTP.K.dette, q = LTP.q, p = P(), rng = ctx.rng;
    var T = p.travail(etat);

    // 1. Résolution des incidents échus avant cette semaine : un incident reste visible au moins la semaine où il éclate.
    etat.incidents.forEach(function (inc) {
      if (inc.resolu || inc.finSemaine >= etat.semaine) return;
      inc.resolu = true;
      if (inc.mttrHeures >= K.seuilMTTRJournal) {
        ctx.noter('incident', 'dette.resolutionLongue', { ilot: inc.ilot, heures: Math.round(inc.mttrHeures) }, { ilot: inc.ilot });
      }
    });

    q.ilots(etat).forEach(function (ilot) {
      var ti = T.ilots[ilot.id], cb = ilot.codebase;
      // 2. Défauts latents qui se manifestent en production.
      var attendu = Math.max(0, cb.defautsLatents) * K.eclosion;
      var bogues = Math.floor(attendu) + (rng.chance(attendu - Math.floor(attendu)) ? 1 : 0);
      bogues = Math.min(bogues, Math.floor(cb.defautsLatents));
      cb.defautsLatents = Math.max(0, cb.defautsLatents - bogues);
      ilot.derniere.bogues += bogues;
      if (bogues >= K.seuilBoguesJournal && p.semainesDepuis(etat, ti, 'bogues') >= K.delaiJournal) {
        ti.avis.bogues = etat.semaine;
        ctx.noter('alerte', 'dette.bogues', { ilot: ilot.id, bogues: bogues }, { ilot: ilot.id });
      }
      // 3. Certains bogues deviennent des incidents : criticité, compréhension et densité de défauts latents.
      var nb = 0;
      var comp = comprehensionIlot(etat, ilot.id);
      var densite = Math.max(0, cb.defautsLatents) / Math.max(1, cb.volume);
      var pIncident = O.borner(K.incidentParBogue * ilot.criticite * (1 + (K.incidentComprehension || 0) * (1 - comp))
        * (1 + (K.incidentDensite || 0) * densite) / (1 + bogues / Math.max(1e-9, K.saturationBogues || Infinity)));
      for (var i = 0; i < bogues && nb < K.maxIncidentsSemaine; i++) {
        if (rng.chance(pIncident)) {
          var g = O.borner(rng.entre(K.gravite[0], K.gravite[1]) + K.graviteCriticite * ilot.criticite
            + (K.graviteComprehension || 0) * (1 - comp));
          var grave = K.seuilGrave != null && g >= K.seuilGrave;
          var inc = q.creerIncident(etat, { ilot: ilot.id, gravite: g, cause: 'bogue' }, grave ? null : ctx);
          if (grave && inc) {
            // La direction voit les pannes graves : confiance −1 à −2 selon la gravité, et le journal dit pourquoi.
            var delta = K.confianceGrave * (1 + (g - K.seuilGrave) / Math.max(1e-9, 1 - K.seuilGrave));
            etat.org.confiance = O.borner(etat.org.confiance - delta, 0, 100);
            ctx.noter('incident', comp < K.seuilDetteGrave ? 'dette.incidentGraveDette' : 'dette.incidentGrave',
              { ilot: ilot.id, heures: Math.round(inc.mttrHeures), delta: O.arrondir(delta, 1) }, { ilot: ilot.id });
          }
          nb++;
        }
      }
    });

    // 4. Pendant un incident ouvert, des développeurs de l'îlot sont mobilisés (la garde d'abord, puis qui comprend le mieux).
    q.ilots(etat).forEach(function (ilot) {
      if (!q.incidentsActifs(etat, ilot.id).length) return;
      var candidats = q.devsDe(etat, ilot.id).filter(function (d) {
        return q.disponible(etat, d) && (d.etat === 'code' || d.etat === 'revue' || d.etat === 'attente' || d.etat === 'incident');
      });
      candidats.sort(function (x, y) {
        var gx = q.aTrait(x, 'garde') ? 1 : 0, gy = q.aTrait(y, 'garde') ? 1 : 0;
        return gy - gx || y.comprehension - x.comprehension || (x.id < y.id ? -1 : 1);
      });
      candidats.slice(0, K.mobilises).forEach(function (d) { d.etat = 'incident'; });
    });
  }

  LTP.systemes.dette = {
    initialiser: function (etat) { P().travail(etat); },
    semaine: semaine,
    incidents: incidents,
    mttr: mttr,
    comprehensionIlot: comprehensionIlot,
  };

  LTP.enregistrerTextes({
    'dette.personneNeComprend': '{ilot} : plus personne ne sait vraiment comment une bonne partie du code fonctionne.',
    'dette.resolutionLongue': '{ilot} : {heures} h pour rétablir le service. Il a d\'abord fallu comprendre ce que faisait le code.',
    'dette.bogues': '{ilot} : {bogues} bogues signalés en production cette semaine.',
    // Incident grave (gravité ≥ K.dette.seuilGrave) : remplace la ligne « incident.bogue » et dit pourquoi la confiance baisse.
    'dette.incidentGrave': '{ilot} : panne grave en production ({heures} h prévues). La direction l\'a vue passer : confiance −{delta}.',
    'dette.incidentGraveDette': '{ilot} : panne grave ({heures} h prévues), et personne ne comprend assez ce code pour réparer vite. Confiance −{delta}.',
    // Clé produite par LTP.q.creerIncident pour la cause « bogue » (20-etat.js).
    'incident.bogue': '{ilot} : un bogue en production tourne à l\'incident.',
  });
})(globalThis.LTP = globalThis.LTP || {});
