/* 25-donnees-secu.js */
/* Données et sécurité (CONCEPTION §4.7, §4.8) : étape 8 de la semaine.
   Fait évoluer org.donnees et org.menace, calcule l'exposition Loi 25 et la surface d'attaque par îlot,
   puis tire les risques de la semaine ; chaque risque réalisé devient un incident (LTP.q.creerIncident).
   La triade (ilot.triade) et les cartes actives viennent de 26-registre (étape 3).

   Champs propres ajoutés à l'état (initialiser), lus par 30-metriques et par les conditions d'événements :
     etat.securite = {
       surfaceAttaque,       // 0–100 : métrique « surfaceAttaque » (0,6 × moyenne + 0,4 × maximum des îlots)
       expositionLoi25,      // 0–100 : métrique « expositionLoi25 » (moyenne des îlots × 100)
       parIlot: { [ilotId]: {
         surface,            // 0–100 : surface d'attaque de l'îlot
         loi25,              // 0–1 : sensibilite × (1 − classification) × sortieDonnees
         sortieDonnees,      // 0–1 : part des données qui sortent (modèle externe, MCP de fournisseurs)
         risques: { injection, paquetHallucine, mcpEmpoisonne, changementSilencieux, destructionProd, fuiteDonnees },
                             // probabilités de la semaine (pour l'interface et l'équilibrage)
       } },
       incidents: { [cause]: n },   // cumul des incidents de sécurité par cause
       detections,           // changements silencieux détectés par la passerelle MCP (cumul)
     }

   Causes d'incident créées ici : injection · paquetHallucine · mcpEmpoisonne · changementSilencieux ·
   destructionProd · fuiteDonnees.

   Drapeaux posés dans org.drapeaux (testables par les événements et le profil de fin) :
     injectionPrompt       au moins une injection de prompt réussie
     paquetHallucine       un paquet inventé par l'agent a été installé (slopsquatting)
     mcpEmpoisonne         un MCP communautaire a exfiltré ou piégé quelque chose
     changementSilencieux  un MCP de fournisseur a changé sans que personne le voie
     changementDetecte     la passerelle MCP a détecté un changement silencieux
     destructionProd       un agent a détruit des données en production
     fuiteDonnees          fuite de données sensibles
     fuiteMajeure          fuite de gravité ≥ K.donneesSecu.fuite.seuilMajeure (profil « Far West ») */
(function (LTP) {
  'use strict';
  var O = LTP.outils, q = LTP.q;

  LTP.K.donneesSecu = {
    // --- Données (CONCEPTION §4.7) ---
    duplicationCode: 0.5,         // duplication + 0,5 × (code écrit cette semaine / volume total)
    duplicationFantome: 0.0015,   // + par application hors TI non encadrée et par semaine
    duplicationEncadree: 0.0004,  // + par application hors TI encadrée
    catalogueDuplication: [0, 0.004, 0.01],     // − par semaine selon le catalogue (aucun, inventaire, sources)
    fraicheurPerte: 0.006,        // fraîcheur perdue par semaine sans catalogue
    catalogueRalentit: [1, 0.6, 0.3],           // × perte de fraîcheur
    catalogueRafraichit: [0, 0, 0.004],         // + fraîcheur par semaine (sources officielles)
    qualiteCible: { base: 1, duplication: 0.5, fraicheur: 0.3, catalogue: [0, 0.05, 0.12] },
    qualiteVitesse: 0.05,         // rapprochement hebdomadaire de la qualité vers sa cible
    classificationCible: { non: 0.1, manuelle: 0.6, auto: 0.9 },
    classificationVitesse: 0.08,
    // --- Menace ---
    // Hausse par semaine, selon le trimestre (T1 à T4) : les attaques visant les agents se multiplient
    // à partir du T3, quand les leviers de gouvernance se débloquent (sinon le joueur subirait sans recours).
    menaceHausse: [0.004, 0.007, 0.014, 0.012],
    // --- Exposition Loi 25 ---
    sortieModeleExterne: 0.7,     // part des données qui sortent avec un modèle infonuagique
    sortieModeleLocal: 0.1,       // avec le modèle local (serveur GPU)
    sortieParFournisseur: 0.15,   // + par MCP de fournisseur actif dans l'îlot
    // --- Usage des agents selon l'autonomie (0 à 4) ---
    usage: [0.1, 0.4, 0.7, 1, 1.2],
    facteurRisqueCarte: 0.2,  // × la probabilité de risque déclarée par une carte (risque.proba, par semaine)
    formation: [0, 0.4, 0.7, 1],  // vigilance acquise selon le programme de formation (aucune → mentorat)
    // --- Injection de prompt (CONCEPTION §4.8) ---
    injection: {
      base: 0.025, triadePartielle: 0.08, // par semaine × menace ; triade incomplète : × 0,08 × branches / 3
      bacASable: [0, 0.5, 0.85],          // part neutralisée (aucun, conteneur, filtré)
      approbationSorties: 0.1,            // × si un humain approuve les sorties
      gravite: 0.4, graviteSensibilite: 0.4, confiance: 1.5,
    },
    // --- Paquet halluciné (slopsquatting) ---
    paquet: {
      base: 0.008, modele: { economique: 6, intermediaire: 1, haut: 0.4, local: 2 }, // × menace
      autonomie: [0, 0.2, 0.5, 1, 1.5], bacASable: [0, 0.4, 0.95],
      gravite: 0.5, confiance: 1.5,
    },
    // --- MCP communautaire empoisonné ---
    empoisonne: {
      base: 0.008, // base × menace + risque de la carte (exfiltration, typosquat), par carte communautaire active
      passerelle: 0.3, bacASable: [0, 0.3, 0.6],
      gravite: 0.42, confiance: 1.5,
    },
    // --- Changement silencieux d'un MCP de fournisseur ---
    silencieux: { base: 0.003, gravite: 0.35, confiance: 1.5, confianceDetection: 1 },
    // --- Destruction en production (accès en écriture et autonomie ≥ 3) ---
    destruction: {
      base: 0.002, autonomie4: 1.6, accesLecture: 0.1, // les cartes « destruction » restent risquées en lecture (× 0,1)
      gravite: 0.6, graviteCriticite: 0.3, confiance: 2,
    },
    // --- Fuite de données sensibles ---
    fuite: {
      base: 0.03, // × exposition Loi 25 de l'îlot × menace × usage
      gravite: 0.25, graviteSensibilite: 0.25, graviteExposition: 0.5, graviteTriade: 0.15, // gravité selon la sensibilité,
                                  // l'exposition Loi 25 et la triade complète
      seuilMajeure: 0.8, confiance: 2, confianceMajeure: 4,
      vigilance: 0.5,             // × (1 − 0,5 × formation)
    },
    incidentsMaxIlot: 1,          // incidents de sécurité au plus par îlot et par semaine
    // --- Surface d'attaque (0–100 par îlot) ---
    surface: {
      autonomie: 2,               // par niveau d'autonomie
      provenance: { maison: 1, fournisseur: 1.5, personnelle: 1.5, communaute: 3 },
      parBrancheCarte: 0.5,       // × (1 + 0,5 × branches de la carte)
      perimee: 1.5,               // × pour une carte périmée
      branche: 5,                 // par branche de la triade allumée
      triade: 20,                 // triade complète
      accesProd: { oui: 10, lecture: 4, non: 0 },
      fantome: 4, encadree: 1,    // par application hors TI
      bacASable: [0, 0.25, 0.45], // part retranchée
      passerelle: 0.75, approbationSorties: 0.85, // multiplicateurs
      echelle: 1.6,               // points bruts → score
    },
  };

  var CAUSES = ['injection', 'paquetHallucine', 'mcpEmpoisonne', 'changementSilencieux', 'destructionProd', 'fuiteDonnees'];

  function securiteVide() {
    var inc = {};
    CAUSES.forEach(function (c) { inc[c] = 0; });
    return { surfaceAttaque: 0, expositionLoi25: 0, parIlot: {}, incidents: inc, detections: 0 };
  }

  function rang(etat, id, ilotId) { return q.rangLevier(etat, id, ilotId); }
  function autonomie(etat, ilotId) { return Math.max(0, Math.min(4, q.levier(etat, 'autonomie', ilotId) | 0)); }
  function actives(etat, ilotId) {
    var reg = LTP.systemes.registre;
    if (reg && reg.cartesActives) return reg.cartesActives(etat, ilotId);
    return (etat.ilots[ilotId].capacites || []).map(function (id) { return { id: id, def: q.defCapacite(id), etc: etat.capacites[id] }; })
      .filter(function (c) { return c.def && c.etc; });
  }
  function risqueCarte(def, types) {
    var r = def.risque;
    return r && types.indexOf(r.type) >= 0 ? Math.max(0, r.proba || 0) : 0;
  }

  // --- Données organisationnelles ---
  function evoluerDonnees(etat) {
    var K = LTP.K.donneesSecu, d = etat.org.donnees, ilots = q.ilots(etat);
    var rc = rang(etat, 'catalogue');
    var volume = Math.max(1, O.somme(ilots, function (i) { return i.codebase.volume; }));
    var code = O.somme(ilots, function (i) { return Math.max(0, i.derniere.code || 0); });
    var fantomes = O.somme(ilots, function (i) { return i.applisFantomes || 0; });
    var encadrees = O.somme(ilots, function (i) { return i.applisEncadrees || 0; });
    d.duplication = O.borner(d.duplication + K.duplicationCode * code / volume + K.duplicationFantome * fantomes
      + K.duplicationEncadree * encadrees - K.catalogueDuplication[rc]);
    d.fraicheur = O.borner(d.fraicheur - K.fraicheurPerte * K.catalogueRalentit[rc] + K.catalogueRafraichit[rc]);
    var qc = K.qualiteCible;
    var cible = O.borner(qc.base - qc.duplication * d.duplication - qc.fraicheur * (1 - d.fraicheur) + qc.catalogue[rc]);
    d.qualite = O.borner(d.qualite + K.qualiteVitesse * (cible - d.qualite));
    var cc = K.classificationCible[q.levier(etat, 'classification')];
    if (cc == null) cc = K.classificationCible.non;
    d.classification = O.borner(d.classification + K.classificationVitesse * (cc - d.classification));
  }

  // --- Exposition et surface d'attaque (sans hasard) ---
  function evaluerIlot(etat, ilotId) {
    var K = LTP.K.donneesSecu, S = K.surface, il = etat.ilots[ilotId];
    var cartes = actives(etat, ilotId), a = autonomie(etat, ilotId);
    var fournisseurs = cartes.filter(function (c) { return c.def.provenance === 'fournisseur'; }).length;
    var externe = q.levier(etat, 'modele') !== 'local';
    var sortie = O.borner((externe ? K.sortieModeleExterne : K.sortieModeleLocal) + K.sortieParFournisseur * fournisseurs);
    var loi25 = O.borner(il.sensibilite * (1 - etat.org.donnees.classification) * sortie);

    var seuilPerime = LTP.K.registre ? LTP.K.registre.seuilPeremption : 0.45;
    var brut = S.autonomie * a;
    cartes.forEach(function (c) {
      var p = S.provenance[c.def.provenance] || 1;
      brut += p * (1 + S.parBrancheCarte * (c.def.branches || []).length) * (c.etc.fraicheur < seuilPerime ? S.perimee : 1);
    });
    var t = il.triade, nb = (t.prive ? 1 : 0) + (t.nonFiable ? 1 : 0) + (t.sortie ? 1 : 0);
    brut += S.branche * nb + (nb === 3 ? S.triade : 0);
    brut += S.accesProd[q.levier(etat, 'accesProd', ilotId)] || 0;
    brut += S.fantome * (il.applisFantomes || 0) + S.encadree * (il.applisEncadrees || 0);
    brut *= 1 - S.bacASable[rang(etat, 'bacASable', ilotId)];
    if (q.levier(etat, 'passerelleMCP') === true) brut *= S.passerelle;
    if (q.levier(etat, 'approbationSorties', ilotId) === true) brut *= S.approbationSorties;
    return { surface: O.borner(brut * S.echelle, 0, 100), loi25: loi25, sortieDonnees: sortie, cartes: cartes, branches: nb, autonomie: a };
  }

  function evaluerTout(etat) {
    var sec = etat.securite, surfaces = [], expos = [];
    q.ilots(etat).forEach(function (il) {
      var ev = evaluerIlot(etat, il.id);
      var anc = sec.parIlot[il.id];
      sec.parIlot[il.id] = { surface: ev.surface, loi25: ev.loi25, sortieDonnees: ev.sortieDonnees, risques: anc ? anc.risques : risquesVides() };
      surfaces.push(ev.surface);
      expos.push(ev.loi25);
    });
    var moy = O.moyenne(surfaces), max = surfaces.length ? Math.max.apply(null, surfaces) : 0;
    sec.surfaceAttaque = O.borner(0.6 * moy + 0.4 * max, 0, 100);
    sec.expositionLoi25 = O.borner(O.moyenne(expos) * 100, 0, 100);
  }
  function risquesVides() {
    var r = {};
    CAUSES.forEach(function (c) { r[c] = 0; });
    return r;
  }

  // --- Probabilités des risques de la semaine pour un îlot ---
  function probabilites(etat, ilotId, ev) {
    var K = LTP.K.donneesSecu, il = etat.ilots[ilotId], menace = etat.org.menace;
    var a = ev.autonomie, usage = K.usage[a];
    var bac = rang(etat, 'bacASable', ilotId);
    var formation = K.formation[rang(etat, 'formation')] || 0;
    var approbation = q.levier(etat, 'approbationSorties', ilotId) === true;
    var passerelle = q.levier(etat, 'passerelleMCP') === true;
    var acces = q.levier(etat, 'accesProd', ilotId);
    var p = risquesVides(), details = { empoisonne: [], silencieux: [] };

    // Injection de prompt : la triade complète, puis les garde-fous.
    var KI = K.injection;
    var facteurTriade = ev.branches === 3 ? 1 : KI.triadePartielle * ev.branches / 3;
    var cartesInjection = K.facteurRisqueCarte * O.somme(ev.cartes, function (c) { return risqueCarte(c.def, ['injection']); });
    p.injection = (KI.base * menace * facteurTriade + cartesInjection) * usage
      * (1 - KI.bacASable[bac]) * (1 - 0.5 * formation) * (approbation ? KI.approbationSorties : 1);

    // Paquet halluciné : modèle économique et autonomie élevée ; le bac à sable filtré le neutralise.
    var KP = K.paquet;
    p.paquetHallucine = KP.base * menace * (KP.modele[q.levier(etat, 'modele')] || 1) * KP.autonomie[a] * (1 - KP.bacASable[bac]);

    // MCP communautaire empoisonné : par carte communautaire active, sans passerelle.
    var KE = K.empoisonne, pasEmp = 1;
    ev.cartes.forEach(function (c) {
      if (c.def.provenance !== 'communaute') return;
      var pc = (KE.base * menace + K.facteurRisqueCarte * risqueCarte(c.def, ['exfiltration', 'typosquat']))
        * (passerelle ? KE.passerelle : 1) * (1 - KE.bacASable[bac]);
      pc = O.borner(pc);
      details.empoisonne.push({ id: c.id, p: pc, type: c.def.risque && c.def.risque.type });
      pasEmp *= 1 - pc;
    });
    p.mcpEmpoisonne = 1 - pasEmp;

    // Changement silencieux d'un MCP de fournisseur : détecté par la passerelle.
    var KS = K.silencieux, pasSil = 1;
    ev.cartes.forEach(function (c) {
      if (c.def.provenance !== 'fournisseur') return;
      var pc = O.borner(KS.base + K.facteurRisqueCarte * risqueCarte(c.def, ['changementSilencieux']));
      details.silencieux.push({ id: c.id, p: pc });
      pasSil *= 1 - pc;
    });
    p.changementSilencieux = 1 - pasSil;

    // Destruction en production : accès en écriture et autonomie ≥ 3 ; cartes à risque « destruction ».
    var KD = K.destruction;
    var cartesDestruction = K.facteurRisqueCarte * O.somme(ev.cartes, function (c) { return risqueCarte(c.def, ['destruction']); });
    var pd = 0;
    if (acces === 'oui' && a >= 3) pd += KD.base * (a === 4 ? KD.autonomie4 : 1);
    if (acces === 'oui') pd += cartesDestruction * usage;
    else if (acces === 'lecture') pd += cartesDestruction * usage * KD.accesLecture;
    p.destructionProd = pd;

    // Fuite de données sensibles : exposition Loi 25 × menace × usage ; cartes à risque « exfiltration ».
    var KF = K.fuite;
    var cartesFuite = K.facteurRisqueCarte * O.somme(ev.cartes, function (c) {
      return c.def.provenance === 'communaute' ? 0 : risqueCarte(c.def, ['exfiltration']); // les communautaires sont dans mcpEmpoisonne
    });
    // La formation apprend aussi à ne pas coller de données sensibles dans l'agent (vigilance).
    p.fuiteDonnees = (KF.base * ev.loi25 * menace + cartesFuite * il.sensibilite) * usage
      * (approbation ? 0.5 : 1) * (1 - (KF.vigilance || 0) * formation);

    for (var k in p) p[k] = O.borner(p[k]);
    return { p: p, details: details, passerelle: passerelle };
  }

  // --- Réalisation d'un risque ---
  function realiser(etat, ctx, ilotId, cause, gravite, confiance, drapeaux, vars) {
    var inc = q.creerIncident(etat, { ilot: ilotId, gravite: gravite, cause: cause }, ctx);
    if (!inc) return null;
    if (vars && vars.carte) inc.carte = vars.carte;
    etat.org.confiance = O.borner(etat.org.confiance - confiance, 0, 100);
    (drapeaux || []).forEach(function (d) { etat.org.drapeaux[d] = true; });
    etat.securite.incidents[cause] = (etat.securite.incidents[cause] || 0) + 1;
    return inc;
  }

  function tirerRisques(etat, ctx, ilotId, ev) {
    var K = LTP.K.donneesSecu, il = etat.ilots[ilotId], rng = ctx.rng;
    var r = probabilites(etat, ilotId, ev), p = r.p;
    etat.securite.parIlot[ilotId].risques = p;
    if (!q.devsDe(etat, ilotId).length) return;
    var n = 0, max = K.incidentsMaxIlot;
    function place() { return n < max; }

    if (place() && rng.chance(p.injection)) {
      realiser(etat, ctx, ilotId, 'injection', K.injection.gravite + K.injection.graviteSensibilite * il.sensibilite,
        K.injection.confiance, ['injectionPrompt']);
      n++;
    }
    if (place() && rng.chance(p.paquetHallucine)) {
      realiser(etat, ctx, ilotId, 'paquetHallucine', K.paquet.gravite, K.paquet.confiance, ['paquetHallucine']);
      n++;
    }
    if (place() && p.mcpEmpoisonne > 0) {
      for (var i = 0; i < r.details.empoisonne.length && place(); i++) {
        var e = r.details.empoisonne[i];
        if (!rng.chance(e.p)) continue;
        realiser(etat, ctx, ilotId, 'mcpEmpoisonne', K.empoisonne.gravite + 0.2 * il.sensibilite, K.empoisonne.confiance,
          ['mcpEmpoisonne'], { carte: e.id });
        n++;
      }
    }
    if (p.changementSilencieux > 0) {
      for (var j = 0; j < r.details.silencieux.length; j++) {
        var s = r.details.silencieux[j];
        if (!rng.chance(s.p)) continue;
        if (r.passerelle) {
          etat.securite.detections += 1;
          etat.org.drapeaux.changementDetecte = true;
          etat.org.confiance = O.borner(etat.org.confiance + K.silencieux.confianceDetection, 0, 100);
          ctx.noter('info', 'donneesSecu.changementDetecte', { carte: s.id, ilot: ilotId }, { ilot: ilotId });
        } else if (place()) {
          realiser(etat, ctx, ilotId, 'changementSilencieux', K.silencieux.gravite, K.silencieux.confiance,
            ['changementSilencieux'], { carte: s.id });
          n++;
        }
        break;
      }
    }
    if (place() && rng.chance(p.destructionProd)) {
      realiser(etat, ctx, ilotId, 'destructionProd', K.destruction.gravite + K.destruction.graviteCriticite * il.criticite,
        K.destruction.confiance, ['destructionProd']);
      n++;
    }
    if (place() && rng.chance(p.fuiteDonnees)) {
      var g = O.borner(K.fuite.gravite + K.fuite.graviteSensibilite * il.sensibilite + K.fuite.graviteExposition * ev.loi25
        + (ev.branches === 3 ? K.fuite.graviteTriade : 0) + rng.bruit(0.05));
      var majeure = g >= K.fuite.seuilMajeure;
      realiser(etat, ctx, ilotId, 'fuiteDonnees', g, majeure ? K.fuite.confianceMajeure : K.fuite.confiance,
        majeure ? ['fuiteDonnees', 'fuiteMajeure'] : ['fuiteDonnees']);
      n++;
    }
  }

  LTP.systemes.donneesSecu = {
    initialiser: function (etat) {
      etat.securite = securiteVide();
      evaluerTout(etat);
    },
    // Étape 8 : données, menace, exposition, surface d'attaque, puis risques de la semaine.
    semaine: function (etat, ctx) {
      if (!etat.securite) etat.securite = securiteVide();
      evoluerDonnees(etat);
      var hausse = LTP.K.donneesSecu.menaceHausse;
      if (Array.isArray(hausse)) hausse = hausse[Math.max(0, Math.min(hausse.length - 1, (etat.trimestre || 1) - 1))];
      etat.org.menace = O.borner(etat.org.menace + (Number(hausse) || 0));
      evaluerTout(etat);
      q.ilots(etat).forEach(function (il) { tirerRisques(etat, ctx, il.id, evaluerIlot(etat, il.id)); });
      evaluerTout(etat);
    },
    // Pour l'interface et les tests : évaluation et probabilités d'un îlot, sans hasard ni mutation.
    evaluer: function (etat, ilotId) {
      var ev = evaluerIlot(etat, ilotId);
      return { surface: ev.surface, loi25: ev.loi25, sortieDonnees: ev.sortieDonnees, risques: probabilites(etat, ilotId, ev).p };
    },
    CAUSES: CAUSES,
  };

  LTP.enregistrerTextes({
    'incident.paquetHallucine': 'Paquet halluciné : l\'agent a installé une dépendance au nom inventé, qu\'un attaquant avait publiée entre-temps. {ilot}.',
    'incident.mcpEmpoisonne': 'MCP communautaire piégé : il envoyait plus que prévu. {ilot}.',
    'incident.changementSilencieux': 'Un MCP de fournisseur a changé sa description sans prévenir ; l\'agent a suivi. {ilot}.',
    'incident.destructionProd': 'Un agent a « nettoyé » la production. Sans WHERE. {ilot}.',
    'incident.fuiteDonnees': 'Fuite de données sensibles : {ilot}. Me Bérubé arrive avec son PDF.',
    'donneesSecu.changementDetecte': 'La passerelle MCP a détecté un changement silencieux de {carte} ({ilot}) : appels limités à la lecture seule.',
  });
})(globalThis.LTP = globalThis.LTP || {});
