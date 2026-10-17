/* 27-personnes.js */
/* Les personnes (CONCEPTION §4.10) : disponibilité (congés forcés, formation, départ du contractuel),
   fatigue, moral, compétence, compétence IA, adhésion, rigueur, démissions, relève de l'organisation,
   réactions immédiates aux leviers et bulles de pensée pour l'interface.
   Champs propres : etat.personnes = { absences, departs, base, chocs, formationVue, formationDebut, juniorsDepart }. */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.personnes = {
    // --- Disponibilité ---
    seuilConge: 0.85,            // fatigue au-delà de laquelle le congé est forcé
    dureeConge: 2,               // semaines de congé forcé
    recuperationConge: 0.3,      // fatigue retirée par semaine de congé
    moralConge: 0.02,            // moral gagné par semaine de congé (on revient un peu reposé)

    // --- Départ du contractuel (Olivier) ---
    contractuel: {
      trait: 'contractuel',      // trait qui marque un contrat à durée fixe
      semaine: 26,               // fin du contrat
      drapeauProlonge: 'olivierReste', // drapeau posé par un événement : contrat prolongé
      prolongation: 13,          // semaines de plus si le contrat est prolongé
    },

    // --- Connaissance capturée et perte au départ ---
    captureParCarte: 0.35,       // part de la compréhension d'une personne capturée par une carte maison ou personnelle à jour
    captureMax: 0.8,             // on ne capture jamais tout
    perteDepart: 0.15,           // perte de compréhension des collègues de l'îlot × compréhension du partant × (1 − capture)

    // --- Formation ---
    // Pendant le délai du levier (préparation et cours), une fraction du temps part en formation,
    // et chaque personne passe une semaine sur periodeRotation en formation intensive.
    formation: {
      occupationCours: { aucune: 0, atelier: 0.08, programme: 0.15, mentorat: 0.18 },
      occupationIntensive: 0.6,  // fraction de la semaine « en salle »
      periodeRotation: 4,        // une semaine sur 4, à tour de rôle
      // Une fois le programme en vigueur : temps d'entretien hebdomadaire et gain de compétence IA.
      occupationContinue: { aucune: 0, atelier: 0, programme: 0.04, mentorat: 0.06 },
      gainCours: 0.025,          // compétence IA gagnée par semaine de cours (× (1 − competenceIA))
      gainContinu: { aucune: 0, atelier: 0.002, programme: 0.006, mentorat: 0.008 },
      mentoratCompetence: 0.004, // compétence de base gagnée par les juniors et le stagiaire avec mentorat
      mentoratOccupation: 0.05,  // temps des seniors consacré au mentorat
      adhesionCours: 0.004,      // l'adhésion monte un peu pendant les cours
    },

    // --- Fatigue : charge − récupération × fatigue (équilibre ≈ charge / récupération) ---
    fatigue: {
      base: 0.03,                // charge normale d'une semaine
      sessions: 0.025,           // par session d'agent au-delà de sessionsLibres
      sessionsLibres: 2,
      garde: 0.015,              // trait « garde » : nuits au téléphone
      gardeIncident: 0.03,       // par incident actif de l'îlot, pour la personne de garde
      incident: 0.04,            // la personne est sur un incident cette semaine
      incidentIlot: 0.008,       // par incident actif de l'îlot, pour tout le monde
      fileRevue: 0.04,           // réviseurs : pression de la file de revue (à pleine charge)
      fileRef: 15,               // PR en file pour une pleine charge de revue
      recuperation: 0.15,        // coefficient de récupération (× fatigue)
    },

    // --- Moral ---
    moral: {
      cible: 0.62,               // point d'équilibre sans autre influence
      rappel: 0.05,              // vitesse de retour vers la cible
      facteurReactions: 1,       // multiplicateur des réactions des personnages (données)
      reactionsMax: 0.015,       // plafond (±) de la somme hebdomadaire des réactions d'une personne : équilibre ± 0,3
      succes: 0.01,              // × (satisfaction de l'îlot − 0,5)
      incident: 0.012,           // par incident de l'îlot cette semaine
      fatigue: 0.04,             // × max(0, fatigue − seuilFatigue)
      seuilFatigue: 0.5,
      incoherence: 0.01,         // × conflit de skills dans l'îlot
      seuilDepart: 0.15,         // sous ce moral : démission
      chocLevier: 3,             // choc immédiat d'une réaction = moral hebdomadaire × chocLevier
      chocMax: 0.05,
    },

    // --- Compétence (apprentissage) ---
    competence: {
      apprentissage: 0.008,      // gain hebdomadaire de base (× (1 − competence))
      niveau: { stagiaire: 1.6, junior: 1.4, inter: 1, senior: 0.5 },
      poidsEcrit: 0.5,           // part du code écrit soi-même
      poidsPair: 0.35,           // pair humain
      poidsRevue: 0.25,          // revue humaine reçue
      pair: { aucun: 0, humain: 1, ia: 0.2, rotation: 0.5 },
      stagnation: 0.1,           // juniors et stagiaire, autonomie ≥ autonomieStagnation sans pair humain
      autonomieStagnation: 3,
      ecritParAutonomie: [0.9, 0.75, 0.55, 0.35, 0.15], // estimation si la production ne fournit pas ecritSoiMeme
    },

    // --- Compétence IA ---
    competenceIA: {
      pratique: 0.006,           // × min(1, sessions / 3) × adhésion, si l'autonomie ≥ 1
      mentoratPair: 0.002,       // pair humain-IA : on apprend aussi à s'en servir
    },

    // --- Adhésion ---
    adhesion: {
      reussite: 0.008,           // × (satisfaction de l'îlot − 0,5)
      echec: 0.01,               // par incident de l'îlot cette semaine, si l'autonomie ≥ 2
      moral: 0.006,              // × (moral − 0,5)
      plafondMentorat: 0.7,      // plafond des réfractaires une fois le mentorat en vigueur
      rappelPlafond: 0.1,        // vitesse de descente vers le plafond
    },

    // --- Rigueur ---
    rigueur: {
      rappel: 0.05,              // retour vers la rigueur naturelle
      fatigue: 0.03,             // × max(0, fatigue − seuilFatigue)
      seuilFatigue: 0.4,
    },

    // --- Relève (org.releve) ---
    releve: {
      base: 0.15,
      poidsCompetence: 0.8,      // × compétence moyenne des juniors et du stagiaire × présence
      embauche: { gel: -0.15, normale: 0, releve: 0.2 },
      vitesse: 0.05,             // convergence hebdomadaire vers la cible
    },

    seuilBulleFatigue: 0.7,      // au-delà, la bulle de fatigue l'emporte sur l'activité
    periodeBulleReaction: 3,     // une semaine sur 3, la bulle d'une réaction active remplace celle de l'activité
  };
  var K = LTP.K.personnes;

  var NIVEAUX_JUNIORS = { junior: true, stagiaire: true };

  function etatPropre(etat) {
    if (!etat.personnes) {
      etat.personnes = { absences: {}, departs: [], base: {}, chocs: {}, formationVue: null, formationDebut: null, juniorsDepart: 0 };
    }
    var p = etat.personnes;
    if (!p.absences) p.absences = {};
    if (!p.departs) p.departs = [];
    if (!p.base) p.base = {};
    if (!p.chocs) p.chocs = {};
    return p;
  }

  // Rigueur naturelle d'un développeur (niveau et traits), vers laquelle la rigueur revient au repos.
  function rigueurBase(etat, dev) {
    var p = etatPropre(etat);
    if (p.base[dev.id] == null) {
      var def = LTP.q.defDev(dev.id);
      var s = def ? LTP.etat.statsDepart(def) : null;
      p.base[dev.id] = s ? s.rigueur : dev.rigueur;
    }
    return p.base[dev.id];
  }

  function initialiser(etat) {
    var p = etatPropre(etat);
    LTP.q.devs(etat).forEach(function (d) { rigueurBase(etat, d); });
    p.juniorsDepart = LTP.q.devs(etat).filter(function (d) { return NIVEAUX_JUNIORS[d.niveau]; }).length;
    p.formationVue = LTP.q.levier(etat, 'formation');
  }

  // --- Connaissance capturée : cartes maison ou personnelles dont la personne est auteure ou propriétaire ---
  function carteExiste(etat, def, c) {
    if (!c) return false;
    if (def.disponible === 'developpement') return c.disponibleSemaine != null && c.disponibleSemaine <= etat.semaine;
    return c.decouverte !== false;
  }
  function cartesDe(etat, devId) {
    return (LTP.donnees.capacites || []).filter(function (def) {
      if (def.provenance !== 'maison' && def.provenance !== 'personnelle') return false;
      var c = etat.capacites && etat.capacites[def.id];
      if (!carteExiste(etat, def, c)) return false;
      return def.auteur === devId || c.proprietaire === devId;
    });
  }
  function capture(etat, devId) {
    var s = 0;
    cartesDe(etat, devId).forEach(function (def) {
      var c = etat.capacites[def.id];
      s += K.captureParCarte * O.borner(c.fraicheur == null ? 1 : c.fraicheur);
    });
    return Math.min(K.captureMax, s);
  }

  // Départ d'une personne (démission ou fin de contrat) : sa compréhension part avec elle, sauf la part capturée.
  function partir(etat, dev, motif, ctx) {
    if (!dev.present) return;
    var p = etatPropre(etat);
    var part = capture(etat, dev.id);
    dev.present = false;
    dev.etat = 'absent';
    dev.sessions = 0;
    delete p.absences[dev.id];
    LTP.q.devsDe(etat, dev.ilot).forEach(function (c) {
      c.comprehension = O.borner(c.comprehension - K.perteDepart * dev.comprehension * (1 - part));
    });
    p.departs.push({ dev: dev.id, ilot: dev.ilot, semaine: etat.semaine, motif: motif, capture: O.arrondir(part, 3) });
    if (ctx && ctx.noter) {
      ctx.noter(motif === 'demission' ? 'alerte' : 'personne', motif === 'demission' ? 'personnes.demission' : 'personnes.finContrat',
        { dev: dev.id, capture: Math.round(part * 100) }, { dev: dev.id, ilot: dev.ilot });
    }
  }

  function semaineFinContrat(etat) {
    var c = K.contractuel;
    return c.semaine + (etat.org.drapeaux[c.drapeauProlonge] ? c.prolongation : 0);
  }

  // Étape de la formation : 'cours' pendant le délai du levier, 'continue' une fois en vigueur.
  function situationFormation(etat) {
    var attente = LTP.q.enAttente(etat, 'formation', null);
    var effective = LTP.q.levier(etat, 'formation');
    if (attente && attente.valeur && attente.valeur !== 'aucune') return { phase: 'cours', valeur: attente.valeur, effective: effective };
    return { phase: effective && effective !== 'aucune' ? 'continue' : 'aucune', valeur: effective, effective: effective };
  }

  // --- Étape 2 : congés, retours, formation, fin de contrat ---
  function disponibilite(etat, ctx) {
    var p = etatPropre(etat), s = etat.semaine, q = LTP.q;
    var KF = K.formation;
    var form = situationFormation(etat);

    // Début d'un cours : une note, une seule fois par changement.
    if (form.phase === 'cours' && p.formationDebut !== form.valeur) {
      p.formationDebut = form.valeur;
      if (ctx && ctx.noter) ctx.noter('info', 'personnes.formationDebut', { valeur: form.valeur });
    } else if (form.phase !== 'cours') p.formationDebut = null;
    p.formationVue = form.effective;

    // Fin de contrat du contractuel.
    q.devsPresents(etat).forEach(function (d) {
      if (q.aTrait(d, K.contractuel.trait) && s >= semaineFinContrat(etat)) partir(etat, d, 'finContrat', ctx);
    });

    var presents = q.devsPresents(etat);
    presents.forEach(function (d, i) {
      var abs = p.absences[d.id];
      if (abs && abs.fin > s) {
        d.etat = abs.motif;
        d.indisponibleJusqua = Math.max(d.indisponibleJusqua || 0, abs.fin);
        return;
      }
      if (abs) delete p.absences[d.id];
      if (d.etat === 'conge' || d.etat === 'formation' || d.etat === 'absent') d.etat = 'code';

      // Congé forcé.
      if (d.fatigue > K.seuilConge) {
        p.absences[d.id] = { motif: 'conge', fin: s + K.dureeConge };
        d.etat = 'conge';
        d.indisponibleJusqua = s + K.dureeConge;
        if (ctx && ctx.noter) ctx.noter('personne', 'personnes.conge', { dev: d.id, semaines: K.dureeConge }, { dev: d.id, ilot: d.ilot });
        return;
      }

      // Formation.
      var occ = d.occupation || (d.occupation = LTP.etat.occupationVide());
      if (form.phase === 'cours') {
        occ.formation = (KF.occupationCours[form.valeur] || 0);
        if ((s + i) % KF.periodeRotation === 0) {
          occ.formation = Math.max(occ.formation, KF.occupationIntensive);
          d.etat = 'formation';
        }
      } else if (form.phase === 'continue') {
        occ.formation = KF.occupationContinue[form.valeur] || 0;
      }
      if (form.effective === 'mentorat' && d.niveau === 'senior') occ.formation += KF.mentoratOccupation;
      occ.formation = O.borner(occ.formation, 0, 0.9);
    });
  }

  // --- Étape 11 : fatigue, moral, compétences, adhésion, rigueur, départs, relève ---
  function semaine(etat, ctx) {
    var q = LTP.q, p = etatPropre(etat);
    var form = situationFormation(etat);
    var KF = K.formation, KFa = K.fatigue, KM = K.moral, KC = K.competence;

    q.devsPresents(etat).forEach(function (d) {
      var ilot = etat.ilots[d.ilot];
      if (!ilot) return;
      var der = d.derniere || {};
      var incActifs = q.incidentsActifs(etat, d.ilot).length;
      var incSemaine = (ilot.derniere && ilot.derniere.incidents) || 0;
      var enConge = d.etat === 'conge';
      var autonomie = Number(q.levier(etat, 'autonomie', d.ilot)) || 0;

      // Fatigue.
      if (enConge) {
        d.fatigue = O.borner(d.fatigue - K.recuperationConge);
        d.moral = O.borner(d.moral + K.moralConge);
      } else {
        var charge = KFa.base + KFa.sessions * Math.max(0, (d.sessions || 0) - KFa.sessionsLibres);
        if (q.aTrait(d, 'garde')) charge += KFa.garde + KFa.gardeIncident * Math.min(3, incActifs);
        if (d.etat === 'incident') charge += KFa.incident;
        charge += KFa.incidentIlot * Math.min(3, incActifs);
        if ((der.revues || 0) > 0 || q.aTrait(d, 'reviseur')) charge += KFa.fileRevue * Math.min(1, (ilot.fileRevue || 0) / KFa.fileRef);
        charge *= q.modTrait(d, 'fatigue');
        d.fatigue = O.borner(d.fatigue + charge - KFa.recuperation * d.fatigue);
      }

      // Moral.
      var dm = KM.rappel * (KM.cible - d.moral);
      var def = q.defDev(d.id), reactions = 0;
      ((def && def.reactions) || []).forEach(function (r) {
        if (!r || !r.levier || !Array.isArray(r.valeurs)) return;
        if (r.valeurs.indexOf(q.levier(etat, r.levier, d.ilot)) >= 0) reactions += KM.facteurReactions * (Number(r.moral) || 0);
      });
      dm += Math.max(-KM.reactionsMax, Math.min(KM.reactionsMax, reactions));
      dm += KM.succes * ((ilot.satisfaction == null ? 0.5 : ilot.satisfaction) - 0.5);
      dm -= KM.incident * Math.min(3, incSemaine);
      dm -= KM.fatigue * Math.max(0, d.fatigue - KM.seuilFatigue);
      dm -= KM.incoherence * O.borner((ilot.mods && ilot.mods.incoherence) || 0);
      d.moral = O.borner(d.moral + dm);

      if (!enConge) {
        // Compétence.
        var ecrit = der.ecritSoiMeme;
        if (!(der.points > 0 || der.code > 0 || ecrit > 0)) ecrit = KC.ecritParAutonomie[Math.max(0, Math.min(4, autonomie))];
        var pairV = q.levier(etat, 'pair');
        var pair = KC.pair[pairV] || 0;
        if (pairV === 'rotation') pair = etat.semaine % 2 === 0 ? 1 : 0;
        var reviseurs = Number(q.levier(etat, 'reviseurs', d.ilot)) || 0;
        var attention = ilot.derniere && ilot.derniere.attention != null ? ilot.derniere.attention : 1;
        var revue = reviseurs > 0 ? O.borner(attention) * (q.levier(etat, 'approbation') === 'auto' ? 0.3 : 1) : 0;
        var gain = KC.poidsEcrit * O.borner(ecrit) + KC.poidsPair * pair + KC.poidsRevue * revue;
        gain *= KC.apprentissage * (KC.niveau[d.niveau] || 1) * q.modTrait(d, 'apprentissage');
        if (NIVEAUX_JUNIORS[d.niveau] && autonomie >= KC.autonomieStagnation && pairV !== 'humain' && pairV !== 'rotation') gain *= KC.stagnation;
        if (form.effective === 'mentorat' && NIVEAUX_JUNIORS[d.niveau]) gain += KF.mentoratCompetence;
        d.competence = O.borner(d.competence + gain * (1 - d.competence));

        // Compétence IA.
        var gIA = 0;
        if (autonomie >= 1) gIA += K.competenceIA.pratique * Math.min(1, (d.sessions || 0) / 3) * d.adhesion;
        if (pairV === 'ia') gIA += K.competenceIA.mentoratPair;
        if (form.phase === 'cours') gIA += KF.gainCours * ((d.occupation && d.occupation.formation) > 0 ? 1 : 0.3);
        else gIA += KF.gainContinu[form.effective] || 0;
        d.competenceIA = O.borner(d.competenceIA + gIA * (1 - d.competenceIA));
      }

      // Adhésion : traits (plafond), réussites et échecs, moral, formation.
      var KA = K.adhesion;
      var da = KA.reussite * ((ilot.satisfaction == null ? 0.5 : ilot.satisfaction) - 0.5) + KA.moral * (d.moral - 0.5);
      if (autonomie >= 2) da -= KA.echec * Math.min(3, incSemaine);
      if (form.phase === 'cours') da += KF.adhesionCours;
      var plafond = q.modTrait(d, 'adhesionMax');
      if (form.effective === 'mentorat') plafond = Math.max(plafond, KA.plafondMentorat);
      plafond = O.borner(plafond);
      var a = d.adhesion + da;
      if (a > plafond) a -= Math.max(KA.rappelPlafond * (a - plafond), Math.min(a - plafond, 0.01));
      d.adhesion = O.borner(a);

      // Rigueur : baisse avec la fatigue, revient au repos.
      var KR = K.rigueur;
      d.rigueur = O.borner(d.rigueur + KR.rappel * (rigueurBase(etat, d) - d.rigueur) - KR.fatigue * Math.max(0, d.fatigue - KR.seuilFatigue));
    });

    // Démissions.
    q.devsPresents(etat).forEach(function (d) {
      if (d.moral < KM.seuilDepart) partir(etat, d, 'demission', ctx);
    });

    // Relève de l'organisation.
    var KRe = K.releve;
    var juniors = q.devsPresents(etat).filter(function (d) { return NIVEAUX_JUNIORS[d.niveau]; });
    var presence = p.juniorsDepart > 0 ? juniors.length / p.juniorsDepart : 1;
    var cible = KRe.base + KRe.poidsCompetence * O.moyenne(juniors, 'competence') * presence + (KRe.embauche[q.levier(etat, 'embauche')] || 0);
    etat.org.releve = O.borner(etat.org.releve + KRe.vitesse * (O.borner(cible) - etat.org.releve));
  }

  // --- Réaction immédiate à un changement de levier ---
  function reactionLevier(etat, action, ctx) {
    if (!action || action.type !== 'levier') return;
    var q = LTP.q, p = etatPropre(etat);
    var meilleur = null;
    q.devsPresents(etat).forEach(function (d) {
      if (action.ilot && d.ilot !== action.ilot) return;
      // Un changement global ne touche pas les îlots qui ont leur propre surcharge.
      if (!action.ilot && etat.leviers.surcharges[d.ilot] && Object.prototype.hasOwnProperty.call(etat.leviers.surcharges[d.ilot], action.id)) return;
      var def = q.defDev(d.id);
      ((def && def.reactions) || []).forEach(function (r) {
        if (!r || r.levier !== action.id || !Array.isArray(r.valeurs) || r.valeurs.indexOf(action.valeur) < 0) return;
        var cle = d.id + '|' + action.id;
        if (p.chocs[cle] === etat.semaine) return;
        p.chocs[cle] = etat.semaine;
        var choc = Math.max(-K.moral.chocMax, Math.min(K.moral.chocMax, (Number(r.moral) || 0) * K.moral.chocLevier));
        d.moral = O.borner(d.moral + choc);
        if (r.bulle && (!meilleur || Math.abs(choc) > Math.abs(meilleur.choc))) meilleur = { dev: d, r: r, choc: choc };
      });
    });
    if (meilleur && ctx && ctx.noter) {
      ctx.noter('personne', 'personnes.reaction', { dev: meilleur.dev.id, bulle: texte(meilleur.r.bulle, etat, meilleur.dev), levier: action.id },
        { dev: meilleur.dev.id, ilot: meilleur.dev.ilot });
    }
  }

  function texte(x, etat, dev) {
    if (typeof x === 'function') {
      try { return String(x(etat, LTP.q, dev)); } catch (e) { return ''; }
    }
    if (Array.isArray(x)) return x.length ? String(x[Math.floor(O.hachage(dev.id + etat.semaine) * x.length)]) : '';
    return x == null ? '' : String(x);
  }

  // --- Bulle de pensée courante (pour l'interface) ---
  function etatBulle(etat, d) {
    if (!d.present) return 'conge';
    if (d.etat === 'conge' || d.etat === 'absent') return 'conge';
    if (d.etat === 'incident') return 'incident';
    if (d.fatigue > K.seuilBulleFatigue) return 'fatigue';
    return d.etat || 'code';
  }
  function choisir(liste, d, etat) {
    var l = (liste || []).filter(function (x) { return typeof x === 'string' && x; });
    if (!l.length) return null;
    var decalage = Math.floor(O.hachage(d.id) * 997);
    return l[(etat.semaine + decalage) % l.length];
  }
  function bulle(etat, devId) {
    var d = etat.devs[devId];
    if (!d) return null;
    var def = LTP.q.defDev(devId) || {};
    var e = etatBulle(etat, d);
    // De temps en temps, la personne pense tout haut à une politique qui la fait réagir.
    if ((e === 'code' || e === 'attente') && (etat.semaine + Math.floor(O.hachage(d.id) * 7)) % K.periodeBulleReaction === 0) {
      var actives = (def.reactions || []).filter(function (r) {
        return r && r.bulle && typeof r.bulle === 'string' && Array.isArray(r.valeurs) && r.valeurs.indexOf(LTP.q.levier(etat, r.levier, d.ilot)) >= 0;
      });
      if (actives.length) return actives[etat.semaine % actives.length].bulle;
    }
    var propres = def.bulles && def.bulles[e];
    var x = choisir(propres, d, etat);
    if (x) return x;
    var generiques = LTP.donnees.bulles && LTP.donnees.bulles[e];
    x = choisir(generiques, d, etat);
    if (x) return x;
    return choisir(LTP.donnees.bulles && LTP.donnees.bulles.code, d, etat);
  }

  LTP.systemes.personnes = {
    initialiser: initialiser,
    disponibilite: disponibilite,
    semaine: semaine,
    reactionLevier: reactionLevier,
    bulle: bulle,
    etatBulle: etatBulle,
    // Outils exposés pour les autres systèmes et les tests.
    capture: capture,
    cartesDe: cartesDe,
    partir: partir,
    situationFormation: situationFormation,
  };

  LTP.enregistrerTextes({
    'personnes.conge': '{dev} n\'en peut plus : congé forcé de {semaines} semaines.',
    'personnes.demission': '{dev} démissionne. Ce que cette personne savait du code s\'en va aussi ({capture}\u202f% capturé dans des skills).',
    'personnes.finContrat': 'Fin du contrat de {dev}. Ce qu\'il savait part avec lui ({capture}\u202f% capturé dans des skills).',
    'personnes.formationDebut': 'La formation commence : l\'équipe passe une partie de ses semaines en salle.',
    'personnes.reaction': '{dev} : « {bulle} »',
  });
})(globalThis.LTP = globalThis.LTP || {});
