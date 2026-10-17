/* 31-boucle.js */
/* La boucle du moteur : jour(), semaine(), action(), trimestres et fin de partie.
   Le joueur avance d'une journée ouvrable à la fois (jour). Du lundi au jeudi, la journée ne fait que
   vivre : réflexions des développeurs (33-reflexions), carte d'événement programmée. Le vendredi, toute la
   semaine est calculée d'un coup par les systèmes (ORDRE), comme un rapport hebdomadaire : l'équilibrage
   hebdomadaire est donc inchangé. Une carte tirée le vendredi est montrée un jour de la semaine suivante.
   Les fonctions publiques ne mutent jamais l'état reçu : elles travaillent sur une copie. */
(function (LTP) {
  'use strict';
  var O = LTP.outils, q = LTP.q;

  // Ordre des étapes d'une semaine : [système, méthode]. Une méthode absente est ignorée.
  var ORDRE = [
    ['effets', 'differes'],
    ['personnes', 'disponibilite'],
    ['registre', 'semaine'],
    ['production', 'semaine'],
    ['qualite', 'semaine'],
    ['dette', 'semaine'],
    ['couts', 'semaine'],
    ['donneesSecu', 'semaine'],
    ['dette', 'incidents'],
    ['demandes', 'semaine'],
    ['personnes', 'semaine'],
    ['evenements', 'tirage'],
    ['metriques', 'semaine'],
  ];
  // Ordre d'appel des initialiser(etat) à la création d'une partie.
  var ORDRE_INIT = ['registre', 'production', 'qualite', 'dette', 'couts', 'donneesSecu', 'demandes', 'personnes', 'evenements', 'metriques', 'fin', 'reflexions'];

  function creerContexte(etat) {
    var journal = [];
    var ctx = {
      rng: LTP.rng.creer(etat),
      K: LTP.K,
      q: q,
      journal: journal,
      // Semaine inscrite au journal : celle que le joueur est en train de jouer. null : etat.semaine
      // (pendant le calcul du vendredi, etat.semaine vient d'être incrémentée et c'est la bonne).
      semaine: null,
      // type : 'info' | 'alerte' | 'incident' | 'evenement' | 'levier' | 'personne' | 'comite' ; cle : clé de texte
      noter: function (type, cle, vars, ref) {
        var e = { semaine: ctx.semaine != null ? ctx.semaine : etat.semaine, jour: etat.jour || 0, type: type, cle: cle, vars: vars || {} };
        if (ref && ref.ilot) e.ilot = ref.ilot;
        if (ref && ref.dev) e.dev = ref.dev;
        journal.push(e);
        return e;
      },
    };
    return ctx;
  }

  function archiverJournal(etat, journal) {
    for (var i = 0; i < journal.length; i++) etat.journal.push(journal[i]);
    var max = LTP.K.etat.journalMax;
    if (etat.journal.length > max) etat.journal = etat.journal.slice(etat.journal.length - max);
  }

  function appliquerLeviersEnAttente(etat, ctx) {
    var restants = [];
    etat.leviers.enAttente.forEach(function (a) {
      if (a.semaine <= etat.semaine) {
        ecrireLevier(etat, a.id, a.valeur, a.ilot);
        // En vigueur au calcul de ce vendredi : la semaine qui vient de se terminer, jour 4.
        noterChangement(etat, a.id, a.valeur, a.ilot, true, { semaine: etat.semaine, jour: 4, effectif: true });
        ctx.noter('levier', 'levier.effectif', { levier: a.id, valeur: a.valeur }, { ilot: a.ilot });
      } else restants.push(a);
    });
    etat.leviers.enAttente = restants;
  }

  // Garde la trace des changements de leviers : les réflexions en parlent, le tableau de bord les marque.
  function noterChangement(etat, id, valeur, ilot, differe, quand) {
    var l = etat.leviers.changements || (etat.leviers.changements = []);
    var c = { id: id, valeur: valeur, ilot: ilot || null,
      semaine: quand ? quand.semaine : q.semaineCourante(etat), jour: quand ? quand.jour : (etat.jour || 0) };
    if (differe) c.differe = true;
    if (quand && quand.effectif) c.effectif = true;
    l.push(c);
    var max = LTP.K.etat.changementsMax || 40;
    if (l.length > max) etat.leviers.changements = l.slice(l.length - max);
  }

  function ecrireLevier(etat, id, valeur, ilotId) {
    if (ilotId) {
      etat.leviers.surcharges[ilotId] = etat.leviers.surcharges[ilotId] || {};
      etat.leviers.surcharges[ilotId][id] = valeur;
    } else etat.leviers.global[id] = valeur;
    if (id === 'objectif') etat.org.objectif = valeur;
  }

  function creer(graine, options) {
    return LTP.etat.creer(graine, options);
  }

  // Avance d'une journée ouvrable. Refuse d'avancer hors de la phase d'exécution ou si un événement attend une décision.
  // Le vendredi, la semaine entière est calculée (calculerSemaine) et on passe au lundi suivant.
  // Renvoie { etat, journal, avance, finSemaine }.
  function jour(etatEntree) {
    if (etatEntree.phase !== 'execution' || etatEntree.evenementEnAttente) return { etat: etatEntree, journal: [], avance: false };
    return avancerJour(O.cloner(etatEntree));
  }

  // Corps de jour(), qui modifie l'état reçu (simuler s'en sert sur sa propre copie, pour aller vite).
  function avancerJour(etat) {
    var ctx = creerContexte(etat);
    var J = LTP.K.etat.joursParSemaine || 5;
    var finSemaine = (etat.jour || 0) >= J - 1;

    if (finSemaine) {
      calculerSemaine(etat, ctx);
      differerEvenement(etat, ctx);
      etat.jour = 0;
      if (etat.phase === 'execution') annoncerDeblocage(etat, ctx);
    } else {
      etat.jour = (etat.jour || 0) + 1;
    }
    ctx.semaine = q.semaineCourante(etat);
    if (etat.phase === 'execution') {
      poserEvenementProgramme(etat, ctx);
      assurerReflexions(etat, ctx);
    }
    archiverJournal(etat, ctx.journal);
    return { etat: etat, journal: ctx.journal, avance: true, finSemaine: finSemaine };
  }

  // Avance jusqu'à la fin de la semaine en cours (vendredi compris). S'arrête plus tôt si une carte d'événement
  // attend une décision : avance vaut alors vrai, mais la semaine n'est pas finie (finSemaine faux).
  function semaine(etatEntree) {
    var etat = etatEntree, journal = [], avance = false, fin = false;
    var J = LTP.K.etat.joursParSemaine || 5;
    for (var i = 0; i < J && !fin; i++) {
      var r = jour(etat);
      if (!r.avance) break;
      avance = true;
      etat = r.etat;
      journal = journal.concat(r.journal);
      fin = !!r.finSemaine;
    }
    return { etat: etat, journal: journal, avance: avance, finSemaine: fin };
  }

  // Le calcul de la semaine (le rapport du vendredi) : remise à zéro des statistiques, leviers programmés, systèmes.
  function calculerSemaine(etat, ctx) {
    etat.semaine += 1;
    ctx.semaine = null;
    LTP.q.ilots(etat).forEach(function (i) { i.derniere = LTP.etat.derniereIlotVide(); });
    LTP.q.devs(etat).forEach(function (d) { d.derniere = LTP.etat.derniereDevVide(); d.occupation = LTP.etat.occupationVide(); });
    appliquerLeviersEnAttente(etat, ctx);

    for (var i = 0; i < ORDRE.length; i++) {
      var s = LTP.systemes[ORDRE[i][0]];
      var f = s && s[ORDRE[i][1]];
      if (typeof f === 'function') f.call(s, etat, ctx);
    }

    finDeSemaine(etat, ctx);
  }

  // Une carte tirée le vendredi attend un jour de la semaine suivante (du mardi au vendredi, selon la graine).
  function differerEvenement(etat, ctx) {
    var KE = LTP.K.evenements || {};
    var att = etat.evenementEnAttente;
    if (!att || KE.differer === false || etat.phase !== 'execution') return;
    var J = LTP.K.etat.joursParSemaine || 5;
    var j = 1 + Math.floor(O.hachage(etat.graine + ':evenement:' + etat.semaine) * (J - 1));
    etat.evenementProgramme = { id: att.id, ref: att.ref, vars: att.vars, jour: Math.min(J - 1, j) };
    etat.evenementEnAttente = null;
    // L'annonce au journal attend le jour où la carte arrive vraiment.
    for (var i = ctx.journal.length - 1; i >= 0; i--) {
      var e = ctx.journal[i];
      if (e.cle === 'evenements.tire' && e.vars && e.vars.evenement === att.id) {
        etat.evenementProgramme.note = { vars: e.vars, ilot: e.ilot || null, dev: e.dev || null };
        ctx.journal.splice(i, 1);
        break;
      }
    }
  }
  function poserEvenementProgramme(etat, ctx) {
    var p = etat.evenementProgramme;
    if (!p || etat.evenementEnAttente || (etat.jour || 0) < p.jour) return;
    etat.evenementEnAttente = { id: p.id, ref: p.ref || {}, vars: p.vars || {} };
    etat.evenementProgramme = null;
    var n = p.note || { vars: { evenement: p.id } };
    var ref = {};
    if (n.ilot) ref.ilot = n.ilot;
    if (n.dev) ref.dev = n.dev;
    ctx.noter('evenement', 'evenements.tire', n.vars, ref);
  }

  // Note de service du calendrier pour la semaine courante (une seule fois par semaine).
  function annoncerDeblocage(etat, ctx) {
    if (etat.options && etat.options.toutDebloque) return;
    var s = q.semaineCourante(etat);
    var c = q.deblocage(s);
    if (!c || (etat.nouveaute && etat.nouveaute.semaine === s)) return;
    etat.nouveaute = { semaine: s, vue: false };
    var sem = ctx.semaine;
    ctx.semaine = s;
    ctx.noter('levier', 'calendrier.deblocage', { semaine: s, titre: c.titre || '', leviers: (c.leviers || []).slice() });
    ctx.semaine = sem;
  }

  // Pendant simuler, les réflexions ne sont calculées qu'à la fin : elles ne changent rien à la simulation.
  var silencieux = false;
  function assurerReflexions(etat, ctx) {
    var r = LTP.systemes.reflexions;
    if (!silencieux && r && r.assurer) r.assurer(etat, ctx);
  }

  function finDeSemaine(etat, ctx) {
    var K = LTP.K.etat, fin = LTP.systemes.fin;
    var seuil = fin && fin.seuilConfiance != null ? fin.seuilConfiance : 15;
    if (etat.org.confiance < seuil) {
      terminer(etat, ctx, 'confiance');
    } else if (etat.semaine >= K.semainesTotal) {
      terminer(etat, ctx, 'annee');
    } else if (etat.semaine % K.semainesParTrimestre === 0) {
      etat.phase = 'comite';
      if (fin && fin.preparerComite) fin.preparerComite(etat, ctx);
      else etat.comite = { trimestre: etat.trimestre, questions: [], reponses: {} };
      ctx.noter('comite', 'comite.convoque', { trimestre: etat.trimestre });
    }
  }

  function terminer(etat, ctx, raison) {
    etat.phase = 'fin';
    etat.evenementEnAttente = null;
    etat.evenementProgramme = null;
    var fin = LTP.systemes.fin;
    if (fin && fin.calculerResultat) etat.resultat = fin.calculerResultat(etat, ctx, raison);
    else etat.resultat = { raison: raison, profil: null, etoiles: 0, score: 0, criteres: {} };
    ctx.noter('info', 'partie.fin', { raison: raison });
  }

  var TRANSITIONS = {
    accueil: ['tutoriel', 'planification'],
    tutoriel: ['planification'],
    planification: ['execution'],
    execution: ['planification'],
    comite: ['planification'],
    fin: [],
  };

  // Applique une action du joueur. Renvoie { etat, journal, erreur? }.
  function action(etatEntree, a) {
    var etat = O.cloner(etatEntree);
    var ctx = creerContexte(etat);
    ctx.semaine = q.semaineCourante(etat);
    var erreur = null;
    try {
      erreur = appliquerAction(etat, a, ctx);
      // Ce qui vient d'une décision du joueur est marqué (le journal peut le distinguer de ce qui arrive tout seul).
      if (!erreur && a.type !== 'phase') ctx.journal.forEach(function (e) { e.joueur = true; });
      if (!erreur && (etat.phase === 'planification' || etat.phase === 'execution')) assurerReflexions(etat, ctx);
    } catch (e) {
      erreur = 'exception:' + (e && e.message);
    }
    if (erreur) return { etat: etatEntree, journal: [], erreur: erreur };
    archiverJournal(etat, ctx.journal);
    return { etat: etat, journal: ctx.journal };
  }

  function appliquerAction(etat, a, ctx) {
    if (!a || !a.type) return 'action-invalide';
    switch (a.type) {
      case 'levier': {
        var d = q.defLevier(a.id);
        if (!d) return 'levier-inconnu';
        if (!q.levierDebloque(etat, a.id)) return 'levier-verrouille';
        if (!d.valeurs.some(function (x) { return x.v === a.valeur; })) return 'valeur-invalide';
        if (!q.valeurDebloquee(etat, a.id, a.valeur)) return 'valeur-verrouillee';
        if (a.ilot && !d.surchargeable) return 'levier-non-surchargeable';
        if (a.ilot && !etat.ilots[a.ilot]) return 'ilot-inconnu';
        if (etat.phase === 'fin') return 'partie-terminee';
        var ilot = a.ilot || null;
        etat.leviers.enAttente = etat.leviers.enAttente.filter(function (x) { return !(x.id === a.id && (x.ilot || null) === ilot); });
        var avant = q.levier(etat, a.id, ilot);
        if (d.delai > 0 && avant !== a.valeur) {
          etat.leviers.enAttente.push({ id: a.id, valeur: a.valeur, ilot: ilot, semaine: etat.semaine + d.delai });
          ctx.noter('levier', 'levier.programme', { levier: a.id, valeur: a.valeur, delai: d.delai }, { ilot: ilot });
        } else {
          ecrireLevier(etat, a.id, a.valeur, ilot);
          ctx.noter('levier', 'levier.change', { levier: a.id, valeur: a.valeur }, { ilot: ilot });
        }
        if (avant !== a.valeur) noterChangement(etat, a.id, a.valeur, ilot, d.delai > 0);
        appelerReaction(etat, ctx, a);
        return null;
      }
      case 'retirerSurcharge': {
        if (etat.leviers.surcharges[a.ilot]) delete etat.leviers.surcharges[a.ilot][a.id];
        etat.leviers.enAttente = etat.leviers.enAttente.filter(function (x) { return !(x.id === a.id && x.ilot === a.ilot); });
        return null;
      }
      case 'equiper': case 'desequiper': case 'developper': case 'proprietaire': {
        var reg = LTP.systemes.registre;
        if (!reg || !reg.action) return 'registre-indisponible';
        return reg.action(etat, a, ctx) || null;
      }
      case 'choix': {
        if (!etat.evenementEnAttente) return 'aucun-evenement';
        var ev = LTP.systemes.evenements;
        if (!ev || !ev.resoudre) return 'evenements-indisponibles';
        return ev.resoudre(etat, a.choix, ctx) || null;
      }
      case 'comite': {
        if (etat.phase !== 'comite') return 'pas-de-comite';
        var fin = LTP.systemes.fin;
        if (!fin || !fin.repondreComite) return 'comite-indisponible';
        return fin.repondreComite(etat, a.question, a.reponse, ctx) || null;
      }
      case 'phase': {
        var permis = TRANSITIONS[etat.phase] || [];
        if (permis.indexOf(a.phase) < 0) return 'transition-interdite';
        if (etat.phase === 'comite') {
          var f2 = LTP.systemes.fin;
          if (f2 && f2.comiteTermine && !f2.comiteTermine(etat)) return 'comite-incomplet';
          etat.trimestre += 1;
          etat.comite = null;
          ctx.noter('info', 'trimestre.debut', { trimestre: etat.trimestre });
          if (f2 && f2.debutTrimestre) f2.debutTrimestre(etat, ctx);
        }
        etat.phase = a.phase;
        if (a.phase === 'planification' || a.phase === 'execution') annoncerDeblocage(etat, ctx);
        return null;
      }
      case 'nouveauteVue': {
        if (etat.nouveaute) etat.nouveaute.vue = true;
        return null;
      }
      case 'option': {
        if (!etat.options.animateur && a.cle !== 'serveurAchete') return 'animateur-requis';
        etat.options[a.cle] = a.valeur;
        return null;
      }
      default:
        return 'action-inconnue';
    }
  }

  function appelerReaction(etat, ctx, a) {
    var p = LTP.systemes.personnes;
    if (p && p.reactionLevier) p.reactionLevier(etat, a, ctx);
    var r = LTP.systemes.reflexions;
    if (!silencieux && r && r.reaction) r.reaction(etat, a, ctx);
  }

  // Signature des déblocages : change à chaque trimestre et, si la stratégie suit le calendrier,
  // à chaque nouveauté (suivreCalendrier: false pour une stratégie qui ne règle qu'en planification).
  function signatureDeblocage(etat, strategie) {
    var n = 0;
    if (!strategie || strategie.suivreCalendrier !== false) {
      var s = q.semaineCourante(etat);
      n = (LTP.donnees.calendrier || []).filter(function (c) { return c.semaine <= s; }).length;
    }
    return etat.graine + ':' + etat.trimestre + ':' + n + ':' + !!(etat.options && etat.options.toutDebloque);
  }

  // Joue n semaines en résolvant tout seul les décisions. strategie (facultative) :
  //   { leviers(etat) → [actions], choix(etat, evenement) → idChoix, comite(etat, question) → idReponse }
  // leviers est appelé au début de chaque trimestre et à chaque nouveauté du calendrier (un lundi),
  // sauf si strategie.suivreCalendrier vaut false (seulement en planification).
  // Sert aux tests, à l'équilibrage et au mode animateur.
  function simuler(etatEntree, n, strategie) {
    var dejaSilencieux = silencieux;
    silencieux = true;
    try {
      var r = simulerCorps(etatEntree, n, strategie);
    } finally {
      silencieux = dejaSilencieux;
    }
    if (!silencieux && (r.etat.phase === 'planification' || r.etat.phase === 'execution')) {
      var ctx = creerContexte(r.etat);
      ctx.semaine = q.semaineCourante(r.etat);
      assurerReflexions(r.etat, ctx);
    }
    return r;
  }

  function simulerCorps(etatEntree, n, strategie) {
    var etat = etatEntree, s = strategie || {};
    var propre = false; // vrai quand etat est une copie qui nous appartient (on peut avancer sur place)
    var journal = [];
    var garde = 0;
    function agir(a) {
      var r = action(etat, a);
      if (!r.erreur) { etat = r.etat; propre = true; journal.push.apply(journal, r.journal); }
      return r;
    }
    // La signature est gardée sur l'objet stratégie : plusieurs appels de simuler ne rejouent pas les mêmes réglages.
    function regler() {
      if (!s.leviers) return;
      var sig = signatureDeblocage(etat, s);
      if (s.__sigLeviers === sig) return;
      s.__sigLeviers = sig;
      (s.leviers(etat) || []).forEach(agir);
    }
    var semainesJouees = 0;
    var J = LTP.K.etat.joursParSemaine || 5;
    while (semainesJouees < n && etat.phase !== 'fin' && garde++ < n * (J + 1) * 4 + 50) {
      if (etat.phase === 'accueil' || etat.phase === 'tutoriel') { agir({ type: 'phase', phase: 'planification' }); continue; }
      if (etat.phase === 'comite') {
        (etat.comite && etat.comite.questions || []).forEach(function (qq) {
          if (etat.comite.reponses && etat.comite.reponses[qq.id] != null) return;
          var rep = s.comite ? s.comite(etat, qq) : (qq.reponses && qq.reponses[0] && qq.reponses[0].id);
          agir({ type: 'comite', question: qq.id, reponse: rep });
        });
        var r = agir({ type: 'phase', phase: 'planification' });
        if (r.erreur) break;
        continue;
      }
      if (etat.phase === 'planification') {
        regler();
        agir({ type: 'phase', phase: 'execution' });
        continue;
      }
      if (etat.evenementEnAttente) {
        var def = q.defEvenement(etat.evenementEnAttente.id);
        var choix = s.choix ? s.choix(etat, def) : (def && def.choix && def.choix[0] && def.choix[0].id);
        var rc = agir({ type: 'choix', choix: choix });
        if (rc.erreur) { etat = O.cloner(etat); etat.evenementEnAttente = null; }
        continue;
      }
      if ((etat.jour || 0) === 0) regler();
      if (etat.phase !== 'execution') continue;
      if (!propre) { etat = O.cloner(etat); propre = true; }
      var res = avancerJour(etat);
      journal.push.apply(journal, res.journal);
      if (res.finSemaine) semainesJouees += 1;
    }
    return { etat: etat, journal: journal };
  }

  LTP.moteur = {
    ORDRE: ORDRE, ORDRE_INIT: ORDRE_INIT,
    creer: creer, jour: jour, semaine: semaine, action: action, simuler: simuler,
    creerContexte: creerContexte, ecrireLevier: ecrireLevier, calculerSemaine: calculerSemaine,
  };

})(globalThis.LTP = globalThis.LTP || {});
