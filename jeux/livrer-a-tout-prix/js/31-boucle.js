/* 31-boucle.js */
/* La boucle du moteur : semaine(), action(), trimestres et fin de partie.
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
  var ORDRE_INIT = ['registre', 'production', 'qualite', 'dette', 'couts', 'donneesSecu', 'demandes', 'personnes', 'evenements', 'metriques', 'fin'];

  function creerContexte(etat) {
    var journal = [];
    var ctx = {
      rng: LTP.rng.creer(etat),
      K: LTP.K,
      q: q,
      journal: journal,
      // type : 'info' | 'alerte' | 'incident' | 'evenement' | 'levier' | 'personne' | 'comite' ; cle : clé de texte
      noter: function (type, cle, vars, ref) {
        var e = { semaine: etat.semaine, type: type, cle: cle, vars: vars || {} };
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
        ctx.noter('levier', 'levier.effectif', { levier: a.id, valeur: a.valeur }, { ilot: a.ilot });
      } else restants.push(a);
    });
    etat.leviers.enAttente = restants;
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

  // Avance d'une semaine. Refuse d'avancer hors de la phase d'exécution ou si un événement attend une décision.
  function semaine(etatEntree) {
    if (etatEntree.phase !== 'execution' || etatEntree.evenementEnAttente) return { etat: etatEntree, journal: [], avance: false };
    var etat = O.cloner(etatEntree);
    etat.semaine += 1;
    var ctx = creerContexte(etat);

    LTP.q.ilots(etat).forEach(function (i) { i.derniere = LTP.etat.derniereIlotVide(); });
    LTP.q.devs(etat).forEach(function (d) { d.derniere = LTP.etat.derniereDevVide(); });
    appliquerLeviersEnAttente(etat, ctx);

    for (var i = 0; i < ORDRE.length; i++) {
      var s = LTP.systemes[ORDRE[i][0]];
      var f = s && s[ORDRE[i][1]];
      if (typeof f === 'function') f.call(s, etat, ctx);
    }

    finDeSemaine(etat, ctx);
    archiverJournal(etat, ctx.journal);
    return { etat: etat, journal: ctx.journal, avance: true };
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
    var erreur = null;
    try {
      erreur = appliquerAction(etat, a, ctx);
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
        if (a.ilot && !d.surchargeable) return 'levier-non-surchargeable';
        if (a.ilot && !etat.ilots[a.ilot]) return 'ilot-inconnu';
        if (etat.phase === 'fin') return 'partie-terminee';
        var ilot = a.ilot || null;
        etat.leviers.enAttente = etat.leviers.enAttente.filter(function (x) { return !(x.id === a.id && (x.ilot || null) === ilot); });
        if (d.delai > 0 && q.levier(etat, a.id, ilot) !== a.valeur) {
          etat.leviers.enAttente.push({ id: a.id, valeur: a.valeur, ilot: ilot, semaine: etat.semaine + d.delai });
          ctx.noter('levier', 'levier.programme', { levier: a.id, valeur: a.valeur, delai: d.delai }, { ilot: ilot });
        } else {
          ecrireLevier(etat, a.id, a.valeur, ilot);
          ctx.noter('levier', 'levier.change', { levier: a.id, valeur: a.valeur }, { ilot: ilot });
        }
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
  }

  // Joue n semaines en résolvant tout seul les décisions. strategie (facultative) :
  //   { leviers(etat) → [actions], choix(etat, evenement) → idChoix, comite(etat, question) → idReponse }
  // Appelé au début de chaque trimestre pour les leviers. Sert aux tests, à l'équilibrage et au mode animateur.
  function simuler(etatEntree, n, strategie) {
    var etat = etatEntree, s = strategie || {};
    var journal = [];
    var garde = 0;
    function agir(a) {
      var r = action(etat, a);
      if (!r.erreur) { etat = r.etat; journal = journal.concat(r.journal); }
      return r;
    }
    var semainesJouees = 0;
    var trimestreLeviers = -1;
    while (semainesJouees < n && etat.phase !== 'fin' && garde++ < n * 20 + 50) {
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
        if (s.leviers && trimestreLeviers !== etat.trimestre) {
          trimestreLeviers = etat.trimestre;
          (s.leviers(etat) || []).forEach(agir);
        }
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
      var res = semaine(etat);
      etat = res.etat;
      journal = journal.concat(res.journal);
      semainesJouees += 1;
    }
    return { etat: etat, journal: journal };
  }

  LTP.moteur = {
    ORDRE: ORDRE, ORDRE_INIT: ORDRE_INIT,
    creer: creer, semaine: semaine, action: action, simuler: simuler,
    creerContexte: creerContexte, ecrireLevier: ecrireLevier,
  };

  LTP.enregistrerTextes({
    'levier.change': 'Levier modifié : {levier}.',
    'levier.programme': 'Changement programmé : {levier}, effectif dans {delai} semaines.',
    'levier.effectif': 'Changement en vigueur : {levier}.',
    'comite.convoque': 'Fin du trimestre {trimestre} : le comité de direction vous attend.',
    'trimestre.debut': 'Début du trimestre {trimestre}.',
    'partie.fin': 'Fin de la partie.',
  });
})(globalThis.LTP = globalThis.LTP || {});
