/* 30-metriques.js */
/* BOUCHON de la phase 0 — à remplacer par l'agent C. Calcul de etat.metriques, historique,
   visibilité des métriques cachées (CONCEPTION §5, 18-donnees-metriques.js). */
(function (LTP) {
  'use strict';
  LTP.K.metriques = {};
  LTP.systemes.metriques = {
    initialiser: function (etat) { LTP.systemes.metriques.semaine(etat); },
    semaine: function (etat, ctx) {
      var q = LTP.q, O = LTP.outils, ilots = q.ilots(etat), devs = q.devsPresents(etat);
      var m = {
        prFusionnees: O.somme(ilots, function (i) { return i.derniere.prFusionnees; }),
        pointsLivres: O.somme(ilots, function (i) { return i.derniere.pointsLivres; }),
        backlog: O.somme(ilots, 'backlog'),
        satisfaction: O.moyenne(ilots, 'satisfaction'),
        incidents: q.incidentsActifs(etat).length,
        volumeCode: O.somme(ilots, function (i) { return i.codebase.volume; }),
        jetonsSemaine: etat.org.jetonsSemaine,
        coutSemaine: etat.org.coutSemaine,
        budgetRestant: etat.org.budget,
        moral: O.moyenne(devs, 'moral'),
        fatigue: O.moyenne(devs, 'fatigue'),
        comprehension: O.moyenne(devs, 'comprehension'),
        confiance: etat.org.confiance,
        releve: etat.org.releve,
      };
      etat.metriques = m;
      for (var k in m) {
        etat.historique[k] = etat.historique[k] || [];
        etat.historique[k][etat.semaine] = m[k];
      }
    },
    // Une métrique cachée devient visible selon les leviers (voir CONCEPTION §5).
    visible: function (etat, id) {
      var d = LTP.donnees.metriquesParId[id];
      return !!d && !d.cachee;
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
