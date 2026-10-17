/* en/18-donnees-metriques.js */
/* Édition anglaise : groupes et métriques du tableau de bord (libellés fixés par le glossaire). Les indices sont
   dits par Ginette quand on survole une métrique cachée. */
(function (LTP) {
  'use strict';

  LTP.traduire('groupesMetriques', {
    livraison: { libelle: 'Delivery' },
    qualite:   { libelle: 'Quality' },
    couts:     { libelle: 'Costs' },
    risques:   { libelle: 'Risks' },
    personnes: { libelle: 'People' },
  });

  LTP.traduire('metriques', {
    prFusionnees:         { libelle: 'PRs merged per week' },
    pointsLivres:         { libelle: 'Value delivered (points)' },
    delai:                { libelle: 'Lead time' },
    frequenceDeploiement: { libelle: 'Deployments per week',
      indice: 'Without a deployment pipeline, nobody keeps count of production releases.' },
    backlog:              { libelle: 'Request backlog (points)' },
    satisfaction:         { libelle: 'Business satisfaction' },

    bogues:               { libelle: 'Production bugs' },
    incidents:            { libelle: 'Incidents' },
    tauxEchec:            { libelle: 'Change failure rate',
      indice: 'You need a CI that runs tests to know which changes break things.' },
    mttr:                 { libelle: 'Mean time to recovery' },
    volumeCode:           { libelle: 'Code volume' },
    couvertureApparente:  { libelle: 'Test coverage (reported)' },
    couvertureReelle:     { libelle: 'Actual coverage',
      indice: 'Add mutation testing, and you\'d know which tests actually check something.' },
    comprehension:        { libelle: 'Code comprehension',
      indice: 'Ask them to explain their code without the AI. You\'ll see.' },

    jetonsSemaine:        { libelle: 'Tokens used per week' },
    coutSemaine:          { libelle: 'AI and tooling cost per week' },
    budgetRestant:        { libelle: 'Remaining budget' },
    retard:               { libelle: 'Late projects' },

    surfaceAttaque:       { libelle: 'Attack surface',
      indice: 'With no gateway and no owners, nobody even knows what\'s installed.' },
    expositionLoi25:      { libelle: 'Law 25 exposure',
      indice: 'You can\'t protect data you haven\'t classified. Counsel Bérubé says so often enough.' },
    applisFantomes:       { libelle: 'Known apps built outside IT' },
    cartesOrphelines:     { libelle: 'Ownerless skills and MCPs',
      indice: 'Require an owner for every card in the registry, and you\'ll know.' },

    moral:                { libelle: 'Average morale' },
    fatigue:              { libelle: 'Average fatigue' },
    releve:               { libelle: 'Succession' },
    connaissanceCapturee: { libelle: 'Captured knowledge' },
    confiance:            { libelle: 'Executive confidence' },
  });
})(globalThis.LTP = globalThis.LTP || {});
