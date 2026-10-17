/* 18-donnees-metriques.js */
/* Définition des métriques du tableau de bord : contrat entre 30-metriques.js (qui calcule
   etat.metriques et etat.historique) et 44-ui-tableau-bord.js (qui les affiche).
   - unite : nb · semaines · h · $ · jetons · kLOC · ratio (0–1, affiché en %) · score (0–100)
   - sens : 'haut' si une valeur haute est bonne, 'bas' si une valeur basse est bonne, 'neutre' (piège)
   - cachee : invisible tant que la condition de visibilite (calculée par 30-metriques) n'est pas remplie
   - indice : ce que Ginette dit quand on survole la métrique cachée */
(function (LTP) {
  'use strict';

  LTP.donnees.groupesMetriques = [
    { id: 'livraison', libelle: 'Livraison' },
    { id: 'qualite',   libelle: 'Qualité' },
    { id: 'couts',     libelle: 'Coûts' },
    { id: 'risques',   libelle: 'Risques' },
    { id: 'personnes', libelle: 'Personnes' },
  ];

  LTP.donnees.metriques = [
    { id: 'prFusionnees',       groupe: 'livraison', unite: 'nb',       sens: 'haut',   libelle: 'PR fusionnées par semaine' },
    { id: 'pointsLivres',       groupe: 'livraison', unite: 'nb',       sens: 'haut',   libelle: 'Valeur livrée (points)' },
    { id: 'delai',              groupe: 'livraison', unite: 'semaines', sens: 'bas',    libelle: 'Délai de livraison' },
    { id: 'frequenceDeploiement', groupe: 'livraison', unite: 'nb',     sens: 'haut',   libelle: 'Déploiements par semaine', cachee: true,
      indice: 'Sans pipeline de déploiement, personne ne compte les mises en production.' },
    { id: 'backlog',            groupe: 'livraison', unite: 'nb',       sens: 'bas',    libelle: 'Demandes en attente (points)' },
    { id: 'satisfaction',       groupe: 'livraison', unite: 'ratio',    sens: 'haut',   libelle: 'Satisfaction des directions' },

    { id: 'bogues',             groupe: 'qualite',   unite: 'nb',       sens: 'bas',    libelle: 'Bogues en production' },
    { id: 'incidents',          groupe: 'qualite',   unite: 'nb',       sens: 'bas',    libelle: 'Incidents' },
    { id: 'tauxEchec',          groupe: 'qualite',   unite: 'ratio',    sens: 'bas',    libelle: 'Taux d\'échec des changements', cachee: true,
      indice: 'Il faut une CI qui teste pour savoir quels changements cassent.' },
    { id: 'mttr',               groupe: 'qualite',   unite: 'h',        sens: 'bas',    libelle: 'Temps moyen de rétablissement' },
    { id: 'volumeCode',         groupe: 'qualite',   unite: 'kLOC',     sens: 'neutre', libelle: 'Volume de code' },
    { id: 'couvertureApparente', groupe: 'qualite',  unite: 'ratio',    sens: 'haut',   libelle: 'Couverture de tests (affichée)' },
    { id: 'couvertureReelle',   groupe: 'qualite',   unite: 'ratio',    sens: 'haut',   libelle: 'Couverture réelle', cachee: true,
      indice: 'Des tests de mutation, et on saurait lesquels vérifient vraiment quelque chose.' },
    { id: 'comprehension',      groupe: 'qualite',   unite: 'ratio',    sens: 'haut',   libelle: 'Compréhension du code', cachee: true,
      indice: 'Demande-leur d\'expliquer leur code sans l\'IA. Tu verrais.' },

    { id: 'jetonsSemaine',      groupe: 'couts',     unite: 'jetons',   sens: 'bas',    libelle: 'Jetons consommés par semaine' },
    { id: 'coutSemaine',        groupe: 'couts',     unite: '$',        sens: 'bas',    libelle: 'Coût de l\'IA et des outils par semaine' },
    { id: 'budgetRestant',      groupe: 'couts',     unite: '$',        sens: 'haut',   libelle: 'Budget restant' },
    { id: 'retard',             groupe: 'couts',     unite: 'ratio',    sens: 'bas',    libelle: 'Projets en retard' },

    { id: 'surfaceAttaque',     groupe: 'risques',   unite: 'score',    sens: 'bas',    libelle: 'Surface d\'attaque', cachee: true,
      indice: 'Sans passerelle ni propriétaires, on ne sait même pas ce qui est installé.' },
    { id: 'expositionLoi25',    groupe: 'risques',   unite: 'score',    sens: 'bas',    libelle: 'Exposition Loi 25', cachee: true,
      indice: 'On ne peut pas protéger des données qu\'on n\'a pas classées. Me Bérubé le répète assez.' },
    { id: 'applisFantomes',     groupe: 'risques',   unite: 'nb',       sens: 'bas',    libelle: 'Applications hors TI connues' },
    { id: 'cartesOrphelines',   groupe: 'risques',   unite: 'nb',       sens: 'bas',    libelle: 'Skills et MCP sans propriétaire', cachee: true,
      indice: 'Exige un propriétaire pour chaque carte du registre, et tu sauras.' },

    { id: 'moral',              groupe: 'personnes', unite: 'ratio',    sens: 'haut',   libelle: 'Moral moyen' },
    { id: 'fatigue',            groupe: 'personnes', unite: 'ratio',    sens: 'bas',    libelle: 'Fatigue moyenne' },
    { id: 'releve',             groupe: 'personnes', unite: 'ratio',    sens: 'haut',   libelle: 'Relève' },
    { id: 'connaissanceCapturee', groupe: 'personnes', unite: 'ratio',  sens: 'haut',   libelle: 'Connaissance capturée' },
    { id: 'confiance',          groupe: 'personnes', unite: 'score',    sens: 'haut',   libelle: 'Confiance de la direction' },
  ];

  LTP.donnees.metriquesParId = {};
  LTP.donnees.metriques.forEach(function (m) { LTP.donnees.metriquesParId[m.id] = m; });
})(globalThis.LTP = globalThis.LTP || {});
