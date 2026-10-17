/* 12-donnees-leviers.js */
/* Tous les leviers du joueur : contrat entre le moteur (qui lit les valeurs avec LTP.q.levier)
   et l'interface (qui génère les panneaux à partir de cette table).
   - panneau : onglet de l'interface
   - semaine (et trimestre, qui en découle) : tirés du calendrier (08-donnees-calendrier.js), source unique des déblocages ;
     une valeur peut avoir sa propre semaine (valeurs[].semaine) quand elle arrive plus tard que le levier
   - surchargeable : le levier peut prendre une valeur différente par îlot (◆ dans la conception)
   - delai : semaines avant que le changement prenne effet (0 : immédiat)
   - valeurs[].v : valeur lue par le moteur (nombre, texte ou booléen) */
(function (LTP) {
  'use strict';

  LTP.donnees.panneaux = [
    { id: 'ia',           libelle: 'Utilisation de l\'IA' },
    { id: 'pratiques',    libelle: 'Pratiques' },
    { id: 'produit',      libelle: 'Produit' },
    { id: 'registre',     libelle: 'Registre' },
    { id: 'gouvernance',  libelle: 'Données et sécurité' },
    { id: 'organisation', libelle: 'Organisation' },
  ];

  function L(def) { return def; }

  LTP.donnees.leviers = [
    // --- Pratiques d'ingénierie (T1) ---
    L({ id: 'pair', panneau: 'pratiques', surchargeable: false, delai: 0, defaut: 'aucun',
      libelle: 'Programmation en binôme',
      aide: 'Deux têtes sur le même code : plus de compréhension, moins de débit brut.',
      valeurs: [
        { v: 'aucun',    libelle: 'Aucun' },
        { v: 'humain',   libelle: 'Humain avec humain' },
        { v: 'ia',       libelle: 'Humain avec l\'IA' },
        { v: 'rotation', libelle: 'Rotation (une semaine sur deux)' },
      ] }),
    L({ id: 'reviseurs', panneau: 'pratiques', surchargeable: true, delai: 0, defaut: 1,
      libelle: 'Réviseurs par PR',
      aide: 'Plus de regards attrapent plus de défauts, mais la file de revue s\'allonge.',
      valeurs: [{ v: 0, libelle: 'Aucun' }, { v: 1, libelle: 'Un' }, { v: 2, libelle: 'Deux' }] }),
    L({ id: 'approbation', panneau: 'pratiques', surchargeable: false, delai: 0, defaut: 'humain',
      libelle: 'Approbation des PR',
      aide: 'Qui a le dernier mot avant la fusion.',
      valeurs: [
        { v: 'auto',         libelle: 'Automatique (revue par l\'IA)' },
        { v: 'humain',       libelle: 'Un humain' },
        { v: 'proprietaire', libelle: 'Le propriétaire du code' },
      ] }),
    L({ id: 'taillePR', panneau: 'pratiques', surchargeable: false, delai: 0, defaut: 'libre',
      libelle: 'Taille maximale des PR',
      aide: 'Une petite PR se relit vraiment. Une grosse se survole.',
      valeurs: [{ v: 'libre', libelle: 'Libre' }, { v: 400, libelle: '400 lignes' }, { v: 200, libelle: '200 lignes' }] }),
    L({ id: 'ci', panneau: 'pratiques', surchargeable: true, delai: 0, defaut: 'build',
      libelle: 'Intégration et déploiement continus',
      aide: 'Chaque marche du pipeline attrape des défauts plus tôt, et coûte un peu de temps.',
      valeurs: [
        { v: 'aucune',  libelle: 'Aucune' },
        { v: 'build',   libelle: 'Compilation' },
        { v: 'tests',   libelle: 'Compilation et tests' },
        { v: 'complet', libelle: 'Pipeline complet et déploiement continu' },
      ] }),
    L({ id: 'lint', panneau: 'pratiques', surchargeable: false, delai: 0, defaut: 'formatage',
      libelle: 'Lint et formatage',
      aide: 'Moins de bruit en revue. Verrouillé : l\'agent ne peut pas désactiver une règle pour passer la CI.',
      valeurs: [
        { v: 'aucun',      libelle: 'Aucun' },
        { v: 'formatage',  libelle: 'Formatage automatique' },
        { v: 'strict',     libelle: 'Lint strict' },
        { v: 'verrouille', libelle: 'Lint strict et verrouillé' },
      ] }),
    L({ id: 'tests', panneau: 'pratiques', surchargeable: true, delai: 0, defaut: 'unitaires',
      libelle: 'Tests',
      aide: 'Les tests de mutation démasquent les tests qui ne vérifient rien.',
      valeurs: [
        { v: 'aucun',       libelle: 'Aucun' },
        { v: 'unitaires',   libelle: 'Unitaires' },
        { v: 'integration', libelle: 'Unitaires et intégration' },
        { v: 'mutation',    libelle: 'Avec tests de mutation' },
      ] }),
    L({ id: 'revuesExplication', panneau: 'pratiques', surchargeable: false, delai: 0, defaut: 'non',
      libelle: 'Revues d\'explication',
      aide: 'La personne explique son code sans l\'IA. Mesure et restaure la compréhension.',
      valeurs: [
        { v: 'non',       libelle: 'Non' },
        { v: 'mensuelles', libelle: 'Mensuelles' },
        { v: 'majeures',  libelle: 'À chaque PR majeure' },
      ] }),

    // --- Utilisation de l'IA ---
    L({ id: 'licences', panneau: 'ia', surchargeable: false, delai: 0, defaut: 'pilote',
      libelle: 'Licences Copilot',
      aide: 'Des licences GitHub Copilot de base. Sans licence, on code comme avant : pas d\'assistant, pas de jetons.',
      valeurs: [
        { v: 'aucune', libelle: 'Aucune' },
        { v: 'pilote', libelle: 'Groupe pilote (7)' },
        { v: 'tous',   libelle: 'Toute l\'équipe (21)' },
      ] }),
    L({ id: 'autonomie', panneau: 'ia', surchargeable: true, delai: 0, defaut: 0,
      libelle: 'Niveau d\'autonomie des agents',
      aide: 'Plus d\'autonomie, plus de débit. Moins de regards humains, moins de compréhension.',
      valeurs: [
        { v: 0, libelle: '0 · Autocomplétion' },
        { v: 1, libelle: '1 · Clavardage' },
        { v: 2, libelle: '2 · Agent avec approbation à chaque étape' },
        { v: 3, libelle: '3 · Agent qui ouvre des PR' },
        { v: 4, libelle: '4 · Agent qui fusionne seul' },
      ] }),
    L({ id: 'sessionsMax', panneau: 'ia', surchargeable: false, delai: 0, defaut: 1,
      libelle: 'Sessions d\'agent par personne',
      aide: 'Le multi-Clauding : plus de débit, en rendement décroissant, et plus de fatigue.',
      valeurs: [1, 2, 3, 4, 5, 6].map(function (n) { return { v: n, libelle: n === 1 ? '1 session' : n + ' sessions' }; }) }),
    L({ id: 'modele', panneau: 'ia', surchargeable: false, delai: 0, defaut: 'intermediaire',
      libelle: 'Modèle par défaut',
      aide: 'Un bon modèle coûte plus cher par jeton, mais se trompe moins. Le modèle local garde les données à l\'interne.',
      valeurs: [
        { v: 'economique',    libelle: 'Économique' },
        { v: 'intermediaire', libelle: 'Intermédiaire' },
        { v: 'haut',          libelle: 'Haut de gamme' },
        { v: 'local',         libelle: 'Local (sur le serveur GPU)' },
      ] }),
    L({ id: 'modeleParTache', panneau: 'ia', surchargeable: false, delai: 0, defaut: false,
      libelle: 'Modèle choisi selon la tâche',
      aide: 'Un petit modèle pour le lint et la documentation, un gros pour l\'architecture.',
      valeurs: [{ v: false, libelle: 'Non' }, { v: true, libelle: 'Oui' }] }),
    L({ id: 'formation', panneau: 'ia', surchargeable: false, delai: 4, defaut: 'aucune',
      libelle: 'Programme de formation',
      aide: 'Coûte du temps et de l\'argent tout de suite, rapporte dans un mois.',
      valeurs: [
        { v: 'aucune',    libelle: 'Aucune' },
        { v: 'atelier',   libelle: 'Un atelier' },
        { v: 'programme', libelle: 'Programme complet' },
        { v: 'mentorat',  libelle: 'Programme et mentorat' },
      ] }),
    L({ id: 'plafondJetons', panneau: 'ia', surchargeable: true, delai: 0, defaut: 'aucun',
      libelle: 'Plafond de jetons',
      aide: 'Coupe les boucles d\'agent. Trop strict, il bloque le travail.',
      valeurs: [{ v: 'aucun', libelle: 'Aucun' }, { v: 'souple', libelle: 'Souple (alerte)' }, { v: 'strict', libelle: 'Strict (coupure)' }] }),

    // --- Produit ---
    L({ id: 'disciplineNon', panneau: 'produit', surchargeable: true, delai: 0, defaut: 'normale',
      libelle: 'Discipline du « non »',
      aide: 'Quand coder devient « pas cher », les demandes se multiplient. Savoir dire non évite le gonflement fonctionnel.',
      valeurs: [{ v: 'laxiste', libelle: 'Laxiste' }, { v: 'normale', libelle: 'Normale' }, { v: 'ferme', libelle: 'Ferme' }] }),
    L({ id: 'specs', panneau: 'produit', surchargeable: false, delai: 0, defaut: 'non',
      libelle: 'Specs avant le code',
      aide: 'Un agent fait exactement ce qu\'on lui demande. Encore faut-il savoir quoi demander.',
      valeurs: [{ v: 'non', libelle: 'Non' }, { v: 'legeres', libelle: 'Légères' }, { v: 'obligatoires', libelle: 'Obligatoires' }] }),

    // --- Données et sécurité (T3) ---
    L({ id: 'catalogue', panneau: 'gouvernance', surchargeable: false, delai: 4, defaut: 'aucun',
      libelle: 'Catalogue de données',
      aide: 'Savoir quelle source est la bonne. Prérequis du MCP du catalogue.',
      valeurs: [{ v: 'aucun', libelle: 'Aucun' }, { v: 'inventaire', libelle: 'Inventaire' }, { v: 'sources', libelle: 'Catalogue avec sources officielles' }] }),
    L({ id: 'classification', panneau: 'gouvernance', surchargeable: false, delai: 2, defaut: 'non',
      libelle: 'Classification des données sensibles',
      aide: 'On ne protège bien que ce qu\'on a étiqueté.',
      valeurs: [{ v: 'non', libelle: 'Non' }, { v: 'manuelle', libelle: 'Manuelle' }, { v: 'auto', libelle: 'Automatique' }] }),
    L({ id: 'bacASable', panneau: 'gouvernance', surchargeable: true, delai: 0, defaut: 'aucun',
      libelle: 'Bac à sable des agents',
      aide: 'L\'agent travaille dans une boîte : un paquet malveillant n\'atteint pas le reste du réseau. Avec le filtrage, une injection ne peut plus envoyer de données n\'importe où.',
      valeurs: [{ v: 'aucun', libelle: 'Aucun' }, { v: 'conteneur', libelle: 'Conteneur' }, { v: 'filtre', libelle: 'Conteneur et réseau filtré' }] }),
    L({ id: 'accesProd', panneau: 'gouvernance', surchargeable: true, delai: 0, defaut: 'oui',
      libelle: 'Accès des agents à la production',
      aide: 'Diagnostiquer vite, ou ne rien casser.',
      valeurs: [{ v: 'oui', libelle: 'Lecture et écriture' }, { v: 'lecture', libelle: 'Lecture seule' }, { v: 'non', libelle: 'Aucun' }] }),
    L({ id: 'approbationSorties', panneau: 'gouvernance', surchargeable: true, delai: 0, defaut: false,
      libelle: 'Approbation humaine des sorties externes',
      aide: 'Courriels, requêtes web, billets externes : un humain confirme. Casse la troisième branche de la triade.',
      valeurs: [{ v: false, libelle: 'Non' }, { v: true, libelle: 'Oui' }] }),
    L({ id: 'politiqueRegistre', panneau: 'gouvernance', surchargeable: false, delai: 0, defaut: 'libre',
      libelle: 'Politique du registre',
      aide: 'Qui peut installer quoi, et qui en répond.',
      valeurs: [
        { v: 'libre',        libelle: 'Libre-service' },
        { v: 'listeBlanche', libelle: 'Liste blanche' },
        { v: 'proprietaires', libelle: 'Liste blanche et propriétaire obligatoire' },
      ] }),
    L({ id: 'passerelleMCP', panneau: 'gouvernance', surchargeable: false, delai: 2, defaut: false,
      libelle: 'Passerelle MCP',
      aide: 'Tous les appels d\'outils passent par un point journalisé, en lecture seule par défaut. Un changement de description y est détecté.',
      valeurs: [{ v: false, libelle: 'Non' }, { v: true, libelle: 'Oui' }] }),
    L({ id: 'politiqueNonTI', panneau: 'gouvernance', surchargeable: false, delai: 0, defaut: 'ignorer',
      libelle: 'Développement hors TI',
      aide: 'Kevin code. La question est de savoir avec quels garde-fous.',
      valeurs: [
        { v: 'ignorer',  libelle: 'Ignorer' },
        { v: 'interdire', libelle: 'Interdire' },
        { v: 'encadrer', libelle: 'Encadrer (registre approuvé et accompagnement)' },
      ] }),

    // --- Organisation (T4) ---
    L({ id: 'observabilite', panneau: 'organisation', surchargeable: false, delai: 3, defaut: 'aucune',
      libelle: 'Observabilité',
      aide: 'Ce qu\'on ne mesure pas, on ne le voit pas.',
      valeurs: [{ v: 'aucune', libelle: 'Aucune' }, { v: 'base', libelle: 'De base' }, { v: 'complete', libelle: 'Complète' }] }),
    L({ id: 'embauche', panneau: 'organisation', surchargeable: false, delai: 0, defaut: 'normale',
      libelle: 'Embauche de juniors',
      aide: 'Si personne n\'apprend aujourd\'hui, qui sera senior dans cinq ans ? Le programme de relève forme les juniors en place et prépare la passation des seniors : il n\'ajoute personne à l\'équipe cette année.',
      valeurs: [{ v: 'gel', libelle: 'Gel' }, { v: 'normale', libelle: 'Normale' }, { v: 'releve', libelle: 'Programme de relève' }] }),
    L({ id: 'objectif', panneau: 'organisation', surchargeable: false, delai: 0, defaut: 'aucun',
      libelle: 'Objectif de la direction',
      aide: 'Quand une mesure devient une cible, elle cesse d\'être une bonne mesure.',
      valeurs: [
        { v: 'aucun',        libelle: 'Aucun' },
        { v: 'pr',           libelle: 'Nombre de PR' },
        { v: 'delai',        libelle: 'Délai de livraison' },
        { v: 'satisfaction', libelle: 'Satisfaction' },
        { v: 'incidents',    libelle: 'Nombre d\'incidents' },
      ] }),
  ];

  LTP.donnees.leviersParId = {};
  LTP.donnees.leviers.forEach(function (l) { LTP.donnees.leviersParId[l.id] = l; });

  // --- Semaines de déblocage, tirées du calendrier ---
  // Un levier absent du calendrier est disponible dès la semaine 1. Le panneau ouvre avec son premier levier
  // (ou à la semaine où le calendrier le nomme dans panneaux).
  function trimestreDe(semaine) { return Math.max(1, Math.ceil(semaine / 13)); }
  LTP.donnees.trimestreDe = trimestreDe;
  var semaineLevier = {}, semaineValeur = {}, semainePanneau = {};
  (LTP.donnees.calendrier || []).forEach(function (c) {
    (c.leviers || []).forEach(function (id) { if (semaineLevier[id] == null) semaineLevier[id] = c.semaine; });
    (c.valeurs || []).forEach(function (x) {
      (x.v || []).forEach(function (v) { semaineValeur[x.levier + '|' + String(v)] = c.semaine; });
    });
    (c.panneaux || []).forEach(function (id) { if (semainePanneau[id] == null) semainePanneau[id] = c.semaine; });
  });
  LTP.donnees.leviers.forEach(function (l) {
    l.semaine = semaineLevier[l.id] || 1;
    l.trimestre = trimestreDe(l.semaine);
    l.valeurs.forEach(function (x) {
      var sv = semaineValeur[l.id + '|' + String(x.v)];
      x.semaine = Math.max(l.semaine, sv || l.semaine);
    });
  });
  LTP.donnees.panneaux.forEach(function (p) {
    var s = semainePanneau[p.id];
    LTP.donnees.leviers.forEach(function (l) { if (l.panneau === p.id && (s == null || l.semaine < s)) s = l.semaine; });
    p.semaine = s || 1;
    p.trimestre = trimestreDe(p.semaine);
  });
})(globalThis.LTP = globalThis.LTP || {});
