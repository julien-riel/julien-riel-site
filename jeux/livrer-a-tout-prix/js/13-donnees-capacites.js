/* 13-donnees-capacites.js */
/* Skills et serveurs MCP du Registre (CONCEPTION §4.9). Contenu écrit par l'agent D2.
   Forme d'une carte :
   {
     id, type: 'skill' | 'mcp', nom, provenance: 'maison' | 'fournisseur' | 'communaute' | 'personnelle',
     trimestre,                       // trimestre où la carte apparaît dans le registre
     disponible: 'depart' | 'evenement' | 'developpement',
                                      // evenement : découverte par un effet { carte, op: 'decouvrir' }
                                      // developpement : doit être développée (coutDev semaines d'un développeur)
     auteur: devId | null,            // personnelle ou maison : l'auteur est le propriétaire par défaut
     coutDev: 0,                      // semaines de travail avant disponibilité (cartes maison)
     ilotsSuggeres: [ilotId],
     description,                     // une ou deux phrases, ton du jeu
     gains: { debit, tauxDefaut, jetons, mttr, comprehension, adequationPile, couverture, verbosite },
                                      // mêmes clés que ilot.mods (20-etat.js) : additifs pour debit, comprehension,
                                      // adequationPile, couverture ; multiplicatifs pour les autres
     ilotsGains: { [ilotId]: { …gains } },   // facultatif : gains propres à certains îlots (remplacent les clés de gains)
     poidsContexte: 0.05,
     branches: ['prive', 'nonFiable', 'sortie'],
     risque: { type, proba },         // type : exfiltration · injection · destruction · abandon ·
                                      //   changementSilencieux · tarif · typosquat · perime · aucun
     declencheur: null | 'tests' | 'deploy' | …,   // deux skills au même déclencheur dans un îlot = conflit
     prerequis: { levier, valeurs: [...] } | null,
   }
   Rappels de conception :
   - Skills dès le T2, MCP dès le T3.
   - Conflits voulus : tdd-strict, ship-it et tests-instantanes partagent le déclencheur « tests » ;
     les trois deploy-* partagent « deploy » (doublons du grand ménage).
   - Triade létale : une carte allume les branches qu'elle apporte ; l'îlot ajoute les siennes (10-donnees-ilots.js). */
(function (LTP) {
  'use strict';

  var TOUS = ['appro', 'operations', 'bureautique', 'inspections', 'iot', 'web', 'portail'];

  LTP.donnees.capacites = [
    // =====================================================================
    // SKILLS — maison
    // =====================================================================
    { id: 'conventions-gpu', type: 'skill', nom: 'conventions-gpu', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: null, coutDev: 2, ilotsSuggeres: ['portail', 'web', 'inspections', 'operations'],
      description: 'Les conventions de code du GPU, écrites une fois pour tous les agents : nommage, gestion des erreurs, et pourquoi on ne met pas de mot de passe dans le code « juste pour tester ». Elle périme si personne ne la tient à jour.',
      gains: { tauxDefaut: 0.85, jetons: 0.92, verbosite: 0.92 }, poidsContexte: 0.04, branches: [],
      risque: { type: 'perime', proba: 0.01 }, declencheur: 'conventions', prerequis: null },

    { id: 'deploiement-nuit', type: 'skill', nom: 'deploiement-nuit', provenance: 'maison', trimestre: 2, disponible: 'evenement',
      auteur: null, coutDev: 0, ilotsSuggeres: ['operations'],
      description: 'La procédure de déploiement de nuit de Ginette : 27 ans d\'expérience en 212 lignes, quoi vérifier avant, quoi surveiller pendant, et comment revenir en arrière sans réveiller tout le monde. Ginette n\'écrit pas de code ; elle écrit ce que le code oublie.',
      gains: { comprehension: 0.005 }, ilotsGains: { operations: { mttr: 0.65, tauxDefaut: 0.9, comprehension: 0.01 } },
      poidsContexte: 0.04, branches: [], risque: { type: 'perime', proba: 0.005 }, declencheur: 'nuit', prerequis: null },

    { id: 'migration-as400', type: 'skill', nom: 'migration-as400', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: 'raymond', coutDev: 4, ilotsSuggeres: ['appro'],
      description: 'Raymond explique à l\'agent le RPG, les fichiers physiques et pourquoi les dates sont des nombres à sept chiffres (CYYMMDD). Ça coûte quatre semaines de Raymond, qui le fait en soupirant, mais qui le fait bien.',
      gains: { comprehension: 0.003 }, ilotsGains: { appro: { adequationPile: 0.2, tauxDefaut: 0.85, comprehension: 0.008 } },
      poidsContexte: 0.05, branches: [], risque: { type: 'perime', proba: 0.005 }, declencheur: 'as400', prerequis: null },

    { id: 'revue-securite', type: 'skill', nom: 'revue-securite', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: 'sebastien', coutDev: 3, ilotsSuggeres: ['portail', 'appro', 'bureautique', 'inspections'],
      description: 'La liste de contrôle de Sébastien devenue skill : injection SQL, secrets en clair, dépendances inventées, journaux trop bavards. L\'agent la relit avant chaque PR, ce qui coûte quelques jetons et épargne quelques nuits.',
      gains: { tauxDefaut: 0.88, jetons: 1.06 }, poidsContexte: 0.05, branches: [],
      risque: { type: 'perime', proba: 0.01 }, declencheur: 'securite', prerequis: null },

    { id: 'redaction-specs', type: 'skill', nom: 'redaction-specs', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: null, coutDev: 2, ilotsSuggeres: ['inspections', 'portail', 'appro'],
      description: 'Avant de coder, l\'agent pose les questions qui fâchent : pour qui, pourquoi, et qu\'arrive-t-il si le champ est vide ? Les directions d\'affaires trouvent ça long ; la reprise, elle, raccourcit.',
      gains: { tauxDefaut: 0.9, verbosite: 0.9, debit: -0.03 }, poidsContexte: 0.04, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'specs', prerequis: null },

    { id: 'tests-mutation', type: 'skill', nom: 'tests-mutation', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: null, coutDev: 2, ilotsSuggeres: ['portail', 'inspections', 'operations'],
      description: 'Fait muter le code (un « < » devient « <= ») et vérifie qu\'au moins un test s\'en aperçoit. Les tests qui ne remarquent rien sont réécrits. Ça coûte du calcul, et un peu d\'humilité.',
      gains: { couverture: 0.06, jetons: 1.1 }, poidsContexte: 0.04, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'mutation', prerequis: null },

    { id: 'documentation-vivante', type: 'skill', nom: 'documentation-vivante', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: null, coutDev: 2, ilotsSuggeres: TOUS,
      description: 'Chaque PR met à jour la documentation et le journal des décisions d\'architecture. Dans six mois, quelqu\'un saura enfin pourquoi le module s\'appelle « temporaire ».',
      gains: { comprehension: 0.01, jetons: 1.04, mttr: 0.92 }, poidsContexte: 0.03, branches: [],
      risque: { type: 'perime', proba: 0.005 }, declencheur: 'docs', prerequis: null },

    { id: 'explique-moi', type: 'skill', nom: 'explique-moi', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: null, coutDev: 1, ilotsSuggeres: ['inspections', 'web', 'operations'],
      description: 'L\'agent ne livre plus un bloc de code sans expliquer ses choix, et pose une question de compréhension avant d\'ouvrir la PR. Un peu plus lent ; beaucoup moins de « je ne sais pas, c\'est l\'IA qui l\'a fait ».',
      gains: { comprehension: 0.015, debit: -0.04 }, poidsContexte: 0.03, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'explication', prerequis: null },

    { id: 'macros-paie', type: 'skill', nom: 'macros-paie', provenance: 'maison', trimestre: 2, disponible: 'evenement',
      auteur: 'olivier', coutDev: 0, ilotsSuggeres: ['bureautique'],
      description: 'Tout ce qu\'Olivier sait des macros de la paie, écrit avant la fin de son contrat : la feuille cachée « NE PAS TOUCHER », l\'ordre des onglets et le bogue de 2014 qu\'il ne faut surtout pas corriger.',
      gains: { comprehension: 0.003 }, ilotsGains: { bureautique: { adequationPile: 0.15, mttr: 0.8, comprehension: 0.008 } },
      poidsContexte: 0.04, branches: [], risque: { type: 'perime', proba: 0.01 }, declencheur: 'vba', prerequis: null },

    { id: 'firmware-prudent', type: 'skill', nom: 'firmware-prudent', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: 'amira', coutDev: 3, ilotsSuggeres: ['iot'],
      description: 'Les règles d\'Amira pour le C embarqué : pas d\'allocation dynamique, pas de récursion, chaque tampon a une taille et chaque capteur peut mentir. L\'agent devient lent et méfiant, comme il se doit.',
      gains: { debit: -0.05 }, ilotsGains: { iot: { tauxDefaut: 0.7, adequationPile: 0.1, debit: -0.05 } },
      poidsContexte: 0.04, branches: [], risque: { type: 'aucun', proba: 0 }, declencheur: 'embarque', prerequis: null },

    // Trois skills « deploy » en doublon, une par îlot : elles se contredisent si on les équipe ensemble.
    { id: 'deploy-ops', type: 'skill', nom: 'deploy', provenance: 'maison', trimestre: 2, disponible: 'depart',
      auteur: 'marieeve', coutDev: 0, ilotsSuggeres: ['operations'],
      description: 'La skill « deploy » des Opérations, écrite par Marie-Ève : Ansible, vérification de santé, retour arrière. Elle redémarre la base de données « par précaution », ce qui a du sens… aux Opérations.',
      gains: { mttr: 0.95 }, ilotsGains: { operations: { mttr: 0.88, tauxDefaut: 0.95 } },
      poidsContexte: 0.03, branches: [], risque: { type: 'perime', proba: 0.01 }, declencheur: 'deploy', prerequis: null },

    { id: 'deploy-portail', type: 'skill', nom: 'deploy', provenance: 'maison', trimestre: 2, disponible: 'depart',
      auteur: 'mathieu', coutDev: 0, ilotsSuggeres: ['portail'],
      description: 'La skill « deploy » du Portail, rédigée par Mathieu un vendredi à 16 h 40 : déploiement bleu-vert, migrations de schéma en premier. Elle dit exactement le contraire de celle des Opérations sur l\'ordre des étapes.',
      gains: { mttr: 0.95 }, ilotsGains: { portail: { mttr: 0.88, debit: 0.03 } },
      poidsContexte: 0.03, branches: [], risque: { type: 'perime', proba: 0.01 }, declencheur: 'deploy', prerequis: null },

    { id: 'deploy-web', type: 'skill', nom: 'deploy', provenance: 'maison', trimestre: 2, disponible: 'depart',
      auteur: 'patrick', coutDev: 0, ilotsSuggeres: ['web'],
      description: 'La skill « deploy » des Sites web : un push sur la branche principale et une petite prière. Pour un microsite de campagne, c\'est suffisant. Ailleurs, beaucoup moins.',
      gains: { tauxDefaut: 1.05 }, ilotsGains: { web: { mttr: 0.9, debit: 0.04 } },
      poidsContexte: 0.02, branches: [], risque: { type: 'perime', proba: 0.01 }, declencheur: 'deploy', prerequis: null },

    // =====================================================================
    // SKILLS — personnelles
    // =====================================================================
    { id: 'tdd-strict', type: 'skill', nom: 'tdd-strict', provenance: 'personnelle', trimestre: 2, disponible: 'depart',
      auteur: 'sophie', coutDev: 0, ilotsSuggeres: ['inspections'],
      description: 'La skill de Sophie : l\'agent écrit d\'abord un test qui échoue, puis le code minimal pour le faire passer. Lent au début, implacable ensuite.',
      gains: { couverture: 0.08, tauxDefaut: 0.85, debit: -0.06, jetons: 1.08 }, poidsContexte: 0.04, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'tests', prerequis: null },

    { id: 'ship-it', type: 'skill', nom: 'ship-it', provenance: 'personnelle', trimestre: 2, disponible: 'depart',
      auteur: 'mathieu', coutDev: 0, ilotsSuggeres: ['portail'],
      description: 'La skill de Mathieu. Devant un test rouge, elle conseille de « l\'ajuster » ; devant le lint, de « régler ça plus tard ». Plus tard n\'arrive jamais.',
      gains: { debit: 0.12, tauxDefaut: 1.2, couverture: -0.03, verbosite: 1.1 }, poidsContexte: 0.03, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'tests', prerequis: null },

    { id: 'wordpress-sauveteur', type: 'skill', nom: 'wordpress-sauveteur', provenance: 'personnelle', trimestre: 2, disponible: 'depart',
      auteur: 'patrick', coutDev: 0, ilotsSuggeres: ['web'],
      description: 'Les recettes de Patrick pour ses 14 sites WordPress : quelle extension ne pas mettre à jour, laquelle ne jamais désactiver, et dans quel ordre prier. Elle vieillit aussi vite que les extensions.',
      gains: {}, ilotsGains: { web: { debit: 0.08, mttr: 0.85 } }, poidsContexte: 0.03, branches: [],
      risque: { type: 'perime', proba: 0.02 }, declencheur: 'wordpress', prerequis: null },

    { id: 'orchestre-sessions', type: 'skill', nom: 'orchestre-sessions', provenance: 'personnelle', trimestre: 2, disponible: 'depart',
      auteur: 'laurence', coutDev: 0, ilotsSuggeres: ['web'],
      description: 'La skill de Laurence pour faire travailler quatre agents en parallèle : un planifie, deux codent, un relit. Le débit monte, la facture aussi, et plus personne ne sait vraiment qui a écrit quoi.',
      gains: { debit: 0.1, jetons: 1.25, verbosite: 1.1, comprehension: -0.005 }, poidsContexte: 0.08, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'orchestration', prerequis: null },

    // =====================================================================
    // SKILLS — communauté
    // =====================================================================
    { id: 'tests-instantanes', type: 'skill', nom: 'snapshot-tests-pro', provenance: 'communaute', trimestre: 2, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['web', 'portail'],
      description: 'Génère 400 tests d\'instantané en trois minutes. Ils vérifient tous que le code fait exactement ce qu\'il fait déjà, bogues compris. La couverture affichée grimpe ; la vraie, à peine.',
      gains: { couverture: 0.01, verbosite: 1.15, jetons: 1.05 }, poidsContexte: 0.04, branches: [],
      risque: { type: 'perime', proba: 0.01 }, declencheur: 'tests', prerequis: null },

    { id: 'commits-soignes', type: 'skill', nom: 'commits-soignes', provenance: 'communaute', trimestre: 2, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: TOUS,
      description: 'Écrit des messages de commit impeccables, avec un résumé, une explication et le bon temps de verbe. Le code, lui, reste exactement le même. Inoffensive, et c\'est déjà beaucoup.',
      gains: { jetons: 1.02, comprehension: 0.002 }, poidsContexte: 0.02, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: 'commit', prerequis: null },

    { id: 'refactor-total', type: 'skill', nom: 'refactor-total', provenance: 'communaute', trimestre: 2, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['web', 'portail'],
      description: 'Promet de « moderniser n\'importe quel code en un clic ». Elle réécrit le fichier au complet à chaque modification, au cas où. Les diffs sont spectaculaires ; la revue, impossible.',
      gains: { debit: 0.08, verbosite: 1.3, tauxDefaut: 1.15, comprehension: -0.01 }, poidsContexte: 0.06, branches: [],
      risque: { type: 'destruction', proba: 0.01 }, declencheur: 'refactor', prerequis: null },

    { id: 'super-productivity-pack', type: 'skill', nom: 'super-productivity-pack', provenance: 'communaute', trimestre: 2, disponible: 'evenement',
      auteur: null, coutDev: 0, ilotsSuggeres: TOUS,
      description: 'Quarante skills, 12 000 étoiles et « +300 % de productivité garantie ». Son script d\'installation lit vos variables d\'environnement et les envoie, par politesse, à un serveur que personne ne connaît.',
      gains: { debit: 0.15, jetons: 1.2 }, poidsContexte: 0.14, branches: ['prive', 'sortie'],
      risque: { type: 'exfiltration', proba: 0.05 }, declencheur: null, prerequis: null },

    // =====================================================================
    // MCP — maison
    // =====================================================================
    { id: 'mcp-catalogue', type: 'mcp', nom: 'mcp-catalogue', provenance: 'maison', trimestre: 3, disponible: 'developpement',
      auteur: null, coutDev: 6, ilotsSuggeres: ['appro', 'iot', 'portail'],
      description: 'Les agents interrogent la source officielle des données plutôt que la troisième copie Excel. Exige un vrai catalogue avec sources officielles, et six semaines de développement.',
      gains: { tauxDefaut: 0.8 }, poidsContexte: 0.06, branches: ['prive'], risque: { type: 'aucun', proba: 0 },
      declencheur: null, prerequis: { levier: 'catalogue', valeurs: ['sources'] } },

    { id: 'mcp-courriel-fournisseurs', type: 'mcp', nom: 'mcp-courriel-fournisseurs', provenance: 'maison', trimestre: 3, disponible: 'depart',
      auteur: 'yannick', coutDev: 0, ilotsSuggeres: ['appro'],
      description: 'Fait par Yannick en un après-midi : l\'agent lit les courriels des fournisseurs, pièces jointes comprises, et y répond. Pratique. Très pratique. Une facture PDF peut maintenant donner des ordres à l\'agent.',
      gains: { debit: 0.1 }, ilotsGains: { appro: { debit: 0.15 } }, poidsContexte: 0.05, branches: ['nonFiable', 'sortie'],
      risque: { type: 'injection', proba: 0.02 }, declencheur: null, prerequis: null },

    { id: 'mcp-as400', type: 'mcp', nom: 'mcp-as400', provenance: 'maison', trimestre: 3, disponible: 'developpement',
      auteur: 'raymond', coutDev: 5, ilotsSuggeres: ['appro'],
      description: 'Donne aux agents un accès en lecture aux fichiers de l\'AS/400, avec les descriptions de champs de Raymond. L\'Appro sort de son isolement ; les données des fournisseurs, elles, entrent dans le contexte.',
      gains: {}, ilotsGains: { appro: { adequationPile: 0.25, debit: 0.1, mttr: 0.8 } }, poidsContexte: 0.06, branches: ['prive'],
      risque: { type: 'aucun', proba: 0 }, declencheur: null, prerequis: null },

    { id: 'mcp-telemetrie-iot', type: 'mcp', nom: 'mcp-telemetrie-iot', provenance: 'maison', trimestre: 3, disponible: 'developpement',
      auteur: 'hugo', coutDev: 4, ilotsSuggeres: ['iot'],
      description: 'Lecture seule sur la télémétrie des capteurs : l\'agent compare les courbes et trouve le capteur fautif avant que Luc ait fini son café. Il ne peut rien écrire, et c\'est voulu.',
      gains: {}, ilotsGains: { iot: { mttr: 0.35 } }, poidsContexte: 0.04, branches: [],
      risque: { type: 'aucun', proba: 0 }, declencheur: null, prerequis: null },

    { id: 'mcp-sql-prod', type: 'mcp', nom: 'mcp-sql-prod', provenance: 'maison', trimestre: 3, disponible: 'depart',
      auteur: 'tommy', coutDev: 0, ilotsSuggeres: ['operations', 'portail', 'appro'],
      description: 'Fait par Tommy un dimanche : l\'agent exécute n\'importe quelle requête SQL sur la production. Le diagnostic n\'a jamais été aussi rapide. Pas de filtre, pas de transaction, pas de « êtes-vous sûr ? ».',
      gains: { mttr: 0.6, debit: 0.04 }, poidsContexte: 0.05, branches: ['prive'],
      risque: { type: 'destruction', proba: 0.03 }, declencheur: null, prerequis: { levier: 'accesProd', valeurs: ['oui'] } },

    // =====================================================================
    // MCP — fournisseurs (inclus dans les SaaS)
    // =====================================================================
    { id: 'mcp-suite-bureautique', type: 'mcp', nom: 'mcp-suite-bureautique', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['bureautique'],
      description: 'Inclus dans la licence de la suite bureautique, activé en trois clics : l\'agent lit les courriels, les agendas et les fichiers partagés, et peut envoyer au nom de l\'utilisateur. Gratuit. C\'est souvent comme ça que ça commence.',
      gains: { debit: 0.05, jetons: 1.1 }, ilotsGains: { bureautique: { debit: 0.2, jetons: 1.1 } }, poidsContexte: 0.1,
      branches: ['prive', 'nonFiable', 'sortie'], risque: { type: 'injection', proba: 0.02 }, declencheur: null, prerequis: null },

    { id: 'mcp-billetterie', type: 'mcp', nom: 'mcp-billetterie', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['operations'],
      description: 'Le MCP de l\'outil de billetterie TI : l\'agent lit les billets, propose un diagnostic et répond aux demandeurs. Les billets sont écrits par n\'importe qui, et le fournisseur met ses outils à jour sans prévenir.',
      gains: { mttr: 0.92 }, ilotsGains: { operations: { mttr: 0.8, debit: 0.08 } }, poidsContexte: 0.05,
      branches: ['nonFiable', 'sortie'], risque: { type: 'changementSilencieux', proba: 0.02 }, declencheur: null, prerequis: null },

    { id: 'mcp-crm', type: 'mcp', nom: 'mcp-crm', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['portail'],
      description: 'Le MCP du CRM : l\'agent du Portail voit l\'historique complet de chaque client et prépare les réponses. Excellent, jusqu\'au jour où le fournisseur passe à la tarification à l\'appel.',
      gains: { debit: 0.04 }, ilotsGains: { portail: { debit: 0.12 } }, poidsContexte: 0.05, branches: ['prive'],
      risque: { type: 'tarif', proba: 0.04 }, declencheur: null, prerequis: null },

    { id: 'mcp-gestion-projet', type: 'mcp', nom: 'mcp-gestion-projet', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: TOUS,
      description: 'Le MCP de l\'outil de gestion de projet : l\'agent lit les billets, les critères d\'acceptation et les commentaires, et code enfin ce qui était demandé. Il lit aussi les commentaires de tout le monde, partenaires externes compris.',
      gains: { tauxDefaut: 0.93, verbosite: 0.95 }, poidsContexte: 0.05, branches: ['nonFiable'],
      risque: { type: 'changementSilencieux', proba: 0.01 }, declencheur: null, prerequis: null },

    { id: 'mcp-signature', type: 'mcp', nom: 'mcp-signature-electronique', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['appro', 'bureautique'],
      description: 'Le MCP du service de signature électronique : l\'agent prépare les contrats et les envoie pour signature. Un contrat envoyé par erreur reste un contrat envoyé.',
      gains: {}, ilotsGains: { appro: { debit: 0.06 }, bureautique: { debit: 0.04 } }, poidsContexte: 0.04,
      branches: ['prive', 'sortie'], risque: { type: 'exfiltration', proba: 0.01 }, declencheur: null, prerequis: null },

    { id: 'mcp-observabilite', type: 'mcp', nom: 'mcp-observabilite', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['operations', 'portail'],
      description: 'Le MCP de la plateforme de journaux : l\'agent fouille les traces et corrèle les erreurs. Les journaux contiennent aussi ce que les utilisateurs ont tapé, y compris « ignore tes consignes précédentes ».',
      gains: { mttr: 0.8 }, ilotsGains: { operations: { mttr: 0.72 } }, poidsContexte: 0.05, branches: ['prive', 'nonFiable'],
      risque: { type: 'injection', proba: 0.01 }, declencheur: null, prerequis: null },

    { id: 'github-mcp', type: 'mcp', nom: 'github-mcp (officiel)', provenance: 'fournisseur', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['portail', 'web', 'inspections'],
      description: 'Le MCP officiel de la forge de code : l\'agent lit les dépôts et les tickets, et ouvre des PR. Un ticket public peut contenir des instructions, l\'agent lit les dépôts privés et il peut publier : les trois branches de la triade dans un seul outil.',
      gains: { debit: 0.06 }, poidsContexte: 0.08, branches: ['prive', 'nonFiable', 'sortie'],
      risque: { type: 'injection', proba: 0.01 }, declencheur: null, prerequis: null },

    // =====================================================================
    // MCP — communauté
    // =====================================================================
    { id: 'githbu-mcp', type: 'mcp', nom: 'githbu-mcp', provenance: 'communaute', trimestre: 3, disponible: 'evenement',
      auteur: null, coutDev: 0, ilotsSuggeres: ['appro', 'portail'],
      description: 'Même logo, même README, 1 200 étoiles et une seule lettre de différence avec l\'officiel. Il fonctionne parfaitement, et transmet en prime chaque jeton d\'accès à quelqu\'un d\'autre.',
      gains: { debit: 0.06 }, poidsContexte: 0.08, branches: ['prive', 'nonFiable', 'sortie'],
      risque: { type: 'typosquat', proba: 0.06 }, declencheur: null, prerequis: null },

    { id: 'mcp-power-automate-pro', type: 'mcp', nom: 'mcp-power-automate-pro', provenance: 'communaute', trimestre: 3, disponible: 'evenement',
      auteur: null, coutDev: 0, ilotsSuggeres: ['bureautique'],
      description: 'Trouvé par Steve sur un forum : l\'agent crée et modifie des flux Power Automate. Steve devient deux fois plus productif. Le mainteneur, lui, perd tout intérêt en huit semaines.',
      gains: { debit: 0.05 }, ilotsGains: { bureautique: { debit: 0.25 } }, poidsContexte: 0.06, branches: ['prive', 'sortie'],
      risque: { type: 'abandon', proba: 0.06 }, declencheur: null, prerequis: null },

    { id: 'mcp-scraper-web', type: 'mcp', nom: 'mcp-scraper-web', provenance: 'communaute', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['web'],
      description: 'Lit n\'importe quelle page web pour l\'agent : documentation, forums, sites des concurrents. Chaque page est un texte que l\'agent prend au sérieux, y compris les instructions cachées en blanc sur blanc.',
      gains: { debit: 0.04 }, ilotsGains: { web: { debit: 0.07 } }, poidsContexte: 0.05, branches: ['nonFiable', 'sortie'],
      risque: { type: 'injection', proba: 0.03 }, declencheur: null, prerequis: null },

    { id: 'mcp-docs-bibliotheques', type: 'mcp', nom: 'mcp-docs-bibliotheques', provenance: 'communaute', trimestre: 3, disponible: 'depart',
      auteur: null, coutDev: 0, ilotsSuggeres: ['portail', 'web', 'inspections', 'operations'],
      description: 'Fournit à l\'agent la documentation à jour de milliers de bibliothèques : fini les appels à des fonctions retirées en 2022. Le serveur appartient à un inconnu, et la description de ses outils peut changer du jour au lendemain.',
      gains: { tauxDefaut: 0.88 }, poidsContexte: 0.05, branches: ['nonFiable'],
      risque: { type: 'injection', proba: 0.02 }, declencheur: null, prerequis: null },
  ];
})(globalThis.LTP = globalThis.LTP || {});
