/* 11-donnees-personnages.js */
/* Les 21 développeurs, les personnages secondaires et la table des traits.
   Les champs structurels (id, ilot, niveau, traits) sont un contrat : ne pas les renommer.
   Les champs de texte (tagline, replique, bulles, reactions) sont du contenu. */
(function (LTP) {
  'use strict';

  // --- Traits ---
  // init : modificateurs appliqués aux statistiques de départ (additifs) par 20-etat.js.
  // mods : modificateurs lus par les systèmes, chacun dans son fichier :
  //   production (21) : debit (multiplicatif, ex. −0,15), sessions (additif), taillePR (multiplicatif), verbosite (multiplicatif)
  //   qualite (22)    : tauxDefaut (multiplicatif), couverture (additif sur la couverture réelle produite)
  //   dette (23)      : comprehension (multiplicatif sur l'apprentissage)
  //   personnes (27)  : fatigue (multiplicatif), apprentissage (multiplicatif), adhesionMax (plafond)
  //   registre (26)   : installeCommunaute (probabilité par semaine d'installer une carte communautaire en libre-service)
  LTP.donnees.traits = {
    legacy:          { libelle: 'Gardien des systèmes hérités', init: { comprehension: 0.15, competence: 0.05 }, mods: {} },
    sceptique:       { libelle: 'Sceptique',             init: { adhesion: -0.25 }, mods: {} },
    donnees:         { libelle: 'Œil pour les données',  init: { rigueur: 0.05 }, mods: { tauxDefaut: 0.9 } },
    rigoureux:       { libelle: 'Rigueur',               init: { rigueur: 0.2 }, mods: { tauxDefaut: 0.85 } },
    curieux:         { libelle: 'Curiosité',             init: { adhesion: 0.15, competenceIA: 0.1 }, mods: { installeCommunaute: 0.04 } },
    integrateur:     { libelle: 'Intégrateur',           init: { competenceIA: 0.05 }, mods: { installeCommunaute: 0.06 } },
    garde:           { libelle: 'De garde la nuit',      init: { fatigue: 0.1 }, mods: { fatigue: 1.2 } },
    stoique:         { libelle: 'Stoïque',               init: { moral: 0.05 }, mods: { fatigue: 0.85 } },
    sre:             { libelle: 'SRE',                   init: { rigueur: 0.1 }, mods: {} },
    apprenant:       { libelle: 'Apprend vite',          init: {}, mods: { apprentissage: 1.5 } },
    automatiseur:    { libelle: 'Automatise tout',       init: { adhesion: 0.2 }, mods: { installeCommunaute: 0.08, debit: 0.05 } },
    enthousiaste:    { libelle: 'Enthousiaste',          init: { adhesion: 0.25, moral: 0.05 }, mods: {} },
    vba:             { libelle: 'Macros VBA',            init: { comprehension: 0.1 }, mods: {} },
    refractaire:     { libelle: 'Réfractaire',           init: { adhesion: -0.4 }, mods: { adhesionMax: 0.3 } },
    contractuel:     { libelle: 'Contractuel',           init: {}, mods: {} },
    tdd:             { libelle: 'TDD',                   init: { rigueur: 0.25 }, mods: { debit: -0.15, couverture: 0.3, tauxDefaut: 0.75 } },
    mobile:          { libelle: 'Mobile',                init: {}, mods: {} },
    pragmatique:     { libelle: 'Pragmatique',           init: { adhesion: 0.05 }, mods: { debit: 0.05 } },
    natifIA:         { libelle: 'Génération IA',        init: { adhesion: 0.3, competenceIA: 0.25, competence: -0.05 }, mods: { verbosite: 1.2 } },
    firmware:        { libelle: 'Firmware',              init: { rigueur: 0.1 }, mods: {} },
    prudent:         { libelle: 'Prudence',              init: { rigueur: 0.15, adhesion: -0.1 }, mods: { tauxDefaut: 0.85 } },
    electronicien:   { libelle: 'Électronicien',         init: {}, mods: {} },
    designer:        { libelle: 'Designer',              init: { competence: -0.05, adhesion: 0.15 }, mods: {} },
    wordpress:       { libelle: 'WordPress',             init: {}, mods: {} },
    multiClauding:   { libelle: 'Multi-Clauding',        init: { adhesion: 0.3 }, mods: { sessions: 2, fatigue: 1.25 } },
    dixX:            { libelle: '« 10x »',               init: { competence: 0.05, rigueur: -0.2, adhesion: 0.3 }, mods: { sessions: 2, taillePR: 3, verbosite: 1.4 } },
    shipIt:          { libelle: 'Ship it',               init: { rigueur: -0.15 }, mods: { debit: 0.1, tauxDefaut: 1.25 } },
    reviseur:        { libelle: 'Réviseuse de référence', init: { rigueur: 0.15 }, mods: {} },
    securite:        { libelle: 'Sécurité',              init: { rigueur: 0.1 }, mods: { tauxDefaut: 0.9 } },
    mesure:          { libelle: 'Mesure tout',           init: { rigueur: 0.05 }, mods: {} },
  };

  // --- Développeurs ---
  // niveau : stagiaire · junior · inter · senior
  // accessoire : objet signature sur le bureau (dessiné par 41-ui-avatars.js)
  // pilote : volontaire du groupe pilote (une personne par îlot) qui reçoit la licence Copilot dès la semaine 1
  // apparence : optionnelle ; sinon dérivée de l'id de façon déterministe
  LTP.donnees.devs = [
    { id: 'raymond',   nom: 'Raymond',   ilot: 'appro',       niveau: 'senior',    traits: ['legacy', 'sceptique'],         accessoire: 'terminal-vert',
      tagline: '31 ans sur l\'AS/400. « L\'IA ? J\'ai vu passer le 4GL aussi. »' },
    { id: 'lucie',     nom: 'Lucie',     ilot: 'appro',       niveau: 'inter',     traits: ['donnees', 'rigoureux'],        accessoire: 'classeurs',
      tagline: 'Connaît par cœur les trois listes de fournisseurs, et laquelle est la bonne.' },
    { id: 'yannick',   nom: 'Yannick',   ilot: 'appro',       niveau: 'inter',     traits: ['curieux', 'integrateur'],      pilote: true, accessoire: 'cables',
      tagline: 'Branche tout sur tout. Premier à installer un MCP trouvé sur un forum.' },

    { id: 'benoit',    nom: 'Benoît',    ilot: 'operations',  niveau: 'senior',    traits: ['garde', 'stoique'],            accessoire: 'tasse',
      tagline: 'De garde la nuit depuis 2009. Son téléphone vibre, il soupire.' },
    { id: 'marieeve',  nom: 'Marie-Ève', ilot: 'operations',  niveau: 'senior',    traits: ['sre', 'mesure'],               pilote: true, accessoire: 'tableau-bord',
      tagline: 'SRE. Ne croit que ce qui a un tableau de bord.' },
    { id: 'tommy',     nom: 'Tommy',     ilot: 'operations',  niveau: 'junior',    traits: ['apprenant'],                   accessoire: 'carnet',
      tagline: 'Apprend vite, mais seulement ce qu\'on lui laisse faire.' },

    { id: 'steve',     nom: 'Steve',     ilot: 'bureautique', niveau: 'inter',     traits: ['automatiseur', 'enthousiaste'], pilote: true, accessoire: 'organigramme',
      tagline: 'Roi du Power Automate. Voisin de bureau de Kevin : ils s\'inspirent mutuellement.' },
    { id: 'diane',     nom: 'Diane',     ilot: 'bureautique', niveau: 'senior',    traits: ['vba', 'refractaire'],          accessoire: 'plante',
      tagline: 'Ses macros Excel font tourner la paie depuis 2007. Ne veut rien savoir.' },
    { id: 'olivier',   nom: 'Olivier',   ilot: 'bureautique', niveau: 'inter',     traits: ['contractuel', 'enthousiaste'], accessoire: 'sac-a-dos',
      tagline: 'Contractuel. Part dans six mois, avec ce qu\'il sait.' },

    { id: 'sophie',    nom: 'Sophie',    ilot: 'inspections', niveau: 'senior',    traits: ['tdd', 'rigoureux'],            accessoire: 'cactus',
      tagline: 'Pas de code sans test rouge d\'abord. Lente au T1, imbattable au T3.' },
    { id: 'karim',     nom: 'Karim',     ilot: 'inspections', niveau: 'inter',     traits: ['mobile', 'pragmatique'],       pilote: true, accessoire: 'telephones',
      tagline: 'L\'application terrain hors ligne, c\'est lui.' },
    { id: 'jeremy',    nom: 'Jérémy',    ilot: 'inspections', niveau: 'stagiaire', traits: ['natifIA', 'apprenant'],        accessoire: 'boisson-energie',
      tagline: 'N\'a jamais codé sans IA. Brillant ou perdu selon vos choix.' },

    { id: 'amira',     nom: 'Amira',     ilot: 'iot',         niveau: 'senior',    traits: ['firmware', 'prudent'],         accessoire: 'oscilloscope',
      tagline: 'Firmware des capteurs. Sait qu\'un capteur ne pardonne pas.' },
    { id: 'luc',       nom: 'Luc',       ilot: 'iot',         niveau: 'inter',     traits: ['electronicien'],               accessoire: 'fer-a-souder',
      tagline: 'Électronicien devenu développeur. Parle aux capteurs.' },
    { id: 'hugo',      nom: 'Hugo',      ilot: 'iot',         niveau: 'inter',     traits: ['donnees', 'curieux'],          pilote: true, accessoire: 'graphique',
      tagline: 'Pipelines Python de télémétrie. Deux millions de mesures par jour.' },

    { id: 'chloe',     nom: 'Chloé',     ilot: 'web',         niveau: 'inter',     traits: ['designer', 'natifIA'],         pilote: true, accessoire: 'tablette-dessin',
      tagline: 'Designer devenue développeuse grâce à l\'IA.' },
    { id: 'patrick',   nom: 'Patrick',   ilot: 'web',         niveau: 'senior',    traits: ['wordpress', 'pragmatique'],    accessoire: 'mug-wp',
      tagline: '14 sites WordPress, dont 3 qu\'il aimerait oublier.' },
    { id: 'laurence',  nom: 'Laurence',  ilot: 'web',         niveau: 'junior',    traits: ['multiClauding', 'enthousiaste'], accessoire: 'ecrans',
      tagline: 'Quatre écrans, quatre sessions, zéro pause.' },

    { id: 'mathieu',   nom: 'Mathieu',   ilot: 'portail',     niveau: 'senior',    traits: ['dixX', 'shipIt'],              pilote: true, accessoire: 'casque',
      tagline: '« 10x » autoproclamé. Des PR de 4 000 lignes.' },
    { id: 'nadia',     nom: 'Nadia',     ilot: 'portail',     niveau: 'senior',    traits: ['reviseur', 'rigoureux'],       accessoire: 'pile-papiers',
      tagline: 'Tout le monde veut sa revue. Elle est le goulot incarné.' },
    { id: 'sebastien', nom: 'Sébastien', ilot: 'portail',     niveau: 'senior',    traits: ['securite', 'mesure'],          accessoire: 'cadenas',
      tagline: 'Architecte sécurité. Voit la triade létale avant tout le monde.' },
  ];

  // Champs de contenu complétés par l'agent de contenu (D1) :
  //   replique : phrase d'accueil dans la fiche
  //   bulles : { code: [], revue: [], incident: [], fatigue: [], formation: [], attente: [], conge: [] }
  //   reactions : [{ levier, valeurs: [...], moral: +0,01 (par semaine où le levier effectif de son îlot est dans valeurs), bulle }]
  // Le contenu est rangé par id ci-dessous, puis fusionné dans LTP.donnees.devs.
  function R(levier, valeurs, moral, bulle) { return { levier: levier, valeurs: valeurs, moral: moral, bulle: bulle }; }

  var CONTENU = {
    raymond: {
      replique: 'J\'ai vu passer le 4GL, le client-serveur, le web 2.0 et la chaîne de blocs. L\'AS/400, lui, est encore là. On verra bien pour votre agent.',
      bulles: {
        code: ['RPG IV. Colonne 6, comme en 1994.', 'L\'agent me propose du Java. Non.', 'Ce programme a mon ancienneté.', 'Un fichier physique de 1994. Il marche encore.', 'Je commente en français. L\'agent, en anglais.'],
        revue: ['Qui a écrit ça ? Ah. L\'agent.', 'Je relis. Lentement. Correctement.', 'Ce SQL est pour PostgreSQL. Ici, c\'est DB2.', 'Une PR de l\'agent pour l\'AS/400. Courageux.', 'Ce n\'est pas faux. C\'est juste pas d\'ici.'],
        incident: ['Le traitement de nuit a planté. Classique.', 'J\'ai déjà vu ce message. En 2003.', 'Je suis le seul à lire ce journal.'],
        fatigue: ['Trente et un ans. Je les sens, ce soir.', 'Mon café a refroidi deux fois.', 'L\'AS/400 ne se fatigue jamais. Moi, oui.', 'Trop de nouveautés pour un seul trimestre.'],
        formation: ['Un prompt, c\'est une spec mal écrite.', 'Bon. Je prends des notes. Sur papier.', 'Il ne connaît pas le RPG, votre agent.'],
        attente: ['J\'attends une revue depuis mardi.', 'Personne ne veut relire du RPG.', 'En attendant, j\'explique la colonne 6 à Lucie.', 'Ma PR attend. Le traitement de nuit, lui, roule.'],
        conge: ['Au chalet. Pas de réseau. Parfait.', 'Je pêche. Le RPG attendra.'],
      },
      reactions: [
        R('autonomie', [3, 4], -0.012, 'Un robot qui fusionne du RPG. J\'aurai tout vu.'),
        R('pair', ['humain', 'rotation'], 0.01, 'Enfin quelqu\'un à qui montrer l\'AS/400.'),
        R('catalogue', ['inventaire', 'sources'], 0.012, 'La vraie liste est dans l\'AS/400. Je vous montre.'),
        R('embauche', ['releve'], 0.01, 'Une relève ? Je vais enfin transmettre le RPG.'),
        R('licences', ['tous'], -0.005, 'Une licence pour moi ? Je vais la ranger avec le 4GL.'),
        R('licences', ['aucune'], 0.005, 'Plus de licences. Le monde a retrouvé son bon sens.'),
      ],
    },
    lucie: {
      replique: 'Il y a trois listes de fournisseurs, et l\'agent prend toujours la première qu\'il trouve. Moi, je doute. C\'est un peu mon métier.',
      bulles: {
        code: ['Je vérifie la source avant d\'écrire.', 'Quelle liste ? La vraie. Toujours.', 'Une jointure de plus, une erreur de moins.', 'Je nomme la source dans chaque requête.', 'Fournisseurs_FINAL_v2_vraie.xlsx. La bonne.'],
        revue: ['L\'agent a pris la liste de 2019.', 'Ce code de fournisseur n\'existe plus.', 'Je compare les trois listes. Encore.', 'La jointure est juste. La table, non.', 'D\'où vient ce code postal ? Je cherche.'],
        incident: ['Facture payée deux fois. Je sais où.', 'Doublon de fournisseur. Je l\'avais dit.'],
        fatigue: ['Trop de listes, pas assez de pauses.', 'Je vois des codes fournisseurs partout.', 'Trois listes, une Lucie. Le compte est mauvais.', 'Je relis la même facture pour la quatrième fois.'],
        formation: ['Comment on dit « doute » à un agent ?', 'Je note : toujours citer la source.'],
        attente: ['Ma PR attend. Ma liste, elle, vieillit.', 'En attente de Raymond. Patience.', 'En attendant, je dédoublonne une liste.', 'Ma PR attend. Les doublons, eux, se multiplient.'],
        conge: ['Congé. Les listes vont se dupliquer sans moi.', 'En vacances. Sans tableur.'],
      },
      reactions: [
        R('catalogue', ['inventaire', 'sources'], 0.015, 'Les trois listes vont enfin se parler.'),
        R('classification', ['manuelle', 'auto'], 0.01, 'Étiqueter les données : enfin du ménage.'),
        R('specs', ['non'], -0.006, 'Sans spec, l\'agent choisit une liste au hasard.'),
        R('tests', ['aucun'], -0.01, 'Pas de tests ? Et on vérifie les données comment ?'),
        R('licences', ['tous'], 0.005, 'Une licence ! Je vais lui apprendre quelle liste est la bonne.'),
      ],
    },
    yannick: {
      replique: 'J\'ai trouvé un serveur MCP qui branche l\'agent sur les courriels des fournisseurs. Douze étoiles sur GitHub ! Je ne sais pas qui l\'a écrit, mais il a une belle icône.',
      bulles: {
        code: ['Je branche le MCP dans le MCP.', 'Trouvé un outil génial sur un forum.', 'Encore un connecteur et c\'est parfait.', 'Un connecteur de plus. Promis, c\'est le dernier.', 'J\'ai trouvé une extension qui fait tout. Tout.'],
        revue: ['Ça marche chez moi. C\'est bon ?', 'L\'agent a ajouté une dépendance. Cool ?', 'Je relis le code du MCP avant de l\'installer. Cette fois.', 'Pourquoi ce paquet a 3 étoiles et 40 000 téléchargements ?'],
        incident: ['Ce MCP faisait ça ? Je savais pas.', 'Oups. Je débranche tout. Un par un.', 'Le serveur MCP a changé tout seul ?'],
        fatigue: ['Quarante onglets. Trente-neuf utiles.', 'Je me débranche. Moi aussi.', 'Trop d\'outils, pas assez de mains.', 'Je ne sais plus ce que j\'ai branché où.'],
        formation: ['La triade létale ? Ah. Ah non.', 'Donc on lit le code avant d\'installer.'],
        attente: ['Ma PR attend. J\'installe un truc ?', 'En attendant, j\'explore un forum.', 'Ma PR attend. Je lis le code d\'un MCP, pour voir.', 'En attente. Je fais le ménage de mes extensions.'],
        conge: ['Congé. Je teste un MCP à la maison.', 'Congé. Mon routeur a douze extensions.'],
      },
      reactions: [
        R('politiqueRegistre', ['libre'], 0.01, 'Libre-service ! J\'ai déjà trois MCP à essayer.'),
        R('politiqueRegistre', ['proprietaires'], -0.008, 'Un propriétaire par MCP ? Bon. Je signe.'),
        R('sessionsMax', [3, 4, 5, 6], 0.008, 'Trois sessions, trois intégrations en même temps.'),
        R('bacASable', ['filtre'], -0.006, 'Le réseau filtré bloque mon serveur préféré.'),
        R('licences', ['aucune'], -0.01, 'On m\'enlève Copilot ? J\'avais tout branché dessus.'),
      ],
    },
    benoit: {
      replique: 'Je suis de garde depuis 2009. J\'ai juste une question pour votre agent : quand il casse quelque chose à 3 h du matin, c\'est lui qu\'on appelle ?',
      bulles: {
        code: ['Un script Ansible. Comme d\'habitude.', 'J\'ajoute une alerte. Juste au cas.', 'L\'agent écrit vite. Je relis lentement.', 'Une alerte de plus, une nuit de moins.', 'Je documente pour Tommy. Il va en avoir besoin.'],
        revue: ['Qui va déboguer ça à 3 h ? Moi.', 'Ce code a l\'air parfait. C\'est suspect.', 'Où sont les journaux ? Pas de journaux.', 'Ça redémarre bien ? Montre-moi.', 'Pas de retour arrière prévu ? Non.'],
        incident: ['Soupir. Café. Terminal.', 'Mon téléphone vibre. Évidemment.', 'Qui a écrit ça ? L\'agent. Super.'],
        fatigue: ['Trois nuits de garde. Tasse vide.', 'Je dors debout. Littéralement.', 'Mon téléphone me fait peur.', 'Ma tasse et moi, on est vides.', 'J\'ai dormi trois heures. En deux fois.'],
        formation: ['Formation à 9 h. Fin de garde à 4 h.', 'Bon. Je prends des notes. Calmement.'],
        attente: ['En attente. Je recharge mon téléphone.', 'Ma PR attend. J\'ai l\'habitude.', 'En attente. Je relis le carnet de garde.', 'Ma PR attend. Mon téléphone, non.'],
        conge: ['En congé. Téléphone éteint. Vraiment.', 'Congé. Tommy a mon numéro. Hélas.'],
      },
      reactions: [
        R('observabilite', ['base', 'complete'], 0.015, 'Des tableaux de bord. Je vais peut-être dormir.'),
        R('accesProd', ['oui'], -0.01, 'Un agent qui écrit en prod. Je charge mon téléphone.'),
        R('autonomie', [4], -0.015, 'Il fusionne seul. Qui se fait réveiller ? Moi.'),
        R('ci', ['complet'], 0.008, 'Des déploiements ennuyants. Mon rêve.'),
      ],
    },
    marieeve: {
      replique: 'Montre-moi un tableau de bord et je te crois. Montre-moi une démo et je te pose trois questions. Pour l\'IA, c\'est pareil : où sont les métriques ?',
      bulles: {
        code: ['J\'instrumente. Ensuite, je code.', 'Une métrique de plus. Ça coûte rien.', 'Pas de SLO, pas de mise en prod.', 'Un SLO avant la fonctionnalité.', 'J\'ajoute un compteur. Juste pour savoir.'],
        revue: ['Où est la métrique ? Je cherche encore.', 'Il manque des journaux structurés.', 'Ça marche ? Prouve-le.', 'Cette PR ajoute du code et retire une métrique. Non.', 'Je veux voir le graphique avant et après.'],
        incident: ['Le MTTR, je le chronomètre.', 'Hypothèse, mesure, correction.', 'Graphique d\'abord, panique jamais.'],
        fatigue: ['Mes tableaux de bord sont rouges. Moi aussi.', 'Mesure du jour : épuisée.', 'Mon tableau de bord a moins de cernes que moi.', 'Trop d\'alertes. Même les miennes.'],
        formation: ['Combien de jetons par tâche ? Je note.', 'Des évaluations, c\'est des tests, non ?'],
        attente: ['La file de revue, je la mesure, au moins.', 'Délai de ma PR : en hausse.', 'En attente : je trace le délai de ma propre PR.', 'Ma PR attend. J\'en fais une courbe.'],
        conge: ['Congé. Je ne regarde aucun graphique.', 'En vacances. Bon, un seul graphique.'],
      },
      reactions: [
        R('observabilite', ['complete'], 0.02, 'Enfin des chiffres que je peux croire.'),
        R('ci', ['tests', 'complet'], 0.01, 'Chaque changement testé et compté. Bien.'),
        R('objectif', ['pr'], -0.015, 'Compter les PR, c\'est mesurer la mauvaise chose.'),
        R('accesProd', ['lecture'], 0.008, 'Lecture seule en prod : le bon compromis.'),
        R('licences', ['pilote'], 0.005, 'Sept pilotes, quatorze témoins. On compare pour vrai.'),
        R('licences', ['tous'], -0.005, 'Tout le monde en même temps ? Je perds mon groupe témoin.'),
      ],
    },
    tommy: {
      replique: 'Benoît dit qu\'on apprend la garde en la faisant. L\'agent dit qu\'il peut la faire à ma place. Je ne sais pas encore lequel des deux écouter.',
      bulles: {
        code: ['Je l\'écris moi-même. Pour comprendre.', 'L\'agent a fini. Moi, j\'ai compris ?', 'Premier script Bash sans aide !', 'Je tape la commande moi-même, comme Benoît dit.', 'L\'agent propose, je cherche pourquoi.'],
        revue: ['Benoît m\'explique. Je note tout.', 'Je relis la PR de l\'agent. Lentement.', 'Je pose une question par PR. Au moins.', 'Je relis la PR de Benoît. J\'apprends.'],
        incident: ['Benoît, c\'est quoi, ce graphique ?', 'Ma première panne. J\'ai peur et j\'aime ça.'],
        fatigue: ['Ça va trop vite. Je décroche un peu.', 'J\'ai appris trop de choses cette semaine.', 'J\'ai trop appris cette semaine. Ma tête déborde.', 'La garde de nuit, c\'est plus dur que prévu.'],
        formation: ['J\'adore les formations. Vraiment.', 'Enfin le pourquoi derrière le quoi.'],
        attente: ['J\'attends ma revue. J\'apprends en attendant.', 'Personne n\'a le temps de me relire.', 'En attendant, je relis la doc d\'Ansible.', 'Ma PR attend. Je recopie les notes de Benoît.'],
        conge: ['Congé. Je lis la doc de Kubernetes.', 'En vacances. Pour vrai, cette fois.'],
      },
      reactions: [
        R('pair', ['humain', 'rotation'], 0.015, 'Benoît m\'apprend la garde. J\'ai peur, j\'apprends.'),
        R('autonomie', [3, 4], -0.01, 'L\'agent fait tout. Moi, j\'apprends quoi ?'),
        R('formation', ['programme', 'mentorat'], 0.012, 'Un vrai programme de formation ! Je suis prêt.'),
        R('embauche', ['releve'], 0.008, 'Un programme de relève ? On va enfin me former à la garde, pour vrai.'),
        R('licences', ['tous'], 0.005, 'Ma licence ! Je promets de lire ce qu\'il propose.'),
      ],
    },
    steve: {
      replique: 'Si une tâche revient deux fois, je l\'automatise. Si elle revient trois fois, Kevin me demande de lui montrer comment. C\'est comme ça que tout a commencé.',
      bulles: {
        code: ['Un flux de plus. Le 147e.', 'J\'automatise l\'automatisation.', 'Déclencheur, condition, action. Magie.', 'Un flux qui déclenche un flux. Élégant.', 'Kevin m\'a demandé un petit truc. Encore.'],
        revue: ['Ce flux appelle un flux qui appelle…', 'Qui a fait ce flux ? Kevin. Évidemment.', 'Qui a mis 14 étapes dans ce flux ? Moi.', 'Ce flux envoie des courriels à qui, au juste ?'],
        incident: ['Le flux tourne en boucle depuis midi.', '3 000 courriels envoyés. Désolé.'],
        fatigue: ['J\'automatiserais bien ma sieste.', 'Trop de flux, pas assez de moi.', 'Trop de flux ouverts. Dans ma tête aussi.', 'Je rêve en organigrammes.'],
        formation: ['On peut automatiser la formation ?', 'Ah. Les sorties externes. Hmm.'],
        attente: ['En attendant, j\'aide Kevin. Encore.', 'Ma PR attend. Je fais un flux.', 'Ma PR attend. Mes flux, non.', 'En attente. J\'automatise mes rappels de revue.'],
        conge: ['En congé. Mes flux travaillent.', 'Congé. Kevin a mon numéro de cellulaire.'],
      },
      reactions: [
        R('politiqueNonTI', ['encadrer'], 0.012, 'On va aider Kevin à faire ça comme du monde.'),
        R('politiqueNonTI', ['interdire'], -0.012, 'Interdire Power Automate ? Bonne chance.'),
        R('politiqueRegistre', ['libre'], 0.008, 'Libre-service ! J\'ai un MCP pour chaque flux.'),
        R('autonomie', [0], -0.01, 'De l\'autocomplétion. En 2026.'),
        R('licences', ['aucune'], -0.012, 'Plus de licences ? Kevin, lui, a la sienne. Payée avec sa carte.'),
      ],
    },
    diane: {
      replique: 'Mes macros font la paie de 4 000 personnes depuis 2007, et personne ne s\'est jamais plaint. Votre agent, lui, il a déjà fait une paie ?',
      bulles: {
        code: ['Module 7. Celui qui fait la paie.', 'Une macro propre, commentée. À ma façon.', 'Pas de robot dans mon VBA.', 'Je renomme rien. Ça marche.', 'Une macro de plus, commentée à la main.'],
        revue: ['Qui a touché à ma macro ?', 'L\'agent a renommé mes variables. Non.', 'Je relis. Avec mes lunettes.', 'L\'agent a mis des noms en anglais. Je corrige.', 'Ce n\'est pas comme ça qu\'on arrondit la paie.'],
        incident: ['La paie ne balance pas. Je regarde.', 'Je sais exactement quelle cellule.'],
        fatigue: ['Je suis fatiguée de dire non.', 'Fin de trimestre, fin de patience.', 'Trop de changements, trop vite.', 'J\'ai fait trois paies dans ma tête cette nuit.'],
        formation: ['Bon. Je regarde. Sans promesse.', 'Ginette m\'explique. Elle, je l\'écoute.', 'Il a compris ma macro ? Hein.'],
        attente: ['J\'attends. J\'ai l\'habitude.', 'Ma PR attend. La paie, elle, n\'attend pas.', 'Ma PR attend. La paie est jeudi.', 'En attente. Je vérifie les totaux, au cas.'],
        conge: ['Congé. La paie est prête pour jeudi.', 'En vacances. Personne ne touche au module 7.'],
      },
      reactions: [
        R('autonomie', [3, 4], -0.02, 'Un robot dans mes macros de paie ? Non.'),
        R('formation', ['mentorat'], 0.015, 'Si c\'est Ginette qui me montre, j\'essaie.'),
        R('pair', ['humain'], 0.008, 'À deux, devant le même écran. Ça, je connais.'),
        R('sessionsMax', [4, 5, 6], -0.008, 'Quatre sessions ? J\'ai un Excel ouvert, c\'est assez.'),
        R('licences', ['tous'], -0.008, 'Une licence pour moi ? Je n\'ai rien demandé.'),
        R('licences', ['aucune'], 0.005, 'Plus de licences. Mes macros respirent.'),
      ],
    },
    olivier: {
      replique: 'Je suis contractuel : mon mandat finit dans six mois. Avec l\'IA, je livre deux fois plus vite. Ce que je comprends, par contre, part avec moi. Écrivez-le quelque part.',
      bulles: {
        code: ['Vite, avant la fin du mandat.', 'Je livre, je livre, je livre.', 'L\'agent et moi, on est efficaces.', 'Je livre et je laisse des notes. Promis.', 'Encore une fonctionnalité avant la fin du contrat.'],
        revue: ['Je documente. Pour celui qui viendra.', 'C\'est clair pour moi. Pour vous ?', 'J\'explique dans la PR ce que je sais. Pour après.', 'Ça, personne d\'autre ne le sait. Je l\'écris.'],
        incident: ['C\'est mon code. Je suis encore là.', 'Je corrige avant de partir. Promis.'],
        fatigue: ['Deux mandats en même temps. Oups.', 'Je cours après la fin du contrat.', 'Je livre à double vitesse. Je dors à demi-vitesse.', 'Fin de contrat, fin de forces.'],
        formation: ['Une formation payée ? Mon CV dit merci.', 'J\'apprends. Et je l\'emporterai.'],
        attente: ['Ma PR attend. Mon contrat, lui, avance.', 'En attente. Le compteur tourne.', 'Ma PR attend. Mon dernier jour approche.', 'En attendant, j\'écris le guide de passation.'],
        conge: ['Entre deux mandats. Ou presque.', 'Congé. Je mets mon profil LinkedIn à jour.'],
      },
      reactions: [
        R('formation', ['programme', 'mentorat'], 0.01, 'Une vraie formation ? Je reste motivé jusqu\'au bout.'),
        R('pair', ['humain', 'rotation'], 0.008, 'Je transmets avant de partir. Promis.'),
        R('sessionsMax', [3, 4, 5, 6], 0.008, 'Plus de sessions, plus de livrables. J\'aime.'),
        R('disciplineNon', ['ferme'], -0.006, 'Dire non ? En agence, on disait toujours oui.'),
        R('licences', ['tous'], 0.005, 'Une licence ! Je vais livrer deux fois plus vite avant mon départ.'),
      ],
    },
    sophie: {
      replique: 'Je n\'écris pas une ligne sans un test rouge d\'abord. Avec un agent, c\'est encore plus vrai : il écrit vite, alors le test doit savoir ce qu\'on veut avant lui.',
      bulles: {
        code: ['Rouge, vert, refactorisation.', 'Le test d\'abord. Toujours.', 'L\'agent écrit le code. Moi, le test.', 'Test rouge. Bien. Maintenant, le code.', 'Un cas limite de plus. Un test de plus.'],
        revue: ['Ce test vérifie quoi, au juste ?', 'assertTrue(true). Sérieusement ?', 'Mutant survivant à la ligne 42.', 'Ce test passe même si on efface la fonction.', 'Où est le test qui échoue avant le correctif ?'],
        incident: ['Pas couvert. Évidemment.', 'J\'écris le test qui l\'aurait attrapé.'],
        fatigue: ['Mon cactus a plus d\'énergie que moi.', 'Trop de tests à relire, pas assez de moi.', 'Trop de tests générés à relire.', 'Même mes tests sont fatigués.'],
        formation: ['Un agent qui fait du TDD ? Montre.', 'J\'enseigne le TDD à l\'agent. Il apprend.'],
        attente: ['Tests au vert. J\'attends la revue.', 'Ma PR est petite. Elle attend quand même.', 'Ma PR attend. Mes tests, eux, sont prêts.', 'En attente. J\'arrose mon cactus.'],
        conge: ['Congé. Quelqu\'un arrose mon cactus ?', 'En vacances. Zéro test. Presque.'],
      },
      reactions: [
        R('tests', ['mutation'], 0.02, 'Des tests de mutation ! Enfin des tests qui mordent.'),
        R('tests', ['aucun'], -0.02, 'Aucun test. Je vais faire semblant de pas avoir lu.'),
        R('ci', ['tests', 'complet'], 0.008, 'La CI lance mes tests. Ça me rend heureuse.'),
        R('specs', ['obligatoires'], 0.008, 'Une spec, c\'est un test qui ne parle pas encore.'),
        R('licences', ['tous'], 0.005, 'Copilot pour tous. Je vais lui apprendre le test rouge.'),
      ],
    },
    karim: {
      replique: 'L\'application marche sans réseau, au fond d\'un sous-sol, avec des gants. L\'agent l\'a testée sur un portable branché à la fibre. On n\'a pas la même définition de « ça marche ».',
      bulles: {
        code: ['Mode avion activé. Ça synchronise ?', 'Conflit de synchro. Mon préféré.', 'Kotlin le matin, Swift l\'après-midi.', 'Je teste au sous-sol. Pas de réseau. Parfait.', 'La synchro gère les conflits ? On va voir.'],
        revue: ['Et sans réseau, ça fait quoi ?', 'Testé avec des gants ? Non ? Bon.', 'Et si l\'inspecteur perd le réseau à la moitié ?', 'Ça marche sur le Wi-Fi du bureau. Pas sur le terrain.'],
        incident: ['Un inspecteur au sous-sol. Pas de 4G.', 'La synchro a dupliqué 300 photos.'],
        fatigue: ['Trois téléphones, zéro batterie.', 'Je synchronise mal, moi aussi.', 'Trois appareils, trois versions, une tête.', 'Je synchronise mal, moi aussi.'],
        formation: ['L\'agent comprend le hors ligne ? On verra.', 'Je prends ce qui marche. Le reste, non.'],
        attente: ['Ma PR attend. Le terrain aussi.', 'En attente. Je teste sur le vieil Android.', 'Ma PR attend. Les inspecteurs aussi.', 'En attente. Je recharge mes trois téléphones.'],
        conge: ['Congé. Hors ligne, pour vrai.', 'En congé. Cellulaire en mode avion.'],
      },
      reactions: [
        R('specs', ['legeres'], 0.008, 'Une spec légère : juste assez pour l\'inspecteur.'),
        R('tests', ['integration', 'mutation'], 0.008, 'Des tests d\'intégration hors ligne. J\'en rêvais.'),
        R('disciplineNon', ['laxiste'], -0.01, 'Tablette, montre, frigo… on dit oui à tout ?'),
        R('autonomie', [4], -0.008, 'Fusionner seul sur la synchro ? Non merci.'),
        R('licences', ['aucune'], -0.008, 'Sans Copilot, le Kotlin redevient long.'),
      ],
    },
    jeremy: {
      replique: 'Honnêtement, je n\'ai jamais codé sans IA. C\'est comme demander si on sait écrire sans correcteur. Euh… on sait, hein ?',
      bulles: {
        code: ['L\'agent a fini. C\'est bon, je pense ?', 'Génial, ça marche ! Pourquoi ?', 'Prompt, Entrée, café. Prompt, Entrée.', 'J\'ai écrit une fonction tout seul. Bon, presque.', 'L\'agent a fini. Je relis. Je comprends ? Un peu.'],
        revue: ['Sophie m\'a posé une question. Gloups.', 'Je lis le code. Il est long.', 'Je ne sais pas quoi commenter. Alors j\'approuve ?', 'Karim m\'explique la synchro. Je prends des notes.'],
        incident: ['C\'est mon code ? C\'est mon code.', 'Je demande à l\'agent ce qui a cassé.'],
        fatigue: ['Trop de boissons énergisantes.', 'Même l\'agent a l\'air fatigué.', 'Trop de PR, pas assez de sommeil.', 'Même ma boisson énergisante est fatiguée.'],
        formation: ['J\'ai écrit une boucle à la main !', 'Oh. C\'est ça, une pile d\'appels.', 'J\'apprends le pourquoi. C\'est nouveau.'],
        attente: ['Personne ne relit ma PR de stagiaire.', 'J\'attends. Je fais une autre PR ?', 'Ma PR attend. J\'en ouvre une autre ?', 'Personne n\'a le temps. Je relis la mienne.'],
        conge: ['Congé. Je regarde des vidéos de code.', 'En vacances. Avec mon agent, au cas.'],
      },
      reactions: [
        R('autonomie', [3, 4], 0.01, 'L\'agent ouvre mes PR. Je suis 10x !'),
        R('pair', ['humain', 'rotation'], 0.012, 'Sophie m\'a fait écrire un test à la main. Wow.'),
        R('revuesExplication', ['majeures'], -0.008, 'Expliquer mon code sans l\'IA ? Euh. Demain ?'),
        R('formation', ['mentorat'], 0.01, 'Un mentor juste pour moi ? Je note tout.'),
        R('licences', ['pilote'], -0.005, 'Je ne suis pas dans le pilote. Je code comme en 2015, paraît-il.'),
        R('licences', ['tous'], 0.008, 'Ma licence ! Je recommence à vivre.'),
      ],
    },
    amira: {
      replique: 'Un site web qui plante, on le recharge. Un capteur qui envoie une mauvaise valeur, on ferme une usine. Alors oui, je vais relire chaque ligne que votre agent écrit.',
      bulles: {
        code: ['Chaque octet compte. Littéralement.', 'Pas d\'allocation dynamique. Jamais.', 'Je vérifie la fiche technique. Deux fois.', 'J\'écris le test du capteur avant le code.', 'Une interruption, trois lignes. Pas une de plus.'],
        revue: ['Débordement possible à la ligne 88.', 'L\'agent a inventé un registre. Non.', 'Ce délai bloque l\'interruption.', 'Un malloc dans le firmware. Non.', 'Ce délai bloque tout. Je le vois d\'ici.'],
        incident: ['Seuil erroné. Rappel des capteurs.', 'Je savais que ce cas limite viendrait.'],
        fatigue: ['Trop de PR, pas assez de café.', 'Tout relire, ça use.', 'Trop de firmware à relire en une semaine.', 'Je vois des registres partout.'],
        formation: ['Un agent pour du C embarqué ? Prudence.', 'Utile pour les tests. Pas pour le reste.'],
        attente: ['J\'attends. Mieux vaut tard que rappelé.', 'Ma PR attend. Le capteur patiente.', 'Ma PR attend. Le capteur peut attendre aussi.', 'En attente. Je relis la fiche technique.'],
        conge: ['Congé. Aucun capteur à moins de 10 km.', 'En vacances. J\'ai laissé des notes.'],
      },
      reactions: [
        R('autonomie', [3, 4], -0.015, 'Un agent qui pousse du firmware seul ? Non.'),
        R('autonomie', [1, 2], 0.005, 'Avec approbation à chaque étape. Je garde la main.'),
        R('reviseurs', [2], 0.008, 'Deux paires d\'yeux sur un capteur : un minimum.'),
        R('tests', ['mutation'], 0.008, 'Des mutants dans le firmware. Bonne idée.'),
        R('licences', ['tous'], -0.005, 'Copilot pour tout le monde, firmware compris ? Prudence.'),
      ],
    },
    luc: {
      replique: 'Je suis électronicien d\'abord. Le code, je l\'ai appris sur le tas, en parlant aux capteurs. L\'agent m\'aide à écrire le C. Moi, je lui explique pourquoi la broche 7 chauffe.',
      bulles: {
        code: ['Je parle au capteur. Il répond.', 'Broche 7, haut. Broche 8, bas.', 'Le C, c\'est de l\'électronique écrite.', 'Le capteur répond. J\'écris ce qu\'il dit.', 'J\'ajuste la broche, puis le code.'],
        revue: ['L\'agent a oublié la résistance de rappel.', 'Ça compile, mais ça chauffe.', 'Ça compile. Mais est-ce que ça chauffe ?', 'L\'agent ne sait pas ce qu\'est une masse flottante.'],
        incident: ['Ça sent le brûlé. Littéralement.', 'J\'apporte le fer à souder.'],
        fatigue: ['Fer à souder froid, tête chaude.', 'Trop de capteurs, pas assez de Luc.', 'Trop de code, pas assez de circuits.', 'Mon fer à souder a plus d\'énergie que moi.'],
        formation: ['J\'apprends le pourquoi du code !', 'Enfin une formation faite pour moi.'],
        attente: ['En attente. Je soude un prototype.', 'Ma PR attend. Mon circuit aussi.', 'Ma PR attend. Je soude en attendant.', 'En attente. Mon prototype clignote.'],
        conge: ['Congé. Je répare la radio de mon père.', 'En vacances. Sans multimètre. Ou presque.'],
      },
      reactions: [
        R('pair', ['humain'], 0.01, 'Amira m\'explique le C, je lui explique le circuit.'),
        R('formation', ['atelier', 'programme', 'mentorat'], 0.01, 'Une formation ! J\'ai pas étudié en informatique, moi.'),
        R('modele', ['economique'], -0.008, 'Le modèle économique croit qu\'un capteur, c\'est un booléen.'),
        R('observabilite', ['complete'], 0.008, 'La télémétrie de la télémétrie. J\'adore.'),
        R('licences', ['tous'], 0.005, 'Une licence ! Il va m\'aider avec le C.'),
      ],
    },
    hugo: {
      replique: 'Mes pipelines avalent deux millions de mesures par jour. Si l\'agent prend le mauvais seuil dans le mauvais fichier, il le propage partout, très vite, sans la moindre hésitation.',
      bulles: {
        code: ['Pandas. Encore Pandas.', 'Ce seuil vient d\'où, déjà ?', 'Pipeline de télémétrie, version 14.', 'J\'ajoute un test sur les unités. Celsius, oui.', 'Une colonne de plus dans le pipeline.'],
        revue: ['Trois fichiers de seuils. Lequel est bon ?', 'L\'agent a pris des degrés Fahrenheit.', 'Ce seuil vient de quel fichier ?', 'Fahrenheit. Encore. Je le savais.'],
        incident: ['Valeur aberrante propagée. Partout.', 'Le pipeline a avalé un mauvais seuil.'],
        fatigue: ['Deux millions de mesures. Je compte mal.', 'Mes courbes ondulent. Moi aussi.', 'Trop de mesures, pas assez de pauses.', 'Mes yeux voient des valeurs aberrantes.'],
        formation: ['Comment forcer l\'agent à citer ses sources ?', 'Je teste un nouvel outil. Pour voir.'],
        attente: ['Ma PR attend. Les mesures s\'empilent.', 'En attente. Je nettoie un jeu de données.', 'Ma PR attend. Deux millions de mesures, non.', 'En attente. Je trace une courbe.'],
        conge: ['Congé. Je mesure la pluie au chalet.', 'En vacances. Mes capteurs, non.'],
      },
      reactions: [
        R('catalogue', ['sources'], 0.015, 'Une source officielle pour les seuils !'),
        R('classification', ['auto'], 0.008, 'Classées automatiquement. Je vérifie quand même.'),
        R('politiqueRegistre', ['libre'], 0.006, 'Libre-service : je teste un MCP de données.'),
        R('sessionsMax', [2, 3], 0.006, 'Deux pipelines en parallèle. Raisonnable.'),
        R('licences', ['aucune'], -0.008, 'Plus de Copilot ? Mes pipelines vont s\'ennuyer.'),
      ],
    },
    chloe: {
      replique: 'J\'étais designer. L\'IA m\'a ouvert la porte du code et je ne la refermerai pas. Mais quand Patrick me demande pourquoi ça marche, je réponds encore « parce que l\'agent l\'a dit ».',
      bulles: {
        code: ['Je dessine, l\'agent intègre.', 'Ce bouton mérite trois pixels de plus.', 'Ça marche ! Je sais pas trop pourquoi.', 'Je dessine le composant, puis je l\'écris.', 'Ça marche ! Je note pourquoi, cette fois.'],
        revue: ['Le contraste est mauvais. Je le vois.', 'Patrick me demande pourquoi. Hmm.', 'Le bouton est beau. Le code, je demande à Patrick.', 'Ce contraste ne passe pas l\'accessibilité.'],
        incident: ['Le site est tout blanc. Normal ?', 'Le CSS a cassé la page d\'accueil.'],
        fatigue: ['Trop de maquettes, trop de PR.', 'Mes yeux voient en gris 50 %.', 'Trop de pixels aujourd\'hui.', 'Je vois flou. C\'est peut-être le CSS.'],
        formation: ['Je comprends enfin la cascade du CSS.', 'Oh, c\'est ça, un état React !'],
        attente: ['Ma PR attend. Je fais une maquette.', 'En attente. Je choisis une police.', 'Ma PR attend. J\'ajuste une marge.', 'En attente. Je fais une maquette de plus.'],
        conge: ['Congé. Je dessine pour le plaisir.', 'En vacances. Sans Figma. Presque.'],
      },
      reactions: [
        R('autonomie', [2, 3], 0.008, 'L\'agent code, je dessine. Le rêve.'),
        R('pair', ['ia'], 0.01, 'Moi et l\'agent, en duo. On s\'entend bien.'),
        R('formation', ['programme', 'mentorat'], 0.012, 'J\'apprends enfin le pourquoi du code.'),
        R('lint', ['verrouille'], -0.005, 'Le lint refuse mon CSS. Mon CSS est beau.'),
        R('licences', ['aucune'], -0.012, 'Sans Copilot, je redeviens designer à temps plein.'),
      ],
    },
    patrick: {
      replique: 'J\'ai 14 sites WordPress, dont trois que j\'aimerais oublier. L\'agent écrit une extension en dix minutes. Le problème, c\'est qu\'il faut ensuite la maintenir pendant dix ans.',
      bulles: {
        code: ['Un 15e site ? Je soupire.', 'Mise à jour d\'extension. Prière.', 'Ce thème date de 2015. Courage.', 'Une extension de moins. Victoire.', 'Je mets à jour le site 9. Prière.'],
        revue: ['L\'agent a ajouté une 38e extension.', 'Simple, efficace. Approuvé.', 'Une 39e extension ? Justifie.', 'Simple et lisible. Je signe.'],
        incident: ['Le site de la campagne est à terre.', 'Extension mise à jour, site par terre.'],
        fatigue: ['Quatorze sites. Une seule tête.', 'Je suis l\'extension de trop.', 'Quatorze sites, une seule fin de semaine.', 'Trop de mises à jour cette semaine.'],
        formation: ['Pratique. Je garde ce qui est concret.', 'L\'agent connaît WordPress ? Il est brave.'],
        attente: ['En attente. Je mets un site à jour.', 'Ma PR attend. Comme mes 14 sites.', 'Ma PR attend. Mes 14 sites aussi.', 'En attente. Je fais une sauvegarde, au cas.'],
        conge: ['Congé. Aucun WordPress en vue.', 'En vacances. Mon site perso est en panne.'],
      },
      reactions: [
        R('ci', ['complet'], 0.008, 'Déployer sans FTP. Une première en 14 sites.'),
        R('disciplineNon', ['ferme'], 0.01, 'Un 15e site ? Non. Merci, la direction.'),
        R('autonomie', [4], -0.006, 'Un agent qui fusionne seul dans WordPress. Aïe.'),
        R('bacASable', ['conteneur', 'filtre'], 0.006, 'Les extensions douteuses restent dans la boîte.'),
        R('licences', ['tous'], 0.005, 'Une licence. Il connaît WordPress mieux que moi, paraît-il.'),
      ],
    },
    laurence: {
      replique: 'Quatre écrans, quatre sessions, quatre agents. Pendant que l\'un écrit, l\'autre teste, le troisième déploie et le quatrième… je ne sais plus, mais il est très occupé.',
      bulles: {
        code: ['Session 1 : fini. Session 2 : presque.', 'Quatre agents, quatre microsites !', 'Je change d\'écran. Encore.', 'Écran 1 : agent. Écran 2 : agent. Écran 3 : café.', 'Quatre microsites avant midi.'],
        revue: ['Je relis quatre PR en même temps.', 'Laquelle de mes sessions a fait ça ?', 'Laquelle de mes sessions a écrit ça ?', 'Je relis vite. Trop vite ?'],
        incident: ['Quel agent a déployé ça ?', 'Écran 3 en rouge. Écran 4 aussi.'],
        fatigue: ['Je ne sais plus quel écran est le mien.', 'Quatre sessions, zéro neurone.', 'Une pause ? C\'est quoi, une pause ?', 'J\'ai oublié de dîner. Deux fois.', 'Mes agents vont bien. Moi, moins.'],
        formation: ['Une seule session pendant la formation ?', 'J\'apprends à déléguer. À moi-même.'],
        attente: ['Quatre PR en attente. Quatre !', 'En attendant, j\'ouvre une cinquième session.', 'Quatre PR en attente. Mes agents s\'impatientent.', 'En attente. J\'ouvre une autre session.'],
        conge: ['Congé forcé. Mes écrans s\'ennuient.', 'En vacances. Un seul écran : le ciel.'],
      },
      reactions: [
        R('sessionsMax', [4, 5, 6], 0.015, 'Enfin ! Mes quatre écrans servent à quelque chose.'),
        R('sessionsMax', [1], -0.012, 'Une session ? Mes trois autres écrans s\'ennuient.'),
        R('plafondJetons', ['strict'], -0.01, 'Coupée en plein élan. Merci, le plafond.'),
        R('licences', ['pilote'], -0.005, 'Pas dans le pilote. Mes quatre écrans pleurent.'),
        R('licences', ['tous'], 0.008, 'Ma licence ! Mes quatre écrans se réveillent.'),
      ],
    },
    mathieu: {
      replique: 'Je suis 10x. Peut-être 20x, maintenant, avec six agents. Le seul problème, c\'est que le reste de l\'équipe relit encore à 1x.',
      bulles: {
        code: ['4 000 lignes. Petite journée.', 'Ship it. Ship it. Ship it.', 'Six agents, un casque, zéro réunion.', 'Une PR, une idée, 3 000 lignes.', 'Mes agents codent, je dirige.'],
        revue: ['LGTM. Suivant.', 'Je relis en diagonale. C\'est un art.', 'Ça a l\'air bon. LGTM.', 'Je relis le titre. C\'est un bon titre.'],
        incident: ['Ça marchait sur ma machine.', 'Pas ma faute. Mon agent, peut-être.'],
        fatigue: ['Même un 10x doit dormir ?', 'Je suis à 3x. Mauvaise semaine.', 'Même mes agents ont besoin d\'une pause.', 'Six sessions, un cerveau. Le calcul est serré.'],
        formation: ['Une formation ? Je pourrais la donner.', 'Les petites PR. Oui. J\'ai compris. Bof.'],
        attente: ['Ma PR attend. Nadia, s\'il te plaît.', 'Pourquoi personne ne relit 4 000 lignes ?', 'Ma PR attend Nadia. Encore.', 'Personne ne relit à ma vitesse.'],
        conge: ['Congé. J\'ai lancé un agent avant.', 'En vacances. Mes agents, non.'],
      },
      reactions: [
        R('reviseurs', [2], -0.02, 'Deux réviseurs ? Ma PR prendra sa retraite avant Ginette.'),
        R('taillePR', [400, 200], -0.015, '400 lignes ? Mes imports en font 300.'),
        R('approbation', ['auto'], 0.01, 'Revue par l\'IA : enfin quelqu\'un qui me suit.'),
        R('autonomie', [4], 0.012, 'Fusion automatique. Ship it !'),
        R('licences', ['aucune'], -0.015, 'Sans Copilot, je suis à 1x. Comme vous autres.'),
        R('licences', ['pilote'], 0.005, 'Pilote. Évidemment.'),
      ],
    },
    nadia: {
      replique: 'Tout le monde veut ma revue. Je suis contente qu\'on me fasse confiance. Je serais encore plus contente si les PR faisaient moins de 4 000 lignes.',
      bulles: {
        code: ['Je code entre deux revues. Rarement.', 'Une heure pour moi. Enfin.', 'Dix minutes de code entre deux revues.', 'Ma propre PR. Petite, elle.'],
        revue: ['PR 14 de 23. Courage.', 'Ligne 2 847. Je suis encore là.', 'Je relis pour vrai. C\'est rare.', 'Encore une PR de Mathieu.', 'Ligne 1 200. Je relis encore.', 'Je pose une question. J\'attends la réponse.'],
        incident: ['J\'avais commenté ce risque. Juste là.', 'Approuvé sans moi ? Je vois.'],
        fatigue: ['La pile monte. Moi, je descends.', 'Je vois des diffs en dormant.', 'Un LGTM de plus et je craque.', 'La pile me regarde.', 'Je relis la même ligne pour la troisième fois.'],
        formation: ['Comment relire du code d\'agent ? Enfin.', 'Une demi-journée sans revue. Merci.'],
        attente: ['Ma propre PR attend. Ironique.', 'Qui relit la réviseuse ?', 'Ma PR attend. Qui relit la réviseuse ?', 'En attente. Je relis celle des autres.'],
        conge: ['Congé. La pile m\'attendra.', 'En vacances. Zéro diff.'],
      },
      reactions: [
        R('autonomie', [3, 4], -0.012, 'Plus d\'agents, plus de PR. Et toujours une seule Nadia.'),
        R('taillePR', ['libre'], -0.01, 'Taille libre. Ligne 3 912. À l\'aide.'),
        R('taillePR', [400, 200], 0.012, 'Des PR de 200 lignes. Je relis pour vrai.'),
        R('reviseurs', [2], -0.008, 'Deux réviseurs : je suis toujours l\'une des deux.'),
        R('licences', ['tous'], -0.005, 'Copilot pour tous : plus de code, et toujours une seule Nadia.'),
      ],
    },
    sebastien: {
      replique: 'Donnez à un agent nos données privées, du contenu venu d\'Internet et le droit d\'envoyer un courriel : c\'est la triade létale. Je ne suis pas contre l\'IA. Je suis contre les trois en même temps.',
      bulles: {
        code: ['Je ferme un port. Puis un autre.', 'Moindre privilège, toujours.', 'OAuth, portée minimale. Point.', 'Je retire un droit. Personne ne le remarquera.', 'Un secret dans le code ? Plus maintenant.'],
        revue: ['Ce jeton a beaucoup trop de droits.', 'D\'où vient ce paquet npm ?', 'Une sortie externe ? Justifie.', 'Ce MCP demande trop de portées.', 'Une requête vers l\'extérieur ici ? Pourquoi ?'],
        incident: ['Injection de prompt. Classique.', 'Qui a installé ce MCP ? Montrez-moi.', 'J\'isole l\'îlot. Tout de suite.'],
        fatigue: ['Trois voyants rouges dans ma tête.', 'Trop de MCP, pas assez de moi.', 'Trop de voyants allumés dans ma tête.', 'Je vois des injections dans mon courriel.'],
        formation: ['Je donne l\'atelier sur l\'injection.', 'Rappel : un courriel n\'est pas un ordre.'],
        attente: ['Ma PR attend. La menace, non.', 'En attente. Je lis des avis de vulnérabilité.', 'Ma PR attend. Les attaquants, non.', 'En attente. Je lis le journal de la passerelle.'],
        conge: ['Congé. Mots de passe changés avant.', 'En vacances. Avec double authentification.'],
      },
      reactions: [
        R('bacASable', ['conteneur', 'filtre'], 0.015, 'Un bac à sable. L\'injection reste dans la boîte.'),
        R('passerelleMCP', [true], 0.015, 'Chaque appel d\'outil journalisé. Je souris.'),
        R('approbationSorties', [true], 0.01, 'Une branche de la triade coupée. Bien.'),
        R('politiqueRegistre', ['libre'], -0.012, 'MCP en libre-service. Je prépare le rapport d\'incident.'),
        R('licences', ['tous'], -0.005, 'Vingt et une licences, vingt et une façons de coller un secret dans une demande.'),
      ],
    },
  };

  LTP.donnees.devs.forEach(function (d) {
    var c = CONTENU[d.id];
    if (!c) return;
    if (d.replique == null) d.replique = c.replique;
    if (!d.bulles) d.bulles = c.bulles;
    if (!d.reactions) d.reactions = c.reactions;
  });

  LTP.donnees.devs.forEach(function (d) {
    if (d.replique === undefined) d.replique = null;
    if (!d.bulles) d.bulles = {};
    if (!d.reactions) d.reactions = [];
  });

  // --- Personnages secondaires ---
  LTP.donnees.personnages = {
    ginette:   { id: 'ginette',   nom: 'Ginette',       role: 'Technicienne senior, TI',            teinte: 285,
                 tagline: '27 ans de service, trois « révolutions » et zéro panique.',
                 replique: 'Trois révolutions en 27 ans : le client-serveur, le web, le nuage. Chaque fois, le code allait s\'écrire tout seul. Chaque fois, il a fallu quelqu\'un pour le comprendre.' },
    kevin:     { id: 'kevin',     nom: 'Kevin',         role: 'Marketing, développeur citoyen',      teinte: 25,
                 tagline: 'Une carte de crédit corporative et un abonnement à tout.',
                 replique: 'J\'ai fait une application de suivi des campagnes en une fin de semaine, avec un agent et la carte corpo. Elle est dans mon OneDrive. C\'est sécuritaire : il y a un mot de passe.' },
    regis:     { id: 'regis',     nom: 'Régis',         role: 'VP Finances',                        teinte: 210,
                 tagline: 'Veut un chiffre. Un seul. Et petit.',
                 replique: 'L\'IA, je n\'ai rien contre. Je veux juste savoir ce qu\'elle coûte par point livré, en dollars, et sans le mot « stratégique ».' },
    berube:    { id: 'berube',    nom: 'Me Bérubé',     role: 'Contentieux et conformité',          teinte: 150,
                 tagline: 'Ne se déplace jamais sans un PDF de 300 pages. Imprimé.',
                 replique: 'Un agent qui lit un courriel peut obéir à ce courriel. Je l\'ai lu dans un rapport de sécurité. Je l\'ai imprimé. Trois fois.' },
    lavigueur: { id: 'lavigueur', nom: 'Mme Lavigueur', role: 'Présidente-directrice générale',     teinte: 340,
                 tagline: 'Pose une question. Attend une réponse. Pas un acronyme.',
                 replique: 'Je ne veux pas savoir combien de lignes de code vous avez écrites. Je veux savoir ce qui marche mieux qu\'avant, et ce qui risque de casser.' },
  };
})(globalThis.LTP = globalThis.LTP || {});
