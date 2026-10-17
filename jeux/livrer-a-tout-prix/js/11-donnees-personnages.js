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
    legacy:          { libelle: 'Gardien du legacy',     init: { comprehension: 0.15, competence: 0.05 }, mods: {} },
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
    natifIA:         { libelle: 'Native de l\'IA',       init: { adhesion: 0.3, competenceIA: 0.25, competence: -0.05 }, mods: { verbosite: 1.2 } },
    firmware:        { libelle: 'Firmware',              init: { rigueur: 0.1 }, mods: {} },
    prudent:         { libelle: 'Prudence',              init: { rigueur: 0.15, adhesion: -0.1 }, mods: { tauxDefaut: 0.85 } },
    electronicien:   { libelle: 'Électronicien',         init: {}, mods: {} },
    designer:        { libelle: 'Designer',              init: { competence: -0.05, adhesion: 0.15 }, mods: {} },
    wordpress:       { libelle: 'WordPress',             init: {}, mods: {} },
    multiClauding:   { libelle: 'Multi-Clauding',        init: { adhesion: 0.3 }, mods: { sessions: 2, fatigue: 1.25 } },
    dixX:            { libelle: '« 10x »',               init: { competence: 0.05, rigueur: -0.2, adhesion: 0.3 }, mods: { sessions: 2, taillePR: 3, verbosite: 1.4 } },
    shipIt:          { libelle: 'Ship it',               init: { rigueur: -0.15 }, mods: { debit: 0.1, tauxDefaut: 1.25 } },
    reviseur:        { libelle: 'Réviseuse de référence', init: { rigueur: 0.15 }, mods: {} },
    securite:        { libelle: 'Sécurité',              init: { rigueur: 0.1 }, mods: { tauxDefaut: 0.9 } },
    mesure:          { libelle: 'Mesure tout',           init: { rigueur: 0.05 }, mods: {} },
  };

  // --- Développeurs ---
  // niveau : stagiaire · junior · inter · senior
  // accessoire : objet signature sur le bureau (dessiné par 41-ui-avatars.js)
  // apparence : optionnelle ; sinon dérivée de l'id de façon déterministe
  LTP.donnees.devs = [
    { id: 'raymond',   nom: 'Raymond',   ilot: 'appro',       niveau: 'senior',    traits: ['legacy', 'sceptique'],         accessoire: 'terminal-vert',
      tagline: '31 ans sur l\'AS/400. « L\'IA ? J\'ai vu passer le 4GL aussi. »' },
    { id: 'lucie',     nom: 'Lucie',     ilot: 'appro',       niveau: 'inter',     traits: ['donnees', 'rigoureux'],        accessoire: 'classeurs',
      tagline: 'Connaît par cœur les trois listes de fournisseurs, et laquelle est la bonne.' },
    { id: 'yannick',   nom: 'Yannick',   ilot: 'appro',       niveau: 'inter',     traits: ['curieux', 'integrateur'],      accessoire: 'cables',
      tagline: 'Branche tout sur tout. Premier à installer un MCP trouvé sur un forum.' },

    { id: 'benoit',    nom: 'Benoît',    ilot: 'operations',  niveau: 'senior',    traits: ['garde', 'stoique'],            accessoire: 'tasse',
      tagline: 'De garde la nuit depuis 2009. Son téléphone vibre, il soupire.' },
    { id: 'marieeve',  nom: 'Marie-Ève', ilot: 'operations',  niveau: 'senior',    traits: ['sre', 'mesure'],               accessoire: 'tableau-bord',
      tagline: 'Ne croit que ce qui a un tableau de bord.' },
    { id: 'tommy',     nom: 'Tommy',     ilot: 'operations',  niveau: 'junior',    traits: ['apprenant'],                   accessoire: 'carnet',
      tagline: 'Apprend vite, mais seulement ce qu\'on lui laisse faire.' },

    { id: 'steve',     nom: 'Steve',     ilot: 'bureautique', niveau: 'inter',     traits: ['automatiseur', 'enthousiaste'], accessoire: 'organigramme',
      tagline: 'Roi du Power Automate, voisin de bureau de Kevin.' },
    { id: 'diane',     nom: 'Diane',     ilot: 'bureautique', niveau: 'senior',    traits: ['vba', 'refractaire'],          accessoire: 'plante',
      tagline: '58 ans, des macros Excel qui font tourner la paie. Ne veut rien savoir.' },
    { id: 'olivier',   nom: 'Olivier',   ilot: 'bureautique', niveau: 'inter',     traits: ['contractuel', 'enthousiaste'], accessoire: 'sac-a-dos',
      tagline: 'Contractuel. Part dans six mois, avec ce qu\'il sait.' },

    { id: 'sophie',    nom: 'Sophie',    ilot: 'inspections', niveau: 'senior',    traits: ['tdd', 'rigoureux'],            accessoire: 'cactus',
      tagline: 'Pas de code sans test rouge d\'abord. Lente au T1, imbattable au T3.' },
    { id: 'karim',     nom: 'Karim',     ilot: 'inspections', niveau: 'inter',     traits: ['mobile', 'pragmatique'],       accessoire: 'telephones',
      tagline: 'L\'application terrain hors ligne, c\'est lui.' },
    { id: 'jeremy',    nom: 'Jérémy',    ilot: 'inspections', niveau: 'stagiaire', traits: ['natifIA', 'apprenant'],        accessoire: 'boisson-energie',
      tagline: 'N\'a jamais codé sans IA. Brillant ou perdu selon vos choix.' },

    { id: 'amira',     nom: 'Amira',     ilot: 'iot',         niveau: 'senior',    traits: ['firmware', 'prudent'],         accessoire: 'oscilloscope',
      tagline: 'Sait qu\'un capteur ne pardonne pas.' },
    { id: 'luc',       nom: 'Luc',       ilot: 'iot',         niveau: 'inter',     traits: ['electronicien'],               accessoire: 'fer-a-souder',
      tagline: 'Électronicien devenu développeur. Parle aux capteurs.' },
    { id: 'hugo',      nom: 'Hugo',      ilot: 'iot',         niveau: 'inter',     traits: ['donnees', 'curieux'],          accessoire: 'graphique',
      tagline: 'Pipelines Python de télémétrie.' },

    { id: 'chloe',     nom: 'Chloé',     ilot: 'web',         niveau: 'inter',     traits: ['designer', 'natifIA'],         accessoire: 'tablette-dessin',
      tagline: 'Designer devenue développeuse grâce à l\'IA.' },
    { id: 'patrick',   nom: 'Patrick',   ilot: 'web',         niveau: 'senior',    traits: ['wordpress', 'pragmatique'],    accessoire: 'mug-wp',
      tagline: '14 sites WordPress, dont 3 qu\'il aimerait oublier.' },
    { id: 'laurence',  nom: 'Laurence',  ilot: 'web',         niveau: 'junior',    traits: ['multiClauding', 'enthousiaste'], accessoire: 'ecrans',
      tagline: 'Quatre écrans, quatre sessions, zéro pause.' },

    { id: 'mathieu',   nom: 'Mathieu',   ilot: 'portail',     niveau: 'senior',    traits: ['dixX', 'shipIt'],              accessoire: 'casque',
      tagline: '« 10x » autoproclamé. Des PR de 4 000 lignes.' },
    { id: 'nadia',     nom: 'Nadia',     ilot: 'portail',     niveau: 'senior',    traits: ['reviseur', 'rigoureux'],       accessoire: 'pile-papiers',
      tagline: 'Tout le monde veut sa revue. Elle est le goulot incarné.' },
    { id: 'sebastien', nom: 'Sébastien', ilot: 'portail',     niveau: 'senior',    traits: ['securite', 'mesure'],          accessoire: 'cadenas',
      tagline: 'Architecte sécurité. Voit la triade létale avant tout le monde.' },
  ];

  // Champs de contenu complétés par l'agent de contenu (D1) :
  //   replique : phrase d'accueil dans la fiche
  //   bulles : { code: [], revue: [], incident: [], fatigue: [], formation: [], attente: [], conge: [] }
  //   reactions : [{ levier, valeurs: [...], moral: +0,01 (par semaine où le levier effectif de son îlot est dans valeurs), bulle }]
  LTP.donnees.devs.forEach(function (d) {
    if (d.replique === undefined) d.replique = null;
    if (!d.bulles) d.bulles = {};
    if (!d.reactions) d.reactions = [];
  });

  // --- Personnages secondaires ---
  LTP.donnees.personnages = {
    ginette:   { id: 'ginette',   nom: 'Ginette',       role: 'Technicienne senior, TI',            teinte: 285,
                 tagline: '27 ans de service, trois « révolutions » et zéro panique.' },
    kevin:     { id: 'kevin',     nom: 'Kevin',         role: 'Marketing, développeur du dimanche', teinte: 25,
                 tagline: 'Carte de crédit corporative et un abonnement à tout.' },
    regis:     { id: 'regis',     nom: 'Régis',         role: 'VP Finances',                        teinte: 210,
                 tagline: 'Veut un chiffre. Un seul. Et petit.' },
    berube:    { id: 'berube',    nom: 'Me Bérubé',     role: 'Contentieux et conformité',          teinte: 150,
                 tagline: 'Ne se déplace jamais sans un PDF de 300 pages.' },
    lavigueur: { id: 'lavigueur', nom: 'Mme Lavigueur', role: 'Présidente-directrice générale',     teinte: 340,
                 tagline: 'Pose une question. Attend une réponse. Pas un acronyme.' },
  };
})(globalThis.LTP = globalThis.LTP || {});
