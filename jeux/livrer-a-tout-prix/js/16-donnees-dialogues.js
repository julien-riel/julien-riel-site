/* 16-donnees-dialogues.js */
/* Tutoriel, indices de Ginette et bulles génériques. Contenu écrit par l'agent D1.
   LTP.donnees.tutoriel : [{ id, region: 'openspace' | 'panneau' | 'tableau' | 'hud' | 'journal', personnage, texte, action? }]
   LTP.donnees.indices.leviers : { [levierId]: { [valeur]: texte } }   // commentaire de Ginette quand on choisit cette valeur
   LTP.donnees.indices.trimestres : { [n]: texte }                       // ouverture de chaque trimestre
   LTP.donnees.bulles : { code: [], revue: [], incident: [], fatigue: [], formation: [], attente: [], conge: [] }
                        // bulles génériques, utilisées quand un développeur n'a pas de bulle propre pour son état
   Les clés de valeur des indices sont les valeurs du levier converties en texte : 0, 400, 'true', 'mutation'… */
(function (LTP) {
  'use strict';

  // --- Tutoriel : Ginette fait visiter, une région à la fois ---
  LTP.donnees.tutoriel = [
    { id: 'bienvenue', region: 'hud', personnage: 'ginette',
      texte: 'Bienvenue au GPU ! Toi, tu diriges 21 développeurs pendant un an, le temps qu\'ils apprennent à travailler avec des agents. En haut : la semaine, le budget de l\'IA, les jetons et la confiance de la direction. Sous 15 de confiance, l\'aventure s\'arrête.' },
    { id: 'openspace', region: 'openspace', personnage: 'ginette',
      texte: 'Voici l\'open space : sept îlots, un par ligne d\'affaires, trois personnes chacun. Clique sur quelqu\'un pour ouvrir sa fiche, ou sur un îlot pour voir ses réglages et ses outils.' },
    { id: 'signaux', region: 'openspace', personnage: 'ginette',
      texte: 'Apprends à lire les bureaux. Les écrans allumés, ce sont des sessions d\'agent. La pile de papiers, des PR qui attendent une revue. La flamme, un incident. Et le brouillard gris, c\'est la dette cognitive : du code que plus personne ne comprend.' },
    { id: 'leviers', region: 'panneau', personnage: 'ginette',
      texte: 'À droite, tes leviers : pratiques, utilisation de l\'IA, produit. D\'autres onglets se débloquent à chaque trimestre. Ceux marqués d\'un losange peuvent se régler îlot par îlot : le firmware des capteurs ne mérite pas le même traitement que le site vitrine.' },
    { id: 'delais', region: 'panneau', personnage: 'ginette',
      texte: 'Certains changements prennent du temps : une formation rapporte après quatre semaines, un MCP maison se développe pendant six. Et quand un choix me fait lever un sourcil, je te le dis en une ligne.' },
    { id: 'tableau', region: 'tableau', personnage: 'ginette',
      texte: 'Le tableau de bord montre les tendances des 13 dernières semaines. Une tuile grisée « non mesuré », c\'est quelque chose que tu ne vois pas encore. Ça ne veut pas dire que ça n\'existe pas. Ça veut dire que tu vas l\'apprendre plus tard, souvent à 3 h du matin.' },
    { id: 'semaines', region: 'hud', personnage: 'ginette',
      texte: 'Le bouton de lecture fait défiler les semaines, à la vitesse de ton choix. Tu peux mettre en pause et changer tes leviers n\'importe quand. Avancer d\'une semaine à la fois, c\'est bien aussi, surtout au début.' },
    { id: 'journal', region: 'journal', personnage: 'ginette',
      texte: 'Le journal raconte ce qui se passe. Quand une carte d\'événement apparaît, tout s\'arrête jusqu\'à ta décision. À la fin du trimestre, le comité te pose des questions sur les vrais chiffres. Mon conseil : dis la vérité, même quand elle n\'est pas belle. Ils ont le tableau de bord, eux aussi.' },
  ];

  // --- Indices de Ginette sur les valeurs de leviers ---
  LTP.donnees.indices = {
    leviers: {
      pair: {
        humain: 'Deux humains, un clavier : ça va moins vite, mais les deux comprennent. Pour les juniors, c\'est de l\'or.',
        ia: 'Humain avec l\'IA, ça va vite. Vérifie juste que l\'humain tient encore le volant.',
        rotation: 'La rotation, c\'est un bon compromis : une semaine on apprend, une semaine on livre.',
      },
      reviseurs: {
        0: 'Aucun réviseur ? Le code part en production tel qu\'il sort de l\'agent. Courageux.',
        2: 'Deux réviseurs, c\'est solide. Regarde quand même la pile sur le bureau de Nadia dans deux semaines.',
      },
      approbation: {
        auto: 'L\'IA qui approuve le code de l\'IA. Mêmes angles morts, même sourire.',
        proprietaire: 'Le propriétaire du code a le dernier mot. Bonne idée, tant qu\'il a le temps de lire.',
      },
      taillePR: {
        libre: 'Taille libre ? Mathieu te remercie. Nadia, un peu moins.',
        400: '400 lignes : à peu près la limite où un humain lit encore vraiment.',
        200: '200 lignes, c\'est très lisible. Surveille seulement que ça ne devienne pas 40 PR par jour.',
      },
      ci: {
        aucune: 'Sans CI, chaque défaut se découvre en production. C\'est une façon d\'apprendre, disons.',
        tests: 'Compilation et tests à chaque PR : les agents commencent à avoir des rails.',
        complet: 'Pipeline complet : les déploiements deviennent ennuyants. C\'est exactement ce qu\'on veut.',
      },
      lint: {
        aucun: 'Pas de lint : les revues vont parler de virgules au lieu de parler de logique.',
        strict: 'Strict, mais pas verrouillé : un agent pressé peut désactiver une règle pour faire passer la CI. Je dis ça de même.',
        verrouille: 'Verrouillé : l\'agent ne peut plus faire taire le lint pour passer. Il doit corriger son code.',
      },
      tests: {
        aucun: 'Aucun test, avec des agents ? Tu leur enlèves les rails et tu montes la vitesse.',
        unitaires: 'Les tests unitaires écrits par l\'agent montent la couverture vite. La couverture affichée, en tout cas.',
        mutation: 'Les tests de mutation modifient le code exprès pour voir si les tests s\'en aperçoivent. Les tests qui ne vérifient rien ne s\'en remettent pas.',
      },
      revuesExplication: {
        mensuelles: 'Une fois par mois, chacun explique son code sans l\'IA. Tu vas apprendre des choses. Eux aussi.',
        majeures: 'À chaque PR majeure : ça coûte du temps, mais tu sauras enfin qui comprend quoi.',
      },
      autonomie: {
        0: 'De l\'autocomplétion seulement. Prudent. Raymond approuve, Laurence boude.',
        2: 'Agent avec approbation à chaque étape : un bon niveau pour du code critique.',
        3: 'L\'agent ouvre ses PR tout seul. Ça va écrire beaucoup. Reste à savoir qui va relire.',
        4: 'Il fusionne seul ? Sur un site vitrine, pourquoi pas. Sur le firmware des capteurs, je retiens mon souffle.',
      },
      sessionsMax: {
        1: 'Une session à la fois : on comprend ce qu\'on fait. On en fait moins, par exemple.',
        3: 'Trois sessions : au-delà, chaque agent de plus rapporte moins et fatigue plus.',
        6: 'Six sessions par personne. Personne ne surveille six agents à la fois. Personne.',
      },
      modele: {
        economique: 'Le modèle économique coûte peu par jeton, mais il invente parfois des noms de paquets. Un jour, quelqu\'un publiera un paquet malveillant sous ce nom-là.',
        haut: 'Le haut de gamme se trompe moins. Regarde le coût par point livré, pas le prix du jeton.',
        local: 'Le modèle local, sur notre serveur GPU : les données restent ici et les jetons sont presque gratuits. Un peu moins brillant, mais il ne parle à personne.',
      },
      modeleParTache: {
        true: 'Un petit modèle pour le lint et la documentation, un gros pour l\'architecture. Régis va t\'aimer.',
      },
      formation: {
        atelier: 'Un atelier, c\'est un début. Dans un mois, on verra ce qu\'il en reste.',
        programme: 'Un programme complet : cher tout de suite, rentable dans un mois. Et les gens deviennent plus méfiants envers les injections.',
        mentorat: 'Le mentorat, c\'est la seule chose qui fasse bouger Diane. Je le sais : c\'est moi qui vais la mentorer.',
      },
      plafondJetons: {
        souple: 'Une alerte avant la coupure : tu vois venir les boucles d\'agent sans bloquer tout le monde.',
        strict: 'Strict : les boucles d\'agent s\'arrêtent net. Le travail aussi, des fois, un vendredi à 16 h.',
      },
      disciplineNon: {
        laxiste: 'Laxiste ? Quand le code a l\'air gratuit, les demandes deviennent infinies. Le volume monte, la valeur, pas tant.',
        ferme: 'Ferme : les directions vont grogner, mais le code que tu n\'écris pas, tu n\'auras pas à le maintenir.',
      },
      specs: {
        non: 'Pas de spec : l\'agent va faire exactement ce qu\'on lui demande. C\'est justement ça, le problème.',
        obligatoires: 'Specs obligatoires : départ plus lent, beaucoup moins de reprise.',
      },
      catalogue: {
        inventaire: 'Un inventaire, c\'est savoir ce qui existe. Il reste à savoir quelle source est la bonne.',
        sources: 'Un catalogue avec les sources officielles. Lucie va pleurer de joie, et les agents vont enfin prendre la bonne liste.',
      },
      classification: {
        non: 'Sans classification, tu ne sais pas quelles données sortent. Me Bérubé, lui, va vouloir le savoir.',
        auto: 'La classification automatique va vite et large. Fais quand même vérifier les cas limites par un humain.',
      },
      bacASable: {
        aucun: 'Pas de bac à sable : un paquet malveillant installé par l\'agent voit tout ce que la machine voit.',
        conteneur: 'Un conteneur, c\'est bien : ce qui casse reste dedans. Mais il peut encore parler à Internet.',
        filtre: 'Conteneur et réseau filtré : même si l\'agent se fait berner, il ne peut rien envoyer dehors.',
      },
      accesProd: {
        oui: 'Les agents écrivent en production. Un jour, quelqu\'un va découvrir l\'UPDATE sans WHERE. Ce sera un mardi.',
        lecture: 'Lecture seule : l\'agent diagnostique vite, et il ne peut rien casser.',
        non: 'Aucun accès : rien ne casse, mais les diagnostics de nuit vont prendre plus de temps. Benoît le sentira.',
      },
      approbationSorties: {
        true: 'Un humain confirme chaque sortie externe : la troisième branche de la triade est coupée.',
      },
      politiqueRegistre: {
        libre: 'Libre-service : Yannick a déjà installé trois MCP trouvés sur un forum. Je les ai comptés.',
        listeBlanche: 'Une liste blanche : moins de surprises. Mais une carte approuvée sans responsable vieillit toute seule.',
        proprietaires: 'Chaque carte a un propriétaire : les orphelines deviennent visibles, et quelqu\'un répond de chacune.',
      },
      passerelleMCP: {
        true: 'Tous les appels d\'outils passent par un point journalisé. Si un MCP change sa description en douce, on le saura.',
      },
      politiqueNonTI: {
        interdire: 'Interdire ? Kevin va continuer. Il va juste arrêter de nous le dire.',
        encadrer: 'Encadrer : des outils approuvés et quelqu\'un pour accompagner. Kevin a un vrai besoin. Il lui manque juste des garde-fous.',
      },
      observabilite: {
        base: 'De l\'observabilité de base : le MTTR devient un vrai chiffre. C\'est déjà ça.',
        complete: 'Observabilité complète : tu vas voir ce qui était caché depuis le début. Prépare-toi, ce n\'est pas toujours beau.',
      },
      embauche: {
        gel: 'Geler l\'embauche des juniors ? Dans cinq ans, qui sera senior ? Moi, je serai au chalet.',
        releve: 'Un programme de relève : le meilleur investissement que personne ne verra au tableau de bord cette année.',
      },
      objectif: {
        pr: 'Un objectif en nombre de PR ? Elles vont rapetisser par magie. La valeur, elle, ne bougera pas.',
        delai: 'Un objectif de délai : la revue va accélérer. En lisant moins.',
        satisfaction: 'La satisfaction comme cible : tout le monde va dire oui à tout. Le backlog va adorer.',
        incidents: 'Un objectif d\'incidents : ils vont baisser. Sur papier. On va les appeler des « demandes ».',
      },
    },

    // --- Ouverture des trimestres ---
    trimestres: {
      1: 'Premier trimestre : l\'euphorie. Tout le monde veut essayer les agents. Tu as les pratiques, l\'utilisation de l\'IA et le produit. Mon conseil : regarde où s\'empilent les PR.',
      2: 'Deuxième trimestre : la facture. Régis a reçu le relevé des jetons. Tu débloques le modèle choisi selon la tâche, la formation, le plafond de jetons et le registre des skills et des MCP.',
      3: 'Troisième trimestre : les fuites. Les agents lisent maintenant des courriels, des factures et des pages web. Tu débloques la gouvernance des données, le bac à sable, l\'accès à la prod, la passerelle MCP et la politique hors TI.',
      4: 'Dernier trimestre : la maturité. Tu débloques l\'observabilité, la relève et les objectifs de la direction. Et j\'ai quelque chose à t\'annoncer. Pas tout de suite. Bientôt.',
    },
  };

  // --- Bulles génériques (cinq par état) ---
  LTP.donnees.bulles = {
    code: ['L\'agent propose. Je dispose.', 'Encore une petite fonction.', 'Ça compile. Bon signe ?', 'Je relis ce que l\'agent a écrit.', 'Une ligne à la fois.'],
    revue: ['Je relis. Pour vrai.', 'Ce diff est long.', 'Qui a écrit ça ? Ah, l\'agent.', 'Un commentaire, deux questions.', 'Approuvé… presque.'],
    incident: ['Ça brûle. Café, puis terminal.', 'Qui comprend ce code ?', 'Les journaux, vite.', 'Ça marchait hier.', 'Je rétablis d\'abord, j\'explique après.'],
    fatigue: ['Café. Encore.', 'Mes yeux piquent.', 'Trop de sessions, pas assez de moi.', 'Je relis la même ligne.', 'Vivement vendredi.'],
    formation: ['J\'apprends à mieux écrire mes demandes.', 'Ah, c\'est donc ça, une injection.', 'Je prends des notes.', 'Bon atelier, finalement.', 'J\'essaie ça demain.'],
    attente: ['J\'attends une revue.', 'Ma PR attend son tour.', 'La file avance. Lentement.', 'En attendant, je documente.', 'Quelqu\'un peut me relire ?'],
    conge: ['En congé. Enfin.', 'Pas de cellulaire.', 'Au chalet.', 'Vacances méritées.', 'Je reviens lundi. Peut-être.'],
  };
})(globalThis.LTP = globalThis.LTP || {});
