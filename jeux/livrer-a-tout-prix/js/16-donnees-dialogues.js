/* 16-donnees-dialogues.js */
/* Tutoriel, indices de Ginette et bulles génériques. Contenu écrit par l'agent D1.
   LTP.donnees.tutoriel : [{ id, cibles?, region: 'openspace' | 'panneau' | 'tableau' | 'hud' | 'journal', panneau?, attendre?, personnage, texte, action?, fait? }]
   LTP.donnees.indices.leviers : { [levierId]: { [valeur]: texte } }   // commentaire de Ginette quand on choisit cette valeur
   LTP.donnees.indices.trimestres : { [n]: texte }                       // ouverture de chaque trimestre
   LTP.donnees.bulles : { code: [], revue: [], incident: [], fatigue: [], formation: [], attente: [], conge: [] }
                        // bulles génériques, utilisées quand un développeur n'a pas de bulle propre pour son état
   Les clés de valeur des indices sont les valeurs du levier converties en texte : 0, 400, 'true', 'mutation'… */
(function (LTP) {
  'use strict';

  // --- Tutoriel : Ginette fait visiter, en montrant de vrais éléments, et fait jouer une journée ---
  // cibles : sélecteurs essayés dans l'ordre (le premier visible gagne) ; region : repli.
  // panneau : onglet de leviers à ouvrir ; attendre : 'levier' (un levier change) ou 'jour' (une journée passe) ;
  // action : la consigne affichée tant que l'action n'est pas faite ; fait : la réplique une fois l'action faite.
  LTP.donnees.tutoriel = [
    { id: 'bienvenue', cibles: ['#hud .hud-stats'], region: 'hud', personnage: 'ginette',
      texte: 'Bienvenue au GPU ! Pendant un an, tu accompagnes 21 développeurs qui découvrent l\'IA, une journée ouvrable à la fois. En haut : la semaine et le jour, le budget de l\'IA, les jetons de la semaine et la confiance de la direction. Sous 15 de confiance, l\'aventure s\'arrête.' },
    { id: 'echos', cibles: ['.echos', '[data-region="openspace"]'], region: 'openspace', personnage: 'ginette',
      texte: 'Voici l\'open space : sept îlots, sept lignes d\'affaires. Chaque jour, quelques personnes pensent tout haut, et c\'est écrit ici, dans « Ce qu\'on entend », avec la cause à côté. C\'est ton meilleur instrument : quand tu changes quelque chose, ce sont eux qui te le disent en premier.' },
    { id: 'licences', cibles: ['[data-levier="licences"]'], region: 'panneau', panneau: 'ia', attendre: 'levier', personnage: 'ginette',
      texte: 'On commence petit : sept licences GitHub Copilot, une personne volontaire par îlot. Tu peux donner Copilot à toute l\'équipe tout de suite, ou garder le groupe pilote quelques semaines pour comparer les pilotes à leurs voisins. Les deux se défendent. Choisis.',
      action: 'Change le levier des licences (ou remets-le sur le groupe pilote) pour continuer.',
      fait: 'C\'est noté. Écoute ce qu\'ils en disent dans les prochains jours. Et quand un choix me fait lever un sourcil, je te le dis en une ligne.' },
    { id: 'tableau', cibles: ['.zone-tableau .is-surveiller', '.zone-tableau .groupe[data-groupe="livraison"]', '[data-region="tableau"]'], region: 'tableau', personnage: 'ginette',
      texte: 'Le tableau de bord, c\'est le rapport du vendredi : les chiffres changent une fois par semaine, pas avant. Les tuiles « à surveiller » sont celles que la dernière note de service te conseille de regarder. Une tuile grisée « non mesuré », ce n\'est pas une absence de problème : c\'est un problème qu\'on ne voit pas encore.' },
    { id: 'jour', cibles: ['#btn-jour'], region: 'hud', attendre: 'jour', personnage: 'ginette',
      texte: 'Joue maintenant une journée. Du lundi au jeudi, l\'open space vit et parle ; le vendredi, la semaine se calcule et le rapport sort. Si une carte d\'événement arrive, tout s\'arrête jusqu\'à ta décision.',
      action: 'Joue une journée pour continuer.',
      fait: 'Une journée de passée. Il en reste quelques-unes, crois-moi.' },
    { id: 'notes', cibles: ['.note-service', '[data-region="journal"]'], region: 'journal', personnage: 'ginette',
      texte: 'Chaque nouveauté arrive par une note de service, une par semaine au plus : le clavardage, le mode agent, les réviseurs, les tests… Mon conseil : change un seul levier à la fois, puis attends un vendredi ou deux. Sinon, tu ne sauras jamais lequel a fait quoi.' },
    { id: 'comite', cibles: ['[data-region="journal"]'], region: 'journal', personnage: 'ginette',
      texte: 'À la fin de chaque trimestre, le comité te pose trois questions sur tes chiffres. Dis la vérité, même quand elle n\'est pas belle : ils ont le tableau de bord, eux aussi. Bon courage. Je suis au bout de l\'open space, à côté de la machine à café.' },
  ];

  // --- Indices de Ginette sur les valeurs de leviers ---
  LTP.donnees.indices = {
    leviers: {
      licences: {
        aucune: 'Plus de licences du tout ? Tu vas avoir des chiffres sans IA pour comparer. Et quelques visages longs.',
        pilote: 'Sept pilotes, quatorze témoins : regarde vendredi si les pilotes livrent plus que leurs voisins. Rappelle-toi seulement que ce sont des volontaires, donc déjà convaincus.',
        tous: 'Copilot pour tout le monde. Ça va écrire plus, et ça va coûter des licences et des jetons. Regarde surtout qui relit tout ce nouveau code.',
      },
      pair: {
        humain: 'Deux humains, un clavier : ça va moins vite, mais les deux comprennent. Pour les juniors, c\'est de l\'or.',
        ia: 'Humain avec l\'IA, ça va vite. Vérifie juste que l\'humain tient encore le volant.',
        rotation: 'La rotation, c\'est un bon compromis : une semaine on apprend, une semaine on livre.',
      },
      reviseurs: {
        0: 'Aucun réviseur ? Le code part en production tel qu\'il sort de l\'agent. Courageux.',
        2: 'Deux réviseurs, c\'est solide. Regarde quand même la pile sur le bureau de Nadia dans deux semaines.',
      },
      approbation: {
        auto: 'L\'IA qui approuve le code de l\'IA. Mêmes angles morts, même sourire.',
        proprietaire: 'Le propriétaire du code a le dernier mot. Bonne idée, tant qu\'il a le temps de lire.',
      },
      taillePR: {
        libre: 'Taille libre ? Mathieu te remercie. Nadia, un peu moins.',
        400: '400 lignes : à peu près la limite où un humain lit encore vraiment.',
        200: '200 lignes, c\'est très lisible. Surveille seulement que ça ne devienne pas 40 PR par jour.',
      },
      ci: {
        aucune: 'Sans CI, chaque défaut se découvre en production. C\'est une façon d\'apprendre, disons.',
        tests: 'Compilation et tests à chaque PR : les agents commencent à avoir des rails.',
        complet: 'Pipeline complet : les déploiements deviennent ennuyants. C\'est exactement ce qu\'on veut.',
      },
      lint: {
        aucun: 'Pas de lint : les revues vont parler de virgules au lieu de parler de logique.',
        strict: 'Strict, mais pas verrouillé : un agent pressé peut désactiver une règle pour faire passer la CI. Je dis ça de même.',
        verrouille: 'Verrouillé : l\'agent ne peut plus faire taire le lint pour passer. Il doit corriger son code.',
      },
      tests: {
        aucun: 'Aucun test, avec des agents ? Tu leur enlèves les rails et tu montes la vitesse.',
        unitaires: 'Les tests unitaires écrits par l\'agent montent la couverture vite. La couverture affichée, en tout cas.',
        mutation: 'Les tests de mutation modifient le code exprès pour voir si les tests s\'en aperçoivent. Les tests qui ne vérifient rien ne s\'en remettent pas.',
      },
      revuesExplication: {
        mensuelles: 'Une fois par mois, chacun explique son code sans l\'IA. Tu vas apprendre des choses. Eux aussi.',
        majeures: 'À chaque PR majeure : ça coûte du temps, mais tu sauras enfin qui comprend quoi.',
      },
      autonomie: {
        0: 'De l\'autocomplétion seulement. Prudent. Raymond approuve, Laurence boude.',
        2: 'Agent avec approbation à chaque étape : un bon niveau pour du code critique.',
        3: 'L\'agent ouvre ses PR tout seul. Ça va écrire beaucoup. Reste à savoir qui va relire.',
        4: 'Il fusionne seul ? Sur un site vitrine, pourquoi pas. Sur le firmware des capteurs, je retiens mon souffle.',
      },
      sessionsMax: {
        1: 'Une session à la fois : on comprend ce qu\'on fait. On en fait moins, par exemple.',
        3: 'Trois sessions : au-delà, chaque agent de plus rapporte moins et fatigue plus.',
        6: 'Six sessions par personne. Personne ne surveille six agents à la fois. Personne.',
      },
      modele: {
        economique: 'Le modèle économique coûte peu par jeton, mais il invente parfois des noms de paquets. Un jour, quelqu\'un publiera un paquet malveillant sous ce nom-là.',
        haut: 'Le haut de gamme se trompe moins. Regarde le coût par point livré, pas le prix du jeton.',
        local: 'Le modèle local, sur notre serveur GPU : les données restent ici et les jetons sont presque gratuits. Un peu moins brillant, mais il ne parle à personne.',
      },
      modeleParTache: {
        true: 'Un petit modèle pour le lint et la documentation, un gros pour l\'architecture. Régis va t\'aimer.',
      },
      formation: {
        atelier: 'Un atelier, c\'est un début. Dans un mois, on verra ce qu\'il en reste.',
        programme: 'Un programme complet : cher tout de suite, rentable dans un mois. Et les gens deviennent plus méfiants envers les injections.',
        mentorat: 'Le mentorat, c\'est la seule chose qui fasse bouger Diane. Je le sais : c\'est moi qui vais la mentorer.',
      },
      plafondJetons: {
        souple: 'Une alerte avant la coupure : tu vois venir les boucles d\'agent sans bloquer tout le monde.',
        strict: 'Strict : les boucles d\'agent s\'arrêtent net. Le travail aussi, des fois, un vendredi à 16 h.',
      },
      disciplineNon: {
        laxiste: 'Laxiste ? Quand le code a l\'air gratuit, les demandes deviennent infinies. Le volume monte, la valeur, pas tant.',
        ferme: 'Ferme : les directions vont grogner, mais le code que tu n\'écris pas, tu n\'auras pas à le maintenir.',
      },
      specs: {
        non: 'Pas de spec : l\'agent va faire exactement ce qu\'on lui demande. C\'est justement ça, le problème.',
        obligatoires: 'Specs obligatoires : départ plus lent, beaucoup moins de reprise.',
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
        aucun: 'Pas de bac à sable : un paquet malveillant installé par l\'agent voit tout ce que la machine voit.',
        conteneur: 'Un conteneur, c\'est bien : ce qui casse reste dedans. Mais il peut encore parler à Internet.',
        filtre: 'Conteneur et réseau filtré : même si l\'agent se fait berner, il ne peut joindre que les adresses autorisées.',
      },
      accesProd: {
        oui: 'Les agents écrivent en production. Un jour, quelqu\'un va découvrir l\'UPDATE sans WHERE. Ce sera un mardi.',
        lecture: 'Lecture seule : l\'agent diagnostique vite, et il ne peut rien casser.',
        non: 'Aucun accès : rien ne casse, mais les diagnostics de nuit vont prendre plus de temps. Benoît le sentira.',
      },
      approbationSorties: {
        true: 'Un humain confirme chaque sortie externe : la troisième branche de la triade est coupée.',
      },
      politiqueRegistre: {
        libre: 'Libre-service : Yannick a déjà installé trois MCP trouvés sur un forum. Je les ai comptés.',
        listeBlanche: 'Une liste blanche : moins de surprises. Mais une carte approuvée sans responsable vieillit toute seule.',
        proprietaires: 'Chaque carte a un propriétaire : les orphelines deviennent visibles, et quelqu\'un répond de chacune.',
      },
      passerelleMCP: {
        true: 'Tous les appels d\'outils passent par un point journalisé. Si un MCP change sa description en douce, on le saura.',
      },
      politiqueNonTI: {
        interdire: 'Interdire ? Kevin va continuer. Il va juste arrêter de nous le dire.',
        encadrer: 'Encadrer : des outils approuvés et quelqu\'un pour accompagner. Kevin a un vrai besoin. Il lui manque juste des garde-fous.',
      },
      observabilite: {
        base: 'De l\'observabilité de base : le MTTR devient un vrai chiffre. C\'est déjà ça.',
        complete: 'Observabilité complète : tu vas voir ce qui était caché depuis le début. Prépare-toi, ce n\'est pas toujours beau.',
      },
      embauche: {
        gel: 'Geler l\'embauche des juniors ? Dans cinq ans, qui sera senior ? Moi, je serai au chalet.',
        releve: 'Un programme de relève : le meilleur investissement que personne ne verra au tableau de bord cette année.',
      },
      objectif: {
        pr: 'Un objectif en nombre de PR ? Elles vont rapetisser par magie. La valeur, elle, ne bougera pas.',
        delai: 'Un objectif de délai : la revue va accélérer. En lisant moins.',
        satisfaction: 'La satisfaction comme cible : tout le monde va dire oui à tout. Le carnet de demandes va déborder.',
        incidents: 'Un objectif d\'incidents : ils vont baisser. Sur papier. On va les appeler des « demandes ».',
      },
    },

    // --- Ouverture des trimestres ---
    trimestres: {
      1: 'Premier trimestre : la découverte. On commence avec sept licences Copilot et de l\'autocomplétion. Les nouveautés arrivent une à une, par note de service. Mon conseil : regarde où s\'empilent les PR, et écoute ce qu\'on dit dans l\'open space.',
      2: 'Deuxième trimestre : la facture. Les agents ouvrent leurs propres PR, le registre des skills ouvre, et Régis a reçu le relevé des jetons. Avant de monter l\'autonomie, demande-toi qui va relire.',
      3: 'Troisième trimestre : les fuites. Les agents lisent maintenant des courriels, des factures et des pages web, et la production n\'est jamais loin. Regarde les trois voyants de chaque îlot.',
      4: 'Dernier trimestre : la maturité. On va enfin pouvoir mesurer ce qui était caché. Et j\'ai quelque chose à t\'annoncer. Pas tout de suite. Bientôt.',
    },
  };

  // --- Bulles génériques (huit par état au moins) ---
  LTP.donnees.bulles = {
    code: ['L\'agent propose. Je dispose.', 'Encore une petite fonction.', 'Ça compile. Bon signe ?', 'Je relis ce que l\'agent a écrit.', 'Une ligne à la fois.',
      'Je nomme mes variables. Pour vrai.', 'Un petit refactor, en passant.', 'Je lis la doc avant de demander.', 'Ça marche. Je vérifie pourquoi.', 'Je découpe la tâche en trois.'],
    revue: ['Je relis. Pour vrai.', 'Ce diff est long.', 'Qui a écrit ça ? Ah, l\'agent.', 'Un commentaire, deux questions.', 'Approuvé… presque.',
      'Il manque un test ici.', 'Ce nom de fonction ment un peu.', 'Je lis les tests en premier.', 'Pourquoi ce fichier a changé ?'],
    incident: ['Ça brûle. Café, puis terminal.', 'Qui comprend ce code ?', 'Les journaux, vite.', 'Ça marchait hier.', 'Je rétablis d\'abord, j\'explique après.',
      'Retour arrière ? Retour arrière.', 'Qu\'est-ce qui a changé mardi ?', 'J\'appelle qui, à cette heure-ci ?'],
    fatigue: ['Café. Encore.', 'Mes yeux piquent.', 'Trop de sessions, pas assez de moi.', 'Je relis la même ligne.', 'Vivement vendredi.',
      'Une pause. Dans cinq minutes.', 'J\'ai oublié de dîner.', 'Mon cerveau est en mode avion.'],
    formation: ['J\'apprends à mieux écrire mes demandes.', 'Ah, c\'est donc ça, une injection.', 'Je prends des notes.', 'Bon atelier, finalement.', 'J\'essaie ça demain.',
      'Petites tâches, contexte clair. Noté.', 'Un courriel n\'est pas un ordre. Noté.', 'On relit tout ce qui sort. Noté.'],
    attente: ['J\'attends une revue.', 'Ma PR attend son tour.', 'La file avance. Lentement.', 'En attendant, je documente.', 'Quelqu\'un peut me relire ?',
      'Ma PR attend. Je la relis moi-même.', 'Le carnet est vide. Je range.', 'J\'attends. Je nettoie mes branches.'],
    conge: ['En congé. Enfin.', 'Pas de cellulaire.', 'Au chalet.', 'Vacances méritées.', 'Je reviens lundi. Peut-être.',
      'Pas de courriel. Promis.', 'Je dors. Pour vrai.', 'Le code attendra.'],
  };
})(globalThis.LTP = globalThis.LTP || {});
