/* 17-donnees-fins.js */
/* Profils de fin et textes du bilan (CONCEPTION §2.3). Les ids sont un contrat avec 32-fin.js (qui choisit le profil).
   Textes écrits par l'agent D1 : titre, sousTitre, texte (chaîne ou fonction (etat, q) → chaîne), replique de Mme Lavigueur,
   lecon (la leçon pédagogique du profil, une phrase).
   Les textes lisent etat.resultat.faits de façon défensive (valeurLivree, incidents, fuites, budgetDepense,
   comprehension, departs) et se rabattent sur l'état lui-même si un fait manque.
   LTP.donnees.bilan.etoiles : une ligne par nombre d'étoiles (0 à 5). */
(function (LTP) {
  'use strict';
  var F = LTP.format;

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }
  function somme(h) {
    if (!Array.isArray(h)) return null;
    var s = 0, n = 0;
    for (var i = 0; i < h.length; i++) if (typeof h[i] === 'number' && isFinite(h[i])) { s += h[i]; n++; }
    return n ? s : null;
  }
  function premier() { for (var i = 0; i < arguments.length; i++) if (arguments[i] != null) return arguments[i]; return null; }

  // Faits de l'année : ceux du moteur de fin d'abord, sinon reconstruits depuis l'état.
  function faits(etat) {
    etat = etat || {};
    var f = (etat.resultat && etat.resultat.faits) || {};
    var org = etat.org || {}, met = etat.metriques || {}, hist = etat.historique || {};
    var inc = Array.isArray(etat.incidents) ? etat.incidents : [];
    var devs = etat.devs ? Object.keys(etat.devs).map(function (k) { return etat.devs[k]; }) : [];
    var annuel = num(org.budgetAnnuel);
    return {
      valeur: premier(num(f.valeurLivree), num(f.pointsLivres), somme(hist.pointsLivres)),
      incidents: premier(num(f.incidents), num(etat.compteurs && etat.compteurs.incidents), inc.length || null),
      fuites: premier(num(f.fuites), inc.filter(function (i) { return i && /fuite|exfil|loi25/i.test(String(i.cause || '')); }).length),
      depense: premier(num(f.budgetDepense), annuel != null && num(org.budget) != null ? annuel - org.budget : null),
      annuel: annuel,
      comprehension: premier(num(f.comprehension), num(met.comprehension)),
      departs: premier(num(f.departs), devs.length ? devs.filter(function (d) { return d && d.present === false; }).length : null),
      semaine: num(etat.semaine),
    };
  }

  function nb(n) { return F.nombre(n, 0); }
  function pl(n, un, plusieurs) { return nb(n) + ' ' + (Math.abs(n) >= 2 ? plusieurs : un); }
  function joindre(phrases) { return phrases.filter(function (p) { return !!p; }).join(' '); }
  function depassement(x) { return x.depense != null && x.annuel ? x.depense / x.annuel - 1 : null; }

  LTP.donnees.fins = [
    { id: 'reussie', titre: 'La transition réussie',
      sousTitre: 'Plus vite, et toujours compris.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          x.valeur != null ? 'En 52 semaines, l\'équipe a livré ' + nb(x.valeur) + ' points de valeur, sans que la revue devienne un mur.'
            : 'En 52 semaines, l\'équipe a livré plus qu\'avant, sans que la revue devienne un mur.',
          x.incidents == null ? null
            : x.incidents === 0 ? 'Aucun incident sérieux de l\'année, et ce n\'est pas de la chance.'
            : x.incidents === 1 ? 'Le seul incident de l\'année a été réglé par des gens qui comprenaient leur code.'
            : 'Il y a eu ' + nb(x.incidents) + ' incidents, réglés par des gens qui comprenaient leur code.',
          x.comprehension != null ? 'La compréhension moyenne a terminé l\'année à ' + F.pourcent(x.comprehension) + '.' : null,
          'Les agents ont eu des rails : des tests qui vérifient vraiment, une CI, des PR qu\'on peut lire. L\'autonomie a suivi la criticité, et les données sensibles sont restées où elles devaient être.',
        ]);
      },
      replique: 'Je n\'ai pas compris tous vos acronymes. Mais j\'ai compris vos chiffres, et ils disaient la vérité. On continue.',
      lecon: 'L\'IA accélère l\'écriture du code ; la transition réussit quand on investit autant dans la revue, les tests, la compréhension et les données que dans les agents.' },

    { id: 'usine', titre: 'L\'usine à PR',
      sousTitre: 'Beaucoup de PR, peu de lecteurs.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          'Les PR ont coulé à flots' + (x.valeur != null ? ' : ' + nb(x.valeur) + ' points livrés, et beaucoup plus de code que ça.' : ', et le volume de code avec elles.'),
          'Les réviseurs n\'ont jamais rattrapé les agents, et les « LGTM » se sont empilés sur le bureau de Nadia.',
          x.incidents ? 'Résultat : ' + pl(x.incidents, 'incident', 'incidents') + (x.comprehension != null ? ', et une compréhension du code tombée à ' + F.pourcent(x.comprehension) + '.' : '.') : null,
          'La chaîne de montage allait vite. Le contrôle de la qualité, lui, était en pause café.',
        ]);
      },
      replique: 'Vous avez fait beaucoup de choses. J\'aimerais maintenant savoir lesquelles servaient à quelque chose.',
      lecon: 'Quand l\'écriture accélère, le goulot se déplace vers la revue : produire plus de PR que l\'équipe ne peut en lire, c\'est fabriquer de la dette et des incidents.' },

    { id: 'musee', titre: 'Le musée',
      sousTitre: 'Rien n\'a cassé. Rien n\'a bougé.',
      texte: function (etat) {
        var x = faits(etat);
        var fantomes = num(etat && etat.metriques && etat.metriques.applisFantomes);
        return joindre([
          x.incidents ? 'Presque rien n\'a cassé : ' + pl(x.incidents, 'incident', 'incidents') + ' en un an.' : 'Rien n\'a cassé cette année, ou presque.',
          'Mais pendant que la TI avançait à petits pas' + (x.valeur != null ? ' (' + nb(x.valeur) + ' points livrés)' : '') + ', les demandes s\'accumulaient et la patience des directions fondait.',
          'Kevin et ses collègues ont comblé le vide' + (fantomes ? ' avec ' + pl(fantomes, 'application', 'applications') + ' hors TI' : ' avec des applications hors TI') + ', que personne n\'a révisées.',
          'La prudence n\'a pas supprimé le risque : elle l\'a déménagé au marketing.',
        ]);
      },
      replique: 'Rien n\'a brûlé, c\'est vrai. Kevin, par contre, a maintenant plus d\'applications que votre équipe.',
      lecon: 'Refuser l\'IA ne supprime pas le risque : la demande trouve un autre chemin, hors des garde-fous de la TI.' },

    { id: 'farwest', titre: 'Le Far West',
      sousTitre: 'Tout était branché. Sur tout.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          'Les agents avaient accès à tout : ils lisaient des données privées, du contenu venu de l\'extérieur, et pouvaient tout envoyer dehors.',
          x.fuites ? pl(x.fuites, 'fuite de données plus tard', 'fuites de données plus tard') + ', la triade létale n\'est plus une théorie pour personne.'
            : 'Il a suffi d\'une fuite pour que la triade létale ne soit plus une théorie pour personne.',
          'Me Bérubé a ouvert un nouveau classeur. Épais. Avec des intercalaires de couleur.',
          x.valeur != null ? 'L\'équipe a livré ' + nb(x.valeur) + ' points cette année, mais c\'est la fuite qu\'on retiendra.' : 'L\'équipe a beaucoup livré, mais c\'est la fuite qu\'on retiendra.',
        ]);
      },
      replique: 'Je ne veux plus jamais apprendre une fuite de données par un journaliste. Plus jamais.',
      lecon: 'Données privées, contenu non fiable et sortie externe dans le même agent ouvrent la porte à l\'exfiltration : il suffit de couper une branche de la triade pour la refermer.' },

    { id: 'facture', titre: 'La facture salée',
      sousTitre: 'Ça marche. Ça coûte.',
      texte: function (etat) {
        var x = faits(etat), d = depassement(x);
        return joindre([
          'Les résultats sont corrects' + (x.valeur != null ? ' : ' + nb(x.valeur) + ' points livrés' : '') + (x.incidents != null ? ', ' + pl(x.incidents, 'incident', 'incidents') : '') + '.',
          x.depense != null && x.annuel ? 'Mais l\'IA a coûté ' + F.argent(x.depense) + ' pour un budget de ' + F.argent(x.annuel)
            + (d != null && d > 0 ? ', soit ' + F.pourcent(d) + ' de dépassement.' : '.')
            : 'Mais la facture de jetons a largement dépassé le budget.',
          'Des sessions en parallèle partout, un seul modèle pour toutes les tâches, des boucles que personne n\'a coupées.',
          'Régis a fait encadrer la facture. Pas pour les bonnes raisons.',
        ]);
      },
      replique: 'Le résultat me plaît. Régis, beaucoup moins. Et c\'est Régis qui signe les chèques.',
      lecon: 'Le coût de l\'IA se pilote comme le reste : le bon modèle pour chaque tâche, des sessions limitées et des plafonds qui coupent les boucles.' },

    { id: 'chateau', titre: 'Le château de cartes',
      sousTitre: 'Beaucoup de code. Personne pour l\'expliquer.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          'Il y a beaucoup de code. Plus que jamais.',
          x.comprehension != null ? 'Mais la compréhension a fini l\'année à ' + F.pourcent(x.comprehension) + ', et elle baissait encore.' : 'Mais la compréhension n\'a jamais cessé de baisser.',
          x.departs ? (x.departs >= 2 ? nb(x.departs) + ' personnes sont parties avec ce qu\'elles savaient' : 'Une personne est partie avec ce qu\'elle savait')
            + ', et la relève n\'a pas suivi.' : 'Et la relève n\'a pas suivi.',
          'Le jour où quelque chose cassera, personne ne saura où regarder. Ça tient. Pour l\'instant.',
        ]);
      },
      replique: 'Ce qui m\'inquiète, ce n\'est pas ce qui est cassé. C\'est que personne ne sache pourquoi le reste marche.',
      lecon: 'La vitesse sans compréhension est une dette : elle reste invisible jusqu\'à l\'incident, ou jusqu\'au départ de la personne qui savait.' },

    { id: 'congedie', titre: 'Merci pour votre contribution',
      sousTitre: 'La direction a tranché avant décembre.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          'La partie s\'est arrêtée' + (x.semaine != null ? ' à la semaine ' + nb(x.semaine) : ' avant la fin de l\'année') + ' : la confiance de la direction est passée sous le seuil.',
          x.incidents != null || x.depense != null
            ? 'Au dossier : ' + joindre([
              x.incidents != null ? pl(x.incidents, 'incident', 'incidents') + (x.depense != null ? ',' : '.') : null,
              x.depense != null ? F.argent(x.depense) + ' dépensés en IA.' : null,
            ])
            : null,
          'Les réponses au comité n\'ont pas tenu face au tableau de bord.',
          'Mme Lavigueur a remercié tout le monde pour sa contribution, ce qui, au GPU, n\'est jamais bon signe.',
        ]);
      },
      replique: 'Merci pour votre contribution. Sincèrement. Ginette va vous raccompagner.',
      lecon: 'La confiance se perd plus vite qu\'elle ne se gagne : des chiffres vrais, même mauvais, valent mieux qu\'une belle histoire contredite par le tableau de bord.' },
  ];

  LTP.donnees.bilan = {
    etoiles: {
      0: 'Zéro étoile. Ginette vous offre un café et une deuxième chance.',
      1: 'Une étoile. L\'équipe a survécu à l\'année, et c\'est déjà quelque chose.',
      2: 'Deux étoiles. De bons coups, mais le tableau de bord a parlé plus fort.',
      3: 'Trois étoiles. Une transition honnête, avec quelques cicatrices.',
      4: 'Quatre étoiles. Le comité vous redemande l\'an prochain.',
      5: 'Cinq étoiles. Même Régis a souri. Brièvement.',
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
