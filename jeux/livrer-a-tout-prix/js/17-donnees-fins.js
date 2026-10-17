/* 17-donnees-fins.js */
/* Profils de fin et textes du bilan (CONCEPTION §2.3). Les ids sont un contrat avec 32-fin.js (qui choisit le profil).
   Textes écrits par l'agent D1 : titre, sousTitre, texte (chaîne ou fonction (etat, q) → chaîne), replique de Mme Lavigueur,
   lecon (la leçon pédagogique du profil, une phrase).
   Règle : chaque phrase factuelle dépend des vrais chiffres de etat.resultat.faits (32-fin.js) ou des leviers réels.
   Une phrase qu'aucune donnée fiable ne soutient est omise plutôt qu'inventée.
   LTP.donnees.bilan.etoiles : une ligne par nombre d'étoiles (0 à 5). */
(function (LTP) {
  'use strict';
  var F = LTP.format;

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }
  function premier() { for (var i = 0; i < arguments.length; i++) if (arguments[i] != null) return arguments[i]; return null; }

  // Faits de l'année : ceux du moteur de fin (etat.resultat.faits), sinon null. Aucun fait n'est deviné.
  function faits(etat) {
    etat = etat || {};
    var f = (etat.resultat && etat.resultat.faits) || {};
    return {
      semaines: premier(num(f.semaines), num(etat.semaine)),
      valeur: premier(num(f.pointsLivres), num(f.valeurLivree)),
      ratioDebit: num(f.ratioDebit),
      partLgtm: num(f.partLgtm),
      incidents: num(f.incidents),
      incidentsSecurite: num(f.incidentsSecurite),
      fuites: num(f.fuites),
      fuitesMajeures: num(f.fuitesMajeures),
      depense: num(f.budgetDepense),
      annuel: num(f.budgetAnnuel),
      ratioBudget: num(f.ratioBudget),
      comprehension: num(f.comprehension),
      comprehensionDepart: num(f.comprehensionDepart),
      volume: num(f.volumeCode),
      volumeDepart: num(f.volumeDepart),
      satisfaction: num(f.satisfaction),
      satisfactionDepart: num(f.satisfactionDepart),
      fantomes: num(f.applisFantomes),
      surface: num(f.surfaceAttaque),
      releve: num(f.releve),
      departs: num(f.departs),
    };
  }

  function nb(n) { return F.nombre(n, 0); }
  function pc(v) { return F.pourcent(v); }
  function pl(n, un, plusieurs) { return nb(n) + ' ' + (Math.abs(n) >= 2 ? plusieurs : un); }
  // En début de phrase : « Un incident », « Une fuite », « 3 incidents ».
  function Pl(n, un, plusieurs, feminin) { return n === 1 ? (feminin ? 'Une ' : 'Un ') + un : pl(n, un, plusieurs); }
  function joindre(phrases) { return phrases.filter(function (p) { return !!p; }).join(' '); }
  function liste(l) { return l.length < 2 ? l.join('') : l.slice(0, -1).join(', ') + ' et ' + l[l.length - 1]; }
  function lev(etat, q, id, ilot) { try { return q && q.levier ? q.levier(etat, id, ilot) : undefined; } catch (e) { return undefined; } }

  // Compréhension : de combien elle a bougé depuis le début de l'année.
  function phraseComprehension(x) {
    if (x.comprehension == null) return null;
    if (x.comprehensionDepart == null) return 'La compréhension moyenne du code a terminé l\'année à ' + pc(x.comprehension) + '.';
    var d = x.comprehension - x.comprehensionDepart;
    if (Math.abs(d) < 0.03) return 'La compréhension moyenne du code a tenu : ' + pc(x.comprehensionDepart) + ' en janvier, ' + pc(x.comprehension) + ' en décembre.';
    return 'La compréhension moyenne du code est ' + (d > 0 ? 'montée' : 'passée') + ' de ' + pc(x.comprehensionDepart) + ' à ' + pc(x.comprehension) + '.';
  }
  function comprehensionTenue(x) { return x.comprehension != null && x.comprehensionDepart != null && x.comprehension >= x.comprehensionDepart - 0.05; }

  function phraseSecurite(x) {
    if (x.fuites == null && x.incidentsSecurite == null) return null;
    if (x.fuites) return 'Une ombre au tableau : ' + (x.fuites === 1 ? 'une fuite de données' : pl(x.fuites, 'fuite de données', 'fuites de données')) + ' dans l\'année.';
    if (x.fuitesMajeures) return 'Une ombre au tableau : une fuite de données majeure.';
    if (x.incidentsSecurite) return Pl(x.incidentsSecurite, 'incident de sécurité', 'incidents de sécurité') + ', mais aucune fuite de données.';
    return 'Aucune fuite de données, aucun incident de sécurité.';
  }

  function phraseBudget(x) {
    if (x.ratioBudget == null) return null;
    if (x.ratioBudget <= 1.05) return 'Et le budget de l\'IA a été respecté.';
    return 'Le budget de l\'IA, lui, a été dépassé de ' + pc(x.ratioBudget - 1) + '.';
  }

  // Les garde-fous réellement en place en fin d'année.
  function rails(etat, q) {
    var l = [];
    if (lev(etat, q, 'tests') === 'mutation') l.push('des tests de mutation');
    else if (lev(etat, q, 'tests') === 'unitaires') l.push('des tests unitaires');
    var ci = lev(etat, q, 'ci');
    if (ci === 'complet' || ci === 'tests') l.push('une CI qui teste chaque PR');
    var t = lev(etat, q, 'taillePR');
    if (t === 400 || t === 200) l.push('des PR de ' + t + ' lignes au plus');
    return l;
  }

  // L'autonomie a-t-elle vraiment suivi la criticité ? Seulement si des réglages par îlot la font varier dans le bon sens.
  function autonomieSelonCriticite(etat, q) {
    var s = (etat && etat.leviers && etat.leviers.surcharges) || {};
    var touche = Object.keys(s).some(function (i) { return s[i] && Object.prototype.hasOwnProperty.call(s[i], 'autonomie'); });
    if (!touche) return false;
    var a = function (i) { var v = lev(etat, q, 'autonomie', i); return typeof v === 'number' ? v : null; };
    var critiques = ['iot', 'operations'].map(a).filter(function (v) { return v != null; });
    var web = a('web');
    return critiques.length > 0 && web != null && Math.max.apply(null, critiques) < web;
  }

  LTP.donnees.fins = [
    { id: 'reussie', titre: 'La transition réussie',
      sousTitre: function (etat) { return comprehensionTenue(faits(etat)) ? 'Plus vite, et toujours compris.' : 'Plus vite. Compris, un peu moins.'; },
      texte: function (etat, q) {
        var x = faits(etat), r = rails(etat, q);
        return joindre([
          x.valeur != null ? 'En ' + nb(x.semaines || 52) + ' semaines, l\'équipe a livré ' + nb(x.valeur) + ' points de valeur'
            + (x.ratioDebit != null && x.ratioDebit >= 1.05 ? ', soit ' + pc(x.ratioDebit - 1) + ' de plus qu\'une équipe sans agents.' : '.') : null,
          x.partLgtm != null && x.partLgtm < 0.25 ? 'La revue a tenu : ' + (x.partLgtm < 0.005 ? 'aucune PR fusionnée' : 'seulement ' + pc(x.partLgtm) + ' des PR fusionnées') + ' sur un simple « LGTM ».' : null,
          x.incidents == null ? null
            : x.incidents === 0 ? 'Aucun incident de l\'année.'
            : (x.incidents === 1 ? 'Un seul incident dans l\'année' : Pl(x.incidents, 'incident', 'incidents') + ' dans l\'année')
              + (comprehensionTenue(x) ? ', réglés par des gens qui comprenaient leur code.' : '.'),
          phraseComprehension(x),
          r.length ? 'Les agents ont eu des rails : ' + liste(r) + '.' : null,
          autonomieSelonCriticite(etat, q) ? 'Et l\'autonomie des agents a suivi la criticité : moins pour les capteurs et les opérations, plus pour le web.' : null,
          phraseSecurite(x),
          phraseBudget(x),
        ]);
      },
      replique: 'Je n\'ai pas compris tous vos acronymes. Mais j\'ai compris vos chiffres, et ils disaient la vérité. On continue.',
      lecon: 'L\'IA accélère l\'écriture du code ; la transition réussit quand on investit autant dans la revue, les tests, la compréhension et les données que dans les agents.' },

    { id: 'usine', titre: 'L\'usine à PR',
      sousTitre: 'Beaucoup de PR, peu de lecteurs.',
      texte: function (etat) {
        var x = faits(etat);
        var lgtm = x.partLgtm != null && x.partLgtm >= 0.3;
        return joindre([
          'Les PR ont coulé à flots' + (x.valeur != null ? ', pour ' + nb(x.valeur) + ' points de valeur livrée.' : '.'),
          x.volume != null && x.volumeDepart ? 'Le code, lui, est passé de ' + nb(x.volumeDepart) + ' kLOC à ' + nb(x.volume) + ' kLOC.' : null,
          lgtm ? 'Les réviseurs n\'ont pas suivi : ' + pc(x.partLgtm) + ' des PR fusionnées l\'ont été sur un simple « LGTM ».' : null,
          x.incidents ? 'Résultat : ' + pl(x.incidents, 'incident', 'incidents') + ' dans l\'année.' : null,
          phraseComprehension(x),
          lgtm ? 'La chaîne de montage allait vite. Le contrôle de la qualité, lui, était en pause café.' : null,
        ]);
      },
      replique: 'Vous avez fait beaucoup de choses. J\'aimerais maintenant savoir lesquelles servaient à quelque chose.',
      lecon: 'Quand l\'écriture accélère, le goulot se déplace vers la revue : produire plus de PR que l\'équipe ne peut en lire, c\'est fabriquer de la dette et des incidents.' },

    { id: 'musee', titre: 'Le musée',
      sousTitre: 'Rien n\'a cassé. Rien n\'a bougé.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          x.incidents == null ? null
            : x.incidents === 0 ? 'Rien n\'a cassé cette année : aucun incident.'
            : 'Presque rien n\'a cassé : ' + (x.incidents === 1 ? 'un seul incident' : pl(x.incidents, 'incident', 'incidents')) + ' en un an.',
          x.valeur != null ? 'Les TI ont avancé à petits pas : ' + nb(x.valeur) + ' points livrés' + (x.ratioDebit != null && x.ratioDebit < 1 ? ', moins qu\'une équipe sans agents.' : '.') : null,
          x.satisfaction != null && x.satisfactionDepart != null && x.satisfaction < x.satisfactionDepart - 0.02
            ? 'Pendant ce temps, la satisfaction des directions est passée de ' + pc(x.satisfactionDepart) + ' à ' + pc(x.satisfaction) + '.' : null,
          x.fantomes ? 'Kevin et ses collègues ont comblé le vide avec ' + pl(x.fantomes, 'application', 'applications') + ' faites hors TI, que personne n\'a révisées.' : null,
          x.fantomes ? 'La prudence n\'a pas supprimé le risque : elle l\'a déménagé au marketing.' : null,
        ]);
      },
      replique: function (etat) {
        return faits(etat).fantomes
          ? 'Rien n\'a brûlé, c\'est vrai. Kevin, par contre, a maintenant plus d\'applications que votre équipe.'
          : 'Rien n\'a brûlé, c\'est vrai. Rien n\'a avancé non plus.';
      },
      lecon: 'Refuser l\'IA ne supprime pas le risque : la demande trouve un autre chemin, hors des garde-fous des TI.' },

    { id: 'farwest', titre: 'Le Far West',
      sousTitre: 'Tout était branché. Sur tout.',
      texte: function (etat) {
        var x = faits(etat);
        var fuite = !!(x.fuites || x.fuitesMajeures);
        return joindre([
          x.surface != null ? 'La surface d\'attaque a fini l\'année à ' + nb(x.surface * 100) + ' sur 100'
            + (x.surface >= 0.6 ? ' : des agents qui lisent des données privées et du contenu venu de l\'extérieur, et qui peuvent tout envoyer dehors.' : '.') : null,
          x.fuites ? Pl(x.fuites, 'fuite de données plus tard', 'fuites de données plus tard', true) + ', la triade létale n\'est plus une théorie pour personne.'
            : x.fuitesMajeures ? 'Une fuite majeure plus tard, la triade létale n\'est plus une théorie pour personne.' : null,
          x.incidentsSecurite ? Pl(x.incidentsSecurite, 'incident de sécurité', 'incidents de sécurité') + ' au total.' : null,
          fuite ? 'Me Bérubé a ouvert un nouveau classeur. Épais. Avec des intercalaires de couleur.' : null,
          x.valeur != null ? 'L\'équipe a livré ' + nb(x.valeur) + ' points cette année' + (fuite ? (x.fuites > 1 ? ', mais ce sont les fuites qu\'on retiendra.' : ', mais c\'est la fuite qu\'on retiendra.') : '.') : null,
        ]);
      },
      replique: function (etat) {
        var x = faits(etat);
        return x.fuites || x.fuitesMajeures
          ? 'Je ne veux plus jamais apprendre une fuite de données par un journaliste. Plus jamais.'
          : 'Tout était branché sur tout, et on s\'en est tirés. La chance n\'est pas une stratégie.';
      },
      lecon: 'Données privées, contenu non fiable et sortie externe dans le même agent ouvrent la porte à l\'exfiltration : il suffit de couper une branche de la triade pour la refermer.' },

    { id: 'facture', titre: 'La facture salée',
      sousTitre: 'Ça marche. Ça coûte.',
      texte: function (etat, q) {
        var x = faits(etat);
        var causes = [];
        var sess = lev(etat, q, 'sessionsMax');
        if (typeof sess === 'number' && sess >= 3) causes.push('jusqu\'à ' + sess + ' sessions d\'agent par personne');
        if (lev(etat, q, 'modeleParTache') !== true) causes.push('un seul modèle pour toutes les tâches');
        if (lev(etat, q, 'plafondJetons') === 'aucun') causes.push('aucun plafond pour couper les boucles');
        return joindre([
          'Les résultats tiennent' + (x.valeur != null ? ' : ' + nb(x.valeur) + ' points livrés' : '') + (x.incidents != null ? ', ' + pl(x.incidents, 'incident', 'incidents') : '') + '.',
          x.depense != null ? 'Mais l\'IA a coûté ' + F.argent(x.depense)
            + (x.ratioBudget != null && x.ratioBudget > 1 ? ', soit ' + pc(x.ratioBudget - 1) + ' de plus que prévu.' : '.') : null,
          causes.length ? 'En fin d\'année, il y avait encore ' + liste(causes) + '.' : null,
          x.ratioBudget != null && x.ratioBudget > 1.2 ? 'Régis a fait encadrer la facture. Pas pour les bonnes raisons.' : null,
        ]);
      },
      replique: 'Le résultat me plaît. Régis, beaucoup moins. Et c\'est Régis qui signe les chèques.',
      lecon: 'Le coût de l\'IA se pilote comme le reste : le bon modèle pour chaque tâche, des sessions limitées et des plafonds qui coupent les boucles.' },

    { id: 'chateau', titre: 'Le château de cartes',
      sousTitre: 'Beaucoup de code. Personne pour l\'expliquer.',
      texte: function (etat, q) {
        var x = faits(etat);
        var gel = lev(etat, q, 'embauche') === 'gel';
        return joindre([
          x.volume != null && x.volumeDepart ? 'Le code est passé de ' + nb(x.volumeDepart) + ' kLOC à ' + nb(x.volume) + ' kLOC.' : null,
          phraseComprehension(x),
          x.departs ? (x.departs >= 2 ? nb(x.departs) + ' personnes sont parties avec ce qu\'elles savaient.' : 'Une personne est partie avec ce qu\'elle savait.') : null,
          x.releve != null && x.releve < 0.4 ? 'La relève est à ' + pc(x.releve) + (gel ? ' : l\'embauche des juniors est gelée.' : '.') : null,
          x.comprehension != null && x.comprehension < 0.5 ? 'Le jour où quelque chose cassera, personne ne saura où regarder. Ça tient. Pour l\'instant.' : null,
        ]);
      },
      replique: 'Ce qui m\'inquiète, ce n\'est pas ce qui est cassé. C\'est que personne ne sache pourquoi le reste marche.',
      lecon: 'La vitesse sans compréhension est une dette : elle reste invisible jusqu\'à l\'incident, ou jusqu\'au départ de la personne qui savait.' },

    { id: 'congedie', titre: 'Merci pour votre contribution',
      sousTitre: 'La direction a tranché avant décembre.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          'La partie s\'est arrêtée' + (x.semaines != null ? ' à la semaine ' + nb(x.semaines) : ' avant la fin de l\'année') + ' : la confiance de la direction est passée sous le seuil.',
          x.incidents != null || x.depense != null
            ? 'Au dossier : ' + joindre([
              x.incidents != null ? pl(x.incidents, 'incident', 'incidents') + (x.depense != null ? ',' : '.') : null,
              x.depense != null ? F.argent(x.depense) + ' dépensés en IA.' : null,
            ])
            : null,
          'Mme Lavigueur a remercié tout le monde pour sa contribution, ce qui, au GPU, n\'est jamais bon signe.',
        ]);
      },
      replique: 'Merci pour votre contribution. Sincèrement. Ginette va vous raccompagner.',
      lecon: 'La confiance se perd plus vite qu\'elle ne se gagne : des chiffres vrais, même mauvais, valent mieux qu\'une belle histoire contredite par le tableau de bord.' },
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
