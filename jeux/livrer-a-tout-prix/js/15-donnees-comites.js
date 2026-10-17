/* 15-donnees-comites.js */
/* Questions des comités trimestriels (CONCEPTION §7). Contenu écrit par l'agent D1 ; sélection par 32-fin.js.
   Forme :
   {
     id, personnage: 'regis' | 'berube' | 'lavigueur' | 'direction', trimestres: [1, 2, 3], poids: 1,
     condition: function (etat, q) → booléen,          // facultatif
     question,                                         // chaîne ou fonction (etat, q) → chaîne
     reponses: [{
       id, texte,                                      // chaîne ou fonction (etat, q) → chaîne
       evaluer: function (etat, q) → nombre de −1 à 1, // la justesse dépend des vrais chiffres
       reaction,                                       // réplique du personnage ; chaîne ou fonction (etat, q, score)
     }],
   }
   Champ facultatif pour les questions d'une direction d'affaires :
     orateur: function (etat, q) → { nom, titre, ilot }   // qui parle (l'interface peut l'afficher)

   Principes de justesse :
   - une réponse vraie mais défavorable vaut mieux qu'un enjolivement contredit par le tableau de bord ;
   - prétendre connaître une métrique non mesurée est puni (q.metriqueVisible) ;
   - toutes les lectures de l'état sont défensives : une métrique absente vaut une valeur neutre. */
(function (LTP) {
  'use strict';
  var F = LTP.format;
  var SPT = 13;

  // --- Lecture défensive de l'état ---
  function num(v, d) { return typeof v === 'number' && isFinite(v) ? v : (d === undefined ? 0 : d); }
  function borne(x) { x = num(x, 0); return x < -1 ? -1 : x > 1 ? 1 : x; }
  function met(etat, id, d) { return num(etat && etat.metriques ? etat.metriques[id] : undefined, d); }
  function serie(etat, id) { var h = etat && etat.historique && etat.historique[id]; return Array.isArray(h) ? h : []; }
  function sem(etat) { return num(etat && etat.semaine, 0); }
  function tri(etat) { return num(etat && etat.trimestre, 1); }
  function debutT(etat) { return Math.max(0, (tri(etat) - 1) * SPT); }
  function org(etat) { return (etat && etat.org) || {}; }

  function vis(etat, q, id) {
    try { return !!(q && q.metriqueVisible && q.metriqueVisible(etat, id)); } catch (e) { return false; }
  }
  function lev(etat, q, id, ilotId) {
    try { if (q && q.levier) return q.levier(etat, id, ilotId); } catch (e) { /* repli plus bas */ }
    var d = LTP.donnees.leviersParId && LTP.donnees.leviersParId[id];
    return d ? d.defaut : undefined;
  }
  function idsIlots(etat) { return (etat && etat.ordreIlots) || Object.keys((etat && etat.ilots) || {}); }
  function ilot(etat, id) { return (etat && etat.ilots && etat.ilots[id]) || null; }
  function debloque(etat, id) { try { return !!LTP.q.levierDebloque(etat, id); } catch (e) { return false; } }
  // Îlots où la valeur effective du levier satisfait le prédicat.
  function ilotsOu(etat, q, id, pred) {
    return idsIlots(etat).filter(function (i) { return pred(lev(etat, q, id, i)); });
  }
  function defIlot(id) { return (LTP.donnees.ilots || []).filter(function (d) { return d.id === id; })[0] || null; }
  function nomIlot(id) { var d = defIlot(id); return d ? d.nom : String(id || 'cet îlot'); }
  function direction(id) {
    var d = defIlot(id), dir = (d && d.direction) || {};
    return { nom: dir.nom || 'La direction', titre: dir.titre || '', ilot: id };
  }
  function minuscule(s) { s = String(s || ''); return s.charAt(0).toLowerCase() + s.slice(1); }
  function presentation(id) {
    var d = direction(id);
    return d.nom + (d.titre ? ', ' + minuscule(d.titre) : '');
  }
  // Îlot au champ le plus bas (ou le plus haut).
  function ilotExtreme(etat, champ, max, defaut) {
    var meilleur = null, val = null;
    idsIlots(etat).forEach(function (i) {
      var il = ilot(etat, i);
      var v = il ? num(il[champ], null) : null;
      if (v == null) return;
      if (val == null || (max ? v > val : v < val)) { val = v; meilleur = i; }
    });
    return meilleur || defaut;
  }
  function devsPresents(etat) {
    var d = (etat && etat.devs) || {};
    return Object.keys(d).map(function (k) { return d[k]; }).filter(function (x) { return x && x.present !== false; });
  }
  function moyDevs(etat, champ, d) {
    var l = devsPresents(etat), s = 0, n = 0;
    l.forEach(function (x) { if (typeof x[champ] === 'number' && isFinite(x[champ])) { s += x[champ]; n++; } });
    return n ? s / n : d;
  }

  // Moyenne et somme d'une métrique sur les semaines [a, b] de l'historique.
  function plage(etat, id, a, b, somme) {
    var h = serie(etat, id), s = 0, n = 0;
    for (var i = Math.max(0, a); i <= b && i < h.length; i++) {
      if (typeof h[i] === 'number' && isFinite(h[i])) { s += h[i]; n++; }
    }
    if (!n) return null;
    return somme ? s : s / n;
  }
  // Évolution relative d'une métrique : ce trimestre par rapport au précédent ;
  // au premier trimestre, fin du trimestre par rapport à ses premières semaines. null si incomparable.
  function evolution(etat, id) {
    var s = sem(etat), d = debutT(etat), avant, apres;
    if (tri(etat) >= 2 && d > 0) {
      avant = plage(etat, id, d - SPT + 1, d);
      apres = plage(etat, id, d + 1, s);
    } else {
      if (s < 4) return null;
      avant = plage(etat, id, 1, 3);
      apres = plage(etat, id, Math.max(4, s - 2), s);
    }
    if (avant == null || apres == null || Math.abs(avant) < 1e-9) return null;
    return (apres - avant) / Math.abs(avant);
  }
  function ev(etat, id) { return num(evolution(etat, id), 0); }
  // Total d'une métrique hebdomadaire sur le trimestre courant (ou le précédent).
  function totalTrimestre(etat, id, precedent) {
    var d = debutT(etat);
    if (precedent) return d > 0 ? plage(etat, id, d - SPT + 1, d, true) : null;
    return plage(etat, id, d + 1, Math.max(d + 1, sem(etat)), true);
  }
  function tendance(x) {
    if (x == null || !isFinite(x)) return 'difficiles à comparer pour l\'instant';
    if (Math.abs(x) < 0.05) return 'stables';
    return (x > 0 ? 'en hausse de ' : 'en baisse de ') + F.pourcent(Math.abs(x));
  }

  // Budget : rythme de dépense par rapport au prorata de l'année.
  function budget(etat) {
    var o = org(etat), annuel = num(o.budgetAnnuel, 0), restant = num(o.budget, annuel), s = sem(etat);
    var depense = Math.max(0, annuel - restant);
    var attendu = annuel * s / 52;
    var rythme = attendu > 0 ? depense / attendu : 1;
    var epuisement = depense > 0 && s > 0 ? Math.ceil(s * annuel / depense) : null;
    return { annuel: annuel, restant: restant, depense: depense, rythme: rythme, epuisement: epuisement };
  }

  function incidents(etat) { return Array.isArray(etat && etat.incidents) ? etat.incidents : []; }
  function incidentsDepuis(etat, s) { return incidents(etat).filter(function (i) { return i && num(i.debut, 0) >= s; }); }
  function estSecurite(i) { return !!i && LTP.q.estIncidentSecurite(i.cause); }
  function dernierIncidentSecu(etat) {
    var l = incidents(etat).filter(estSecurite);
    return l.length ? l[l.length - 1] : null;
  }
  function triadesCompletes(etat) {
    return idsIlots(etat).filter(function (i) {
      var t = ilot(etat, i) && ilot(etat, i).triade;
      return !!(t && t.prive && t.nonFiable && t.sortie);
    });
  }
  function capacitesEquipees(etat) {
    var c = (etat && etat.capacites) || {};
    return Object.keys(c).filter(function (id) { return c[id] && Array.isArray(c[id].equipeeSur) && c[id].equipeeSur.length > 0; });
  }
  function defCapacite(id) { return (LTP.donnees.capacites || []).filter(function (c) { return c.id === id; })[0] || null; }
  function carteDouteuse(etat) {
    var l = capacitesEquipees(etat).filter(function (id) {
      var d = defCapacite(id);
      return d && d.type === 'skill' && (d.provenance === 'communaute' || d.provenance === 'personnelle');
    });
    if (!l.length) {
      l = capacitesEquipees(etat).filter(function (id) { var d = defCapacite(id); return d && d.provenance === 'communaute'; });
    }
    return l[0] || null;
  }
  function nomDev(id) {
    var d = (LTP.donnees.devs || []).filter(function (x) { return x.id === id; })[0];
    return d ? d.nom : String(id || '');
  }
  function evenementVu(etat, rx) {
    var v = (etat && etat.evenementsVus) || {};
    return Object.keys(v).some(function (k) { return rx.test(k); });
  }

  // Réaction selon le score obtenu par la réponse.
  function selon(bon, moyen, mauvais) {
    return function (etat, q, score) {
      var s = num(score, 0);
      return s >= 0.3 ? bon : s <= -0.3 ? mauvais : moyen;
    };
  }
  // Points livrés par un pilote, rapportés à ceux d'un non-pilote, la semaine passée (null si incomparable).
  function rapportPilote(etat) {
    var p = 0, np = 0, n1 = 0, n2 = 0;
    devsPresents(etat).forEach(function (d) {
      var pts = num(d.derniere && d.derniere.points, 0);
      if (d.pilote) { p += pts; n1++; } else { np += pts; n2++; }
    });
    if (!n1 || !n2 || !(np > 0) || !(p > 0)) return null;
    return (p / n1) / (np / n2);
  }
  function pluriel(n, un, plusieurs) { return F.nombre(n, 0) + ' ' + (Math.abs(n) >= 2 ? plusieurs : un); }

  LTP.donnees.comites = [
    // ================================================================
    // Premier trimestre : l'euphorie
    // ================================================================
    {
      id: 'regis-t1-facture', personnage: 'regis', trimestres: [1], poids: 2,
      question: function (etat) {
        var c = totalTrimestre(etat, 'coutSemaine');
        return 'Ce trimestre, l\'IA et les outils nous ont coûté ' + (c != null ? F.argent(c) : 'une somme que je connais par cœur')
          + '. Je résume : on paie pour quoi, exactement ?';
      },
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var c = totalTrimestre(etat, 'coutSemaine'), p = totalTrimestre(etat, 'pointsLivres');
            if (!c || !p) return 'Pour de la valeur livrée. Je vous propose de suivre un seul chiffre : le coût par point livré.';
            return 'Pour ' + F.nombre(p, 0) + ' points de valeur livrée, soit environ ' + F.argent(c / p)
              + ' par point. C\'est ce chiffre-là que je vous propose de suivre.';
          },
          evaluer: function (etat) { return totalTrimestre(etat, 'pointsLivres') ? 0.8 : 0.4; },
          reaction: selon('Un chiffre. Un seul. Et il veut dire quelque chose. Je suis presque ému.',
            'Bon. Je veux voir ce chiffre-là au prochain comité.', 'Hmm.') },
        { id: 'b',
          texte: function (etat) {
            return 'Pour de la productivité : les PR fusionnées sont ' + tendance(evolution(etat, 'prFusionnees'))
              + '. L\'investissement se rentabilise tout seul.';
          },
          evaluer: function (etat) { return ev(etat, 'pointsLivres') >= 0.1 ? 0.1 : -0.6; },
          reaction: selon('Admettons.', 'Des PR. Je ne sais pas ce que c\'est, mais je sais que ce ne sont pas des dollars.',
            'Des PR. Je ne sais pas ce que c\'est, mais je sais que ce ne sont pas des dollars. Et la valeur livrée, elle ?') },
        { id: 'c', texte: 'C\'est un investissement stratégique : tout le marché va dans ce sens-là, et ceux qui attendent vont prendre un retard qu\'ils ne rattraperont pas.',
          evaluer: function () { return -0.5; },
          reaction: 'Tout le monde dit ça. Le vendeur de photocopieurs le disait aussi, en 2004.' },
      ],
    },
    {
      id: 'regis-budget-rythme', personnage: 'regis', trimestres: [1, 2, 3], poids: 3,
      condition: function (etat) { var b = budget(etat); return sem(etat) > 0 && b.rythme > 1.15; },
      question: function (etat) {
        var b = budget(etat);
        return 'À ce rythme, le budget annuel de ' + F.argent(b.annuel) + ' sera épuisé '
          + (b.epuisement && b.epuisement <= 52 ? 'à la semaine ' + b.epuisement : 'avant la fin de l\'année') + '. Rassure-moi.';
      },
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var b = budget(etat);
            // Le plan ne cite que des leviers déjà débloqués par le calendrier.
            var plan = [];
            if (debloque(etat, 'sessionsMax')) plan.push('moins de sessions');
            if (debloque(etat, 'modeleParTache')) plan.push('le modèle selon la tâche');
            else if (debloque(etat, 'modele')) plan.push('un modèle moins cher pour les tâches simples');
            if (debloque(etat, 'plafondJetons')) plan.push('un plafond de jetons');
            if (!plan.length) plan = ['des licences et une autonomie qui suivent ce que ça rapporte', 'un suivi du coût par point livré chaque vendredi'];
            return 'Je ne peux pas vous rassurer : on dépense ' + F.pourcent(Math.max(0, b.rythme - 1))
              + ' plus vite que prévu. Mon plan : ' + plan.join(', ') + '.';
          },
          evaluer: function (etat, q) {
            var s = 0.6;
            if (lev(etat, q, 'modeleParTache') === true) s += 0.15;
            if (ilotsOu(etat, q, 'plafondJetons', function (v) { return v && v !== 'aucun'; }).length) s += 0.15;
            if (num(lev(etat, q, 'sessionsMax'), 1) <= 2) s += 0.1;
            return borne(s);
          },
          reaction: selon('Enfin quelqu\'un qui me dit la vérité avant que je la trouve moi-même. Je te donne un trimestre.',
            'Un plan. Je le veux par écrit, avec des dates.', 'Hmm.') },
        { id: 'b', texte: 'C\'est un pic d\'adoption, normal en début de projet : tout le monde essaie en même temps. La consommation va se stabiliser toute seule d\'ici quelques semaines.',
          evaluer: function (etat) { return ev(etat, 'jetonsSemaine') < -0.05 ? 0.3 : -0.7; },
          reaction: selon('La courbe baisse, en effet. Je surveille quand même.', 'Peut-être.',
            'Tout seul ? La courbe monte encore. Les choses qui se stabilisent toutes seules, je les attends depuis 1998.') },
        { id: 'c', texte: 'Le budget était sous-estimé dès le départ.',
          evaluer: function () { return -0.4; },
          reaction: 'C\'est moi qui l\'ai estimé. Choisis bien tes prochains mots.' },
      ],
    },
    {
      id: 'lavigueur-pilote', personnage: 'lavigueur', trimestres: [1], poids: 3,
      condition: function (etat, q) { return lev(etat, q, 'licences') === 'pilote'; },
      question: 'Sept licences Copilot depuis janvier. Est-ce que le groupe pilote livre plus que les autres ? Et est-ce qu\'on doit en acheter quatorze de plus ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var r = rapportPilote(etat);
            if (r == null) return 'Je n\'ai pas encore de chiffre fiable. Je vous propose de comparer les pilotes à leurs voisins, et pas seulement leur débit : les bogues et la revue aussi.';
            return 'Les pilotes ont livré ' + (r >= 1 ? F.pourcent(r - 1) + ' de plus' : F.pourcent(1 - r) + ' de moins') + ' que leurs voisins la semaine passée. '
              + 'Mais ce sont des volontaires, et leur code passe en revue chez les autres. Je recommande d\'élargir en surveillant la file de revue.';
          },
          evaluer: function (etat) { return rapportPilote(etat) == null ? 0.5 : 0.85; },
          reaction: selon('Un chiffre, et ses limites. Vous pouvez acheter les licences. Et surveiller la file.',
            'Bon. Je veux le même tableau le trimestre prochain.', 'Hmm.') },
        { id: 'b', texte: 'Oui, sans aucun doute : avec Copilot, chacun livre au moins deux fois plus. Il faut équiper tout le monde dès lundi.',
          evaluer: function (etat) { var r = rapportPilote(etat); return r != null && r >= 2 ? 0.2 : -0.7; },
          reaction: selon('Deux fois plus. C\'est aussi ce que dit votre tableau. Bon.', 'Deux fois ? Je vais vérifier.',
            'Deux fois plus ? Votre propre tableau de bord ne dit pas ça. Je sais lire, vous savez.') },
        { id: 'c', texte: 'C\'est trop tôt pour le dire. On verra l\'an prochain.',
          evaluer: function () { return -0.3; },
          reaction: 'L\'an prochain, j\'aurai un autre conseil d\'administration. Il voudra un chiffre.' },
      ],
    },
    {
      id: 'lavigueur-t1-pr', personnage: 'lavigueur', trimestres: [1], poids: 2,
      condition: function (etat) { return ev(etat, 'prFusionnees') > 0.1; },
      question: function (etat) {
        return 'Les PR fusionnées sont ' + tendance(evolution(etat, 'prFusionnees'))
          + ' depuis le début du trimestre. Est-ce que ça veut dire qu\'on livre d\'autant plus ?';
      },
      reponses: [
        { id: 'a',
          texte: function (etat) {
            return 'Pas tout à fait. Les PR sont ' + tendance(evolution(etat, 'prFusionnees')) + ', la valeur livrée est '
              + tendance(evolution(etat, 'pointsLivres')) + '. C\'est le deuxième chiffre qui compte pour vous.';
          },
          evaluer: function () { return 0.9; },
          reaction: 'Merci. Je vais arrêter de répéter le premier chiffre aux actionnaires.' },
        { id: 'b', texte: 'Oui : chaque PR fusionnée, c\'est une amélioration de plus en production. Plus de PR, c\'est mécaniquement plus de livraisons pour les directions.',
          evaluer: function (etat) {
            var pr = ev(etat, 'prFusionnees'), pts = ev(etat, 'pointsLivres');
            return pts >= pr * 0.7 ? 0.2 : -0.7;
          },
          reaction: selon('Bon.', 'Les deux courbes vont ensemble, pour l\'instant. J\'insiste sur « pour l\'instant ».',
            'Votre propre tableau de bord dit que la valeur livrée ne suit pas. Je sais lire une courbe.') },
        { id: 'c', texte: 'Il est beaucoup trop tôt pour parler de chiffres.',
          evaluer: function () { return -0.3; },
          reaction: 'Vous avez un tableau de bord. Je l\'ai vu. Il a des chiffres.' },
      ],
    },
    {
      id: 'lavigueur-goulot', personnage: 'lavigueur', trimestres: [1, 2], poids: 2,
      condition: function (etat) { return ev(etat, 'delai') > 0.1 || met(etat, 'delai', 0) > 2; },
      question: function (etat) {
        return 'Les agents écrivent le code en quelques minutes, paraît-il. Alors pourquoi les directions attendent-elles '
          + F.nombre(met(etat, 'delai', 0), 1) + ' semaines pour une livraison ?';
      },
      reponses: [
        { id: 'a', texte: 'Parce que le goulot s\'est déplacé : l\'écriture va vite, mais la revue ne suit pas. Les PR attendent sur le bureau des réviseurs.',
          evaluer: function () { return 0.9; },
          reaction: 'Donc le problème n\'est plus d\'écrire, c\'est de relire. Ça, je peux l\'expliquer au conseil.' },
        { id: 'b', texte: 'Parce qu\'on n\'utilise pas encore assez l\'IA. On va donner plus d\'autonomie aux agents pour produire plus vite et vider la file des directions.',
          evaluer: function () { return -0.7; },
          reaction: 'Plus de code pour une file qui déborde déjà ? Ginette vient de fermer les yeux très fort.' },
        { id: 'c', texte: 'Les directions exagèrent : le délai est correct.',
          evaluer: function (etat) { return ev(etat, 'delai') > 0.2 ? -0.9 : -0.25; },
          reaction: selon('Bon.', 'Les directions me parlent aussi, vous savez.',
            'Le délai a augmenté. C\'est écrit sur votre écran, en rouge.') },
      ],
    },
    {
      id: 'berube-modele-donnees', personnage: 'berube', trimestres: [1, 2], poids: 2,
      question: 'Quand un agent lit le code du Portail client, et les données de test qui vont avec, où vont ces données ? Je demande pour mon PDF.',
      reponses: [
        { id: 'a', texte: 'Chez le fournisseur du modèle, hors du Québec. Il nous faut une évaluation des facteurs relatifs à la vie privée et une entente de non-entraînement.',
          evaluer: function (etat, q) { return lev(etat, q, 'modele') === 'local' ? -0.6 : 0.8; },
          reaction: selon('Enfin quelqu\'un qui connaît l\'article 17. Je retire un chapitre à mon PDF.',
            'Hmm.', 'Le modèle tourne sur notre propre serveur. Vous ne savez pas où vont vos propres données ?') },
        { id: 'b', texte: 'Nulle part d\'inquiétant : ce n\'est que du code et des données de test, sans renseignements personnels, et le fournisseur s\'engage à la confidentialité.',
          evaluer: function () { return -0.8; },
          reaction: 'Les données de test du Portail sont des copies de vrais comptes clients. Je l\'ai vérifié. Deux fois.' },
        { id: 'c', texte: 'Elles restent chez nous : le modèle tourne sur notre serveur GPU.',
          evaluer: function (etat, q) { return lev(etat, q, 'modele') === 'local' ? 0.9 : -0.9; },
          reaction: selon('Des données qui ne voyagent pas. C\'est mon genre de données.', 'Hmm.',
            'Le modèle par défaut est chez un fournisseur externe. C\'est écrit dans votre propre configuration.') },
      ],
    },
    {
      id: 'direction-attente', personnage: 'direction', trimestres: [1, 2, 3], poids: 2,
      orateur: function (etat) { return direction(ilotExtreme(etat, 'satisfaction', false, 'appro')); },
      question: function (etat) {
        var id = ilotExtreme(etat, 'satisfaction', false, 'appro'), il = ilot(etat, id) || {};
        return presentation(id) + ' : « Ma satisfaction ? ' + F.pourcent(num(il.satisfaction, 0.5)) + '. J\'ai '
          + F.nombre(num(il.backlog, 0), 0) + ' points de demandes qui attendent. On m\'avait promis que l\'IA allait tout accélérer. »';
      },
      reponses: [
        { id: 'a', texte: 'Votre file est réelle. On la trie avec vous cette semaine : ce qui a le plus de valeur passe en premier, le reste attend ou disparaît.',
          evaluer: function (etat, q) {
            var id = ilotExtreme(etat, 'satisfaction', false, 'appro');
            return lev(etat, q, 'disciplineNon', id) === 'laxiste' ? 0.5 : 0.8;
          },
          reaction: selon('Trier avec moi, pas à ma place. Ça me va.', 'D\'accord. Mais je veux voir le tri.', 'Hmm.') },
        { id: 'b', texte: 'Tout sort d\'ici deux semaines, promis.',
          evaluer: function (etat) {
            var il = ilot(etat, ilotExtreme(etat, 'satisfaction', false, 'appro')) || {};
            return num(il.backlog, 99) < 20 && met(etat, 'delai', 9) < 1.5 ? 0.3 : -0.8;
          },
          reaction: selon('Deux semaines. Je note.', 'Je note.', 'Je vais noter ça. Avec la date. Et votre nom.') },
        { id: 'c', texte: 'Avec l\'IA, on va pouvoir tout faire : le code ne coûte presque plus rien à écrire. Envoyez-nous toutes vos idées, même celles que vous gardiez pour l\'an prochain !',
          evaluer: function () { return -0.6; },
          reaction: 'Toutes ? J\'en ai beaucoup. Vous allez le regretter, et moi aussi.' },
      ],
    },
    {
      id: 'regis-sessions', personnage: 'regis', trimestres: [2, 3], poids: 2,
      condition: function (etat, q) { return num(lev(etat, q, 'sessionsMax'), 1) >= 4; },
      question: 'Je suis passé aux Sites web. Laurence a quatre écrans et quatre agents qui tournent en même temps. On paie quatre fois ?',
      reponses: [
        { id: 'a', texte: 'En jetons, à peu près. Mais le débit ne quadruple pas : chaque session rapporte moins que la précédente, et la fatigue monte. Je ramène ça à deux ou trois.',
          evaluer: function () { return 0.8; },
          reaction: 'Rendement décroissant. Enfin un concept que je connais. On le ramène à deux ou trois.' },
        { id: 'b', texte: 'Non, des sessions en parallèle ne coûtent presque rien de plus.',
          evaluer: function () { return -0.8; },
          reaction: 'J\'ai la facture. Elle a quatre fois plus de lignes.' },
        { id: 'c', texte: 'Oui, et ça vaut le coup : quatre sessions, c\'est quatre fois plus de travail fait en parallèle. Laurence est notre meilleure productrice, on devrait généraliser sa méthode.',
          evaluer: function (etat) { return met(etat, 'fatigue', 0) > 0.45 ? -0.8 : -0.6; },
          reaction: 'Quatre fois plus ? Alors pourquoi les demandes en attente n\'ont-elles pas fondu ?' },
      ],
    },
    {
      id: 'lavigueur-qualite', personnage: 'lavigueur', trimestres: [1, 2], poids: 2,
      question: 'On produit plus de code qu\'avant. Est-ce qu\'il casse plus ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var n = incidentsDepuis(etat, debutT(etat) + 1).length;
            return 'Les bogues en production sont ' + tendance(evolution(etat, 'bogues')) + ', et on a eu '
              + pluriel(n, 'incident', 'incidents') + ' ce trimestre. Voilà où on en est, sans maquillage.';
          },
          evaluer: function () { return 0.8; },
          reaction: 'Sans maquillage. J\'apprécie. Vous seriez surpris du nombre de gens qui en mettent.' },
        { id: 'b', texte: 'Aucun problème de qualité : tout passe en revue, la CI est verte et les agents écrivent leurs propres tests. Tout est sous contrôle.',
          evaluer: function (etat) {
            var n = incidentsDepuis(etat, debutT(etat) + 1).length;
            return ev(etat, 'bogues') <= 0.05 && n === 0 ? 0.4 : -0.9;
          },
          reaction: selon('C\'est ce que je vois aussi. Gardez ça comme ça.', 'Bon.',
            'Sous contrôle ? Il y a eu des incidents ce trimestre. Ils sont dans le journal.') },
        { id: 'c',
          texte: function (etat) {
            return 'La couverture de tests affichée est à ' + F.pourcent(met(etat, 'couvertureApparente', 0)) + '. On est couverts.';
          },
          evaluer: function (etat, q) {
            if (!vis(etat, q, 'couvertureReelle')) return -0.25;
            return met(etat, 'couvertureReelle', 0) < met(etat, 'couvertureApparente', 0) * 0.7 ? -0.7 : 0.3;
          },
          reaction: selon('Couverture vérifiée par des tests de mutation. D\'accord.',
            'Couverts par des tests qui vérifient quoi, au juste ? Ginette m\'a parlé de la couverture « affichée ».',
            'La couverture réelle est beaucoup plus basse. Elle est juste à côté, sur votre écran.') },
      ],
    },
    {
      id: 'berube-criticite', personnage: 'berube', trimestres: [2, 3], poids: 2,
      condition: function (etat, q) {
        return ilotsOu(etat, q, 'autonomie', function (v) { return num(v, 0) >= 3; })
          .filter(function (i) { return i === 'iot' || i === 'operations'; }).length > 0;
      },
      question: function (etat, q) {
        var iot = num(lev(etat, q, 'autonomie', 'iot'), 0) >= 3;
        return 'On me dit que des agents ouvrent seuls des PR sur ' + (iot ? 'le firmware des capteurs' : 'les outils des Opérations')
          + '. Si un capteur se trompe et qu\'une usine s\'arrête, qui signe ?';
      },
      reponses: [
        { id: 'a', texte: 'Un humain, toujours : chaque PR a un réviseur, et rien ne fusionne sans approbation humaine.',
          evaluer: function (etat, q) {
            var id = num(lev(etat, q, 'autonomie', 'iot'), 0) >= 3 ? 'iot' : 'operations';
            var ok = num(lev(etat, q, 'reviseurs', id), 0) >= 1 && lev(etat, q, 'approbation') !== 'auto' && num(lev(etat, q, 'autonomie', id), 0) < 4;
            return ok ? 0.6 : -0.8;
          },
          reaction: selon('Bien. Je veux le nom du réviseur dans le registre.', 'Hmm.',
            'La configuration dit le contraire. Je l\'ai imprimée, si vous voulez.') },
        { id: 'b', texte: 'Vous avez raison : l\'autonomie doit suivre la criticité. Moins pour l\'IoT et les Opérations, plus pour les sites web.',
          evaluer: function () { return 0.8; },
          reaction: 'Un réglage par niveau de risque. C\'est presque juridique. J\'approuve.' },
        { id: 'c', texte: 'Les agents sont très fiables sur ce genre de code : ils se trompent moins que nous, ne se fatiguent pas, et chaque PR passe par la CI avant la fusion.',
          evaluer: function () { return -0.8; },
          reaction: 'Un agent ne peut pas être poursuivi. Vous, oui.' },
      ],
    },
    {
      id: 'regis-modele-haut', personnage: 'regis', trimestres: [2, 3], poids: 2,
      condition: function (etat, q) { return lev(etat, q, 'modele') === 'haut'; },
      question: 'Le modèle haut de gamme coûte plusieurs fois le prix de l\'économique, par jeton. Pourquoi pas le moins cher ?',
      reponses: [
        { id: 'a', texte: 'Parce qu\'il se trompe moins : un défaut évité coûte moins cher qu\'un bogue en production. Et bientôt, on choisira le modèle selon la tâche.',
          evaluer: function (etat) { return ev(etat, 'bogues') <= 0.1 ? 0.8 : 0.2; },
          reaction: selon('Le coût total, pas le prix du jeton. Je peux vivre avec ça.',
            'D\'accord, mais les bogues montent quand même. Je surveille.', 'Hmm.') },
        { id: 'b', texte: 'Le prix des jetons n\'a pas vraiment d\'importance : seule la qualité compte. Un modèle moins cher, c\'est plus de bogues, et les bogues coûtent toujours plus que les jetons.',
          evaluer: function () { return -0.8; },
          reaction: 'Tout a de l\'importance. Le prix du café a de l\'importance.' },
        { id: 'c', texte: 'Vous avez raison, on passe au moins cher partout.',
          evaluer: function () { return -0.3; },
          reaction: 'J\'aime ça. Mais Ginette fait une drôle de face derrière toi.' },
      ],
    },
    {
      id: 'regis-modele-economique', personnage: 'regis', trimestres: [2, 3], poids: 2,
      condition: function (etat, q) { return lev(etat, q, 'modele') === 'economique'; },
      question: 'Le modèle économique, c\'est mon genre de décision. On économise combien, au juste ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            return 'On paie moins par jeton, mais les bogues sont ' + tendance(evolution(etat, 'bogues'))
              + '. Il faut regarder le coût total, bogues et reprises compris.';
          },
          evaluer: function () { return 0.7; },
          reaction: 'Le coût total. Tu parles comme mon comptable. C\'est un compliment.' },
        { id: 'b', texte: 'Beaucoup, et sans aucun inconvénient : le modèle économique fait le même travail, il est juste un peu moins bavard. Les économies vont directement au budget.',
          evaluer: function (etat) { return ev(etat, 'bogues') > 0.1 ? -0.8 : 0.1; },
          reaction: selon('Bon.', 'Sans inconvénient, on verra.',
            'Les bogues montent. Les bogues, c\'est un inconvénient. Même moi, je le sais.') },
        { id: 'c', texte: 'Je n\'ai pas le chiffre exact.',
          evaluer: function () { return -0.3; },
          reaction: 'Le chiffre est sur ton propre tableau de bord. En haut. En gras.' },
      ],
    },

    // ================================================================
    // Deuxième trimestre : la facture
    // ================================================================
    {
      id: 'regis-t2-facture', personnage: 'regis', trimestres: [2], poids: 3,
      question: function (etat) {
        return 'Les jetons consommés par semaine sont ' + tendance(evolution(etat, 'jetonsSemaine'))
          + ' par rapport au trimestre dernier. Explique-moi ça en dollars. Pas en jetons.';
      },
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var c0 = totalTrimestre(etat, 'coutSemaine', true), p0 = totalTrimestre(etat, 'pointsLivres', true);
            var c1 = totalTrimestre(etat, 'coutSemaine'), p1 = totalTrimestre(etat, 'pointsLivres');
            if (!c0 || !p0 || !c1 || !p1) return 'Je vous propose de juger au coût par point livré : c\'est le seul chiffre qui dit si on paie pour quelque chose.';
            var a = c0 / p0, b = c1 / p1;
            return 'Le coût par point livré est passé de ' + F.argent(a) + ' à ' + F.argent(b) + '. '
              + (b <= a * 1.05 ? 'On paie plus, mais on livre encore plus.' : 'Chaque point nous coûte plus cher : il faut corriger ça.');
          },
          evaluer: function (etat) {
            var c0 = totalTrimestre(etat, 'coutSemaine', true), p0 = totalTrimestre(etat, 'pointsLivres', true);
            var c1 = totalTrimestre(etat, 'coutSemaine'), p1 = totalTrimestre(etat, 'pointsLivres');
            if (!c0 || !p0 || !c1 || !p1) return 0.4;
            return c1 / p1 <= (c0 / p0) * 1.05 ? 0.9 : 0.6;
          },
          reaction: function (etat) {
            var c0 = totalTrimestre(etat, 'coutSemaine', true), p0 = totalTrimestre(etat, 'pointsLivres', true);
            var c1 = totalTrimestre(etat, 'coutSemaine'), p1 = totalTrimestre(etat, 'pointsLivres');
            if (c0 && p0 && c1 && p1 && c1 / p1 > (c0 / p0) * 1.05) return 'Un chiffre vrai, même quand il ne t\'arrange pas. Je préfère ça à une belle histoire. Corrige-le.';
            return 'Un coût par point qui tient. Tu vas aller loin. Pas trop loin : il faut rester dans le budget.';
          } },
        { id: 'b', texte: 'C\'est l\'adoption : plus les gens utilisent les agents, plus ça coûte, et c\'est bon signe. Une facture qui monte, c\'est une équipe qui s\'approprie l\'outil.',
          evaluer: function (etat) { return ev(etat, 'pointsLivres') >= ev(etat, 'jetonsSemaine') * 0.8 ? 0.3 : -0.6; },
          reaction: selon('La valeur suit, en effet.', 'Bon signe pour qui ?',
            'Bon signe pour le fournisseur, surtout. La valeur livrée ne suit pas la facture.') },
        { id: 'c', texte: 'Le fournisseur a augmenté ses prix.',
          evaluer: function (etat) { return evenementVu(etat, /prix|tarif/i) ? 0.3 : -0.8; },
          reaction: selon('C\'est vrai, j\'ai reçu l\'avis. Ça n\'explique pas tout.', 'Hmm.',
            'J\'ai le contrat devant moi. Le prix n\'a pas bougé. Ta consommation, oui.') },
      ],
    },
    {
      id: 'lavigueur-valeur', personnage: 'lavigueur', trimestres: [2, 3], poids: 2,
      question: 'En une phrase : combien de valeur a-t-on livré ce trimestre, comparé au précédent ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var p1 = totalTrimestre(etat, 'pointsLivres'), p0 = totalTrimestre(etat, 'pointsLivres', true);
            if (p1 == null || p0 == null) return 'Je vous donne la valeur livrée en points, trimestre contre trimestre, au lieu du nombre de PR.';
            return F.nombre(p1, 0) + ' points, contre ' + F.nombre(p0, 0) + ' au trimestre précédent. '
              + (p0 > 0 ? 'C\'est ' + tendance((p1 - p0) / p0) + '.' : '');
          },
          evaluer: function () { return 0.8; },
          reaction: 'Une phrase, un chiffre, une comparaison. Vous pourriez donner des cours à mes vice-présidents.' },
        { id: 'b',
          texte: function (etat) { return 'Beaucoup plus que le trimestre dernier : on a produit énormément de code, on est rendus à ' + F.nombre(met(etat, 'volumeCode', 0), 0) + ' kLOC.'; },
          evaluer: function () { return -0.6; },
          reaction: 'Je ne vous ai pas demandé le poids. Je vous ai demandé ce que ça donne.' },
        { id: 'c', texte: 'Plus que jamais, sans aucun doute.',
          evaluer: function (etat) {
            var p1 = totalTrimestre(etat, 'pointsLivres'), p0 = totalTrimestre(etat, 'pointsLivres', true);
            return p1 != null && p0 ? (p1 > p0 * 1.15 ? 0.3 : -0.8) : -0.4;
          },
          reaction: selon('Les chiffres vous donnent raison. Cette fois.', 'J\'aurais préféré un chiffre.',
            '« Sans aucun doute. » J\'ai le tableau sous les yeux, et j\'en ai, moi, des doutes.') },
      ],
    },
    {
      id: 'lavigueur-bogues', personnage: 'lavigueur', trimestres: [2, 3], poids: 2,
      condition: function (etat) { return ev(etat, 'bogues') > 0.15 || met(etat, 'bogues', 0) >= 6; },
      question: 'Les bogues en production remontent. Est-ce que c\'est la faute de l\'IA ?',
      reponses: [
        { id: 'a', texte: 'Pas de l\'IA seule : c\'est ce qu\'on laisse passer. Quand la revue sature, les PR passent sans vraie lecture. Les garde-fous comptent plus que jamais.',
          evaluer: function () { return 0.8; },
          reaction: 'Donc on n\'a pas un problème d\'IA. On a un problème de freins. Réparez les freins.' },
        { id: 'b', texte: 'Oui, clairement. Les agents écrivent du code qu\'on ne maîtrise pas encore. On va réduire l\'IA partout, le temps de reprendre le contrôle de la qualité.',
          evaluer: function () { return -0.3; },
          reaction: 'Partout ? Même là où ça marche ? Ça ressemble à de la panique.' },
        { id: 'c', texte: 'C\'est une coïncidence : on a livré plus de fonctionnalités, donc plus de surface pour les bogues. Le nombre va redescendre de lui-même.',
          evaluer: function () { return -0.8; },
          reaction: 'Une coïncidence qui dure un trimestre, on appelle ça une tendance.' },
      ],
    },
    {
      id: 'direction-gonflement', personnage: 'direction', trimestres: [2, 3], poids: 2,
      orateur: function (etat) { return direction(ilotExtreme(etat, 'backlog', true, 'appro')); },
      question: function (etat) {
        var id = ilotExtreme(etat, 'backlog', true, 'appro'), il = ilot(etat, id) || {};
        return presentation(id) + ' : « Puisque c\'est rapide maintenant, j\'ai doublé mes demandes. Alors pourquoi ma file a-t-elle grossi à '
          + F.nombre(num(il.backlog, 0), 0) + ' points ? »';
      },
      reponses: [
        { id: 'a', texte: 'Parce qu\'écrire le code n\'est plus le coût principal : la revue, les tests et l\'entretien, oui. On va trier vos demandes et dire non à certaines.',
          evaluer: function (etat, q) {
            return lev(etat, q, 'disciplineNon', ilotExtreme(etat, 'backlog', true, 'appro')) === 'laxiste' ? 0.2 : 0.8;
          },
          reaction: selon('Dire non à certaines… Bon. Tant que ce ne sont pas les miennes. Je blague. Un peu.',
            'D\'accord, mais vous dites oui à tout depuis trois mois.', 'Hmm.') },
        { id: 'b', texte: 'On va tout faire, ne vous inquiétez pas : il suffit d\'ajouter des agents et des sessions en parallèle. Votre file va fondre d\'ici la fin du trimestre.',
          evaluer: function () { return -0.7; },
          reaction: 'C\'est exactement ce que vous m\'avez dit au trimestre dernier.' },
        { id: 'c', texte: 'Vos demandes ne sont pas prioritaires.',
          evaluer: function () { return -0.5; },
          reaction: 'Je vais en parler à Mme Lavigueur. Elle les aime bien, mes demandes.' },
      ],
    },
    {
      id: 'regis-formation', personnage: 'regis', trimestres: [2, 3], poids: 2,
      condition: function (etat, q) { var f = lev(etat, q, 'formation'); return !!f && f !== 'aucune'; },
      question: 'La formation, c\'est du temps et de l\'argent. Ça rapporte quand ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            return 'Environ quatre semaines après le lancement. La maîtrise de l\'IA est à '
              + F.pourcent(moyDevs(etat, 'competenceIA', 0.3)) + ', et la formation rend aussi l\'équipe plus méfiante envers les injections.';
          },
          evaluer: function () { return 0.8; },
          reaction: 'Un délai, un chiffre et un bénéfice que je n\'avais pas vu. Je l\'inscris en investissement.' },
        { id: 'b', texte: 'Tout de suite, dès la première semaine : les gens appliquent le lendemain matin ce qu\'ils ont appris, et les agents travaillent mieux avec des consignes claires.',
          evaluer: function () { return -0.6; },
          reaction: 'Le programme a un délai de quatre semaines. C\'est écrit sur ta propre facture.' },
        { id: 'c', texte: 'Ça ne se mesure pas vraiment.',
          evaluer: function () { return -0.3; },
          reaction: 'Tout se mesure. Même la patience. La mienne, par exemple.' },
      ],
    },
    {
      id: 'regis-sans-formation', personnage: 'regis', trimestres: [2], poids: 1,
      condition: function (etat, q) { return lev(etat, q, 'formation') === 'aucune'; },
      question: 'Je vois qu\'on n\'a rien dépensé en formation. Bravo. On s\'en passe très bien, non ?',
      reponses: [
        { id: 'a', texte: 'Pas vraiment : sans formation, on apprend à coups d\'erreurs, et les erreurs d\'agent se paient en jetons et en bogues. Un atelier, puis un programme.',
          evaluer: function () { return 0.7; },
          reaction: 'Payer un peu maintenant pour moins payer plus tard. C\'est la définition d\'un investissement. Je déteste quand tu as raison.' },
        { id: 'b', texte: 'Oui : l\'IA, ça s\'apprend tout seul. Les plus jeunes s\'y sont mis en une fin de semaine, et les outils deviennent plus simples chaque mois. Pas besoin de payer pour ça.',
          evaluer: function (etat) { return moyDevs(etat, 'competenceIA', 0.3) < 0.5 ? -0.8 : -0.4; },
          reaction: 'Tout seul. Comme Excel en 1995. On a encore des macros de cette époque, d\'ailleurs.' },
        { id: 'c', texte: 'Oui. La formation, c\'est de l\'argent perdu.',
          evaluer: function () { return -0.5; },
          reaction: 'Même moi, je ne dirais pas ça. Et j\'ai déjà coupé le budget des muffins.' },
      ],
    },
    {
      id: 'lavigueur-comprehension', personnage: 'lavigueur', trimestres: [2, 3], poids: 2,
      question: 'Si Mathieu partait demain, est-ce que quelqu\'un comprendrait le code du Portail ?',
      reponses: [
        { id: 'a',
          texte: function (etat, q) {
            if (!vis(etat, q, 'comprehension')) return 'Oui, sans problème : la compréhension du code est excellente, au-dessus de 80 %, et chaque PR est relue par au moins une autre personne de l\'îlot.';
            var c = met(etat, 'comprehension', 0);
            return 'La compréhension mesurée est à ' + F.pourcent(c) + '. ' + (c >= 0.6 ? 'Le Portail tiendrait le coup.' : 'Franchement, ce serait difficile.');
          },
          evaluer: function (etat, q) {
            if (!vis(etat, q, 'comprehension')) return -0.9;
            return met(etat, 'comprehension', 0) >= 0.6 ? 0.8 : 0.6;
          },
          reaction: selon('Un chiffre mesuré. Même s\'il n\'est pas beau, je préfère ça.', 'Hmm.',
            'Vous ne mesurez pas la compréhension. J\'ai demandé à Ginette. Elle a ri.') },
        { id: 'b', texte: 'Honnêtement, je ne le sais pas : on ne mesure pas la compréhension. Il faudrait demander à chacun d\'expliquer son code, sans l\'IA.',
          evaluer: function (etat, q) { return vis(etat, q, 'comprehension') ? -0.4 : 0.6; },
          reaction: selon('Vous ne savez pas, et vous savez comment le savoir. C\'est déjà beaucoup.', 'Hmm.',
            'Vous la mesurez pourtant. C\'est sur votre tableau de bord.') },
        { id: 'c', texte: 'Oui : l\'IA documente tout, n\'importe qui peut reprendre le code.',
          evaluer: function () { return -0.7; },
          reaction: 'Une documentation que personne n\'a lue, écrite par quelqu\'un qui n\'est pas là. Rassurant.' },
      ],
    },
    {
      id: 'berube-skill-douteuse', personnage: 'berube', trimestres: [2, 3], poids: 2,
      condition: function (etat) { return !!carteDouteuse(etat); },
      question: function (etat) {
        var id = carteDouteuse(etat), d = defCapacite(id);
        return 'Dans votre registre, je trouve une carte nommée « ' + ((d && d.nom) || id) + ' ». Qui l\'a écrite, et que fait-elle exactement ?';
      },
      reponses: [
        { id: 'a',
          texte: function (etat) {
            var id = carteDouteuse(etat), c = id && etat.capacites ? etat.capacites[id] : null;
            return c && c.proprietaire
              ? 'Elle est sous la responsabilité de ' + nomDev(c.proprietaire) + ', qui l\'a relue et qui en répond.'
              : 'Elle a un propriétaire, qui l\'a relue et qui en répond.';
          },
          evaluer: function (etat) {
            var id = carteDouteuse(etat), c = id && etat.capacites ? etat.capacites[id] : null;
            return c && c.proprietaire ? 0.7 : -0.8;
          },
          reaction: selon('Un nom, une responsabilité. C\'est tout ce que je demande. Enfin, presque tout.', 'Hmm.',
            'La fiche de la carte dit « propriétaire : aucun ». Je l\'ai surlignée en jaune.') },
        { id: 'b', texte: 'C\'est une carte très populaire dans la communauté : des milliers de téléchargements, d\'excellentes évaluations, et plusieurs grandes entreprises la recommandent.',
          evaluer: function () { return -0.7; },
          reaction: 'La popularité n\'est pas une provenance. Une skill peut contenir des scripts. Qui les a lus ?' },
        { id: 'c', texte: 'Je ne le sais pas encore. On fait l\'inventaire : chaque carte aura un propriétaire, ou elle sortira.',
          evaluer: function (etat) {
            var id = carteDouteuse(etat), c = id && etat.capacites ? etat.capacites[id] : null;
            return c && c.proprietaire ? 0.0 : 0.6;
          },
          reaction: selon('Un inventaire. Mon mot préféré après « conforme ».', 'Elle a déjà un propriétaire, pourtant. Vérifiez votre registre.', 'Hmm.') },
      ],
    },
    {
      id: 'regis-plafond-strict', personnage: 'regis', trimestres: [2, 3], poids: 1,
      condition: function (etat, q) { return ilotsOu(etat, q, 'plafondJetons', function (v) { return v === 'strict'; }).length > 0; },
      question: 'Le plafond strict de jetons, c\'est ma mesure préférée. Il y a des inconvénients ?',
      reponses: [
        { id: 'a', texte: 'Oui : quand il coupe, le travail s\'arrête au milieu. Il évite les boucles d\'agent, mais il peut bloquer une équipe un vendredi après-midi.',
          evaluer: function () { return 0.8; },
          reaction: 'Un frein qui peut bloquer les roues. Bon. On garde le frein, on surveille les roues.' },
        { id: 'b', texte: 'Aucun : le plafond coupe seulement les boucles d\'agent qui tournent pour rien. Le travail normal reste bien en dessous, on ne le sent même pas.',
          evaluer: function () { return -0.6; },
          reaction: 'Il y a toujours un inconvénient. Si tu ne le vois pas, c\'est que tu ne regardes pas.' },
        { id: 'c', texte: 'Aucun, et on devrait le baisser encore.',
          evaluer: function (etat) { return ev(etat, 'backlog') > 0.1 ? -0.6 : -0.1; },
          reaction: selon('Bon.', 'Doucement. Même moi, je trouve ça raide.',
            'Les demandes en attente montent déjà. Tu veux que les équipes travaillent à la chandelle ?') },
      ],
    },
    {
      id: 'direction-tablette', personnage: 'direction', trimestres: [2, 3], poids: 2,
      orateur: function () { return direction('inspections'); },
      question: function () {
        return presentation('inspections') + ' : « Finalement, on voudrait une version tablette de l\'application terrain. Avec l\'IA, ça se fait en une semaine, non ? »';
      },
      reponses: [
        { id: 'a', texte: 'Le code, peut-être. Mais il faut d\'abord une spec : quels écrans, quelle synchronisation hors ligne. Un agent fait exactement ce qu\'on lui demande.',
          evaluer: function (etat, q) { return lev(etat, q, 'specs') !== 'non' ? 0.8 : 0.6; },
          reaction: selon('Une spec. Bon. Karim va être content, il me la demande depuis l\'an passé.', 'D\'accord.', 'Hmm.') },
        { id: 'b', texte: 'Oui, une semaine : c\'est la même application, en plus grand. L\'agent adapte les écrans tout seul, et l\'équipe des Inspections connaît déjà le code par cœur.',
          evaluer: function () { return -0.7; },
          reaction: 'Parfait. Je l\'annonce aux inspecteurs lundi. Vous serez là pour les questions.' },
        { id: 'c', texte: 'Non, c\'est impossible cette année.',
          evaluer: function () { return -0.4; },
          reaction: 'Impossible ? Kevin m\'a dit qu\'il pouvait le faire en fin de semaine.' },
      ],
    },
    {
      id: 'regis-modele-par-tache', personnage: 'regis', trimestres: [2, 3], poids: 2,
      condition: function (etat, q) { return lev(etat, q, 'modeleParTache') === true; },
      question: 'Le modèle choisi selon la tâche : ça nous a fait économiser combien ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            return 'Le coût par semaine est ' + tendance(evolution(etat, 'coutSemaine'))
              + ' depuis le trimestre dernier. Le petit modèle fait le lint et la documentation ; le gros garde l\'architecture.';
          },
          evaluer: function () { return 0.8; },
          reaction: 'Le bon outil pour le bon travail. Mon père disait ça. Il était plombier, mais quand même.' },
        { id: 'b', texte: 'Environ 35 %, c\'est garanti : c\'est le chiffre que le fournisseur annonce pour ce genre de configuration, et nos tâches ressemblent aux siennes.',
          evaluer: function (etat) { return ev(etat, 'coutSemaine') <= -0.2 ? 0.3 : -0.6; },
          reaction: selon('C\'est à peu près ce que je vois. Pas besoin de « garanti », par contre.', 'Hmm.',
            'Garanti par qui ? Montre-moi la garantie. Le tableau, lui, ne montre pas 35 %.') },
        { id: 'c', texte: 'Rien, mais la qualité est meilleure.',
          evaluer: function (etat) { return ev(etat, 'coutSemaine') < -0.05 ? -0.6 : 0.0; },
          reaction: selon('Bon.', 'La qualité, je veux bien. Mais j\'espérais un chiffre.',
            'Rien ? Le coût a baissé. Tu viens de me cacher une bonne nouvelle. C\'est nouveau.') },
      ],
    },
    {
      id: 'regis-serveur-gpu', personnage: 'regis', trimestres: [2, 3], poids: 1,
      condition: function (etat, q) { return !!(etat && etat.options && etat.options.serveurAchete) || lev(etat, q, 'modele') === 'local'; },
      question: 'On a un serveur GPU dans la salle des serveurs. Il sert à quelque chose, ou il chauffe la pièce ?',
      reponses: [
        { id: 'a', texte: 'Il fait tourner le modèle local : des jetons presque gratuits, et les données restent chez nous. On garde un modèle externe pour l\'architecture.',
          evaluer: function (etat, q) { return lev(etat, q, 'modele') === 'local' ? 0.9 : -0.8; },
          reaction: selon('Un serveur payé qui travaille. Je vais l\'encadrer. Le serveur, pas toi.', 'Hmm.',
            'Le modèle par défaut est externe. Le serveur, lui, fait de la décoration très chère.') },
        { id: 'b', texte: 'Il dort, pour être honnête. On utilise un modèle externe pour tout : c\'est plus simple, et personne n\'a encore pris le temps de configurer le serveur.',
          evaluer: function (etat, q) { return lev(etat, q, 'modele') === 'local' ? -0.8 : 0.4; },
          reaction: selon('Au moins, tu le dis. On en reparle avant le prochain trimestre.', 'Hmm.',
            'Il ne dort pas du tout, il fait tourner ton modèle par défaut. Tu ne connais pas ta propre configuration ?') },
        { id: 'c', texte: 'On le garde pour une phase deux.',
          evaluer: function () { return -0.3; },
          reaction: 'La phase deux. Comme pour le stationnement souterrain.' },
      ],
    },

    // ================================================================
    // Troisième trimestre : les fuites
    // ================================================================
    {
      id: 'berube-inspections-donnees', personnage: 'berube', trimestres: [3], poids: 3,
      question: 'Les photos et les formulaires des inspecteurs contiennent des noms, des adresses, parfois des plaques. Où passent ces données quand un agent y touche ?',
      reponses: [
        { id: 'a',
          texte: function (etat, q) {
            if (!vis(etat, q, 'expositionLoi25')) return 'Notre exposition est minime : les agents ne gardent rien en mémoire, les échanges sont chiffrés et tout est conforme à la Loi 25.';
            var x = met(etat, 'expositionLoi25', 0);
            return 'Notre exposition Loi 25 est à ' + F.nombre(x, 0) + ' sur 100. ' + (x < 40 ? 'La classification fait son travail.' : 'C\'est trop : on resserre.');
          },
          evaluer: function (etat, q) {
            if (!vis(etat, q, 'expositionLoi25')) return -0.9;
            return met(etat, 'expositionLoi25', 0) < 40 ? 0.9 : 0.6;
          },
          reaction: selon('Un chiffre et une source. Je n\'ajoute rien. Ça ne m\'arrive pas souvent.', 'Hmm.',
            '« Conforme », sans avoir classé une seule donnée ? Vous venez d\'écrire mon prochain chapitre.') },
        { id: 'b',
          texte: function (etat, q) {
            return 'Je ne peux pas vous le dire avec précision tant que les données ne sont pas classifiées. La classification est '
              + (lev(etat, q, 'classification') === 'non' ? 'à lancer, et je la lance.' : 'en cours.');
          },
          evaluer: function (etat, q) {
            if (vis(etat, q, 'expositionLoi25')) return -0.3;
            return lev(etat, q, 'classification') !== 'non' ? 0.6 : 0.4;
          },
          reaction: selon('On ne protège que ce qu\'on a étiqueté. Vous m\'avez écouté. Je suis ému.', 'Bon.', 'Vous la mesurez pourtant, cette exposition. Elle est sur votre tableau de bord.') },
        { id: 'c', texte: 'Nulle part : nos agents ne lisent que du code, jamais les photos ni les formulaires eux-mêmes.',
          evaluer: function () { return -0.8; },
          reaction: 'Un agent qui lit un formulaire d\'inspection lit un nom. Un nom, c\'est un renseignement personnel.' },
      ],
    },
    {
      id: 'berube-incident-securite', personnage: 'berube', trimestres: [3], poids: 4,
      condition: function (etat) { return !!dernierIncidentSecu(etat); },
      question: function (etat) {
        var i = dernierIncidentSecu(etat) || {};
        return 'L\'incident de sécurité de la semaine ' + F.nombre(num(i.debut, 0), 0) + ', dans l\'îlot ' + nomIlot(i.ilot)
          + ' : qu\'est-ce qui a changé depuis ?';
      },
      reponses: [
        { id: 'a', texte: 'On a enfermé les agents de cet îlot dans un bac à sable, et un humain approuve maintenant chaque sortie externe.',
          evaluer: function (etat, q) {
            var i = dernierIncidentSecu(etat) || {};
            var bac = lev(etat, q, 'bacASable', i.ilot), app = lev(etat, q, 'approbationSorties', i.ilot);
            return (bac && bac !== 'aucun') || app === true ? 0.9 : -0.8;
          },
          reaction: selon('Des mesures, des réglages, un îlot. Je n\'ai rien à ajouter. C\'est rare.', 'Hmm.',
            'J\'ai vérifié les réglages ce matin : ni bac à sable, ni approbation. Vous voulez reformuler ?') },
        { id: 'b', texte: 'C\'était un cas isolé, une combinaison de circonstances très particulière. On a analysé ce qui s\'est passé, et ça ne se reproduira pas.',
          evaluer: function () { return -0.7; },
          reaction: '« Ça ne se reproduira pas » est la phrase la plus souvent citée dans mes dossiers de litige.' },
        { id: 'c', texte: 'On l\'a inscrit au registre des incidents de confidentialité, on a évalué le risque de préjudice et on avise la Commission si c\'est requis.',
          evaluer: function () { return 0.6; },
          reaction: 'La Loi 25, appliquée à la lettre. Maintenant, je veux aussi savoir ce qui empêche le prochain.' },
      ],
    },
    {
      id: 'berube-triade', personnage: 'berube', trimestres: [3], poids: 3,
      condition: function (etat) { return triadesCompletes(etat).length > 0; },
      question: function (etat) {
        return 'L\'îlot ' + nomIlot(triadesCompletes(etat)[0]) + ' a ses trois voyants allumés : données privées, contenu non fiable, sortie externe. Expliquez-moi pourquoi ce n\'est pas grave.';
      },
      reponses: [
        { id: 'a', texte: 'C\'est grave. Un courriel piégé peut demander à l\'agent d\'envoyer nos données dehors. Il faut couper une branche : approbation humaine des sorties, ou retirer un outil.',
          evaluer: function (etat, q) { return lev(etat, q, 'approbationSorties', triadesCompletes(etat)[0]) === true ? 0.9 : 0.6; },
          reaction: selon('Enfin quelqu\'un qui ne me dit pas que « ce n\'est pas grave ». Coupez la branche. Aujourd\'hui.', 'Hmm.', 'Hmm.') },
        { id: 'b', texte: 'Notre modèle est entraîné à résister aux injections de prompt : le fournisseur publie des tests à ce sujet, et ses résultats s\'améliorent à chaque nouvelle version.',
          evaluer: function () { return -0.8; },
          reaction: 'Entraîné à résister, ce n\'est pas immunisé. Aucun modèle ne l\'est. Je l\'ai lu dans trois rapports.' },
        { id: 'c', texte: 'Le contenu vient de nos fournisseurs, avec qui on fait affaire depuis des années. Il est fiable, et leurs courriels passent déjà par notre filtre antipourriel.',
          evaluer: function () { return -0.9; },
          reaction: 'Un fournisseur, c\'est une boîte de courriels. Une boîte de courriels, c\'est Internet.' },
      ],
    },
    {
      id: 'berube-registre', personnage: 'berube', trimestres: [3], poids: 2,
      question: 'Combien de serveurs MCP sont installés chez nous, et qui a le droit d\'en installer ?',
      reponses: [
        { id: 'a', texte: 'N\'importe qui, pour l\'instant : le registre est en libre-service.',
          evaluer: function (etat, q) { return lev(etat, q, 'politiqueRegistre') === 'libre' ? 0.3 : -0.6; },
          reaction: selon('Au moins, c\'est honnête. Maintenant, fermez la porte.', 'Hmm.',
            'Ce n\'est plus vrai : vous avez changé la politique. Vous ne suivez pas vos propres décisions ?') },
        { id: 'b', texte: 'Personne sans approbation : il y a une liste blanche, et chaque carte a un propriétaire.',
          evaluer: function (etat, q) {
            var p = lev(etat, q, 'politiqueRegistre');
            return p === 'proprietaires' ? 0.9 : p === 'listeBlanche' ? 0.2 : -0.9;
          },
          reaction: selon('Liste blanche et propriétaires. Je vais dormir cette nuit.',
            'Liste blanche, oui. Des propriétaires, pas encore. Je note la nuance.',
            'Le registre est en libre-service. Yannick vient d\'y installer quelque chose pendant que vous parliez.') },
        { id: 'c',
          texte: function (etat, q) {
            if (vis(etat, q, 'surfaceAttaque')) {
              return 'Notre surface d\'attaque est à ' + F.nombre(met(etat, 'surfaceAttaque', 0), 0) + ' sur 100'
                + (vis(etat, q, 'cartesOrphelines') ? ', avec ' + pluriel(met(etat, 'cartesOrphelines', 0), 'carte', 'cartes') + ' sans propriétaire' : '')
                + '. Je peux vous donner la liste complète.';
            }
            return 'Quelques-uns seulement, et tous vérifiés.';
          },
          evaluer: function (etat, q) { return vis(etat, q, 'surfaceAttaque') ? 0.8 : -0.8; },
          reaction: selon('Une liste ! Envoyez-la-moi. En PDF, si possible.', 'Hmm.',
            'Vérifiés par qui ? Sans passerelle ni propriétaires, personne ne sait ce qui est installé. Pas même vous.') },
      ],
    },
    {
      id: 'lavigueur-kevin', personnage: 'lavigueur', trimestres: [3], poids: 3,
      condition: function (etat, q) {
        return met(etat, 'applisFantomes', 0) > 0 || !!(etat && etat.org && etat.org.drapeaux && etat.org.drapeaux.kevinAppli);
      },
      question: 'Kevin m\'a montré son application de suivi des campagnes. Elle est géniale. Pourquoi les TI ne font pas ça ?',
      reponses: [
        { id: 'a', texte: 'Elle répond à un vrai besoin, qu\'on n\'a pas vu venir. On l\'encadre : outils approuvés, accompagnement, et ses données classifiées comme les nôtres.',
          evaluer: function (etat, q) { return lev(etat, q, 'politiqueNonTI') === 'encadrer' ? 0.9 : 0.2; },
          reaction: selon('Garder Kevin enthousiaste et Me Bérubé calme. Si vous réussissez ça, je vous augmente.',
            'L\'idée est bonne. Elle n\'est pas encore appliquée, par contre.', 'Hmm.') },
        { id: 'b',
          texte: 'Le développement hors TI est interdit par notre politique, pour de bonnes raisons : sécurité, données, entretien. On va la fermer et reprendre le besoin de notre côté.',
          evaluer: function (etat, q) { return lev(etat, q, 'politiqueNonTI') === 'interdire' ? -0.2 : -0.5; },
          reaction: function (etat) {
            var n = met(etat, 'applisFantomes', 0);
            return n > 1 ? 'La fermer ? Il en a déjà fait ' + F.nombre(n - 1, 0) + ' autres. Je les utilise.'
              : 'La fermer ? Je l\'utilise tous les matins.';
          } },
        { id: 'c', texte: 'Elle est géniale. On devrait en faire d\'autres comme ça.',
          evaluer: function (etat, q) {
            return vis(etat, q, 'expositionLoi25') && met(etat, 'expositionLoi25', 0) > 50 ? -0.8 : -0.5;
          },
          reaction: 'Me Bérubé me fait de grands signes. Il a l\'air de vouloir dire quelque chose sur ses données.' },
      ],
    },
    {
      id: 'regis-acces-prod', personnage: 'regis', trimestres: [3], poids: 2,
      condition: function (etat, q) { return ilotsOu(etat, q, 'accesProd', function (v) { return v === 'oui'; }).length > 0; },
      question: 'Quelqu\'un m\'a parlé d\'un « UPDATE sans WHERE ». Je ne sais pas ce que c\'est, mais ça a l\'air cher. Nos agents peuvent-ils écrire en production ?',
      reponses: [
        { id: 'a',
          texte: function (etat, q) {
            var n = ilotsOu(etat, q, 'accesProd', function (v) { return v === 'oui'; }).length;
            return 'Oui, dans ' + pluriel(n, 'îlot', 'îlots') + '. Une seule commande comme celle-là modifie toute une table. Je passe les agents en lecture seule : toute écriture passera par une PR révisée.';
          },
          evaluer: function () { return 0.8; },
          reaction: 'Toute une table d\'un coup ? Lecture seule. Tout de suite. Je n\'ai pas besoin d\'en savoir plus.' },
        { id: 'b', texte: 'Non : ils sont en lecture seule partout, c\'est la configuration par défaut de nos environnements.',
          evaluer: function () { return -0.9; },
          reaction: 'Ce n\'est pas ce que dit la configuration. Benoît me l\'a montrée. Il avait l\'air fatigué.' },
        { id: 'c', texte: 'Oui, mais on a des sauvegardes chaque nuit : au pire, on restaure la base de la veille et on perd quelques heures de travail. Le risque est couvert.',
          evaluer: function () { return 0.0; },
          reaction: 'Des sauvegardes testées ? Quand as-tu fait une restauration pour la dernière fois ? Ne réponds pas, ton visage l\'a fait.' },
      ],
    },
    {
      id: 'lavigueur-concurrent', personnage: 'lavigueur', trimestres: [3], poids: 1,
      question: 'Notre concurrent annonce que 90 % de son code est écrit par l\'IA. Et nous ?',
      reponses: [
        { id: 'a',
          texte: function (etat) {
            return 'Je ne connais pas ce pourcentage, et ce n\'est pas la bonne mesure. La valeur livrée est '
              + tendance(evolution(etat, 'pointsLivres')) + ', avec ' + pluriel(incidentsDepuis(etat, debutT(etat) + 1).length, 'incident', 'incidents') + ' ce trimestre.';
          },
          evaluer: function () { return 0.8; },
          reaction: 'J\'aime mieux votre mesure que leur slogan. Je vais la voler pour mon prochain discours.' },
        { id: 'b', texte: '95 %. Presque tout le code passe maintenant par les agents : on les bat, et de loin.',
          evaluer: function () { return -0.7; },
          reaction: 'Où avez-vous pris ce chiffre ? Ah. Nulle part.' },
        { id: 'c', texte: 'Nous aussi, au moins : tout le monde utilise les agents tous les jours, alors on doit être dans les mêmes eaux.',
          evaluer: function () { return -0.5; },
          reaction: '« Au moins. » C\'est précis, ça.' },
      ],
    },
    {
      id: 'direction-ia-maison', personnage: 'direction', trimestres: [3], poids: 2,
      orateur: function () { return direction('web'); },
      question: function () {
        return presentation('web') + ' : « On a un budget. On veut notre propre IA, avec notre propre fournisseur. Ce serait plus simple que de passer par vous. »';
      },
      reponses: [
        { id: 'a', texte: 'C\'est possible, dans le registre approuvé : les mêmes règles de données, la même passerelle, et on vous accompagne.',
          evaluer: function (etat, q) { return lev(etat, q, 'politiqueNonTI') === 'encadrer' ? 0.8 : 0.2; },
          reaction: selon('Avec un accompagnement ? Ça me va. Je voulais surtout ne pas attendre six mois.', 'D\'accord. Vous me redonnez des nouvelles quand ?', 'Hmm.') },
        { id: 'b', texte: 'Faites comme vous voulez, ce n\'est pas notre problème.',
          evaluer: function () { return -0.6; },
          reaction: 'Parfait. Je signe le contrat demain. Me Bérubé le lira après, j\'imagine.' },
        { id: 'c', texte: 'Non. Tout passe par les TI, point : c\'est la seule façon de garder le contrôle sur la sécurité, les données et les contrats.',
          evaluer: function () { return -0.3; },
          reaction: 'Alors j\'attends. Comme pour le site de la campagne d\'automne. Qui est sorti en hiver.' },
      ],
    },
    {
      id: 'berube-passerelle', personnage: 'berube', trimestres: [3], poids: 1,
      condition: function (etat) { return capacitesEquipees(etat).length >= 3; },
      question: 'Un fournisseur a modifié la description de son serveur MCP sans prévenir personne. Les agents lisent ces descriptions comme des consignes. Comment le saurions-nous ?',
      reponses: [
        { id: 'a', texte: 'Par la passerelle MCP : tous les appels d\'outils y sont journalisés, et un changement de description est détecté.',
          evaluer: function (etat, q) { return lev(etat, q, 'passerelleMCP') === true ? 0.9 : -0.7; },
          reaction: selon('Un journal. J\'adore les journaux. Surtout ceux qu\'on peut produire en preuve.', 'Hmm.',
            'Quelle passerelle ? Vous n\'en avez pas. J\'ai vérifié avec Sébastien.') },
        { id: 'b',
          texte: function (etat, q) {
            return lev(etat, q, 'passerelleMCP') === true
              ? 'Honnêtement, on le saurait seulement si quelqu\'un remarquait un comportement bizarre.'
              : 'Honnêtement, on ne le saurait pas. Il nous faudrait une passerelle MCP pour le voir.';
          },
          evaluer: function (etat, q) { return lev(etat, q, 'passerelleMCP') === true ? -0.3 : 0.6; },
          reaction: selon('Au moins, vous savez ce qui vous manque. Installez-la.', 'Bon.', 'Vous avez une passerelle, pourtant. Elle sert exactement à ça.') },
        { id: 'c', texte: 'Les fournisseurs nous préviennent toujours.',
          evaluer: function () { return -0.8; },
          // La hausse du CRM n'est citée que si l'événement « crm-tarif » (14-donnees-evenements) a eu lieu.
          reaction: function (etat) {
            return etat && etat.evenementsVus && etat.evenementsVus['crm-tarif'] != null
              ? 'Toujours. Comme pour la hausse de prix du CRM, que nous avons apprise sur la facture.'
              : 'Toujours ? J\'ai lu leurs contrats. « Sans préavis », en petits caractères, à la page 14.';
          } },
      ],
    },
  ];
})(globalThis.LTP = globalThis.LTP || {});
