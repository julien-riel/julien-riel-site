/* en/15-donnees-comites.js */
/* Édition anglaise : questions des comités trimestriels (voir en/LISEZMOI.md).
   Les fonctions de texte gardent les mêmes lectures de l'état et les mêmes branches que js/15-donnees-comites.js ;
   les outils de lecture ci-dessous en sont une copie (ils sont privés à ce fichier-là). evaluer, condition et poids ne sont pas touchés. */
(function (LTP) {
  'use strict';
  var F = LTP.format;
  var SPT = 13;

  // --- Lecture défensive de l'état (copie de js/15) ---
  function num(v, d) { return typeof v === 'number' && isFinite(v) ? v : (d === undefined ? 0 : d); }
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
  function ilotsOu(etat, q, id, pred) {
    return idsIlots(etat).filter(function (i) { return pred(lev(etat, q, id, i)); });
  }
  function defIlot(id) { return (LTP.donnees.ilots || []).filter(function (d) { return d.id === id; })[0] || null; }
  function nomIlot(id) { var d = defIlot(id); return d ? d.nom : String(id || 'unknown'); }
  function direction(id) {
    var d = defIlot(id), dir = (d && d.direction) || {};
    return { nom: dir.nom || 'The business', titre: dir.titre || '', ilot: id };
  }
  // En anglais, le titre garde ses majuscules après le nom : « Francine Gagnon, Director of Procurement ».
  function presentation(id) {
    var d = direction(id);
    return d.nom + (d.titre ? ', ' + d.titre : '');
  }
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
  function plage(etat, id, a, b, somme) {
    var h = serie(etat, id), s = 0, n = 0;
    for (var i = Math.max(0, a); i <= b && i < h.length; i++) {
      if (typeof h[i] === 'number' && isFinite(h[i])) { s += h[i]; n++; }
    }
    if (!n) return null;
    return somme ? s : s / n;
  }
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
  function totalTrimestre(etat, id, precedent) {
    var d = debutT(etat);
    if (precedent) return d > 0 ? plage(etat, id, d - SPT + 1, d, true) : null;
    return plage(etat, id, d + 1, Math.max(d + 1, sem(etat)), true);
  }
  function tendance(x) {
    if (x == null || !isFinite(x)) return 'hard to compare for now';
    if (Math.abs(x) < 0.05) return 'stable';
    return (x > 0 ? 'up ' : 'down ') + F.pourcent(Math.abs(x));
  }
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
  function selon(bon, moyen, mauvais) {
    return function (etat, q, score) {
      var s = num(score, 0);
      return s >= 0.3 ? bon : s <= -0.3 ? mauvais : moyen;
    };
  }
  function rapportPilote(etat) {
    var p = 0, np = 0, n1 = 0, n2 = 0;
    devsPresents(etat).forEach(function (d) {
      var pts = num(d.derniere && d.derniere.points, 0);
      if (d.pilote) { p += pts; n1++; } else { np += pts; n2++; }
    });
    if (!n1 || !n2 || !(np > 0) || !(p > 0)) return null;
    return (p / n1) / (np / n2);
  }
  // Pluriel anglais : le singulier pour 1 seulement (0 incidents, 1 incident, 2 incidents).
  function pluriel(n, un, plusieurs) { return F.nombre(n, 0) + ' ' + (Math.abs(n) === 1 ? un : plusieurs); }

  LTP.traduire('comites', {
    // ================================================================
    // Premier trimestre : l'euphorie
    // ================================================================
    'regis-t1-facture': {
      question: function (etat) {
        var c = totalTrimestre(etat, 'coutSemaine');
        return 'This quarter, AI and tooling cost us ' + (c != null ? F.argent(c) : 'a sum I know by heart')
          + '. To sum up: what exactly are we paying for?';
      },
      reponses: {
        a: {
          texte: function (etat) {
            var c = totalTrimestre(etat, 'coutSemaine'), p = totalTrimestre(etat, 'pointsLivres');
            if (!c || !p) return 'For value delivered. I suggest we track a single number: cost per point delivered.';
            return 'For ' + F.nombre(p, 0) + ' points of value delivered, or about ' + F.argent(c / p)
              + ' per point. That\'s the number I suggest we track.';
          },
          reaction: selon('A number. Just one. And it means something. I\'m almost moved.',
            'Fine. I want to see that number at the next committee.', 'Hmm.'),
        },
        b: {
          texte: function (etat) {
            return 'For productivity: merged PRs are ' + tendance(evolution(etat, 'prFusionnees'))
              + '. The investment pays for itself.';
          },
          reaction: selon('Granted.', 'PRs. I don\'t know what those are, but I know they aren\'t dollars.',
            'PRs. I don\'t know what those are, but I know they aren\'t dollars. And value delivered?'),
        },
        c: {
          texte: 'It\'s a strategic investment: the whole market is going this way, and whoever waits will fall behind and never catch up.',
          reaction: 'Everyone says that. The photocopier salesman said it too, in 2004.',
        },
      },
    },
    'regis-budget-rythme': {
      question: function (etat) {
        var b = budget(etat);
        return 'At this rate, the annual budget of ' + F.argent(b.annuel) + ' will run out '
          + (b.epuisement && b.epuisement <= 52 ? 'in week ' + b.epuisement : 'before year-end') + '. Reassure me.';
      },
      reponses: {
        a: {
          texte: function (etat) {
            var b = budget(etat);
            // Le plan ne cite que des leviers déjà débloqués par le calendrier.
            var plan = [];
            if (debloque(etat, 'sessionsMax')) plan.push('fewer sessions');
            if (debloque(etat, 'modeleParTache')) plan.push('the model chosen per task');
            else if (debloque(etat, 'modele')) plan.push('a cheaper model for simple tasks');
            if (debloque(etat, 'plafondJetons')) plan.push('a token cap');
            if (!plan.length) plan = ['licences and autonomy that track what they bring in', 'cost per point delivered, tracked every Friday'];
            return 'I can\'t reassure you: we\'re spending ' + F.pourcent(Math.max(0, b.rythme - 1))
              + ' faster than planned. My plan: ' + plan.join(', ') + '.';
          },
          reaction: selon('Finally, someone who tells me the truth before I find it myself. You have one quarter.',
            'A plan. I want it in writing, with dates.', 'Hmm.'),
        },
        b: {
          texte: 'It\'s an adoption spike, normal at the start of a project: everyone is trying it at once. Usage will level off on its own within a few weeks.',
          reaction: selon('The curve is going down, indeed. I\'m watching anyway.', 'Maybe.',
            'On its own? The curve is still climbing. Things that level off on their own—I\'ve been waiting for those since 1998.'),
        },
        c: {
          texte: 'The budget was underestimated from the start.',
          reaction: 'I\'m the one who estimated it. Choose your next words carefully.',
        },
      },
    },
    'lavigueur-pilote': {
      question: 'Seven Copilot licences since January. Is the pilot group delivering more than the others? And should we buy fourteen more?',
      reponses: {
        a: {
          texte: function (etat) {
            var r = rapportPilote(etat);
            if (r == null) return 'I don\'t have a reliable number yet. I suggest comparing the pilots with their neighbours, and not just on throughput: bugs and review too.';
            return 'Last week, the pilots were ' + (r >= 1 ? F.pourcent(r - 1) + ' ahead of' : F.pourcent(1 - r) + ' behind') + ' their neighbours. '
              + 'But they\'re volunteers, and their code gets reviewed by the others. I recommend expanding while watching the review queue.';
          },
          reaction: selon('A number, and its limits. You can buy the licences. And watch the queue.',
            'Fine. I want the same table next quarter.', 'Hmm.'),
        },
        b: {
          texte: 'Yes, no doubt about it: with Copilot, everyone delivers at least twice as much. We should equip everyone starting Monday.',
          reaction: selon('Twice as much. That\'s what your table says too. Fine.', 'Twice? I\'ll check.',
            'Twice as much? Your own dashboard doesn\'t say that. I can read, you know.'),
        },
        c: {
          texte: 'It\'s too early to tell. We\'ll see next year.',
          reaction: 'Next year, I\'ll be facing the board again. It will want a number.',
        },
      },
    },
    'lavigueur-t1-pr': {
      question: function (etat) {
        return 'Merged PRs are ' + tendance(evolution(etat, 'prFusionnees'))
          + ' since the start of the quarter. Does that mean we\'re delivering that much more?';
      },
      reponses: {
        a: {
          texte: function (etat) {
            return 'Not quite. PRs are ' + tendance(evolution(etat, 'prFusionnees')) + ', value delivered is '
              + tendance(evolution(etat, 'pointsLivres')) + '. The second number is the one that matters to you.';
          },
          reaction: 'Thank you. I\'ll stop repeating the first number to the shareholders.',
        },
        b: {
          texte: 'Yes: every merged PR is one more improvement in production. More PRs mechanically means more deliveries for the business lines.',
          reaction: selon('Fine.', 'The two curves move together, for now. I stress “for now.”',
            'Your own dashboard says value delivered isn\'t keeping up. I can read a curve.'),
        },
        c: {
          texte: 'It\'s much too early to talk numbers.',
          reaction: 'You have a dashboard. I\'ve seen it. It has numbers.',
        },
      },
    },
    'lavigueur-goulot': {
      question: function (etat) {
        return 'Agents write the code in a few minutes, I\'m told. So why do the business lines wait '
          + F.nombre(met(etat, 'delai', 0), 1) + ' weeks for a delivery?';
      },
      reponses: {
        a: {
          texte: 'Because the bottleneck has moved: writing is fast, but review can\'t keep up. PRs are sitting on the reviewers\' desks.',
          reaction: 'So the problem is no longer writing, it\'s reviewing. That I can explain to the board.',
        },
        b: {
          texte: 'Because we\'re not using AI enough yet. We\'ll give the agents more autonomy to produce faster and clear the business lines\' queue.',
          reaction: 'More code for a queue that\'s already overflowing? Ginette just shut her eyes very tight.',
        },
        c: {
          texte: 'The business lines are exaggerating: the lead time is fine.',
          reaction: selon('Fine.', 'The business lines talk to me too, you know.',
            'Lead time went up. It\'s on your screen, in red.'),
        },
      },
    },
    'berube-modele-donnees': {
      question: 'When an agent reads the Customer Portal code, and the test data that goes with it, where does that data go? Asking for my PDF.',
      reponses: {
        a: {
          texte: 'To the model provider, outside Quebec. We need a privacy impact assessment and a no-training agreement.',
          reaction: selon('Finally, someone who knows section 17. I\'m cutting a chapter from my PDF.',
            'Hmm.', 'The model runs on our own server. You don\'t know where your own data goes?'),
        },
        b: {
          texte: 'Nowhere worrying: it\'s only code and test data, with no personal information, and the provider is bound by confidentiality.',
          reaction: 'The Portal\'s test data are copies of real customer accounts. I checked. Twice.',
        },
        c: {
          texte: 'It stays here: the model runs on our GPU server.',
          reaction: selon('Data that doesn\'t travel. My kind of data.', 'Hmm.',
            'The default model is with an external provider. It\'s in your own configuration.'),
        },
      },
    },
    'direction-attente': {
      orateur: function (etat) { return direction(ilotExtreme(etat, 'satisfaction', false, 'appro')); },
      question: function (etat) {
        var id = ilotExtreme(etat, 'satisfaction', false, 'appro'), il = ilot(etat, id) || {};
        return presentation(id) + ': “My satisfaction? ' + F.pourcent(num(il.satisfaction, 0.5)) + '. I have '
          + F.nombre(num(il.backlog, 0), 0) + ' points of requests waiting. I was promised AI would speed everything up.”';
      },
      reponses: {
        a: {
          texte: 'Your queue is real. We\'ll sort it with you this week: what has the most value goes first, the rest waits or goes away.',
          reaction: selon('Sorting with me, not for me. That works.', 'Okay. But I want to see the sorting.', 'Hmm.'),
        },
        b: {
          texte: 'Everything ships within two weeks, promise.',
          reaction: selon('Two weeks. Noted.', 'Noted.', 'I\'ll write that down. With the date. And your name.'),
        },
        c: {
          texte: 'With AI, we can do it all: code costs almost nothing to write now. Send us all your ideas, even the ones you were saving for next year!',
          reaction: 'All of them? I have a lot. You\'ll regret it, and so will I.',
        },
      },
    },
    'regis-sessions': {
      question: 'I stopped by Websites. Laurence has four screens and four agents running at once. Are we paying four times?',
      reponses: {
        a: {
          texte: 'In tokens, roughly. But throughput doesn\'t quadruple: each session yields less than the one before, and fatigue climbs. I\'m bringing it down to two or three.',
          reaction: 'Diminishing returns. Finally, a concept I know. Bring it down to two or three.',
        },
        b: {
          texte: 'No, parallel sessions cost almost nothing extra.',
          reaction: 'I have the invoice. It has four times as many lines.',
        },
        c: {
          texte: 'Yes, and it\'s worth it: four sessions means four times the work done in parallel. Laurence is our top producer; we should roll out her method.',
          reaction: 'Four times as much? Then why hasn\'t the request backlog melted?',
        },
      },
    },
    'lavigueur-qualite': {
      question: 'We\'re producing more code than before. Is it breaking more?',
      reponses: {
        a: {
          texte: function (etat) {
            var n = incidentsDepuis(etat, debutT(etat) + 1).length;
            return 'Production bugs are ' + tendance(evolution(etat, 'bogues')) + ', and we had '
              + pluriel(n, 'incident', 'incidents') + ' this quarter. That\'s where we stand, no sugar-coating.';
          },
          reaction: 'No sugar-coating. I appreciate it. You\'d be surprised how many people bring sugar.',
        },
        b: {
          texte: 'No quality problem: everything gets reviewed, CI is green and the agents write their own tests. Everything is under control.',
          reaction: selon('That\'s what I see too. Keep it that way.', 'Fine.',
            'Under control? There were incidents this quarter. They\'re in the log.'),
        },
        c: {
          texte: function (etat) {
            return 'Reported test coverage is at ' + F.pourcent(met(etat, 'couvertureApparente', 0)) + '. We\'re covered.';
          },
          reaction: selon('Coverage verified by mutation testing. All right.',
            'Covered by tests that check what, exactly? Ginette told me about “reported” coverage.',
            'Actual coverage is much lower. It\'s right next to it, on your screen.'),
        },
      },
    },
    'berube-criticite': {
      question: function (etat, q) {
        var iot = num(lev(etat, q, 'autonomie', 'iot'), 0) >= 3;
        return 'I\'m told agents are opening PRs on their own on ' + (iot ? 'the sensor firmware' : 'the Operations tools')
          + '. If a sensor gets it wrong and a plant shuts down, who signs?';
      },
      reponses: {
        a: {
          texte: 'A human, always: every PR has a reviewer, and nothing merges without human approval.',
          reaction: selon('Good. I want the reviewer\'s name on file.', 'Hmm.',
            'The configuration says otherwise. I printed it, if you like.'),
        },
        b: {
          texte: 'You\'re right: autonomy should follow criticality. Less for IoT and Operations, more for the websites.',
          reaction: 'One setting per risk level. That\'s almost legal. I approve.',
        },
        c: {
          texte: 'Agents are very reliable on this kind of code: they make fewer mistakes than we do, don\'t get tired, and every PR goes through CI before the merge.',
          reaction: 'An agent can\'t be sued. You can.',
        },
      },
    },
    'regis-modele-haut': {
      question: 'The high-end model costs several times as much as the economy one, per token. Why not the cheapest?',
      reponses: {
        a: {
          texte: 'Because it makes fewer mistakes: a defect avoided costs less than a bug in production. And soon, we\'ll choose the model per task.',
          reaction: selon('Total cost, not the token price. I can live with that.',
            'Okay, but bugs are going up anyway. I\'m watching.', 'Hmm.'),
        },
        b: {
          texte: 'The token price doesn\'t really matter: only quality counts. A cheaper model means more bugs, and bugs always cost more than tokens.',
          reaction: 'Everything matters. The price of coffee matters.',
        },
        c: {
          texte: 'You\'re right, we\'ll switch to the cheapest everywhere.',
          reaction: 'I like that. But Ginette is making a funny face behind you.',
        },
      },
    },
    'regis-modele-economique': {
      question: 'The economy model is my kind of decision. How much are we saving, exactly?',
      reponses: {
        a: {
          texte: function (etat) {
            return 'We pay less per token, but bugs are ' + tendance(evolution(etat, 'bogues'))
              + '. We have to look at total cost, bugs and rework included.';
          },
          reaction: 'Total cost. You sound like my accountant. That\'s a compliment.',
        },
        b: {
          texte: 'A lot, with no downside: the economy model does the same work, it\'s just a little less chatty. The savings go straight to the budget.',
          reaction: selon('Fine.', 'No downside. We\'ll see.',
            'Bugs are going up. Bugs are a downside. Even I know that.'),
        },
        c: {
          texte: 'I don\'t have the exact number.',
          reaction: 'The number is on your own dashboard. At the top. In bold.',
        },
      },
    },

    // ================================================================
    // Deuxième trimestre : la facture
    // ================================================================
    'regis-t2-facture': {
      question: function (etat) {
        return 'Tokens used per week are ' + tendance(evolution(etat, 'jetonsSemaine'))
          + ' compared with last quarter. Explain it to me in dollars. Not in tokens.';
      },
      reponses: {
        a: {
          texte: function (etat) {
            var c0 = totalTrimestre(etat, 'coutSemaine', true), p0 = totalTrimestre(etat, 'pointsLivres', true);
            var c1 = totalTrimestre(etat, 'coutSemaine'), p1 = totalTrimestre(etat, 'pointsLivres');
            if (!c0 || !p0 || !c1 || !p1) return 'I suggest we judge by cost per point delivered: it\'s the only number that tells us whether we\'re paying for something.';
            var a = c0 / p0, b = c1 / p1;
            return 'Cost per point delivered went from ' + F.argent(a) + ' to ' + F.argent(b) + '. '
              + (b <= a * 1.05 ? 'We pay more, but we deliver even more.' : 'Each point costs us more: we need to fix that.');
          },
          reaction: function (etat) {
            var c0 = totalTrimestre(etat, 'coutSemaine', true), p0 = totalTrimestre(etat, 'pointsLivres', true);
            var c1 = totalTrimestre(etat, 'coutSemaine'), p1 = totalTrimestre(etat, 'pointsLivres');
            if (c0 && p0 && c1 && p1 && c1 / p1 > (c0 / p0) * 1.05) return 'A true number, even when it doesn\'t suit you. I prefer that to a nice story. Fix it.';
            return 'A cost per point that holds up. You\'ll go far. Not too far: we have to stay on budget.';
          },
        },
        b: {
          texte: 'It\'s adoption: the more people use the agents, the more it costs, and that\'s a good sign. A rising bill means a team making the tool its own.',
          reaction: selon('Value is keeping up, indeed.', 'A good sign for whom?',
            'A good sign for the vendor, mostly. Value delivered isn\'t keeping up with the bill.'),
        },
        c: {
          texte: 'The vendor raised its prices.',
          reaction: selon('True, I got the notice. That doesn\'t explain everything.', 'Hmm.',
            'I have the contract in front of me. The price hasn\'t moved. Your usage has.'),
        },
      },
    },
    'lavigueur-valeur': {
      question: 'In one sentence: how much value did we deliver this quarter, compared with last?',
      reponses: {
        a: {
          texte: function (etat) {
            var p1 = totalTrimestre(etat, 'pointsLivres'), p0 = totalTrimestre(etat, 'pointsLivres', true);
            if (p1 == null || p0 == null) return 'I\'ll give you value delivered in points, quarter over quarter, instead of the number of PRs.';
            return F.nombre(p1, 0) + ' points, versus ' + F.nombre(p0, 0) + ' last quarter. '
              + (p0 > 0 ? 'That\'s ' + tendance((p1 - p0) / p0) + '.' : '');
          },
          reaction: 'One sentence, one number, one comparison. You could teach my vice-presidents a thing or two.',
        },
        b: {
          texte: function (etat) { return 'Much more than last quarter: we produced an enormous amount of code; we\'re up to ' + F.nombre(met(etat, 'volumeCode', 0), 0) + ' kLOC.'; },
          reaction: 'I didn\'t ask for the weight. I asked what it adds up to.',
        },
        c: {
          texte: 'More than ever, no doubt about it.',
          reaction: selon('The numbers prove you right. This time.', 'I would have preferred a number.',
            '“No doubt about it.” I have the dashboard in front of me, and I have doubts.'),
        },
      },
    },
    'lavigueur-bogues': {
      question: 'Production bugs are climbing again. Is it AI\'s fault?',
      reponses: {
        a: {
          texte: 'Not AI alone: it\'s what we let through. When review is saturated, PRs get through without a real read. Guardrails matter more than ever.',
          reaction: 'So we don\'t have an AI problem. We have a brakes problem. Fix the brakes.',
        },
        b: {
          texte: 'Yes, clearly. The agents write code we haven\'t mastered yet. We\'ll scale back AI everywhere while we regain control of quality.',
          reaction: 'Everywhere? Even where it works? That sounds like panic.',
        },
        c: {
          texte: 'It\'s a coincidence: we shipped more features, so there\'s more surface for bugs. The number will come down on its own.',
          reaction: 'A coincidence that lasts a whole quarter is called a trend.',
        },
      },
    },
    'direction-gonflement': {
      orateur: function (etat) { return direction(ilotExtreme(etat, 'backlog', true, 'appro')); },
      question: function (etat) {
        var id = ilotExtreme(etat, 'backlog', true, 'appro'), il = ilot(etat, id) || {};
        return presentation(id) + ': “Since it\'s fast now, I doubled my requests. So why has my queue grown to '
          + F.nombre(num(il.backlog, 0), 0) + ' points?”';
      },
      reponses: {
        a: {
          texte: 'Because writing code is no longer the main cost: review, testing and maintenance are. We\'ll sort your requests and say no to some.',
          reaction: selon('Say no to some… Fine. As long as they\'re not mine. Kidding. Sort of.',
            'Okay, but you\'ve been saying yes to everything for three months.', 'Hmm.'),
        },
        b: {
          texte: 'We\'ll do it all, don\'t worry: we just add agents and parallel sessions. Your queue will melt away by the end of the quarter.',
          reaction: 'That\'s exactly what you told me last quarter.',
        },
        c: {
          texte: 'Your requests aren\'t a priority.',
          reaction: 'I\'ll mention that to Ms. Lavigueur. She likes my requests.',
        },
      },
    },
    'regis-formation': {
      question: 'Training costs time and money. When does it pay off?',
      reponses: {
        a: {
          texte: function (etat) {
            return 'About four weeks after launch. AI proficiency is at '
              + F.pourcent(moyDevs(etat, 'competenceIA', 0.3)) + ', and training also makes the team warier of injections.';
          },
          reaction: 'A timeline, a number and a benefit I hadn\'t seen. I\'m booking it as an investment.',
        },
        b: {
          texte: 'Right away, from the first week: people apply what they learned the very next morning, and the agents work better with clear instructions.',
          reaction: 'The program takes four weeks to kick in. It\'s on your own invoice.',
        },
        c: {
          texte: 'It can\'t really be measured.',
          reaction: 'Everything can be measured. Even patience. Mine, for example.',
        },
      },
    },
    'regis-sans-formation': {
      question: 'I see we spent nothing on training. Well done. We\'re doing just fine without it, right?',
      reponses: {
        a: {
          texte: 'Not really: without training, we learn by making mistakes, and agent mistakes are paid for in tokens and bugs. A workshop, then a program.',
          reaction: 'Pay a little now to pay less later. That\'s the definition of an investment. I hate it when you\'re right.',
        },
        b: {
          texte: 'Yes: AI is self-taught. The younger ones picked it up in a weekend, and the tools get simpler every month. No need to pay for that.',
          reaction: 'Self-taught. Like Excel in 1995. We still have macros from that era, by the way.',
        },
        c: {
          texte: 'Yes. Training is money down the drain.',
          reaction: 'Even I wouldn\'t say that. And I already cut the muffin budget.',
        },
      },
    },
    'lavigueur-comprehension': {
      question: 'If Mathieu left tomorrow, would anyone understand the Portal code?',
      reponses: {
        a: {
          texte: function (etat, q) {
            if (!vis(etat, q, 'comprehension')) return 'Yes, no problem: code comprehension is excellent, above 80%, and every PR is reviewed by at least one other person on the pod.';
            var c = met(etat, 'comprehension', 0);
            return 'Measured comprehension is at ' + F.pourcent(c) + '. ' + (c >= 0.6 ? 'The Portal would hold up.' : 'Frankly, it would be difficult.');
          },
          reaction: selon('A measured number. Even if it isn\'t pretty, I prefer it.', 'Hmm.',
            'You don\'t measure comprehension. I asked Ginette. She laughed.'),
        },
        b: {
          texte: 'Honestly, I don\'t know: we don\'t measure comprehension. We\'d have to ask everyone to explain their code, without AI.',
          reaction: selon('You don\'t know, and you know how to find out. That\'s already a lot.', 'Hmm.',
            'But you do measure it. It\'s on your dashboard.'),
        },
        c: {
          texte: 'Yes: AI documents everything; anyone can pick up the code.',
          reaction: 'Documentation nobody has read, written by someone who isn\'t here. Reassuring.',
        },
      },
    },
    'berube-skill-douteuse': {
      question: function (etat) {
        var id = carteDouteuse(etat), d = defCapacite(id);
        return 'In your registry, I find a card named “' + ((d && d.nom) || id) + '”. Who wrote it, and what exactly does it do?';
      },
      reponses: {
        a: {
          texte: function (etat) {
            var id = carteDouteuse(etat), c = id && etat.capacites ? etat.capacites[id] : null;
            return c && c.proprietaire
              ? 'It belongs to ' + nomDev(c.proprietaire) + ', who reviewed it and answers for it.'
              : 'It has an owner, who reviewed it and answers for it.';
          },
          reaction: selon('A name, an accountability. That\'s all I ask. Well, almost all.', 'Hmm.',
            'The card\'s record says “owner: none.” I highlighted it in yellow.'),
        },
        b: {
          texte: 'It\'s a very popular card in the community: thousands of downloads, excellent ratings, and several large companies recommend it.',
          reaction: 'Popularity isn\'t provenance. A skill can contain scripts. Who read them?',
        },
        c: {
          texte: 'I don\'t know yet. We\'re taking inventory: every card will have an owner, or it goes.',
          reaction: selon('An inventory. My favourite word after “compliant.”', 'It already has an owner, though. Check your registry.', 'Hmm.'),
        },
      },
    },
    'regis-plafond-strict': {
      question: 'The hard token cap is my favourite measure. Any downsides?',
      reponses: {
        a: {
          texte: 'Yes: when it cuts off, work stops midway. It prevents agent loops, but it can block a team on a Friday afternoon.',
          reaction: 'A brake that can lock the wheels. Fine. We keep the brake, we watch the wheels.',
        },
        b: {
          texte: 'None: the cap only cuts off agent loops spinning for nothing. Normal work stays well below it; you don\'t even feel it.',
          reaction: 'There\'s always a downside. If you don\'t see it, you\'re not looking.',
        },
        c: {
          texte: 'None, and we should lower it further.',
          reaction: selon('Fine.', 'Easy. Even I think that\'s harsh.',
            'The request backlog is already climbing. You want the teams working by candlelight?'),
        },
      },
    },
    'direction-tablette': {
      orateur: function () { return direction('inspections'); },
      question: function () {
        return presentation('inspections') + ': “Actually, we\'d like a tablet version of the field app. With AI, that\'s a week\'s work, right?”';
      },
      reponses: {
        a: {
          texte: 'The code, maybe. But first we need a spec: which screens, what offline sync. An agent does exactly what it\'s asked.',
          reaction: selon('A spec. Fine. Karim will be happy, he\'s been asking me for one since last year.', 'Okay.', 'Hmm.'),
        },
        b: {
          texte: 'Yes, one week: it\'s the same app, only bigger. The agent adapts the screens on its own, and the Inspections team already knows the code by heart.',
          reaction: 'Perfect. I\'ll announce it to the inspectors on Monday. You\'ll be there for questions.',
        },
        c: {
          texte: 'No, it\'s impossible this year.',
          reaction: 'Impossible? Kevin told me he could do it over the weekend.',
        },
      },
    },
    'regis-modele-par-tache': {
      question: 'Choosing the model per task: how much has that saved us?',
      reponses: {
        a: {
          texte: function (etat) {
            return 'Weekly cost is ' + tendance(evolution(etat, 'coutSemaine'))
              + ' since last quarter. The small model handles lint and documentation; the big one keeps the architecture.';
          },
          reaction: 'The right tool for the right job. My father used to say that. He was a plumber, but still.',
        },
        b: {
          texte: 'About 35%, guaranteed: that\'s the figure the vendor advertises for this kind of setup, and our tasks look like theirs.',
          reaction: selon('That\'s roughly what I see. No need for “guaranteed,” though.', 'Hmm.',
            'Guaranteed by whom? Show me the guarantee. The dashboard doesn\'t show 35%.'),
        },
        c: {
          texte: 'Nothing, but quality is better.',
          reaction: selon('Fine.', 'Quality, sure. But I was hoping for a number.',
            'Nothing? Costs went down. You just hid good news from me. That\'s new.'),
        },
      },
    },
    'regis-serveur-gpu': {
      question: 'We have a GPU server in the server room. Is it doing anything, or just heating the room?',
      reponses: {
        a: {
          texte: 'It runs the local model: nearly free tokens, and the data stays with us. We keep an external model for architecture.',
          reaction: selon('A paid-for server that actually works. I\'m going to frame it. The server, not you.', 'Hmm.',
            'The default model is external. The server is very expensive furniture.'),
        },
        b: {
          texte: 'It\'s asleep, to be honest. We use an external model for everything: it\'s simpler, and nobody has taken the time to configure the server yet.',
          reaction: selon('At least you say so. We\'ll revisit it before next quarter.', 'Hmm.',
            'It isn\'t asleep at all; it runs your default model. You don\'t know your own configuration?'),
        },
        c: {
          texte: 'We\'re keeping it for phase two.',
          reaction: 'Phase two. Like the underground parking.',
        },
      },
    },

    // ================================================================
    // Troisième trimestre : les fuites
    // ================================================================
    'berube-inspections-donnees': {
      question: 'The inspectors\' photos and forms contain names, addresses, sometimes licence plates. Where does that data go when an agent touches it?',
      reponses: {
        a: {
          texte: function (etat, q) {
            if (!vis(etat, q, 'expositionLoi25')) return 'Our exposure is minimal: agents keep nothing in memory, exchanges are encrypted and everything complies with Quebec\'s Law 25.';
            var x = met(etat, 'expositionLoi25', 0);
            return 'Our Law 25 exposure is at ' + F.nombre(x, 0) + ' out of 100. ' + (x < 40 ? 'Classification is doing its job.' : 'That\'s too high: we\'re tightening up.');
          },
          reaction: selon('A number and a source. I have nothing to add. That doesn\'t happen often.', 'Hmm.',
            '“Compliant,” without having classified a single piece of data? You just wrote my next chapter.'),
        },
        b: {
          texte: function (etat, q) {
            return 'I can\'t tell you precisely until the data is classified. Classification is '
              + (lev(etat, q, 'classification') === 'non' ? 'still to launch, and I\'m launching it.' : 'under way.');
          },
          reaction: selon('You only protect what you\'ve labelled. You listened to me. I\'m moved.', 'Fine.', 'But you do measure that exposure. It\'s on your dashboard.'),
        },
        c: {
          texte: 'Nowhere: our agents only read code, never the photos or the forms themselves.',
          reaction: 'An agent that reads an inspection form reads a name. A name is personal information.',
        },
      },
    },
    'berube-incident-securite': {
      question: function (etat) {
        var i = dernierIncidentSecu(etat) || {};
        return 'The security incident in week ' + F.nombre(num(i.debut, 0), 0) + ', in the ' + nomIlot(i.ilot)
          + ' pod: what has changed since?';
      },
      reponses: {
        a: {
          texte: 'We locked that pod\'s agents in a sandbox, and a human now approves every external output.',
          reaction: selon('Measures, settings, a pod. I have nothing to add. That\'s rare.', 'Hmm.',
            'I checked the settings this morning: no sandbox, no approval. Would you like to rephrase?'),
        },
        b: {
          texte: 'It was an isolated case, a very particular combination of circumstances. We analyzed what happened, and it won\'t happen again.',
          reaction: '“It won\'t happen again” is the most quoted sentence in my litigation files.',
        },
        c: {
          texte: 'We logged it in the confidentiality incident register, assessed the risk of serious injury, and we\'ll notify the CAI, Quebec\'s privacy regulator, if required.',
          reaction: 'Law 25, to the letter. Now I also want to know what prevents the next one.',
        },
      },
    },
    'berube-triade': {
      question: function (etat) {
        return 'The ' + nomIlot(triadesCompletes(etat)[0]) + ' pod has all three lights on: private data, untrusted content, external output. Explain to me why that isn\'t serious.';
      },
      reponses: {
        a: {
          texte: 'It is serious. A booby-trapped email can ask the agent to send our data outside. We have to cut one leg: human approval of outputs, or remove a tool.',
          reaction: selon('Finally, someone who doesn\'t tell me “it isn\'t serious.” Cut the leg. Today.', 'Hmm.', 'Hmm.'),
        },
        b: {
          texte: 'Our model is trained to resist prompt injection: the vendor publishes tests on it, and the results improve with every new version.',
          reaction: 'Trained to resist isn\'t immune. No model is. I read it in three reports.',
        },
        c: {
          texte: 'The content comes from our suppliers, whom we\'ve dealt with for years. It\'s reliable, and their emails already go through our spam filter.',
          reaction: 'A supplier is an inbox. An inbox is the internet.',
        },
      },
    },
    'berube-registre': {
      question: 'How many MCP servers are installed here, and who is allowed to install them?',
      reponses: {
        a: {
          texte: 'Anyone, for now: the registry is self-service.',
          reaction: selon('At least that\'s honest. Now close the door.', 'Hmm.',
            'That\'s no longer true: you changed the policy. You don\'t follow your own decisions?'),
        },
        b: {
          texte: 'Nobody without approval: there\'s an allowlist, and every card has an owner.',
          reaction: selon('Allowlist and owners. I\'ll sleep tonight.',
            'An allowlist, yes. Owners, not yet. I note the nuance.',
            'The registry is self-service. Yannick just installed something while you were talking.'),
        },
        c: {
          texte: function (etat, q) {
            if (vis(etat, q, 'surfaceAttaque')) {
              return 'Our attack surface is at ' + F.nombre(met(etat, 'surfaceAttaque', 0), 0) + ' out of 100'
                + (vis(etat, q, 'cartesOrphelines') ? ', with ' + pluriel(met(etat, 'cartesOrphelines', 0), 'ownerless card', 'ownerless cards') : '')
                + '. I can give you the full list.';
            }
            return 'Only a few, and all vetted.';
          },
          reaction: selon('A list! Send it to me. As a PDF, if possible.', 'Hmm.',
            'Vetted by whom? With no gateway and no owners, nobody knows what\'s installed. Not even you.'),
        },
      },
    },
    'lavigueur-kevin': {
      question: 'Kevin showed me his campaign-tracking app. It\'s brilliant. Why doesn\'t IT do things like that?',
      reponses: {
        a: {
          texte: 'It meets a real need we didn\'t see coming. We\'ll support it: approved tools, coaching, and its data classified like ours.',
          reaction: selon('Keeping Kevin enthusiastic and Counsel Bérubé calm. If you pull that off, you get a raise.',
            'The idea is good. It isn\'t applied yet, though.', 'Hmm.'),
        },
        b: {
          texte: 'Development outside IT is banned by our policy, for good reasons: security, data, maintenance. We\'ll shut it down and take over the need on our side.',
          reaction: function (etat) {
            var n = met(etat, 'applisFantomes', 0);
            return n > 1 ? 'Shut it down? He\'s already built ' + F.nombre(n - 1, 0) + ' others. I use them.'
              : 'Shut it down? I use it every morning.';
          },
        },
        c: {
          texte: 'It\'s brilliant. We should build more like it.',
          reaction: 'Counsel Bérubé is waving at me. He looks like he has something to say about his data.',
        },
      },
    },
    'regis-acces-prod': {
      question: 'Someone mentioned an “UPDATE without a WHERE.” I don\'t know what that is, but it sounds expensive. Can our agents write to production?',
      reponses: {
        a: {
          texte: function (etat, q) {
            var n = ilotsOu(etat, q, 'accesProd', function (v) { return v === 'oui'; }).length;
            return 'Yes, in ' + pluriel(n, 'pod', 'pods') + '. One command like that changes an entire table. I\'m switching the agents to read-only: every write will go through a reviewed PR.';
          },
          reaction: 'A whole table at once? Read-only. Right now. I don\'t need to know more.',
        },
        b: {
          texte: 'No: they\'re read-only everywhere; it\'s the default configuration of our environments.',
          reaction: 'That\'s not what the configuration says. Benoît showed it to me. He looked tired.',
        },
        c: {
          texte: 'Yes, but we have nightly backups: worst case, we restore yesterday\'s database and lose a few hours of work. The risk is covered.',
          reaction: 'Tested backups? When did you last do a restore? Don\'t answer; your face just did.',
        },
      },
    },
    'lavigueur-concurrent': {
      question: 'Our competitor announces that 90% of its code is written by AI. What about us?',
      reponses: {
        a: {
          texte: function (etat) {
            return 'I don\'t know that percentage, and it\'s not the right measure. Value delivered is '
              + tendance(evolution(etat, 'pointsLivres')) + ', with ' + pluriel(incidentsDepuis(etat, debutT(etat) + 1).length, 'incident', 'incidents') + ' this quarter.';
          },
          reaction: 'I prefer your measure to their slogan. I\'m going to steal it for my next speech.',
        },
        b: {
          texte: '95%. Nearly all our code goes through the agents now: we beat them, by a mile.',
          reaction: 'Where did you get that number? Ah. Nowhere.',
        },
        c: {
          texte: 'At least as much: everyone uses the agents every day, so we must be in the same ballpark.',
          reaction: '“At least.” Very precise.',
        },
      },
    },
    'direction-ia-maison': {
      orateur: function () { return direction('web'); },
      question: function () {
        return presentation('web') + ': “We have a budget. We want our own AI, with our own vendor. It would be simpler than going through you.”';
      },
      reponses: {
        a: {
          texte: 'It\'s possible, within the approved registry: the same data rules, the same gateway, and we\'ll coach you.',
          reaction: selon('With coaching? That works for me. Mostly, I didn\'t want to wait six months.', 'Okay. When will I hear back from you?', 'Hmm.'),
        },
        b: {
          texte: 'Do what you want, it\'s not our problem.',
          reaction: 'Perfect. I\'ll sign the contract tomorrow. Counsel Bérubé will read it afterwards, I imagine.',
        },
        c: {
          texte: 'No. Everything goes through IT, period: it\'s the only way to keep control over security, data and contracts.',
          reaction: 'Then I\'ll wait. Like the fall campaign site. Which launched in winter.',
        },
      },
    },
    'berube-passerelle': {
      question: 'A vendor changed its MCP server\'s description without telling anyone. Agents read those descriptions as instructions. How would we know?',
      reponses: {
        a: {
          texte: 'Through the MCP gateway: every tool call is logged there, and a changed description gets flagged.',
          reaction: selon('A log. I love logs. Especially the kind you can enter into evidence.', 'Hmm.',
            'What gateway? You don\'t have one. I checked with Sébastien.'),
        },
        b: {
          texte: function (etat, q) {
            return lev(etat, q, 'passerelleMCP') === true
              ? 'Honestly, we\'d only know if someone noticed odd behaviour.'
              : 'Honestly, we wouldn\'t know. We\'d need an MCP gateway to see it.';
          },
          reaction: selon('At least you know what you\'re missing. Install it.', 'Fine.', 'But you have a gateway. That\'s exactly what it\'s for.'),
        },
        c: {
          texte: 'Vendors always warn us.',
          // La hausse du CRM n'est citée que si l'événement « crm-tarif » (14-donnees-evenements) a eu lieu.
          reaction: function (etat) {
            return etat && etat.evenementsVus && etat.evenementsVus['crm-tarif'] != null
              ? 'Always. Like the CRM price increase, which we found out about from the invoice.'
              : 'Always? I\'ve read their contracts. “Without notice,” in fine print, on page 14.';
          },
        },
      },
    },
  });
})(globalThis.LTP = globalThis.LTP || {});
