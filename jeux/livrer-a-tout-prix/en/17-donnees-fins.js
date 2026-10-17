/* en/17-donnees-fins.js */
/* Édition anglaise : profils de fin et textes du bilan (voir en/LISEZMOI.md).
   Les fonctions de texte reprennent la logique de js/17-donnees-fins.js (mêmes faits, mêmes seuils, mêmes branches) ;
   seules les phrases changent. Les petits outils sont recopiés ici parce que ceux du français sont privés.
   Pluriel anglais : singulier pour 1 seulement (0 incidents, 1 incident, 2 incidents). */
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
  function pl(n, un, plusieurs) { return nb(n) + ' ' + (Math.abs(n) === 1 ? un : plusieurs); }
  // En début de phrase : « One incident », « 3 incidents ».
  function Pl(n, un, plusieurs) { return n === 1 ? 'One ' + un : pl(n, un, plusieurs); }
  function joindre(phrases) { return phrases.filter(function (p) { return !!p; }).join(' '); }
  function liste(l) { return l.length < 2 ? l.join('') : l.slice(0, -1).join(', ') + ' and ' + l[l.length - 1]; }
  function lev(etat, q, id, ilot) { try { return q && q.levier ? q.levier(etat, id, ilot) : undefined; } catch (e) { return undefined; } }

  function phraseComprehension(x) {
    if (x.comprehension == null) return null;
    if (x.comprehensionDepart == null) return 'Average code comprehension ended the year at ' + pc(x.comprehension) + '.';
    var d = x.comprehension - x.comprehensionDepart;
    if (Math.abs(d) < 0.03) return 'Average code comprehension held steady: ' + pc(x.comprehensionDepart) + ' in January, ' + pc(x.comprehension) + ' in December.';
    return 'Average code comprehension ' + (d > 0 ? 'rose' : 'fell') + ' from ' + pc(x.comprehensionDepart) + ' to ' + pc(x.comprehension) + '.';
  }
  function comprehensionTenue(x) { return x.comprehension != null && x.comprehensionDepart != null && x.comprehension >= x.comprehensionDepart - 0.05; }

  function phraseSecurite(x) {
    if (x.fuites == null && x.incidentsSecurite == null) return null;
    if (x.fuites) return 'One blot on the record: ' + (x.fuites === 1 ? 'a data leak' : pl(x.fuites, 'data leak', 'data leaks')) + ' during the year.';
    if (x.fuitesMajeures) return 'One blot on the record: a major data leak.';
    if (x.incidentsSecurite) return Pl(x.incidentsSecurite, 'security incident', 'security incidents') + ', but no data leaks.';
    return 'No data leaks, no security incidents.';
  }

  function phraseBudget(x) {
    if (x.ratioBudget == null) return null;
    if (x.ratioBudget <= 1.05) return 'And the AI budget held.';
    return 'The AI budget, on the other hand, came in ' + pc(x.ratioBudget - 1) + ' over.';
  }

  // Les garde-fous réellement en place en fin d'année.
  function rails(etat, q) {
    var l = [];
    if (lev(etat, q, 'tests') === 'mutation') l.push('mutation testing');
    else if (lev(etat, q, 'tests') === 'unitaires') l.push('unit tests');
    var ci = lev(etat, q, 'ci');
    if (ci === 'complet' || ci === 'tests') l.push('CI that tests every PR');
    var t = lev(etat, q, 'taillePR');
    if (t === 400 || t === 200) l.push('PRs capped at ' + t + ' lines');
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

  LTP.traduire('fins', {
    reussie: {
      titre: 'The Successful Transition',
      sousTitre: function (etat) { return comprehensionTenue(faits(etat)) ? 'Faster, and still understood.' : 'Faster. Understood, not quite as well.'; },
      texte: function (etat, q) {
        var x = faits(etat), r = rails(etat, q);
        return joindre([
          x.valeur != null ? 'In ' + nb(x.semaines || 52) + ' weeks, the team delivered ' + nb(x.valeur) + ' points of value'
            + (x.ratioDebit != null && x.ratioDebit >= 1.05 ? ', ' + pc(x.ratioDebit - 1) + ' more than a team without agents.' : '.') : null,
          x.partLgtm != null && x.partLgtm < 0.25 ? 'Review held up: ' + (x.partLgtm < 0.005 ? 'not a single merged PR' : 'only ' + pc(x.partLgtm) + ' of merged PRs') + ' went through on a bare “LGTM.”' : null,
          x.incidents == null ? null
            : x.incidents === 0 ? 'Not one incident all year.'
            : (x.incidents === 1 ? 'Just one incident all year' : Pl(x.incidents, 'incident', 'incidents') + ' during the year')
              + (comprehensionTenue(x) ? ', fixed by people who understood their code.' : '.'),
          phraseComprehension(x),
          r.length ? 'The agents ran on guardrails: ' + liste(r) + '.' : null,
          autonomieSelonCriticite(etat, q) ? 'And agent autonomy followed criticality: kept low for the sensors and operations, higher for the web.' : null,
          phraseSecurite(x),
          phraseBudget(x),
        ]);
      },
      replique: 'I didn\'t understand all your acronyms. But I understood your numbers, and they told the truth. We carry on.',
      lecon: 'AI speeds up writing code; the transition succeeds when you invest as much in review, tests, comprehension and data as in the agents.',
    },

    usine: {
      titre: 'The PR Factory',
      sousTitre: 'Lots of PRs, few readers.',
      texte: function (etat) {
        var x = faits(etat);
        var lgtm = x.partLgtm != null && x.partLgtm >= 0.3;
        return joindre([
          'The PRs kept coming' + (x.valeur != null ? ', for ' + nb(x.valeur) + ' points of value delivered.' : '.'),
          x.volume != null && x.volumeDepart ? 'The code, meanwhile, went from ' + nb(x.volumeDepart) + ' kLOC to ' + nb(x.volume) + ' kLOC.' : null,
          lgtm ? 'The reviewers couldn\'t keep up: ' + pc(x.partLgtm) + ' of merged PRs went through on a bare “LGTM.”' : null,
          x.incidents ? 'The result: ' + pl(x.incidents, 'incident', 'incidents') + ' during the year.' : null,
          phraseComprehension(x),
          lgtm ? 'The assembly line was fast. Quality control was on a coffee break.' : null,
        ]);
      },
      replique: 'You did a lot of things. Now I\'d like to know which ones were actually useful.',
      lecon: 'When writing speeds up, the bottleneck moves to review: producing more PRs than the team can read means manufacturing debt and incidents.',
    },

    musee: {
      titre: 'The Museum',
      sousTitre: 'Nothing broke. Nothing moved.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          x.incidents == null ? null
            : x.incidents === 0 ? 'Nothing broke this year: not one incident.'
            : 'Almost nothing broke: ' + (x.incidents === 1 ? 'a single incident' : pl(x.incidents, 'incident', 'incidents')) + ' in a year.',
          x.valeur != null ? 'IT moved in small steps: ' + nb(x.valeur) + ' points delivered' + (x.ratioDebit != null && x.ratioDebit < 1 ? ', fewer than a team without agents.' : '.') : null,
          x.satisfaction != null && x.satisfactionDepart != null && x.satisfaction < x.satisfactionDepart - 0.02
            ? 'Meanwhile, business satisfaction went from ' + pc(x.satisfactionDepart) + ' to ' + pc(x.satisfaction) + '.' : null,
          x.fantomes ? 'Kevin and his colleagues filled the gap with ' + pl(x.fantomes, 'app', 'apps') + ' built outside IT, which nobody reviewed.' : null,
          x.fantomes ? 'Caution didn\'t remove the risk: it moved it to Marketing.' : null,
        ]);
      },
      replique: function (etat) {
        return faits(etat).fantomes
          ? 'Nothing burned down, it\'s true. Kevin, on the other hand, now has more apps than your team.'
          : 'Nothing burned down, it\'s true. Nothing moved forward either.';
      },
      lecon: 'Refusing AI doesn\'t remove the risk: the demand finds another path, outside IT\'s guardrails.',
    },

    farwest: {
      titre: 'The Wild West',
      sousTitre: 'Everything was plugged in. Into everything.',
      texte: function (etat) {
        var x = faits(etat);
        var fuite = !!(x.fuites || x.fuitesMajeures);
        return joindre([
          x.surface != null ? 'The attack surface ended the year at ' + nb(x.surface * 100) + ' out of 100'
            + (x.surface >= 0.6 ? ': agents reading private data and outside content, and able to send anything out.' : '.') : null,
          x.fuites ? Pl(x.fuites, 'data leak later', 'data leaks later') + ', the lethal trifecta is no longer a theory for anyone.'
            : x.fuitesMajeures ? 'One major leak later, the lethal trifecta is no longer a theory for anyone.' : null,
          x.incidentsSecurite ? Pl(x.incidentsSecurite, 'security incident', 'security incidents') + ' in total.' : null,
          fuite ? 'Counsel Bérubé has started a new binder. A thick one. With colour-coded tabs.' : null,
          x.valeur != null ? 'The team delivered ' + nb(x.valeur) + ' points this year' + (fuite ? (x.fuites > 1 ? ', but the leaks are what people will remember.' : ', but the leak is what people will remember.') : '.') : null,
        ]);
      },
      replique: function (etat) {
        var x = faits(etat);
        return x.fuites || x.fuitesMajeures
          ? 'I never want to hear about a data leak from a journalist again. Never again.'
          : 'Everything was plugged into everything, and we got away with it. Luck is not a strategy.';
      },
      lecon: 'Private data, untrusted content and external output in the same agent open the door to exfiltration: cutting one leg of the trifecta is enough to close it.',
    },

    facture: {
      titre: 'The Eye-Watering Bill',
      sousTitre: 'It works. It costs.',
      texte: function (etat, q) {
        var x = faits(etat);
        var causes = [];
        var sess = lev(etat, q, 'sessionsMax');
        if (typeof sess === 'number' && sess >= 3) causes.push('up to ' + sess + ' agent sessions per person');
        if (lev(etat, q, 'modeleParTache') !== true) causes.push('a single model for every task');
        if (lev(etat, q, 'plafondJetons') === 'aucun') causes.push('no cap to cut off the loops');
        return joindre([
          'The results hold up' + (x.valeur != null ? ': ' + nb(x.valeur) + ' points delivered' : '') + (x.incidents != null ? ', ' + pl(x.incidents, 'incident', 'incidents') : '') + '.',
          x.depense != null ? 'But the AI cost ' + F.argent(x.depense)
            + (x.ratioBudget != null && x.ratioBudget > 1 ? ', ' + pc(x.ratioBudget - 1) + ' more than planned.' : '.') : null,
          causes.length ? 'At year end, the setup still had ' + liste(causes) + '.' : null,
          x.ratioBudget != null && x.ratioBudget > 1.2 ? 'Régis had the bill framed. Not for the right reasons.' : null,
        ]);
      },
      replique: 'I like the result. Régis, not nearly as much. And Régis signs the cheques.',
      lecon: 'AI cost is managed like everything else: the right model for each task, limited sessions and caps that cut off the loops.',
    },

    chateau: {
      titre: 'The House of Cards',
      sousTitre: 'Lots of code. Nobody to explain it.',
      texte: function (etat, q) {
        var x = faits(etat);
        var gel = lev(etat, q, 'embauche') === 'gel';
        return joindre([
          x.volume != null && x.volumeDepart ? 'The code went from ' + nb(x.volumeDepart) + ' kLOC to ' + nb(x.volume) + ' kLOC.' : null,
          phraseComprehension(x),
          x.departs ? (x.departs >= 2 ? nb(x.departs) + ' people left, taking what they knew with them.' : 'One person left, taking what they knew with them.') : null,
          x.releve != null && x.releve < 0.4 ? 'Succession stands at ' + pc(x.releve) + (gel ? ': junior hiring is frozen.' : '.') : null,
          x.comprehension != null && x.comprehension < 0.5 ? 'The day something breaks, nobody will know where to look. It holds. For now.' : null,
        ]);
      },
      replique: 'What worries me isn\'t what\'s broken. It\'s that nobody knows why the rest works.',
      lecon: 'Speed without comprehension is a debt: it stays invisible until the incident, or until the person who knew walks out the door.',
    },

    congedie: {
      titre: 'Thank You for Your Contribution',
      sousTitre: 'Management made its call before December.',
      texte: function (etat) {
        var x = faits(etat);
        return joindre([
          'The game ended' + (x.semaines != null ? ' in week ' + nb(x.semaines) : ' before the end of the year') + ': executive confidence fell below the threshold.',
          x.incidents != null || x.depense != null
            ? 'On file: ' + joindre([
              x.incidents != null ? pl(x.incidents, 'incident', 'incidents') + (x.depense != null ? ',' : '.') : null,
              x.depense != null ? F.argent(x.depense) + ' spent on AI.' : null,
            ])
            : null,
          'Ms. Lavigueur thanked everyone for their contribution, which, at GPU, is never a good sign.',
        ]);
      },
      replique: 'Thank you for your contribution. Sincerely. Ginette will see you out.',
      lecon: 'Trust is lost faster than it is won: true numbers, even bad ones, are worth more than a nice story the dashboard contradicts.',
    },
  });

  LTP.traduire('bilan', {
    etoiles: {
      0: 'Zero stars. Ginette offers you a coffee and a second chance.',
      1: 'One star. The team survived the year, and that\'s something.',
      2: 'Two stars. Some good moves, but the dashboard spoke louder.',
      3: 'Three stars. An honest transition, with a few scars.',
      4: 'Four stars. The committee wants you back next year.',
      5: 'Five stars. Even Régis smiled. Briefly.',
    },
  });
})(globalThis.LTP = globalThis.LTP || {});
