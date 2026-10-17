/* 30-metriques.js */
/* Métriques du tableau de bord (CONCEPTION §5, 18-donnees-metriques.js) : calcul hebdomadaire de etat.metriques,
   historique etat.historique[id][semaine], visibilité des métriques cachées, cumuls de la partie pour le bilan.
   Les champs produits par les autres systèmes sont lus défensivement (ils peuvent manquer).
   Champs propres : etat.metriquesDetail (valeurs exactes et détail par îlot), etat.cumuls (totaux de la partie). */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.metriques = {
    fenetreMttr: 8,               // semaines d'incidents prises en compte dans le MTTR
    mttrDefaut: 8,                // heures, si aucun incident et aucun modèle de MTTR
    // Arrondi du MTTR affiché selon le niveau d'observabilité (0 aucune, 1 base, 2 complète), en heures.
    arrondiMttr: [4, 1, 0.1],
    seuilRetardSemaines: 4,       // un îlot est en retard si son backlog dépasse n semaines de demande
    seuilRetardDelai: 3,          // … ou si son délai de livraison dépasse n semaines
    // Surface d'attaque calculée ici si 25-donnees-secu ne la fournit pas (score 0–100).
    surface: {
      branche: 0.4 / 3,           // par branche de la triade allumée
      provenance: { communaute: 0.12, fournisseur: 0.07, personnelle: 0.06, maison: 0.03 },
      cartesMax: 0.4,
      autonomie: 0.2,             // × autonomie / 4
      bacASable: [1, 0.75, 0.5],  // multiplicateur selon le rang du levier
      passerelle: 0.8,
      accesProd: [1, 0.9, 0.8],
    },
    // Exposition Loi 25 calculée ici si 25-donnees-secu ne la fournit pas : sensibilité × (1 − classification) × sortie.
    loi25: { sortieExterne: 0.7, sortieLocale: 0.15, parFournisseur: 0.1 },
    // Causes d'incident comptées comme des incidents de sécurité, et parmi elles comme des fuites de données.
    // Comparaison par sous-chaîne, en minuscules (causes de 25-donnees-secu : injection, paquetHallucine, mcpEmpoisonne,
    // changementSilencieux, destructionProd, fuiteDonnees ; les événements peuvent en créer d'autres).
    causesSecurite: ['injection', 'exfiltr', 'fuite', 'typosquat', 'slopsquat', 'paquet', 'empoison', 'silencieux', 'destruction', 'loi25', 'secu'],
    causesFuite: ['fuite', 'exfiltr', 'empoison', 'loi25'],
    graviteMajeure: 0.6,          // gravité à partir de laquelle une fuite est « majeure »
    seuilPerime: 0.4,             // fraîcheur sous laquelle une carte est périmée
    // Seuils de visibilité (rangs dans la liste des valeurs du levier).
    majoriteIlots: 0.5,           // part des îlots qui doivent avoir le bon réglage (leviers surchargeables)
    observabiliteReclasses: 2,    // niveau d'observabilité qui révèle les incidents reclassés en demandes (Goodhart)
  };
  var K = LTP.K.metriques;

  function num(x, defaut) { return typeof x === 'number' && isFinite(x) ? x : (defaut == null ? 0 : defaut); }
  function der(il, champ) { return num(il.derniere && il.derniere[champ]); }

  function initialiser(etat) {
    etat.cumuls = etat.cumuls || nouveauxCumuls();
    etat.metriquesDetail = etat.metriquesDetail || {};
    calculer(etat, null, false);
  }

  function nouveauxCumuls() {
    return { semaines: 0, pointsLivres: 0, points: 0, prFusionnees: 0, code: 0, bogues: 0, incidents: 0, incidentsGraves: 0,
      incidentsSecurite: 0, fuites: 0, fuitesMajeures: 0, lgtm: 0, pr: 0, defautsEchappes: 0, deploiements: 0, jetons: 0 };
  }

  // Niveau d'observabilité : 0 aucune, 1 base, 2 complète (levier, ou jauge org.observabilite).
  function niveauObservabilite(etat) {
    var r = LTP.q.defLevier('observabilite') ? LTP.q.rangLevier(etat, 'observabilite') : 0;
    var o = num(etat.org.observabilite);
    var parJauge = o >= 0.9 ? 2 : o >= 0.4 ? 1 : 0;
    return Math.max(r, parJauge);
  }

  // Part des îlots dont la valeur effective du levier est dans la liste.
  function partIlots(etat, levier, valeurs) {
    var ids = etat.ordreIlots || [];
    if (!ids.length) return 0;
    var n = 0;
    ids.forEach(function (id) { if (valeurs.indexOf(LTP.q.levier(etat, levier, id)) >= 0) n += 1; });
    return n / ids.length;
  }
  function majorite(etat, levier, valeurs) { return partIlots(etat, levier, valeurs) >= K.majoriteIlots; }

  // --- Visibilité (CONCEPTION §5) ---
  var REGLES = {
    frequenceDeploiement: function (etat) { return majorite(etat, 'ci', ['complet']); },
    tauxEchec: function (etat) { return majorite(etat, 'ci', ['tests', 'complet']); },
    couvertureReelle: function (etat) { return majorite(etat, 'tests', ['mutation']) || niveauObservabilite(etat) >= 1; },
    comprehension: function (etat) {
      var r = LTP.q.levier(etat, 'revuesExplication');
      return (r != null && r !== 'non') || niveauObservabilite(etat) >= 2;
    },
    surfaceAttaque: function (etat) {
      return LTP.q.levier(etat, 'passerelleMCP') === true || LTP.q.levier(etat, 'politiqueRegistre') === 'proprietaires';
    },
    expositionLoi25: function (etat) {
      var c = LTP.q.levier(etat, 'classification');
      return c != null && c !== 'non';
    },
    cartesOrphelines: function (etat) { return LTP.q.levier(etat, 'politiqueRegistre') === 'proprietaires'; },
  };

  function visible(etat, id) {
    var d = LTP.donnees.metriquesParId && LTP.donnees.metriquesParId[id];
    if (!d) return false;
    if (etat.options && etat.options.voirCachees) return true;
    if (!d.cachee) return true;
    var r = REGLES[id];
    if (!r) return false;
    try { return !!r(etat); } catch (e) { return false; }
  }

  // --- Calculs auxiliaires ---
  function incidentsDeLaSemaine(etat) {
    return (etat.incidents || []).filter(function (i) { return i && i.debut === etat.semaine; });
  }
  function correspond(cause, liste) {
    var c = String(cause || '').toLowerCase();
    return liste.some(function (m) { return c.indexOf(m) >= 0; });
  }
  function masque(etat, i) {
    // Goodhart « incidents » : un incident reclassé en demande n'apparaît qu'avec l'observabilité.
    return (i.reclasse || i.masque) && niveauObservabilite(etat) < K.observabiliteReclasses;
  }
  function incidentsVisiblesSemaine(etat) {
    var s = etat.semaine;
    return (etat.incidents || []).filter(function (i) {
      if (!i || masque(etat, i)) return false;
      if (i.debut === s) return true;
      if (i.debut > s) return false;
      if (typeof i.finSemaine === 'number') return i.finSemaine >= s;
      return !i.resolu;
    }).length;
  }

  function mttrExact(etat) {
    var depuis = etat.semaine - K.fenetreMttr;
    var recents = (etat.incidents || []).filter(function (i) { return i && i.debut > depuis && typeof i.mttrHeures === 'number'; });
    if (recents.length) return O.moyenne(recents, 'mttrHeures');
    var dette = LTP.systemes.dette;
    var ids = etat.ordreIlots || [];
    if (dette && typeof dette.mttr === 'function' && ids.length) {
      var s = 0;
      for (var k = 0; k < ids.length; k++) {
        var v;
        try { v = dette.mttr(etat, ids[k], 0.5); } catch (e) { v = K.mttrDefaut; }
        s += num(v, K.mttrDefaut);
      }
      return s / ids.length;
    }
    return K.mttrDefaut;
  }
  function arrondirA(v, pas) { return pas > 0 ? Math.round(v / pas) * pas : v; }

  function retardIlot(etat, il) {
    if (typeof il.retard === 'boolean') return il.retard ? 1 : 0;
    if (typeof il.retard === 'number' && isFinite(il.retard)) return O.borner(il.retard);
    var def = LTP.q.defIlot(il.id) || {};
    var demande = num(def.demandeBase, 10);
    if (num(il.backlog) > K.seuilRetardSemaines * demande) return 1;
    if (der(il, 'delai') > K.seuilRetardDelai) return 1;
    return 0;
  }

  function carteDef(id) { return LTP.q.defCapacite(id) || {}; }

  function surfaceIlot(etat, il) {
    var KS = K.surface, q = LTP.q;
    var t = il.triade || {};
    var s = KS.branche * ((t.prive ? 1 : 0) + (t.nonFiable ? 1 : 0) + (t.sortie ? 1 : 0));
    var c = 0;
    (il.capacites || []).forEach(function (id) { c += KS.provenance[carteDef(id).provenance] || 0; });
    s += Math.min(KS.cartesMax, c);
    s += KS.autonomie * (num(Number(q.levier(etat, 'autonomie', il.id))) / 4);
    s *= KS.bacASable[q.defLevier('bacASable') ? q.rangLevier(etat, 'bacASable', il.id) : 0] || 1;
    s *= KS.accesProd[q.defLevier('accesProd') ? q.rangLevier(etat, 'accesProd', il.id) : 0] || 1;
    if (q.levier(etat, 'passerelleMCP') === true) s *= KS.passerelle;
    return O.borner(s);
  }

  function loi25Ilot(etat, il) {
    var KL = K.loi25, q = LTP.q;
    var classif = O.borner(num(etat.org.donnees && etat.org.donnees.classification));
    var sortie = q.levier(etat, 'modele') === 'local' ? KL.sortieLocale : KL.sortieExterne;
    (il.capacites || []).forEach(function (id) { if (carteDef(id).provenance === 'fournisseur') sortie += KL.parFournisseur; });
    return O.borner(num(il.sensibilite) * (1 - classif) * O.borner(sortie));
  }

  // Lit un score 0–100 fourni par 25-donnees-secu (etat.securite.surfaceAttaque, etat.securite.expositionLoi25).
  function scoreExterne(etat, champ) {
    var v = etat.securite && etat.securite[champ];
    return typeof v === 'number' && isFinite(v) ? Math.max(0, Math.min(100, v)) : null;
  }
  // Détail par îlot fourni par 25-donnees-secu : parIlot[id].surface (0–100) et parIlot[id].loi25 (0–1).
  function detailExterne(etat, ilotId, champ, echelle) {
    var p = etat.securite && etat.securite.parIlot && etat.securite.parIlot[ilotId];
    var v = p && p[champ];
    return typeof v === 'number' && isFinite(v) ? O.borner(v / echelle) : null;
  }

  function cartesOrphelines(etat) {
    var n = 0;
    Object.keys(etat.capacites || {}).forEach(function (id) {
      var c = etat.capacites[id];
      if (!c || !c.equipeeSur || !c.equipeeSur.length) return;
      var p = c.proprietaire && etat.devs[c.proprietaire];
      if (!p || !p.present) n += 1;
    });
    return n;
  }
  function cartesPerimees(etat) {
    return Object.keys(etat.capacites || {}).filter(function (id) {
      var c = etat.capacites[id];
      return c && c.equipeeSur && c.equipeeSur.length && num(c.fraicheur, 1) < K.seuilPerime;
    }).length;
  }

  function connaissanceCapturee(etat) {
    var p = LTP.systemes.personnes;
    var porteurs = LTP.q.devsPresents(etat).filter(function (d) {
      return d.niveau === 'senior' || LTP.q.aTrait(d, 'contractuel');
    });
    if (!porteurs.length || !p || typeof p.capture !== 'function') return 0;
    return O.borner(O.moyenne(porteurs.map(function (d) { return p.capture(etat, d.id); })));
  }

  function ou(v, f) { return v == null ? f() : v; }

  function moyennePonderee(ilots, f, poidsF) {
    var s = 0, w = 0;
    ilots.forEach(function (il) { var p = Math.max(0, poidsF(il)); s += f(il) * p; w += p; });
    return w > 0 ? s / w : O.moyenne(ilots, f);
  }

  // --- Calcul de toutes les métriques ---
  function calculer(etat, ctx, cumuler) {
    var q = LTP.q, ilots = q.ilots(etat), devs = q.devsPresents(etat);
    var obs = niveauObservabilite(etat);
    var parIlot = {};
    ilots.forEach(function (il) {
      parIlot[il.id] = {
        comprehension: O.moyenne(q.devsDe(etat, il.id), 'comprehension'),
        satisfaction: num(il.satisfaction),
        retard: retardIlot(etat, il),
        surfaceAttaque: ou(detailExterne(etat, il.id, 'surface', 100), function () { return surfaceIlot(etat, il); }),
        expositionLoi25: ou(detailExterne(etat, il.id, 'loi25', 1), function () { return loi25Ilot(etat, il); }),
        incidents: der(il, 'incidents'),
      };
    });

    var deploiements = O.somme(ilots, function (il) { return der(il, 'deploiements'); });
    var echappes = O.somme(ilots, function (il) { return der(il, 'defautsEchappes'); });
    var prF = O.somme(ilots, function (il) { return der(il, 'prFusionnees'); });
    var mttr = mttrExact(etat);
    var precedent = etat.metriques || {};

    var surface = scoreExterne(etat, 'surfaceAttaque');
    if (surface == null) surface = 100 * O.moyenne(ilots.map(function (il) { return parIlot[il.id].surfaceAttaque; }));
    var loi25 = scoreExterne(etat, 'expositionLoi25');
    if (loi25 == null) loi25 = 100 * O.moyenne(ilots.map(function (il) { return parIlot[il.id].expositionLoi25; }));

    var calc = {
      prFusionnees: function () { return prF; },
      pointsLivres: function () { return O.somme(ilots, function (il) { return der(il, 'pointsLivres'); }); },
      delai: function () {
        return moyennePonderee(ilots, function (il) { return der(il, 'delai'); }, function (il) { return der(il, 'prFusionnees') || der(il, 'pr'); });
      },
      frequenceDeploiement: function () { return deploiements; },
      backlog: function () { return O.somme(ilots, function (il) { return num(il.backlog); }); },
      satisfaction: function () { return O.moyenne(ilots, function (il) { return num(il.satisfaction); }); },
      bogues: function () { return O.somme(ilots, function (il) { return der(il, 'bogues'); }); },
      incidents: function () { return incidentsVisiblesSemaine(etat); },
      tauxEchec: function () {
        if (deploiements > 0) return O.borner(echappes / deploiements);
        return num(precedent.tauxEchec);
      },
      mttr: function () { return arrondirA(mttr, K.arrondiMttr[obs] == null ? 1 : K.arrondiMttr[obs]); },
      volumeCode: function () { return O.somme(ilots, function (il) { return num(il.codebase && il.codebase.volume); }); },
      couvertureApparente: function () {
        return O.borner(moyennePonderee(ilots, function (il) { return num(il.codebase && il.codebase.couvertureApparente); },
          function (il) { return num(il.codebase && il.codebase.volume); }));
      },
      couvertureReelle: function () {
        return O.borner(moyennePonderee(ilots, function (il) { return num(il.codebase && il.codebase.couvertureReelle); },
          function (il) { return num(il.codebase && il.codebase.volume); }));
      },
      comprehension: function () { return O.borner(O.moyenne(devs, 'comprehension')); },
      jetonsSemaine: function () { return num(etat.org.jetonsSemaine); },
      coutSemaine: function () { return num(etat.org.coutSemaine); },
      budgetRestant: function () { return num(etat.org.budget); },
      retard: function () { return O.borner(O.moyenne(ilots.map(function (il) { return parIlot[il.id].retard; }))); },
      surfaceAttaque: function () { return Math.max(0, Math.min(100, surface)); },
      expositionLoi25: function () { return Math.max(0, Math.min(100, loi25)); },
      applisFantomes: function () { return O.somme(ilots, function (il) { return num(il.applisFantomes); }); },
      cartesOrphelines: function () { return cartesOrphelines(etat); },
      moral: function () { return O.borner(O.moyenne(devs, 'moral')); },
      fatigue: function () { return O.borner(O.moyenne(devs, 'fatigue')); },
      releve: function () { return O.borner(num(etat.org.releve)); },
      connaissanceCapturee: function () { return connaissanceCapturee(etat); },
      confiance: function () { return num(etat.org.confiance); },
    };

    var m = {};
    (LTP.donnees.metriques || []).forEach(function (def) {
      var f = calc[def.id], v;
      try {
        v = f ? f() : (typeof etat.org[def.id] === 'number' ? etat.org[def.id] : 0);
      } catch (e) { v = 0; }
      m[def.id] = num(v);
    });
    etat.metriques = m;
    etat.metriquesDetail = {
      mttrExact: num(mttr),
      observabilite: obs,
      cartesPerimees: cartesPerimees(etat),
      parIlot: parIlot,
    };

    var s = etat.semaine;
    for (var k in m) {
      if (!Object.prototype.hasOwnProperty.call(m, k)) continue;
      var h = etat.historique[k] || (etat.historique[k] = []);
      for (var j = h.length; j < s; j++) h[j] = j > 0 && h[j - 1] != null ? h[j - 1] : m[k];
      h[s] = m[k];
    }

    if (cumuler) cumulerSemaine(etat, ilots);
  }

  function cumulerSemaine(etat, ilots) {
    var c = etat.cumuls || (etat.cumuls = nouveauxCumuls());
    c.semaines += 1;
    ['pointsLivres', 'points', 'prFusionnees', 'code', 'bogues', 'lgtm', 'pr', 'defautsEchappes', 'deploiements', 'jetons'].forEach(function (champ) {
      c[champ] += O.somme(ilots, function (il) { return der(il, champ); });
    });
    incidentsDeLaSemaine(etat).forEach(function (i) {
      c.incidents += 1;
      var g = num(i.gravite, 0.5);
      if (g >= K.graviteMajeure) c.incidentsGraves += 1;
      if (correspond(i.cause, K.causesSecurite)) c.incidentsSecurite += 1;
      if (correspond(i.cause, K.causesFuite)) {
        c.fuites += 1;
        if (g >= K.graviteMajeure) c.fuitesMajeures += 1;
      }
    });
  }

  LTP.systemes.metriques = {
    initialiser: initialiser,
    semaine: function (etat, ctx) { calculer(etat, ctx, true); },
    visible: visible,
    // Exposés pour 32-fin, l'interface et les tests.
    niveauObservabilite: niveauObservabilite,
    mttrExact: mttrExact,
    surfaceIlot: surfaceIlot,
    loi25Ilot: loi25Ilot,
  };
})(globalThis.LTP = globalThis.LTP || {});
