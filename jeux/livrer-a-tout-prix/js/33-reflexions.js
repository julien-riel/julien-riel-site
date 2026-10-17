/* 33-reflexions.js */
/* Les réflexions des développeurs (CONCEPTION §8.3) : chaque journée ouvrable, quelques personnes pensent
   tout haut, et ce qu'elles disent a une cause qu'on peut nommer. Priorité, dans l'ordre :
     1. un levier qui vient de changer et qui les touche (leurs réactions propres, puis le contenu par levier) ;
     2. la note de service du calendrier (les deux premiers jours de la semaine) ;
     3. une situation réelle de la simulation (file de revue, fatigue, licence absente, incident…) ;
     4. leur activité du moment (les bulles des personnages).
   Le choix est déterministe et n'utilise pas le générateur du moteur (hachage de la graine, de la semaine,
   du jour et de la personne) : les réflexions ne changent rien à la simulation.
   État : etat.reflexions = { semaine, jour, liste: [Reflexion], fil: [Reflexion] }
     Reflexion = { dev, ilot, texte, motif, levier?, valeur?, ton, semaine, jour, immediat? }
     motif : 'reaction' | 'levier' | 'nouveaute' | 'situation:<id>' | 'activite:<etat>'
     ton : 'plus' | 'moins' | 'neutre'
   Le texte garde les gabarits {dev} et {ilot}, remplis par l'interface avec les noms.
   Contenu : LTP.donnees.reflexions (09-donnees-reflexions.js). */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  LTP.K.reflexions = {
    parJour: 5,                 // réflexions par journée
    parIlotMax: 2,              // au plus deux personnes du même îlot le même jour
    leviersMaxParJour: 3,       // au plus trois réflexions par jour sur des changements de leviers
    filMax: 40,                 // réflexions gardées dans le fil (fiche, liste « Ce qu'on entend »)
    memoireJours: 10,           // une personne ne répète pas la même phrase avant ce nombre de jours
    reposJours: 1,              // une personne qui vient de parler se tait un peu (poids réduit)
    ageChangementMax: 8,        // jours ouvrables pendant lesquels un changement de levier fait parler
    joursNouveaute: 2,          // la note de service fait parler le lundi et le mardi
    immediatesMax: 2,           // réactions tout de suite après un changement de levier
    poids: { reaction: 6, levier: 5, nouveaute: 4, situation: 3, activite: 1 },
  };
  var K = LTP.K.reflexions;

  function etatPropre(etat) {
    var r = etat.reflexions;
    if (!r || typeof r !== 'object') r = etat.reflexions = { semaine: 0, jour: 0, liste: [], fil: [] };
    if (!Array.isArray(r.liste)) r.liste = [];
    if (!Array.isArray(r.fil)) r.fil = [];
    return r;
  }

  function J() { return LTP.K.etat.joursParSemaine || 5; }
  function jourAbsolu(semaine, jour) { return (Math.max(1, semaine) - 1) * J() + (jour || 0); }
  function maintenant(etat) { return jourAbsolu(LTP.q.semaineCourante(etat), etat.jour || 0); }
  function hasard(etat, cle) { return O.hachage(String(etat.graine) + ':' + cle); }

  // Texte d'une entrée de contenu : chaîne, fonction (etat, q, dev) ou objet { texte }.
  function texteDe(x, etat, dev) {
    if (x && typeof x === 'object' && !Array.isArray(x)) x = x.texte;
    if (typeof x === 'function') {
      try { x = x(etat, LTP.q, dev); } catch (e) { return ''; }
    }
    return x == null ? '' : String(x);
  }

  // Une entrée de contenu s'applique-t-elle à cette personne ?
  //   { texte, devs?: [ids], traits?: [traits], niveaux?: [niveaux], pilote?: bool, licence?: bool, effet?: bool,
  //     condition?: (etat, q, dev) → bool, ton? }
  function convient(entree, etat, dev, extra) {
    if (entree == null) return false;
    if (typeof entree === 'string' || typeof entree === 'function') return true;
    var q = LTP.q;
    if (Array.isArray(entree.devs) && entree.devs.indexOf(dev.id) < 0) return false;
    if (Array.isArray(entree.traits) && !entree.traits.some(function (t) { return q.aTrait(dev, t); })) return false;
    if (Array.isArray(entree.niveaux) && entree.niveaux.indexOf(dev.niveau) < 0) return false;
    if (entree.pilote != null && !!dev.pilote !== !!entree.pilote) return false;
    if (entree.licence != null && q.aLicence(etat, dev) !== !!entree.licence) return false;
    if (entree.effet != null && extra && extra.effet != null && !!entree.effet !== !!extra.effet) return false;
    if (typeof entree.condition === 'function') {
      try { if (!entree.condition(etat, q, dev)) return false; } catch (e) { return false; }
    }
    return true;
  }
  // Les entrées propres à quelqu'un (devs, traits) passent devant les génériques.
  function specificite(entree) {
    if (!entree || typeof entree !== 'object') return 1;
    if (Array.isArray(entree.devs)) return 2;
    if (Array.isArray(entree.traits) || Array.isArray(entree.niveaux) || entree.pilote != null) return 1.5;
    return 1;
  }
  function tonDe(entree, defaut) {
    return entree && typeof entree === 'object' && entree.ton ? entree.ton : (defaut || 'neutre');
  }

  // Le changement touche-t-il cette personne ? (un changement global ne touche pas un îlot qui a sa propre surcharge)
  function touche(etat, c, dev) {
    if (c.ilot) return c.ilot === dev.ilot;
    var s = etat.leviers.surcharges && etat.leviers.surcharges[dev.ilot];
    return !(s && Object.prototype.hasOwnProperty.call(s, c.id));
  }

  // --- Candidats d'une personne pour la journée ---
  function candidatsChangement(etat, dev, c, age, poidsBase) {
    var res = [];
    var q = LTP.q, def = q.defDev(dev.id) || {};
    var decroit = 1 / (1 + age * 0.3);
    // Le changement a-t-il déjà passé un vendredi ? (son effet se voit alors dans les chiffres)
    var effet = c.semaine < q.semaineCourante(etat);
    (def.reactions || []).forEach(function (r) {
      if (!r || r.levier !== c.id || !Array.isArray(r.valeurs) || r.valeurs.indexOf(c.valeur) < 0 || !r.bulle) return;
      var t = Array.isArray(r.bulle) ? r.bulle[Math.floor(hasard(etat, dev.id + c.id + age) * r.bulle.length)] : r.bulle;
      res.push({ texte: texteDe(t, etat, dev), poids: K.poids.reaction * decroit * poidsBase, motif: 'reaction', levier: c.id, valeur: c.valeur,
        ton: (Number(r.moral) || 0) > 0 ? 'plus' : (Number(r.moral) || 0) < 0 ? 'moins' : 'neutre' });
    });
    var parLevier = (LTP.donnees.reflexions && LTP.donnees.reflexions.leviers || {})[c.id] || {};
    var liste = [].concat(parLevier[String(c.valeur)] || [], parLevier['*'] || []);
    liste.forEach(function (e) {
      if (!convient(e, etat, dev, { effet: effet })) return;
      res.push({ texte: texteDe(e, etat, dev), poids: K.poids.levier * decroit * specificite(e) * poidsBase, motif: 'levier', levier: c.id, valeur: c.valeur, ton: tonDe(e) });
    });
    return res;
  }

  function candidats(etat, dev) {
    var q = LTP.q, D = LTP.donnees.reflexions || {};
    var res = [];
    var present = dev.present && dev.etat !== 'absent';
    if (!present) return res;
    var ilot = etat.ilots[dev.ilot];
    var auBureau = dev.etat !== 'conge' && dev.etat !== 'formation';

    if (auBureau) {
      // 1. Leviers qui viennent de changer.
      var now = maintenant(etat);
      (etat.leviers.changements || []).forEach(function (c) {
        var age = now - jourAbsolu(c.semaine, c.jour);
        if (age < 0 || age > K.ageChangementMax || !touche(etat, c, dev)) return;
        res = res.concat(candidatsChangement(etat, dev, c, age, 1));
      });

      // 2. La note de service de la semaine.
      var nv = etat.nouveaute;
      if (nv && nv.semaine === q.semaineCourante(etat) && (etat.jour || 0) < K.joursNouveaute) {
        var cal = q.deblocage(nv.semaine) || {};
        (cal.reactions || []).forEach(function (r) {
          if (r && r.dev === dev.id && r.texte) res.push({ texte: texteDe(r.texte, etat, dev), poids: K.poids.nouveaute * 1.5, motif: 'nouveaute', levier: (cal.leviers || [])[0], ton: tonDe(r) });
        });
        var parNv = D.nouveautes || {};
        [].concat(cal.leviers || [], (cal.valeurs || []).map(function (x) { return x.levier + ':' + (x.v || []).join(','); }), cal.panneaux || []).forEach(function (cle) {
          (parNv[cle] || []).forEach(function (e) {
            if (!convient(e, etat, dev)) return;
            res.push({ texte: texteDe(e, etat, dev), poids: K.poids.nouveaute * specificite(e), motif: 'nouveaute', levier: String(cle).split(':')[0], ton: tonDe(e) });
          });
        });
      }

      // 3. Situations.
      (D.situations || []).forEach(function (s) {
        if (!s || typeof s.quand !== 'function') return;
        var intensite;
        try { intensite = s.quand(etat, q, dev, ilot); } catch (e) { intensite = false; }
        if (intensite === true) intensite = 1;
        if (!(intensite > 0)) return;
        (s.lignes || []).forEach(function (e) {
          if (!convient(e, etat, dev)) return;
          res.push({ texte: texteDe(e, etat, dev), poids: K.poids.situation * (s.poids || 1) * Math.min(1.5, intensite) * specificite(e),
            motif: 'situation:' + s.id, levier: s.levier || null, ton: tonDe(e, s.ton) });
        });
      });
    }

    // 4. Activité du moment (bulles des personnages, puis bulles génériques).
    var P = LTP.systemes.personnes;
    var cle = P && P.etatBulle ? P.etatBulle(etat, dev) : (dev.etat || 'code');
    var def = q.defDev(dev.id) || {};
    var propres = (def.bulles && def.bulles[cle]) || [];
    var generiques = (LTP.donnees.bulles && LTP.donnees.bulles[cle]) || [];
    propres.forEach(function (t) { if (typeof t === 'string' && t) res.push({ texte: t, poids: K.poids.activite * 1.3, motif: 'activite:' + cle, ton: 'neutre' }); });
    generiques.forEach(function (t) { if (typeof t === 'string' && t) res.push({ texte: t, poids: K.poids.activite, motif: 'activite:' + cle, ton: 'neutre' }); });
    return res.filter(function (c) { return c.texte; });
  }

  // Ce qu'une personne a dit récemment (pour ne pas se répéter).
  function ditRecemment(etat, devId, texte) {
    var r = etatPropre(etat), now = maintenant(etat);
    return r.fil.some(function (x) { return x.dev === devId && x.texte === texte && now - jourAbsolu(x.semaine, x.jour) < K.memoireJours; });
  }
  function aParleRecemment(etat, devId) {
    var r = etatPropre(etat), now = maintenant(etat);
    return r.fil.some(function (x) { return x.dev === devId && now - jourAbsolu(x.semaine, x.jour) <= K.reposJours && now !== jourAbsolu(x.semaine, x.jour); });
  }

  // Tirage pondéré déterministe parmi les candidats.
  function tirer(etat, liste, cle) {
    var total = O.somme(liste, function (c) { return c.poids; });
    if (!(total > 0)) return null;
    var x = hasard(etat, cle) * total;
    for (var i = 0; i < liste.length; i++) { x -= liste[i].poids; if (x <= 0) return liste[i]; }
    return liste[liste.length - 1];
  }

  function choisirPour(etat, dev, cands, sel) {
    var frais = cands.filter(function (c) { return !ditRecemment(etat, dev.id, c.texte); });
    if (!frais.length) frais = cands;
    // On garde les candidats de la catégorie la plus forte (un levier qui vient de changer l'emporte sur l'activité).
    var max = Math.max.apply(null, frais.map(function (c) { return c.poids; }));
    var forts = frais.filter(function (c) { return c.poids >= max * 0.45; });
    return tirer(etat, forts, 'choix:' + sel + ':' + dev.id);
  }

  function enregistrer(etat, dev, c, immediat) {
    var r = etatPropre(etat);
    var x = {
      dev: dev.id, ilot: dev.ilot, texte: c.texte, motif: c.motif, ton: c.ton || 'neutre',
      semaine: LTP.q.semaineCourante(etat), jour: etat.jour || 0,
    };
    if (c.levier) x.levier = c.levier;
    if (c.valeur !== undefined) x.valeur = c.valeur;
    if (immediat) x.immediat = true;
    r.fil.push(x);
    if (r.fil.length > K.filMax) r.fil = r.fil.slice(r.fil.length - K.filMax);
    return x;
  }

  // Calcule les réflexions de la journée courante (sans rien faire si c'est déjà fait).
  function assurer(etat) {
    var r = etatPropre(etat), q = LTP.q;
    var sc = q.semaineCourante(etat), j = etat.jour || 0;
    if (r.semaine === sc && r.jour === j && r.calcule) return r.liste;
    r.semaine = sc;
    r.jour = j;
    r.calcule = true;
    r.liste = [];
    var sel = sc + ':' + j;
    var parDev = q.devsPresents(etat).map(function (d) {
      var cands = candidats(etat, d);
      var max = cands.length ? Math.max.apply(null, cands.map(function (c) { return c.poids; })) : 0;
      var score = max * (aParleRecemment(etat, d.id) ? 0.35 : 1) + hasard(etat, 'ordre:' + sel + ':' + d.id) * 2.2;
      return { dev: d, cands: cands, score: score };
    }).filter(function (x) { return x.cands.length; });
    parDev.sort(function (a, b) { return b.score - a.score; });
    var parIlot = {}, leviers = 0;
    for (var i = 0; i < parDev.length && r.liste.length < K.parJour; i++) {
      var x = parDev[i];
      if ((parIlot[x.dev.ilot] || 0) >= K.parIlotMax) continue;
      var cands = x.cands;
      if (leviers >= K.leviersMaxParJour) cands = cands.filter(function (c) { return c.motif !== 'levier' && c.motif !== 'reaction'; });
      if (!cands.length) continue;
      var c = choisirPour(etat, x.dev, cands, sel);
      if (!c) continue;
      if (c.motif === 'levier' || c.motif === 'reaction') leviers += 1;
      parIlot[x.dev.ilot] = (parIlot[x.dev.ilot] || 0) + 1;
      r.liste.push(enregistrer(etat, x.dev, c, false));
    }
    return r.liste;
  }

  // Réaction tout de suite après un changement de levier : une ou deux personnes touchées le disent.
  function reaction(etat, action) {
    if (!action || action.type !== 'levier') return;
    var q = LTP.q, r = etatPropre(etat);
    if (!(etat.phase === 'planification' || etat.phase === 'execution')) return;
    assurer(etat);
    var c = { id: action.id, valeur: action.valeur, ilot: action.ilot || null, semaine: q.semaineCourante(etat), jour: etat.jour || 0 };
    var sel = 'reaction:' + c.semaine + ':' + c.jour + ':' + c.id + ':' + String(c.valeur) + ':' + (c.ilot || '');
    var choix = q.devsPresents(etat).filter(function (d) {
      return d.etat !== 'absent' && d.etat !== 'conge' && touche(etat, c, d);
    }).map(function (d) {
      var cands = candidatsChangement(etat, d, c, 0, 1);
      var max = cands.length ? Math.max.apply(null, cands.map(function (x) { return x.poids; })) : 0;
      return { dev: d, cands: cands, score: max + hasard(etat, sel + ':' + d.id) * 2 };
    }).filter(function (x) { return x.cands.length; });
    choix.sort(function (a, b) { return b.score - a.score; });
    var faits = 0, ilots = {};
    for (var i = 0; i < choix.length && faits < K.immediatesMax; i++) {
      if (ilots[choix[i].dev.ilot]) continue;
      var cc = choisirPour(etat, choix[i].dev, choix[i].cands, sel);
      if (!cc) continue;
      ilots[choix[i].dev.ilot] = true;
      // Une réaction remplace la réflexion du jour de cette personne, s'il y en avait une.
      r.liste = r.liste.filter(function (x) { return x.dev !== choix[i].dev.id; });
      r.liste.unshift(enregistrer(etat, choix[i].dev, cc, true));
      faits += 1;
    }
    if (r.liste.length > K.parJour + K.immediatesMax) r.liste = r.liste.slice(0, K.parJour + K.immediatesMax);
  }

  // Dernières réflexions d'une personne (fiche).
  function dernieres(etat, devId, n) {
    var r = etatPropre(etat);
    return r.fil.filter(function (x) { return x.dev === devId; }).slice(-(n || 3)).reverse();
  }

  LTP.systemes.reflexions = {
    initialiser: function (etat) { etatPropre(etat); },
    assurer: assurer,
    reaction: reaction,
    candidats: candidats,
    dernieres: dernieres,
    jourAbsolu: jourAbsolu,
  };
})(globalThis.LTP = globalThis.LTP || {});
