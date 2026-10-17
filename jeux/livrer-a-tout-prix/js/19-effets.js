/* 19-effets.js */
/* Effets déclaratifs : le seul moyen pour le contenu (événements, comités, capacités) de modifier l'état.
   Forme d'un effet (voir ARCHITECTURE.md §4.4) :
     { cible, chemin, op, valeur, pendant?, dans?, filtre? }
       cible : 'org' | 'ilot' | 'dev' | 'ilots' | 'devs' | 'ilot:<id>' | 'dev:<id>'
               'ilot' et 'dev' désignent la référence courante (cible de l'événement)
       op : 'add' | 'mul' | 'set' ; pendant : n semaines puis annulé ; dans : appliqué dans n semaines
       filtre (pour 'ilots' et 'devs') : { ilot, trait, niveau, sauf }
     { drapeau: 'nom', valeur?: true }
     { carte: 'id', op: 'decouvrir' | 'retirer' | 'equiper', ilot? }
     { incident: { gravite, cause }, ilot? }
     { levier: 'id', valeur, ilot? }                 // force un levier (sans délai)
     { journal: 'cle', vars? }
     { evenement: 'id', dans?: n }                   // planifie un événement de suite
     { confiance: +5 }  { budget: -20000 }           // raccourcis vers org */
(function (LTP) {
  'use strict';
  var O = LTP.outils;

  function bornesDe(type, chemin) {
    var b = LTP.etat.BORNES[type];
    return b ? b[chemin] : undefined;
  }

  function entites(etat, cible, ref, filtre) {
    var c = String(cible || 'org');
    if (c === 'org') return [{ type: 'org', obj: etat.org }];
    if (c === 'ilot') return ref && ref.ilot && etat.ilots[ref.ilot] ? [{ type: 'ilot', obj: etat.ilots[ref.ilot] }] : [];
    if (c === 'dev') return ref && ref.dev && etat.devs[ref.dev] ? [{ type: 'dev', obj: etat.devs[ref.dev] }] : [];
    if (c.indexOf('ilot:') === 0) { var i = etat.ilots[c.slice(5)]; return i ? [{ type: 'ilot', obj: i }] : []; }
    if (c.indexOf('dev:') === 0) { var d = etat.devs[c.slice(4)]; return d ? [{ type: 'dev', obj: d }] : []; }
    if (c === 'ilots') {
      return LTP.q.ilots(etat).filter(function (x) {
        return !filtre || ((!filtre.ilot || x.id === filtre.ilot) && (!filtre.sauf || x.id !== filtre.sauf));
      }).map(function (x) { return { type: 'ilot', obj: x }; });
    }
    if (c === 'devs') {
      return LTP.q.devsPresents(etat).filter(function (x) {
        if (!filtre) return true;
        if (filtre.ilot && x.ilot !== (filtre.ilot === 'courant' ? ref && ref.ilot : filtre.ilot)) return false;
        if (filtre.trait && x.traits.indexOf(filtre.trait) < 0) return false;
        if (filtre.niveau && x.niveau !== filtre.niveau) return false;
        if (filtre.sauf && x.id === filtre.sauf) return false;
        return true;
      }).map(function (x) { return { type: 'dev', obj: x }; });
    }
    return [];
  }

  function appliquerValeur(ent, chemin, op, valeur) {
    var avant = O.lireChemin(ent.obj, chemin);
    var apres;
    if (op === 'set') apres = valeur;
    else if (op === 'mul') apres = (typeof avant === 'number' ? avant : 0) * valeur;
    else apres = (typeof avant === 'number' ? avant : 0) + valeur;
    if (typeof apres === 'number') {
      var b = bornesDe(ent.type, chemin);
      if (b) apres = O.borner(apres, b[0], b[1]);
      else if (!(LTP.etat.NEGATIFS_PERMIS[ent.type] || {})[chemin] && apres < 0) apres = 0;
      if (!isFinite(apres)) apres = 0;
    }
    O.ecrireChemin(ent.obj, chemin, apres);
    return { avant: avant, apres: apres };
  }

  // Applique une liste d'effets. ref = { ilot, dev, source }. ctx (facultatif) = contexte de la boucle.
  function appliquer(etat, effets, ref, ctx) {
    (effets || []).forEach(function (e) { appliquerUn(etat, e, ref || {}, ctx); });
  }

  function appliquerUn(etat, e, ref, ctx) {
    if (!e) return;
    if (e.dans > 0) {
      var copie = O.cloner(e);
      delete copie.dans;
      etat.differes.push({ semaine: etat.semaine + e.dans, effets: [copie], ref: O.cloner(ref) });
      return;
    }
    if (e.confiance != null) return appliquerUn(etat, { cible: 'org', chemin: 'confiance', op: 'add', valeur: e.confiance }, ref, ctx);
    if (e.budget != null) return appliquerUn(etat, { cible: 'org', chemin: 'budget', op: 'add', valeur: e.budget }, ref, ctx);
    if (e.drapeau) { etat.org.drapeaux[e.drapeau] = e.valeur === undefined ? true : e.valeur; return; }
    if (e.journal) { noter(etat, ctx, 'evenement', e.journal, e.vars || {}, ref); return; }
    if (e.evenement) {
      etat.evenementsForces.push({ id: e.evenement, semaine: etat.semaine + (e.dans || 1), ref: O.cloner(ref) });
      return;
    }
    if (e.levier) {
      if (e.ilot) {
        etat.leviers.surcharges[e.ilot] = etat.leviers.surcharges[e.ilot] || {};
        etat.leviers.surcharges[e.ilot][e.levier] = e.valeur;
      } else etat.leviers.global[e.levier] = e.valeur;
      if (e.levier === 'objectif') etat.org.objectif = e.valeur;
      return;
    }
    if (e.incident) {
      var il = e.ilot === 'ilot' || !e.ilot ? ref.ilot : e.ilot;
      if (il) LTP.q.creerIncident(etat, { ilot: il, gravite: e.incident.gravite, cause: e.incident.cause }, ctx);
      return;
    }
    if (e.carte) {
      var etc = etat.capacites[e.carte];
      if (!etc) return;
      var ilotCarte = e.ilot === 'ilot' || (!e.ilot && e.op === 'equiper') ? ref.ilot : e.ilot;
      if (e.op === 'decouvrir') etc.decouverte = true;
      else if (e.op === 'retirer') {
        (ilotCarte ? [ilotCarte] : etc.equipeeSur.slice()).forEach(function (iid) { retirerCarte(etat, e.carte, iid); });
      } else if (e.op === 'equiper' && ilotCarte) {
        etc.decouverte = true;
        if (etc.equipeeSur.indexOf(ilotCarte) < 0) etc.equipeeSur.push(ilotCarte);
        if (etat.ilots[ilotCarte].capacites.indexOf(e.carte) < 0) etat.ilots[ilotCarte].capacites.push(e.carte);
      }
      return;
    }
    if (e.chemin) {
      var op = e.op || 'add';
      entites(etat, e.cible, ref, e.filtre).forEach(function (ent) {
        var r = appliquerValeur(ent, e.chemin, op, e.valeur);
        if (e.pendant > 0 && typeof r.apres === 'number') {
          var inverse;
          if (op === 'add') inverse = { op: 'add', valeur: -(r.apres - (typeof r.avant === 'number' ? r.avant : 0)) };
          else if (op === 'mul') inverse = e.valeur ? { op: 'mul', valeur: 1 / e.valeur } : { op: 'set', valeur: r.avant };
          else inverse = { op: 'set', valeur: r.avant };
          var cibleInv = ent.type === 'org' ? 'org' : ent.type + ':' + ent.obj.id;
          etat.differes.push({
            semaine: etat.semaine + e.pendant,
            effets: [{ cible: cibleInv, chemin: e.chemin, op: inverse.op, valeur: inverse.valeur }],
            ref: O.cloner(ref),
          });
        }
      });
    }
  }

  function retirerCarte(etat, carteId, ilotId) {
    var etc = etat.capacites[carteId], il = etat.ilots[ilotId];
    if (etc) etc.equipeeSur = etc.equipeeSur.filter(function (x) { return x !== ilotId; });
    if (il) il.capacites = il.capacites.filter(function (x) { return x !== carteId; });
  }

  function noter(etat, ctx, type, cle, vars, ref) {
    if (ctx && ctx.noter) ctx.noter(type, cle, vars, ref);
    else etat.journal.push({ semaine: etat.semaine, type: type, cle: cle, vars: vars || {}, ilot: ref && ref.ilot, dev: ref && ref.dev });
  }

  // Étape 1 de la boucle : applique les effets différés arrivés à échéance.
  function differes(etat, ctx) {
    var restants = [], echus = [];
    etat.differes.forEach(function (d) { (d.semaine <= etat.semaine ? echus : restants).push(d); });
    etat.differes = restants;
    echus.forEach(function (d) { appliquer(etat, d.effets, d.ref, ctx); });
  }

  // Liste des chemins numériques d'une entité (pour valider les effets du contenu).
  function cheminsNumeriques(obj, prefixe, acc) {
    acc = acc || [];
    for (var k in obj) {
      if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
      var v = obj[k], p = prefixe ? prefixe + '.' + k : k;
      if (typeof v === 'number' || typeof v === 'boolean') acc.push(p);
      else if (v && typeof v === 'object' && !Array.isArray(v)) cheminsNumeriques(v, p, acc);
    }
    return acc;
  }

  LTP.effets = { appliquer: appliquer, retirerCarte: retirerCarte, cheminsNumeriques: cheminsNumeriques, entites: entites };
  LTP.systemes.effets = { differes: differes };
})(globalThis.LTP = globalThis.LTP || {});
