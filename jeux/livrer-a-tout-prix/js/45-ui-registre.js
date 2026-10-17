/* 45-ui-registre.js */
/* Registre des skills et MCP (onglet du panneau) : cartes découvertes et débloquées, filtres,
   provenance, effets, poids de contexte, branches de la triade, risque, propriétaire, fraîcheur ;
   équiper ou retirer sur l'îlot choisi, développer une carte maison, changer le propriétaire ;
   voyants de la triade par îlot. Appelé par 43-ui-leviers.js : LTP.ui.registre.rendre(racine, etat, ui, app).
   Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  var h = LTP.ui.h, t = LTP.t, F = LTP.ui.F;
  var etat_ = { type: 'tous', provenance: 'toutes', erreurs: {}, choixDev: {} };

  // Sens d'un gain : +1 si une hausse est bonne, −1 si une baisse est bonne. mult : multiplicatif.
  var GAINS = {
    debit: { bon: 1, mult: false }, comprehension: { bon: 1, mult: false }, adequationPile: { bon: 1, mult: false },
    couverture: { bon: 1, mult: false }, qualiteDonnees: { bon: 1, mult: false },
    tauxDefaut: { bon: -1, mult: true }, jetons: { bon: -1, mult: true }, mttr: { bon: -1, mult: true }, verbosite: { bon: -1, mult: true },
  };

  function puces(gains) {
    var f = LTP.format, res = [];
    Object.keys(gains || {}).forEach(function (k) {
      var v = gains[k];
      if (typeof v !== 'number') return;
      var g = GAINS[k] || { bon: 1, mult: false };
      var delta = g.mult ? v - 1 : v;
      if (Math.abs(delta) < 1e-9) return;
      var texte = (delta > 0 ? '+' : '−') + (g.mult || k === 'debit' ? f.pourcent(Math.abs(delta), 0) : t('unite.pts', { n: f.nombre(Math.abs(delta) * 100, 0) }));
      var bon = (delta > 0) === (g.bon > 0);
      res.push(h('span', { class: 'puce puce--' + (bon ? 'ok' : 'crit') },
        h('span', { 'aria-hidden': 'true' }, bon ? '▲ ' : '▼ '), LTP.aTexte('gain.' + k) ? t('gain.' + k) : k, ' ', h('b', { class: 'mono' }, texte)));
    });
    return res;
  }

  function etatDev(def, c, etat) {
    if (def.disponible !== 'developpement') return 'pret';
    if (c.enDeveloppementPar && (c.disponibleSemaine == null || c.disponibleSemaine > etat.semaine)) return 'encours';
    if (c.disponibleSemaine != null && c.disponibleSemaine <= etat.semaine) return 'pret';
    if (c.enDeveloppementPar) return 'encours';
    return 'afaire';
  }

  function cartesVisibles(etat) {
    return (LTP.donnees.capacites || []).filter(function (def) {
      var c = etat.capacites && etat.capacites[def.id];
      if (!c || !c.decouverte) return false;
      return etat.options.toutDebloque || !def.trimestre || def.trimestre <= etat.trimestre;
    });
  }

  function apercu(etat, ilotId) {
    var reg = LTP.systemes.registre;
    if (!ilotId || !reg || typeof reg.apercu !== 'function') return null;
    try { return reg.apercu(etat, ilotId) || null; } catch (e) { return null; }
  }

  function poidsIlot(etat, ilotId) {
    var il = etat.ilots[ilotId];
    if (!il) return 0;
    if (il.mods && typeof il.mods.poidsContexte === 'number' && il.mods.poidsContexte > 0) return il.mods.poidsContexte;
    return LTP.outils.somme(il.capacites || [], function (id) { var d = LTP.q.defCapacite(id); return d && d.poidsContexte || 0; });
  }

  function seuilContexte() {
    var k = LTP.K.registre || {};
    return typeof k.seuilContexte === 'number' ? k.seuilContexte : 0.25;
  }

  function devsOptions(etat, selection, avecPersonne) {
    var opts = [];
    if (avecPersonne) opts.push(h('option', { value: '', selected: !selection }, t('registre.personne')));
    (etat.ordreIlots || []).forEach(function (ilotId) {
      var devs = LTP.q.devsDe(etat, ilotId);
      if (!devs.length) return;
      opts.push(h('optgroup', { label: LTP.ui.nomIlot(ilotId) }, devs.map(function (d) {
        return h('option', { value: d.id, selected: d.id === selection }, LTP.ui.nomDev(d.id));
      })));
    });
    return opts;
  }

  function agir(app, carteId, action) {
    var r = app.dispatch(action);
    if (r && r.erreur) { etat_.erreurs[carteId] = r.erreur; LTP.ui.annoncer(t('divers.etiquette', { libelle: t('erreur.titre'), valeur: F.erreur(r.erreur) })); app.notifier(); }
    else { delete etat_.erreurs[carteId]; }
    return r;
  }

  function carte(def, etat, ilotId, app) {
    var c = etat.capacites[def.id];
    var equipeeIci = !!(ilotId && (c.equipeeSur || []).indexOf(ilotId) >= 0);
    var dev = etatDev(def, c, etat);
    var fraicheur = typeof c.fraicheur === 'number' ? c.fraicheur : 1;
    var risque = def.risque || { type: 'aucun', proba: 0 };
    var branches = def.branches || [];
    var tri = { prive: branches.indexOf('prive') >= 0, nonFiable: branches.indexOf('nonFiable') >= 0, sortie: branches.indexOf('sortie') >= 0 };
    var proprio = c.proprietaire;
    var cle = 'cap-' + def.id;

    var prerequis = null;
    if (def.prerequis && def.prerequis.levier) {
      var ok = (def.prerequis.valeurs || []).indexOf(LTP.q.levier(etat, def.prerequis.levier)) >= 0;
      prerequis = h('p', { class: 'cap-prerequis' + (ok ? ' is-ok' : ' is-manque') },
        h('span', { 'aria-hidden': 'true' }, ok ? '✓ ' : '! '),
        t('registre.prerequis', { levier: LTP.ui.libelleLevier(def.prerequis.levier),
          valeurs: (def.prerequis.valeurs || []).map(function (v) { return LTP.ui.libelleValeur(def.prerequis.levier, v); }).join(t('divers.ou')) }));
    }

    var actions = [];
    if (dev === 'afaire') {
      var choisi = etat_.choixDev[def.id] || def.auteur || ((LTP.q.devsDe(etat, (def.ilotsSuggeres || [])[0] || '')[0] || {}).id) || '';
      etat_.choixDev[def.id] = choisi;
      var idSel = 'devsel-' + def.id;
      actions.push(h('div', { class: 'cap-dev' },
        h('label', { for: idSel }, t('registre.developperAvec')),
        h('select', { id: idSel, 'data-cle': cle + '-devsel', onchange: function (ev) { etat_.choixDev[def.id] = ev.target.value; } }, devsOptions(etat, choisi, false)),
        h('button', { type: 'button', class: 'btn btn--petit btn--primaire', 'data-cle': cle + '-dev',
          onclick: function () { agir(app, def.id, { type: 'developper', carte: def.id, dev: etat_.choixDev[def.id] }); } },
          t('registre.developperDuree', { n: def.coutDev || 0 }))));
    } else if (dev === 'encours') {
      actions.push(h('p', { class: 'cap-encours' }, F.icone('reglage', 14), ' ',
        c.disponibleSemaine != null ? t('registre.enDeveloppement', { dev: LTP.ui.nomDev(c.enDeveloppementPar), semaine: c.disponibleSemaine })
          : t('registre.enDeveloppementCourt', { dev: LTP.ui.nomDev(c.enDeveloppementPar) })));
    } else if (ilotId) {
      actions.push(equipeeIci
        ? h('button', { type: 'button', class: 'btn btn--petit', 'data-cle': cle + '-eq',
            onclick: function () { agir(app, def.id, { type: 'desequiper', carte: def.id, ilot: ilotId }); } }, t('registre.desequiper', { ilot: LTP.ui.nomIlot(ilotId) }))
        : h('button', { type: 'button', class: 'btn btn--petit btn--primaire', 'data-cle': cle + '-eq',
            onclick: function () { agir(app, def.id, { type: 'equiper', carte: def.id, ilot: ilotId }); } }, t('registre.equiper', { ilot: LTP.ui.nomIlot(ilotId) })));
    }
    var idProp = 'prop-' + def.id;
    actions.push(h('div', { class: 'cap-proprio-choix' },
      h('label', { for: idProp }, t('registre.changerProprietaire')),
      h('select', { id: idProp, 'data-cle': cle + '-prop', onchange: function (ev) {
        agir(app, def.id, { type: 'proprietaire', carte: def.id, dev: ev.target.value || null });
      } }, devsOptions(etat, proprio, true))));

    var equipee = (c.equipeeSur || []).map(LTP.ui.nomIlot);
    var poids = def.poidsContexte || 0;

    return h('article', { class: 'cap cap--' + def.type + (equipeeIci ? ' is-equipee' : '') + (fraicheur < 0.4 ? ' is-perimee' : ''), 'data-carte': def.id, 'aria-labelledby': cle + '-nom' },
      h('header', { class: 'cap-tete' },
        h('span', { class: 'cap-type mono' }, def.type === 'mcp' ? t('registre.mcp') : t('registre.skill')),
        h('h4', { class: 'cap-nom mono', id: cle + '-nom' }, def.nom || def.id),
        h('span', { class: 'prov prov--' + def.provenance }, t('registre.provenance.' + (def.provenance || 'maison')))),
      def.description ? h('p', { class: 'cap-desc' }, F.texte(def.description, etat)) : null,
      h('dl', { class: 'cap-attr' },
        h('div', { class: 'cap-gains' }, h('dt', null, t('registre.gains')), h('dd', null, puces(def.gains).length ? puces(def.gains) : '—')),
        h('div', null, h('dt', null, t('registre.contexte')), h('dd', null,
          h('span', { class: 'mini-barre', 'aria-hidden': 'true' }, h('i', { style: { width: Math.min(100, Math.round(poids / seuilContexte() * 100)) + '%' } })),
          h('span', { class: 'mono' }, LTP.format.nombre(poids, 2)))),
        h('div', null, h('dt', null, t('registre.branches')), h('dd', null, branches.length ? F.voyants(tri) : h('span', { class: 'muted' }, t('registre.aucuneBranche')))),
        h('div', null, h('dt', null, t('registre.risque')), h('dd', { class: risque.type && risque.type !== 'aucun' ? 'cap-risque' : 'muted' },
          t(LTP.aTexte('risque.' + risque.type) ? 'risque.' + risque.type : 'risque.aucun'),
          risque.proba ? h('span', { class: 'mono' }, ' · ' + t('risque.parSemaine', { p: LTP.format.pourcent(risque.proba, risque.proba < 0.1 ? 1 : 0) })) : null)),
        h('div', null, h('dt', null, t('registre.proprietaireLib')), h('dd', { class: proprio ? '' : 'cap-orpheline' },
          proprio ? LTP.ui.nomDev(proprio) : h('span', null, h('span', { 'aria-hidden': 'true' }, '⚠ '), t('registre.orpheline')))),
        h('div', null, h('dt', null, t('registre.fraicheur')), h('dd', null, F.jauge(fraicheur, { sens: 'haut', texte: fraicheur < 0.4 ? t('registre.perimeeLib') : null })))),
      def.declencheur ? h('p', { class: 'cap-note mono' }, t('registre.declencheur', { declencheur: def.declencheur })) : null,
      prerequis,
      h('p', { class: 'cap-equipee' }, equipee.length ? t('registre.equipeeSur', { liste: equipee.join(', ') }) : h('span', { class: 'muted' }, t('registre.nonEquipee'))),
      etat_.erreurs[def.id] ? h('p', { class: 'erreur', role: 'alert' }, F.erreur(etat_.erreurs[def.id])) : null,
      h('div', { class: 'cap-actions' }, actions));
  }

  function enteteIlot(etat, ilotId, app) {
    if (!ilotId) {
      // Vue d'ensemble : voyants de la triade de chaque îlot.
      return h('div', { class: 'triades' },
        h('p', { class: 'eyebrow' }, t('triade.titre')),
        h('ul', { class: 'triades-liste' }, LTP.q.ilots(etat).map(function (il) {
          var n = (il.capacites || []).length;
          return h('li', null, h('button', { type: 'button', class: 'triade-ligne' + (LTP.q.triadeComplete(etat, il.id) ? ' is-complete' : ''), 'data-cle': 'tri-' + il.id,
            onclick: function () { app.setUI({ ilotEdition: il.id }); } },
            h('span', { class: 'triade-ilot' }, LTP.ui.nomIlot(il.id)),
            h('span', { class: 'triade-n mono' }, t('registre.cartesIlot', { n: n })),
            F.voyants(il.triade)));
        })),
        h('p', { class: 'muted petit' }, t('registre.choisirIlot')));
    }
    var il = etat.ilots[ilotId];
    var ap = apercu(etat, ilotId) || {};
    var triade = ap.triade || il.triade;
    var poids = typeof ap.poidsContexte === 'number' ? ap.poidsContexte : poidsIlot(etat, ilotId);
    var seuil = typeof ap.seuil === 'number' ? ap.seuil : seuilContexte();
    var complete = !!(triade && triade.prive && triade.nonFiable && triade.sortie);
    var alertes = [].concat(ap.alertes || [], ap.conflits || []).map(function (a) { return typeof a === 'string' ? a : (a && (a.texte || a.libelle || a.id)) || ''; }).filter(Boolean);
    return h('div', { class: 'apercu' + (complete ? ' is-complete' : '') },
      h('div', { class: 'apercu-tete' }, h('p', { class: 'eyebrow' }, t('registre.apercu') + ' · ' + LTP.ui.nomIlot(ilotId))),
      F.voyants(triade, { etiquettes: true }),
      complete ? h('p', { class: 'apercu-alerte', role: 'alert' }, h('span', { 'aria-hidden': 'true' }, '⚠ '), t('triade.complete')) : null,
      h('div', { class: 'apercu-contexte' },
        F.jauge(Math.min(1, poids / (seuil * 1.5)), { libelle: t('registre.contexteIlot'), sens: 'bas', texte: LTP.format.nombre(poids, 2) + ' / ' + LTP.format.nombre(seuil, 2) }),
        poids > seuil ? h('p', { class: 'apercu-sature' }, t('registre.saturationLib')) : null),
      alertes.length ? h('ul', { class: 'apercu-alertes' }, alertes.map(function (a) { return h('li', null, a); })) : null);
  }

  LTP.ui.registre = {
    rendre: function (racine, etat, ui, app) {
      if (!racine._monte) {
        racine._monte = true;
        racine._entete = h('div', { class: 'registre-entete' });
        racine._filtres = h('div', { class: 'registre-filtres' });
        racine._liste = h('div', { class: 'registre-liste' });
        LTP.ui.remplacer(racine,
          h('h3', { class: 'registre-titre' }, t('registre.titre')),
          h('p', { class: 'registre-intro' }, t('registre.intro')),
          racine._entete, racine._filtres, racine._liste);
      }
      var ilotId = ui.ilotEdition && etat.ilots[ui.ilotEdition] ? ui.ilotEdition : null;
      var ap = apercu(etat, ilotId);
      var sigEntete = JSON.stringify([ilotId, LTP.q.ilots(etat).map(function (i) { return [i.triade, (i.capacites || []).length, i.mods && i.mods.poidsContexte]; }), ap]);
      F.reconstruire(racine._entete, sigEntete, function () { return enteteIlot(etat, ilotId, app); });

      F.reconstruire(racine._filtres, etat_.type + '|' + etat_.provenance, function () {
        return [
          h('div', { class: 'filtre' }, h('span', { class: 'filtre-lib', id: 'flt-type' }, t('registre.filtreType')),
            F.choix({ cle: 'flt-type', idLibelle: 'flt-type', mode: 'segmente', courant: etat_.type,
              valeurs: [{ v: 'tous', libelle: t('registre.tous') }, { v: 'skill', libelle: t('registre.skills') }, { v: 'mcp', libelle: t('registre.mcp') }],
              onChoisir: function (v) { etat_.type = v; app.notifier(); } })),
          h('div', { class: 'filtre' }, h('span', { class: 'filtre-lib', id: 'flt-prov' }, t('registre.filtreProvenance')),
            F.choix({ cle: 'flt-prov', idLibelle: 'flt-prov', mode: 'segmente', courant: etat_.provenance,
              valeurs: ['toutes', 'maison', 'fournisseur', 'communaute', 'personnelle'].map(function (p) {
                return { v: p, libelle: p === 'toutes' ? t('registre.toutes') : t('registre.provenance.' + p) };
              }),
              onChoisir: function (v) { etat_.provenance = v; app.notifier(); } })),
        ];
      });

      var cartes = cartesVisibles(etat);
      var sigListe = JSON.stringify([ilotId, etat_.type, etat_.provenance, etat.trimestre, etat_.erreurs, etat.phase,
        cartes.map(function (d) {
          var c = etat.capacites[d.id];
          return [d.id, c.equipeeSur, c.proprietaire, Math.round((c.fraicheur == null ? 1 : c.fraicheur) * 10), c.disponibleSemaine, c.enDeveloppementPar,
            d.prerequis ? LTP.q.levier(etat, d.prerequis.levier) : null];
        })]);
      F.reconstruire(racine._liste, sigListe, function () {
        if (!cartes.length) return h('p', { class: 'registre-vide' }, t('registre.aucune'));
        var filtrees = cartes.filter(function (d) {
          return (etat_.type === 'tous' || d.type === etat_.type) && (etat_.provenance === 'toutes' || d.provenance === etat_.provenance);
        });
        if (!filtrees.length) return h('p', { class: 'registre-vide' }, t('registre.vide'));
        var rang = function (d) {
          if (!ilotId) return 0;
          if ((etat.capacites[d.id].equipeeSur || []).indexOf(ilotId) >= 0) return 0;
          if ((d.ilotsSuggeres || []).indexOf(ilotId) >= 0) return 1;
          return 2;
        };
        return filtrees.slice().sort(function (a, b) { return rang(a) - rang(b); }).map(function (d) { return carte(d, etat, ilotId, app); });
      });
    },
  };
})(globalThis.LTP = globalThis.LTP || {});
