/* 14-donnees-evenements.js */
/* Cartes d'événement (CONCEPTION §6). Contenu écrit par l'agent D2.
   Forme d'une carte :
   {
     id, theme, trimestres: [1, 2], poids: 1, delaiRetirage: 8, unique: false,
     personnage: 'ginette' | 'regis' | 'berube' | 'lavigueur' | 'kevin' | devId | 'direction',
     condition: function (etat, q) → booléen,          // facultatif
     choisirCible: function (etat, q, rng) → { ilot?, dev? },   // facultatif ; sinon cible au hasard pondéré
     titre, texte,                                     // chaînes, ou fonctions (etat, q, ref) → chaîne ; {ilot} et {dev} remplacés
     choix: [{ id, libelle, effets: [ … effets déclaratifs (19-effets.js) … ], consequence }],
   }
   Seules condition, choisirCible et les champs de texte peuvent être des fonctions. */
(function (LTP) {
  'use strict';
  LTP.donnees.evenements = [
    { id: 'pr-geante', theme: 'revue', trimestres: [1, 2], poids: 1, delaiRetirage: 10, unique: false,
      personnage: 'nadia',
      condition: function (etat, q) { return q.levier(etat, 'taillePR') === 'libre'; },
      choisirCible: function () { return { ilot: 'portail', dev: 'mathieu' }; },
      titre: 'La PR de 4 000 lignes',
      texte: 'Mathieu a ouvert une PR de 4 000 lignes générée en une nuit. Nadia la regarde comme on regarde une facture d\'Hydro en janvier.',
      choix: [
        { id: 'approuver', libelle: 'On fait confiance, on fusionne', consequence: 'Fusionnée en 12 minutes. Personne ne l\'a vraiment lue.',
          effets: [{ cible: 'ilot', chemin: 'codebase.defautsLatents', op: 'add', valeur: 12 }, { cible: 'dev:nadia', chemin: 'moral', op: 'add', valeur: -0.05 }] },
        { id: 'decouper', libelle: 'Exiger qu\'elle soit découpée', consequence: 'Mathieu soupire. Huit PR plus tard, Nadia respire.',
          effets: [{ cible: 'dev:mathieu', chemin: 'moral', op: 'add', valeur: -0.04 }, { cible: 'ilot', chemin: 'fileRevue', op: 'add', valeur: 3 }] },
      ] },
  ];
})(globalThis.LTP = globalThis.LTP || {});
