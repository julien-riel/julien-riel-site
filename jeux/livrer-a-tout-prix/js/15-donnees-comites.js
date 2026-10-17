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
   } */
(function (LTP) {
  'use strict';
  LTP.donnees.comites = [];
})(globalThis.LTP = globalThis.LTP || {});
