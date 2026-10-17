/* 17-donnees-fins.js */
/* Profils de fin et textes du bilan (CONCEPTION §2.3). Les ids sont un contrat avec 32-fin.js (qui choisit le profil).
   Textes écrits par l'agent D1 : titre, sousTitre, texte (chaîne ou fonction (etat, q) → chaîne), replique de Mme Lavigueur. */
(function (LTP) {
  'use strict';
  LTP.donnees.fins = [
    { id: 'reussie',  titre: 'La transition réussie',  sousTitre: '', texte: '', replique: '' },
    { id: 'usine',    titre: 'L\'usine à PR',          sousTitre: '', texte: '', replique: '' },
    { id: 'musee',    titre: 'Le musée',               sousTitre: '', texte: '', replique: '' },
    { id: 'farwest',  titre: 'Le Far West',            sousTitre: '', texte: '', replique: '' },
    { id: 'facture',  titre: 'La facture salée',       sousTitre: '', texte: '', replique: '' },
    { id: 'chateau',  titre: 'Le château de cartes',   sousTitre: '', texte: '', replique: '' },
    { id: 'congedie', titre: 'Merci pour votre contribution', sousTitre: '', texte: '', replique: '' },
  ];
})(globalThis.LTP = globalThis.LTP || {});
