/* 05-textes.fr.js */
/* Chaînes de l'interface en français. Les modules du moteur enregistrent leurs propres clés de journal
   (préfixées par leur système) avec LTP.enregistrerTextes(). Propriétaire : agent F. */
(function (LTP) {
  'use strict';
  LTP.enregistrerTextes({
    'jeu.titre': 'Livrer à tout prix',
    'jeu.sousTitre': 'Une année pour faire passer 21 développeurs au développement agentique.',
    'hud.semaine': 'Semaine {semaine}',
    'hud.trimestre': 'T{trimestre}',
    'action.jouer': 'Jouer',
    'action.pause': 'Pause',
    'action.semaine': 'Semaine suivante',
    'action.commencer': 'Commencer',
    'action.tutoriel': 'Tutoriel avec Ginette',
    'action.lancer': 'Lancer le trimestre',
    'action.continuer': 'Continuer',
    'action.recommencer': 'Nouvelle partie',
    'incident.bogue': 'Incident en production : {ilot}.',
    'incident.injection': 'Injection de prompt : {ilot}.',
  });
})(globalThis.LTP = globalThis.LTP || {});
