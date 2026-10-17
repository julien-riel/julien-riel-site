/* 16-donnees-dialogues.js */
/* Tutoriel, indices de Ginette et bulles génériques. Contenu écrit par l'agent D1.
   LTP.donnees.tutoriel : [{ id, region: 'openspace' | 'panneau' | 'tableau' | 'hud' | 'journal', personnage, texte, action? }]
   LTP.donnees.indices.leviers : { [levierId]: { [valeur]: texte } }   // commentaire de Ginette quand on choisit cette valeur
   LTP.donnees.indices.trimestres : { [n]: texte }                       // ouverture de chaque trimestre
   LTP.donnees.bulles : { code: [], revue: [], incident: [], fatigue: [], formation: [], attente: [], conge: [] }
                        // bulles génériques, utilisées quand un développeur n'a pas de bulle propre pour son état */
(function (LTP) {
  'use strict';
  LTP.donnees.tutoriel = [];
  LTP.donnees.indices = { leviers: {}, trimestres: {} };
  LTP.donnees.bulles = { code: ['Je code.'], revue: ['Je relis.'], incident: ['Ça brûle.'], fatigue: ['Café.'], formation: ['J\'apprends.'], attente: ['J\'attends une revue.'], conge: ['En congé.'] };
})(globalThis.LTP = globalThis.LTP || {});
