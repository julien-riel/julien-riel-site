/* 09-donnees-reflexions.js */
/* Contenu des réflexions des développeurs (lu par 33-reflexions.js).
   Une entrée est une chaîne, ou un objet :
     { texte, devs?: [ids], traits?: [traits], niveaux?: [niveaux], pilote?: bool, licence?: bool,
       effet?: bool (leviers : vrai = seulement une fois que le changement a passé un vendredi, faux = seulement avant),
       condition?: (etat, q, dev) → bool, ton?: 'plus' | 'moins' | 'neutre' }
   Les gabarits {dev} et {ilot} sont remplis par l'interface.
   - leviers[id][valeur] : ce qu'on pense d'un changement de levier qui nous touche ('*' : n'importe quelle valeur)
   - nouveautes[cle] : ce qu'on dit en lisant la note de service (cle : id de levier, 'levier:v1,v2' ou id de panneau)
   - situations : [{ id, quand(etat, q, dev, ilot) → bool | intensité, poids?, ton?, levier?, lignes: [entrées] }] */
(function (LTP) {
  'use strict';

  LTP.donnees.reflexions = {
    leviers: {
      licences: {
        tous: [
          { texte: 'J\'ai ma licence Copilot. Voyons voir.', licence: true, pilote: false, effet: false },
          { texte: 'L\'autocomplétion devine la fin de mes lignes. Des fois.', licence: true, effet: true },
        ],
        aucune: [
          { texte: 'Plus de Copilot ? Je m\'y étais habitué.', pilote: true, ton: 'moins' },
        ],
      },
    },
    nouveautes: {
      licences: [
        { texte: 'Le groupe pilote, c\'est moi. Je vous en reparle.', pilote: true, ton: 'plus' },
      ],
    },
    situations: [
      { id: 'sansLicence', poids: 0.8,
        quand: function (etat, q, dev) { return !q.aLicence(etat, dev) && q.devsDe(etat, dev.ilot).some(function (d) { return q.aLicence(etat, d); }); },
        lignes: [
          { texte: 'Mon voisin a Copilot. Moi, j\'ai mon clavier.', ton: 'moins' },
        ] },
      { id: 'fileRevue', poids: 1.2, ton: 'moins', levier: 'reviseurs',
        quand: function (etat, q, dev, ilot) {
          var reviseur = (dev.derniere && dev.derniere.revues > 0) || q.aTrait(dev, 'reviseur');
          return reviseur && ilot && ilot.fileRevue > 8 ? Math.min(1.5, ilot.fileRevue / 15) : false;
        },
        lignes: ['Encore des PR à relire. Elles poussent la nuit.'] },
    ],
  };
})(globalThis.LTP = globalThis.LTP || {});
