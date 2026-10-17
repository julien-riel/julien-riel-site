/* 13-donnees-capacites.js */
/* Skills et serveurs MCP du Registre (CONCEPTION §4.9). Contenu écrit par l'agent D2.
   Forme d'une carte :
   {
     id, type: 'skill' | 'mcp', nom, provenance: 'maison' | 'fournisseur' | 'communaute' | 'personnelle',
     trimestre,                       // trimestre où la carte apparaît dans le registre
     disponible: 'depart' | 'evenement' | 'developpement',
                                      // evenement : découverte par un effet { carte, op: 'decouvrir' }
                                      // developpement : doit être développée (coutDev semaines d'un développeur)
     auteur: devId | null,            // personnelle ou maison : l'auteur est le propriétaire par défaut
     coutDev: 0,                      // semaines de travail avant disponibilité (cartes maison)
     ilotsSuggeres: [ilotId],
     description,                     // une ou deux phrases, ton du jeu
     gains: { debit, tauxDefaut, jetons, mttr, comprehension, adequationPile, couverture, verbosite },
                                      // mêmes clés que ilot.mods (20-etat.js) : additifs pour debit, comprehension,
                                      // adequationPile, couverture ; multiplicatifs pour les autres
     ilotsGains: { [ilotId]: { …gains } },   // facultatif : gains propres à certains îlots
     poidsContexte: 0.05,
     branches: ['prive', 'nonFiable', 'sortie'],
     risque: { type, proba },         // type : exfiltration · injection · destruction · abandon ·
                                      //   changementSilencieux · tarif · typosquat · perime · aucun
     declencheur: null | 'tests' | 'deploy' | …,   // deux skills au même déclencheur dans un îlot = conflit
     prerequis: { levier, valeurs: [...] } | null,
   } */
(function (LTP) {
  'use strict';
  LTP.donnees.capacites = [
    { id: 'conventions-gpu', type: 'skill', nom: 'conventions-gpu', provenance: 'maison', trimestre: 2, disponible: 'developpement',
      auteur: null, coutDev: 2, ilotsSuggeres: ['portail', 'web', 'inspections', 'operations'],
      description: 'Les conventions de code du GPU, écrites une fois pour tous les agents.',
      gains: { tauxDefaut: 0.85, jetons: 0.92 }, poidsContexte: 0.04, branches: [], risque: { type: 'perime', proba: 0.01 },
      declencheur: 'conventions', prerequis: null },
    { id: 'mcp-catalogue', type: 'mcp', nom: 'mcp-catalogue', provenance: 'maison', trimestre: 3, disponible: 'developpement',
      auteur: null, coutDev: 6, ilotsSuggeres: ['appro', 'iot', 'portail'],
      description: 'Les agents interrogent la source officielle plutôt que la troisième copie Excel.',
      gains: { tauxDefaut: 0.8 }, poidsContexte: 0.06, branches: ['prive'], risque: { type: 'aucun', proba: 0 },
      declencheur: null, prerequis: { levier: 'catalogue', valeurs: ['sources'] } },
    { id: 'mcp-courriel-fournisseurs', type: 'mcp', nom: 'mcp-courriel-fournisseurs', provenance: 'maison', trimestre: 3, disponible: 'depart',
      auteur: 'yannick', coutDev: 0, ilotsSuggeres: ['appro'],
      description: 'L\'agent lit les courriels des fournisseurs et y répond. Pratique. Très pratique.',
      gains: { debit: 0.1 }, poidsContexte: 0.05, branches: ['nonFiable', 'sortie'], risque: { type: 'injection', proba: 0.02 },
      declencheur: null, prerequis: null },
  ];
})(globalThis.LTP = globalThis.LTP || {});
