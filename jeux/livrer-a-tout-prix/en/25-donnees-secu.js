/* en/25-donnees-secu.js */
/* Édition anglaise : textes des incidents de sécurité. Les données de 25 (LTP.K.donneesSecu) ne portent aucun texte. */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }
  // {carte} est un id : on affiche le nom de la carte (anglais), comme le registre.
  function nomCarte(id) {
    var d = id != null && LTP.q.defCapacite ? LTP.q.defCapacite(id) : null;
    return d && d.nom ? d.nom : id;
  }

  LTP.enregistrerTextes({
    'incident.paquetHallucine': 'Hallucinated package: the agent installed a dependency with a made-up name, which an attacker had published in the meantime. {ilot}.',
    'incident.mcpEmpoisonne': 'Poisoned community MCP: it was sending out more than advertised. {ilot}.',
    'incident.changementSilencieux': 'A vendor MCP rewrote its tool definitions without warning; the agent followed along. {ilot}.',
    'incident.destructionProd': 'An agent “cleaned up” production. Without a WHERE. {ilot}.',
    'incident.fuiteDonnees': 'Sensitive data leak: {ilot}. Counsel Bérubé is on the way, PDF in hand.',
    'donneesSecu.changementDetecte': function (v) {
      return 'The MCP gateway caught a silent change in ' + (v.carte == null ? '{carte}' : nomCarte(v.carte))
        + ' on the ' + val(v, 'ilot') + ' pod: its calls are now limited to read-only.';
    },
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
