/* en/26-registre.js */
/* Édition anglaise : textes du journal du registre (cartes, développement, conflits). */
(function (LTP) {
  'use strict';

  function val(v, k) { return v[k] == null ? '{' + k + '}' : String(v[k]); }
  // {a} et {b} sont des id de cartes : on affiche leur nom (anglais).
  function nomCarte(id) {
    var d = id != null && LTP.q.defCapacite ? LTP.q.defCapacite(id) : null;
    return d && d.nom ? d.nom : (id == null ? '?' : String(id));
  }
  // {declencheur} est un id technique : quelques-uns sont des mots français.
  var DECLENCHEURS = { nuit: 'night', securite: 'security', explication: 'explanation', embarque: 'embedded' };

  LTP.enregistrerTextes({
    'registre.equipee': 'The {nom} card is now equipped on the {ilot} pod.',
    'registre.desequipee': 'The {nom} card has been removed from the {ilot} pod.',
    'registre.developpement': function (v) {
      var n = Number(v.semaines);
      return val(v, 'dev') + ' starts developing ' + val(v, 'nom') + ' (' + val(v, 'semaines') + (n === 1 ? ' week).' : ' weeks).');
    },
    'registre.developpementInterrompu': 'Development of {nom} has stopped: the person in charge of it is gone.',
    'registre.carteDisponible': 'The {nom} card is ready: {dev} just finished it.',
    'registre.proprietaire': '{dev} becomes the owner of {nom}.',
    'registre.installationLibre': '{dev} installed {nom} on their own initiative. “It looked handy.”',
    'registre.perimee': 'The {nom} card is stale: it still documents the old API.',
    'registre.abandonnee': 'The {nom} card is no longer maintained by its community. Last commit: a long time ago.',
    'registre.conflit': function (v) {
      var d = v.declencheur == null ? '{declencheur}' : (DECLENCHEURS[v.declencheur] || String(v.declencheur));
      var a = nomCarte(v.a), b = nomCarte(v.b);
      if (a === b) { a = String(v.a); b = String(v.b); } // les trois skills « deploy » portent le même nom
      return 'Conflict on the ' + val(v, 'ilot') + ' pod: ' + a + ' and ' + b + ' are fighting over the “' + d + '” trigger.';
    },
    'registre.saturation': 'Context overload on the {ilot} pod: too many skills and MCPs, and the agent gets lost.',
  }, 'en');
})(globalThis.LTP = globalThis.LTP || {});
