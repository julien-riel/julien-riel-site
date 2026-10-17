// Outils communs aux tests des systèmes 21 à 24 (production, qualité, dette, coûts).
// Construit une partie avec des leviers donnés et la fait avancer semaine par semaine,
// en écartant les cartes d'événement (on teste le moteur du travail, pas le contenu).
import { charger } from "./charger.js";

export const LTP = charger();
export const M = LTP.moteur;

// Réglages de référence des tests de systèmes : toute l'équipe a une licence, et l'autonomie de départ
// est le clavardage (niveau 1). Les leviers passés à partie() les remplacent.
export const REFERENCE = [{ id: "licences", valeur: "tous" }, { id: "autonomie", valeur: 1 }];

// Partie en exécution, tout débloqué, avec des actions de leviers appliquées.
// options.preparer(etat) : retouche l'état (JSON) avant la première semaine.
export function partie(graine, leviers = [], options = {}) {
  let e = M.creer(graine, { toutDebloque: true, serveurAchete: !!options.serveurAchete });
  e = M.action(e, { type: "phase", phase: "planification" }).etat;
  for (const a of [...REFERENCE, ...leviers]) {
    const r = M.action(e, { type: "levier", ...a });
    if (r.erreur) throw new Error(`levier ${a.id}=${a.valeur} : ${r.erreur}`);
    e = r.etat;
  }
  e = M.action(e, { type: "phase", phase: "execution" }).etat;
  if (options.preparer) {
    e = JSON.parse(JSON.stringify(e));
    options.preparer(e);
  }
  return e;
}

// Avance de n semaines. avant(etat) retouche l'état avant chaque semaine ; apres(etat, s) recueille des mesures.
export function avancer(e, n, { avant, apres } = {}) {
  for (let s = 0; s < n; s++) {
    if (e.phase !== "execution") {
      if (e.phase === "fin") break;
      e = M.action(e, { type: "phase", phase: "planification" }).etat;
      e = M.action(e, { type: "phase", phase: "execution" }).etat;
    }
    if (e.evenementEnAttente || avant) {
      e = JSON.parse(JSON.stringify(e));
      e.evenementEnAttente = null;
      if (avant) avant(e);
    }
    // Une semaine complète, du jour courant au vendredi, en écartant les cartes qui arrivent en chemin.
    for (let j = 0; j < 10; j++) {
      if (e.evenementEnAttente) { e = JSON.parse(JSON.stringify(e)); e.evenementEnAttente = null; }
      const r = M.jour(e);
      if (!r.avance) break;
      e = r.etat;
      if (r.finSemaine) break;
    }
    if (apres) apres(e, s);
  }
  return e;
}

// Moyenne d'une mesure sur plusieurs graines : mesure(graine) → nombre.
export function moyenneGraines(graines, mesure) {
  let s = 0;
  for (const g of graines) s += mesure(g);
  return s / graines.length;
}

export const GRAINES = [1, 2, 3, 4, 5, 6];

export const somme = (liste, f) => liste.reduce((s, x) => s + f(x), 0);
export const ilots = (e) => Object.values(e.ilots);
export const devs = (e) => Object.values(e.devs);

// Carnet de demandes très garni : la production n'est plus limitée par la demande.
export function carnetPlein(e) {
  for (const il of Object.values(e.ilots)) il.backlog = Math.max(il.backlog, 1000);
}
