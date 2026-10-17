// Charge les modules du moteur et des données (fichiers 00 à 39) dans un contexte vm isolé, sans DOM.
// Chaque appel renvoie un LTP neuf : les tests ne se contaminent pas entre eux.
// charger({ langue: 'en' }) ajoute les surcouches de l'édition anglaise (en/), dans l'ordre de l'assemblage.
import vm from "node:vm";
import { readFileSync } from "node:fs";
import { fichiersJs } from "../sources.js";

export function fichiersMoteur(langue = "fr") {
  return fichiersJs(langue, { moteurSeulement: true }).map((f) => (f.dossier === "js" ? f.fichier : `${f.dossier}/${f.fichier}`));
}

export function charger({ langue = "fr" } = {}) {
  const contexte = vm.createContext({ console });
  for (const f of fichiersJs(langue, { moteurSeulement: true })) {
    vm.runInContext(readFileSync(f.chemin, "utf8"), contexte, { filename: f.dossier === "js" ? f.fichier : `${f.dossier}/${f.fichier}` });
  }
  return contexte.LTP;
}

// Copie JSON : les objets créés dans le contexte vm n'ont pas les mêmes prototypes que ceux du test.
export function json(x) {
  return JSON.parse(JSON.stringify(x));
}

// Parcourt toutes les feuilles d'un objet : f(chemin, valeur).
export function parcourir(obj, f, chemin = "") {
  if (obj === null || typeof obj !== "object") return f(chemin, obj);
  for (const k of Object.keys(obj)) parcourir(obj[k], f, chemin ? `${chemin}.${k}` : k);
}

// Stratégie aléatoire mais déterministe : change quelques leviers débloqués à chaque trimestre.
export function strategieAleatoire(LTP, graine) {
  let s = graine >>> 0 || 1;
  const hasard = () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 4294967296);
  const choisir = (l) => l[Math.floor(hasard() * l.length)];
  return {
    suivreCalendrier: false,
    leviers(etat) {
      const actions = [];
      for (const l of LTP.donnees.leviers) {
        if (!LTP.q.levierDebloque(etat, l.id) || hasard() < 0.5) continue;
        actions.push({ type: "levier", id: l.id, valeur: choisir(l.valeurs).v });
        if (l.surchargeable && hasard() < 0.3) {
          actions.push({ type: "levier", id: l.id, valeur: choisir(l.valeurs).v, ilot: choisir(etat.ordreIlots) });
        }
      }
      for (const c of LTP.donnees.capacites || []) {
        if (hasard() < 0.2) actions.push({ type: "equiper", carte: c.id, ilot: choisir(etat.ordreIlots) });
      }
      return actions;
    },
    choix(etat, def) {
      return def && def.choix && def.choix.length ? choisir(def.choix).id : undefined;
    },
    comite(etat, question) {
      const def = (LTP.donnees.comites || []).find((c) => c.id === question.id);
      return def && def.reponses.length ? choisir(def.reponses).id : undefined;
    },
  };
}
