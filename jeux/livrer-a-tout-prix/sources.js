// Ordre des sources JS d'une édition, partagé par le script d'assemblage et le chargeur de tests.
// Édition française : js/*.js. Édition anglaise : js/*.js, plus en/*.js. Chaque fichier de en/ porte le nom
// du fichier qu'il traduit et se place juste après lui (en/00-langue.js, lui, passe avant 00-noyau.js).
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const dossierJeu = dirname(fileURLToPath(import.meta.url));
export const langues = ["fr", "en"];

export function fichiersJs(langue = "fr", { moteurSeulement = false } = {}) {
  const lister = (dossier) =>
    readdirSync(join(dossierJeu, dossier))
      .filter((f) => /^\d\d-.*\.js$/.test(f))
      .filter((f) => !moteurSeulement || parseInt(f, 10) < 40)
      .map((f) => ({ dossier, fichier: f, chemin: join(dossierJeu, dossier, f), cle: f.replace(/\.js$/, "") + (dossier === "en" ? "￿" : "") }));
  const liste = lister("js").concat(langue === "en" ? lister("en") : []);
  return liste.sort((a, b) => (a.cle < b.cle ? -1 : a.cle > b.cle ? 1 : 0));
}
