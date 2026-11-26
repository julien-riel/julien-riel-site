// Copie le jeu « Rouges contre Bleus » (déjà construit, voir jeux/rouges-contre-bleus/LISEZMOI.md)
// dans _site/projets/rouges-contre-bleus/. L'étape Vite d'Eleventy ne garde que ce qu'elle a
// elle-même empaqueté : la copie doit donc venir après elle. En développement (eleventy --serve),
// c'est la copie directe déclarée dans eleventy.config.js qui sert les mêmes fichiers.
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const racine = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(racine, "jeux/rouges-contre-bleus/dist");
const cible = join(racine, "_site/projets/rouges-contre-bleus");

if (!existsSync(join(source, "index.html"))) {
  throw new Error("jeux/rouges-contre-bleus/dist/index.html est absent : reconstruire le jeu (voir LISEZMOI.md).");
}
rmSync(cible, { recursive: true, force: true });
cpSync(source, cible, { recursive: true });
console.log("Copied jeux/rouges-contre-bleus/dist → _site/projets/rouges-contre-bleus/");
