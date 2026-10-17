// Assemble « Livrer à tout prix » et son édition anglaise « Ship at Any Price », chacune en une seule page HTML.
// Sources : jeux/livrer-a-tout-prix/{page.fr.html, page.en.html, styles/*.css, js/*.js, en/*.js}
// Sorties : src/projets/livrer-a-tout-prix.html et src/projets/ship-at-any-price.html (versionnées, traitées par Eleventy)
// L'édition anglaise ajoute les surcouches de en/ (voir en/LISEZMOI.md) ; l'ordre est fixé par jeux/livrer-a-tout-prix/sources.js.
//
//   node scripts/build-livrer-a-tout-prix.js              écrit les deux pages du site
//   node scripts/build-livrer-a-tout-prix.js --check      échoue si une page n'est pas à jour
//   node scripts/build-livrer-a-tout-prix.js --autonome <chemin.html> [--langue en]
//                                                         écrit une page ouvrable en file://
//                                                         (sans front matter ni balises Nunjucks, non minifiée)
//
// La page du site est minifiée avec esbuild (JS et CSS) ; la version autonome reste lisible pour le débogage.
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { fichiersJs, langues } from "../jeux/livrer-a-tout-prix/sources.js";

const racine = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sources = join(racine, "jeux/livrer-a-tout-prix");
const sorties = {
  fr: join(racine, "src/projets/livrer-a-tout-prix.html"),
  en: join(racine, "src/projets/ship-at-any-price.html"),
};

function etiqueter(nom, contenu) {
  return `/* ${nom} */\n` + contenu.replace(/^\/\* [^*]+? \*\/\n/, "");
}

function lireDossier(dossier, extension) {
  return readdirSync(join(sources, dossier))
    .filter((f) => f.endsWith(extension))
    .sort()
    .map((f) => etiqueter(f, readFileSync(join(sources, dossier, f), "utf8")));
}

function minifier(code, loader) {
  // charset utf8 : garde les caractères accentués et les espaces insécables tels quels au lieu de \uXXXX.
  return transformSync(code, { loader, minify: true, charset: "utf8", legalComments: "none", target: "es2018" }).code.trim();
}

export function assembler({ autonome = false, langue = "fr" } = {}) {
  let page = readFileSync(join(sources, `page.${langue}.html`), "utf8");
  let css = lireDossier("styles", ".css").join("\n");
  let js = fichiersJs(langue)
    .map((f) => etiqueter(f.dossier === "js" ? f.fichier : `${f.dossier}/${f.fichier}`, readFileSync(f.chemin, "utf8")))
    .join("\n");
  if (!autonome) {
    css = minifier(css, "css");
    js = minifier(js, "js");
  }
  if (js.includes("</script")) throw new Error("Un module contient « </script » : la page serait coupée.");
  if (autonome) page = page.replace(/^---\n[\s\S]*?\n---\n/, "");
  const entete = "<!-- Fichier généré par scripts/build-livrer-a-tout-prix.js à partir de jeux/livrer-a-tout-prix/. Ne pas éditer. -->\n";
  const script = autonome ? js : `{% raw %}\n${js}\n{% endraw %}`;
  const style = autonome ? css : `{% raw %}\n${css}\n{% endraw %}`;
  page = page
    .replace("<!-- @styles -->", () => `<style>\n${style}\n</style>`)
    .replace("<!-- @scripts -->", () => `<script>\n${script}\n</script>`);
  // L'en-tête vient après le front matter (Eleventy exige que celui-ci soit en tout début de fichier).
  page = autonome ? entete + page : page.replace(/^(---\n[\s\S]*?\n---\n)/, `$1${entete}`);
  return page;
}

const args = process.argv.slice(2);
const option = (nom) => (args.indexOf(nom) >= 0 ? args[args.indexOf(nom) + 1] : undefined);
const relatif = (chemin) => chemin.slice(racine.length + 1);
if (args[0] === "--check") {
  let ok = true;
  for (const langue of langues) {
    const actuel = existsSync(sorties[langue]) ? readFileSync(sorties[langue], "utf8") : "";
    if (assembler({ langue }) !== actuel) {
      console.error(`${relatif(sorties[langue])} n'est pas à jour : lancez npm run build:livrer`);
      ok = false;
    }
  }
  if (!ok) process.exit(1);
  console.log("Livrer à tout prix : pages à jour (fr, en).");
} else if (args[0] === "--autonome") {
  const langue = option("--langue") || "fr";
  const chemin = resolve(args[1] && !args[1].startsWith("--") ? args[1] : join(racine, `_livrer-autonome.${langue}.html`));
  mkdirSync(dirname(chemin), { recursive: true });
  writeFileSync(chemin, assembler({ autonome: true, langue }));
  console.log(`Livrer à tout prix (autonome, ${langue}) : ${chemin}`);
} else {
  for (const langue of langues) {
    const page = assembler({ langue });
    writeFileSync(sorties[langue], page);
    console.log(`Livrer à tout prix (${langue}) : ${relatif(sorties[langue])} (${Math.round(page.length / 1024)} Ko)`);
  }
}
