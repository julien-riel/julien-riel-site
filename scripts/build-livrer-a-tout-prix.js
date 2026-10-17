// Assemble « Livrer à tout prix » en une seule page HTML.
// Sources : jeux/livrer-a-tout-prix/{page.fr.html, styles/*.css, js/*.js}
// Sortie : src/projets/livrer-a-tout-prix.html (versionnée, traitée par Eleventy)
//
//   node scripts/build-livrer-a-tout-prix.js              écrit la page du site
//   node scripts/build-livrer-a-tout-prix.js --check      échoue si la page n'est pas à jour
//   node scripts/build-livrer-a-tout-prix.js --autonome <chemin.html>
//                                                         écrit une page ouvrable en file://
//                                                         (sans front matter ni balises Nunjucks, non minifiée)
//
// La page du site est minifiée avec esbuild (JS et CSS) ; la version autonome reste lisible pour le débogage.
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";

const racine = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sources = join(racine, "jeux/livrer-a-tout-prix");
const sortie = join(racine, "src/projets/livrer-a-tout-prix.html");

function lireDossier(dossier, extension) {
  return readdirSync(join(sources, dossier))
    .filter((f) => f.endsWith(extension))
    .sort()
    .map((f) => `/* ${f} */\n` + readFileSync(join(sources, dossier, f), "utf8").replace(/^\/\* [^*]+? \*\/\n/, ""));
}

function minifier(code, loader) {
  // charset utf8 : garde les caractères accentués et les espaces insécables tels quels au lieu de \uXXXX.
  return transformSync(code, { loader, minify: true, charset: "utf8", legalComments: "none", target: "es2018" }).code.trim();
}

export function assembler({ autonome = false } = {}) {
  let page = readFileSync(join(sources, "page.fr.html"), "utf8");
  let css = lireDossier("styles", ".css").join("\n");
  let js = lireDossier("js", ".js").join("\n");
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
if (args[0] === "--check") {
  const attendu = assembler();
  const actuel = existsSync(sortie) ? readFileSync(sortie, "utf8") : "";
  if (attendu !== actuel) {
    console.error("src/projets/livrer-a-tout-prix.html n'est pas à jour : lancez npm run build:livrer");
    process.exit(1);
  }
  console.log("Livrer à tout prix : page à jour.");
} else if (args[0] === "--autonome") {
  const chemin = resolve(args[1] || join(racine, "_livrer-autonome.html"));
  mkdirSync(dirname(chemin), { recursive: true });
  writeFileSync(chemin, assembler({ autonome: true }));
  console.log(`Livrer à tout prix (autonome) : ${chemin}`);
} else {
  const page = assembler();
  writeFileSync(sortie, page);
  console.log(`Livrer à tout prix : ${sortie} (${Math.round(page.length / 1024)} Ko)`);
}
