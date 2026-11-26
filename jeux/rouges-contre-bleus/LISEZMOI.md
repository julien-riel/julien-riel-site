# Rouges contre Bleus

Jeu de stratégie de flux en 3D (TypeScript, Vite, Three.js, simulation dans un Web Worker).
Son code source vit dans un autre dépôt (`army-game`) ; ce dossier ne contient que le résultat
du build, publié tel quel à l'adresse `/projets/rouges-contre-bleus/`.

## Mettre à jour le jeu

Depuis le dépôt `army-game`, à côté de celui-ci :

```sh
npx vite build --base=/projets/rouges-contre-bleus/ \
  --outDir ../julien-riel-site/jeux/rouges-contre-bleus/dist --emptyOutDir
```

Puis, dans `dist/`, retirer les pages de développement (`demo.html`, `play.html`, `debug.html`)
et les fichiers de `assets/` que plus rien ne référence.

## Comment il arrive dans le site

- En production : `scripts/copy-rouges-contre-bleus.js`, lancé par `npm run build` après Eleventy
  (l'étape Vite d'Eleventy ne conserve pas les fichiers qu'elle n'a pas empaquetés).
- En développement : copie directe déclarée dans `eleventy.config.js`.
- La fiche de la page « Mes projets » est dans `src/_data/projets.json`.
