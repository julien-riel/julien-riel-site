# Site julien-riel.com
```
claude mcp add playwright npx '@playwright/mcp@latest'
```

Fait avec eleventy.dev 

Le site traite de l'informatique et de l'intelligence artificielle

C'est un site qui contient des posts

Il utilise vite pour build le javascript

On veut avoir un haut score lighthouse

On peut faire de la recherche sur le site grâce à lunr.js

On peut tagger les post avec des tags. Il existe un page qui liste toutes les tags et permet de naviguer vers les articles

Les articles sont rédigés en markdown. Les triple backticks peuvent afficher du plantuml ou du mermaid. Si on met du code, il y a la syntaxe highlight

Sur la page d'un article, il y a une table des matières qui permet de naviguer vers une section. On retrouve aussi "articles connexes" qui contient des liens vers des articles avec des tags en commun

Tu peux utiliser le skill frontend-design pour concevoir le UI. 

## Carte des vignobles de Niagara-on-the-Lake

La page `/projets/vignobles-niagara/` s'appuie sur des GeoJSON versionnés dans
`src/assets/data/`. Pour les régénérer à partir des sources ouvertes
(OpenStreetMap, Niagara Open Data, Ontario GeoHub) :

```
node scripts/build-notl-data.js
```

Les réponses d'Overpass sont mises en cache sous `.cache/overpass/` : les
serveurs publics limitent le débit, et le cache évite de les solliciter deux
fois pour la même requête. Supprimez ce dossier pour forcer un rafraîchissement.
