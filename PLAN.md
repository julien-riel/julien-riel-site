# Plan d'implémentation - julien-riel.com

## Vue d'ensemble

Site de blog sur l'informatique et l'IA, construit avec Eleventy + Vite.

---

## Phase 1 : Structure du projet

### 1.1 Initialisation
- [x] Initialiser npm (`package.json`)
- [x] Installer Eleventy (`@11ty/eleventy`)
- [x] Installer Vite (`vite`)
- [x] Configurer l'intégration Eleventy + Vite

### 1.2 Arborescence des dossiers
```
julien-riel-site/
├── src/
│   ├── _data/              # Données globales (metadata, navigation)
│   ├── _includes/
│   │   ├── layouts/        # Templates de base
│   │   └── components/     # Composants réutilisables
│   ├── assets/
│   │   ├── css/            # Styles
│   │   ├── js/             # JavaScript client
│   │   └── images/         # Images
│   ├── posts/              # Articles en Markdown
│   ├── tags/               # Page de listing des tags
│   └── index.njk           # Page d'accueil
├── .eleventy.js            # Configuration Eleventy
├── vite.config.js          # Configuration Vite
└── package.json
```

### 1.3 Fichiers de configuration
- [x] `eleventy.config.js` - Configuration Eleventy
- [x] `vite.config.js` - Configuration Vite
- [x] `.gitignore`

---

## Phase 2 : Layouts et templates

### 2.1 Layout de base
- [x] `_includes/layouts/base.njk` - HTML shell, meta tags, liens CSS/JS
- [x] `_includes/layouts/post.njk` - Template article (hérite de base)
- [x] `_includes/layouts/page.njk` - Template page simple

### 2.2 Composants
- [x] `_includes/components/header.njk` - Navigation principale
- [x] `_includes/components/footer.njk` - Pied de page
- [x] `_includes/components/post-card.njk` - Carte de prévisualisation article
- [x] `_includes/components/tag-list.njk` - Liste de tags cliquables
- [x] `_includes/components/toc.njk` - Table des matières
- [x] `_includes/components/related-posts.njk` - Articles connexes
- [x] `_includes/components/search.njk` - Barre de recherche

---

## Phase 3 : Système de contenu

### 3.1 Configuration des collections
- [x] Collection `posts` - Tous les articles triés par date
- [x] Collection `tagList` - Liste de tous les tags uniques
- [x] Pages générées par tag (`/tags/{tag}/`)

### 3.2 Frontmatter des articles
```yaml
---
title: "Titre de l'article"
date: 2024-01-15
tags:
  - intelligence-artificielle
  - python
description: "Description pour SEO"
draft: false
---
```

### 3.3 Markdown avancé
- [x] Syntax highlighting avec Prism.js ou Shiki
- [x] Support PlantUML (rendu côté client ou build)
- [x] Support Mermaid (rendu côté client)
- [x] Génération automatique de la table des matières

---

## Phase 4 : Fonctionnalités

### 4.1 Recherche (Lunr.js)
- [x] Générer un index JSON des articles au build
- [x] Intégrer Lunr.js côté client
- [x] Interface de recherche avec résultats instantanés
- [x] Indexer : titre, contenu, tags

### 4.2 Table des matières
- [x] Parser automatiquement les headings (h2, h3)
- [x] Générer une navigation sticky sur la page article
- [x] Highlight de la section active au scroll

### 4.3 Articles connexes
- [x] Algorithme basé sur les tags en commun
- [x] Afficher 3-5 articles similaires
- [x] Exclure l'article courant

### 4.4 Système de tags
- [x] Page `/tags/` listant tous les tags avec compteur
- [x] Pages individuelles `/tags/{tag}/` avec articles filtrés
- [x] Tags cliquables sur chaque article

---

## Phase 5 : Design UI (skill frontend-design)

### 5.1 Thème visuel - "Copper Circuit"
- [x] Palette de couleurs (cuivre/ambre + teal + crème chaud)
- [x] Typographie (DM Serif Display + Outfit + JetBrains Mono)
- [x] Espacement et rythme vertical (système de spacing CSS variables)

### 5.2 Pages à designer
- [x] Page d'accueil - Hero avec gradient animé + décorations géométriques
- [x] Page article - Layout avec sidebar TOC sticky + related posts
- [x] Page tags - Grille de tags avec compteurs
- [x] Page résultats recherche - Dropdown avec highlighting

### 5.3 Composants UI
- [x] Cards d'articles avec hover effects
- [x] Blocs de code avec copie + indicateur de langage
- [x] Diagrammes PlantUML/Mermaid stylisés
- [x] Navigation responsive avec menu hamburger animé
- [x] Footer avec liens et icônes sociales

---

## Phase 6 : Performance (Lighthouse)

### 6.1 Optimisations
- [x] Minification CSS/JS via Vite
- [x] Images optimisées (formats modernes, lazy loading)
- [x] Fonts optimisées (preload, font-display)
- [x] Critical CSS inline
- [x] Code splitting si nécessaire

### 6.2 SEO
- [x] Meta tags (title, description, og:*)
- [x] Sitemap XML
- [x] robots.txt
- [x] Données structurées (JSON-LD pour articles)

### 6.3 Accessibilité
- [x] Navigation au clavier
- [x] Contrastes suffisants
- [x] Labels ARIA
- [x] Skip links

---

## Phase 7 : Déploiement

### 7.1 Build
- [ ] Script de build production
- [ ] Vérification Lighthouse en CI


## Phase 8 : Hébergement

### 8.1 OVH
- [ ] Configuration pour hébergeur statique (Netlify, Vercel, Cloudflare Pages)
- [ ] Redirections si nécessaire
- [ ] Headers de cache

---

## Ordre de développement suggéré

1. **Phase 1** - Structure de base fonctionnelle
2. **Phase 2** - Layouts minimaux
3. **Phase 3** - Premier article de test
4. **Phase 5** - Design complet (skill frontend-design)
5. **Phase 4** - Fonctionnalités avancées
6. **Phase 6** - Optimisations performance
7. **Phase 7** - Déploiement

---

## Dépendances npm prévues

```json
{
  "devDependencies": {
    "@11ty/eleventy": "^3.0.0",
    "vite": "^5.0.0",
    "@11ty/eleventy-plugin-vite": "^4.0.0",
    "markdown-it": "^14.0.0",
    "markdown-it-anchor": "^8.0.0",
    "prismjs": "^1.29.0",
    "lunr": "^2.3.9"
  }
}
```

---

## Notes techniques

### Intégration Vite + Eleventy
Utiliser `@11ty/eleventy-plugin-vite` pour une intégration native. Vite gère le JS/CSS, Eleventy gère le HTML/Markdown.

### PlantUML
Option 1 : Rendu côté client avec plantuml-encoder + image externe
Option 2 : Plugin Eleventy custom pour générer les images au build

### Mermaid
Rendu côté client avec la librairie Mermaid.js, initialisation après le DOM ready.
