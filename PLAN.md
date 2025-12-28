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
- [ ] Collection `posts` - Tous les articles triés par date
- [ ] Collection `tagList` - Liste de tous les tags uniques
- [ ] Pages générées par tag (`/tags/{tag}/`)

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
- [ ] Syntax highlighting avec Prism.js ou Shiki
- [ ] Support PlantUML (rendu côté client ou build)
- [ ] Support Mermaid (rendu côté client)
- [ ] Génération automatique de la table des matières

---

## Phase 4 : Fonctionnalités

### 4.1 Recherche (Lunr.js)
- [ ] Générer un index JSON des articles au build
- [ ] Intégrer Lunr.js côté client
- [ ] Interface de recherche avec résultats instantanés
- [ ] Indexer : titre, contenu, tags

### 4.2 Table des matières
- [ ] Parser automatiquement les headings (h2, h3)
- [ ] Générer une navigation sticky sur la page article
- [ ] Highlight de la section active au scroll

### 4.3 Articles connexes
- [ ] Algorithme basé sur les tags en commun
- [ ] Afficher 3-5 articles similaires
- [ ] Exclure l'article courant

### 4.4 Système de tags
- [ ] Page `/tags/` listant tous les tags avec compteur
- [ ] Pages individuelles `/tags/{tag}/` avec articles filtrés
- [ ] Tags cliquables sur chaque article

---

## Phase 5 : Design UI (skill frontend-design)

### 5.1 Thème visuel
- [ ] Palette de couleurs (tech/moderne)
- [ ] Typographie (lisibilité code + prose)
- [ ] Espacement et rythme vertical

### 5.2 Pages à designer
- [ ] Page d'accueil - Hero + liste des derniers articles
- [ ] Page article - Layout avec sidebar (TOC + related)
- [ ] Page tags - Grille de tags
- [ ] Page résultats recherche

### 5.3 Composants UI
- [ ] Cards d'articles
- [ ] Blocs de code avec copie
- [ ] Diagrammes PlantUML/Mermaid stylisés
- [ ] Navigation responsive
- [ ] Footer avec liens

---

## Phase 6 : Performance (Lighthouse)

### 6.1 Optimisations
- [ ] Minification CSS/JS via Vite
- [ ] Images optimisées (formats modernes, lazy loading)
- [ ] Fonts optimisées (preload, font-display)
- [ ] Critical CSS inline
- [ ] Code splitting si nécessaire

### 6.2 SEO
- [ ] Meta tags (title, description, og:*)
- [ ] Sitemap XML
- [ ] robots.txt
- [ ] Données structurées (JSON-LD pour articles)

### 6.3 Accessibilité
- [ ] Navigation au clavier
- [ ] Contrastes suffisants
- [ ] Labels ARIA
- [ ] Skip links

---

## Phase 7 : Déploiement

### 7.1 Build
- [ ] Script de build production
- [ ] Vérification Lighthouse en CI

### 7.2 Hébergement
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
