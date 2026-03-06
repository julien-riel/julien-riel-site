---
title: "Maîtriser les Skills Claude Code : Création et Coordination"
date: 2026-03-05
tags:
  - claude
  - ia
  - développement
  - productivité
description: "Guide complet pour créer et coordonner plusieurs skills Claude Code afin d'optimiser votre workflow de développement avec l'IA."
---

Claude Code est devenu un outil incontournable pour les développeurs qui souhaitent augmenter leur productivité. L'une de ses fonctionnalités les plus puissantes, mais souvent méconnue, est le système de skills personnalisés. Cet article vous guide à travers la création et la coordination de plusieurs skills pour transformer Claude en assistant sur mesure.

## Qu'est-ce qu'un Skill Claude ?

Un **skill** est un ensemble d'instructions spécialisées qui confère à Claude une expertise dans un domaine particulier. Plutôt que de répéter les mêmes instructions à chaque session, vous définissez une fois pour toutes le comportement souhaité dans un fichier de configuration.

### Pourquoi utiliser des skills ?

- **Cohérence** : Claude applique les mêmes standards à chaque fois
- **Expertise** : Instructions détaillées pour des domaines complexes
- **Efficacité** : Pas besoin de réexpliquer votre contexte
- **Modularité** : Combinez plusieurs compétences selon vos besoins

## Anatomie d'un Skill

Un skill est défini dans votre répertoire `.claude/skills/` et se compose de deux éléments :

### 1. Le fichier de définition (skill.md)

```markdown
# Nom du Skill

## Description
Description concise du domaine d'expertise

## Déclencheurs
Quand ce skill doit-il être activé ?

## Instructions
Instructions détaillées pour Claude

## Exemples
Exemples concrets d'utilisation
```

### 2. La structure de dossier

```
.claude/
  skills/
    frontend-design/
      SKILL.md
      examples/
        component-example.jsx
    backend-api/
      SKILL.md
      templates/
        endpoint-template.js
    database-design/
      SKILL.md
```

## Créer votre premier Skill

### Exemple : Skill de revue de code

Créons un skill spécialisé dans la revue de code selon vos standards :

```markdown
# Code Review Expert

## Description
Expert en revue de code qui analyse la qualité, la sécurité 
et les bonnes pratiques selon les standards de l'équipe.

## Déclencheurs
- L'utilisateur demande une revue de code
- Mots-clés : "review", "analyse", "vérifie le code"
- Avant un merge ou un commit important

## Instructions

### Critères d'évaluation

1. **Sécurité**
   - Vérifier les injections SQL/XSS
   - Validation des entrées utilisateur
   - Gestion des secrets et credentials
   - Authentification et autorisation

2. **Performance**
   - Complexité algorithmique
   - Optimisations possibles
   - N+1 queries
   - Cache et memoization

3. **Maintenabilité**
   - Lisibilité du code
   - Documentation et commentaires
   - Respect du DRY
   - Séparation des responsabilités

4. **Tests**
   - Couverture de tests
   - Cas limites testés
   - Tests unitaires et d'intégration

### Format de sortie

Pour chaque problème identifié :

```markdown
## [PRIORITÉ] Catégorie - Titre

**Fichier** : `chemin/vers/fichier.js:42`

**Problème** :
Description du problème

**Impact** :
- Sécurité / Performance / Maintenabilité

**Solution recommandée** :
\`\`\`javascript
// Code corrigé
\`\`\`

**Justification** :
Pourquoi cette solution est préférable
```

### Priorités
- 🔴 **CRITIQUE** : À corriger immédiatement
- 🟠 **IMPORTANT** : À corriger avant le merge
- 🟡 **MINEUR** : Amélioration suggérée
- 🔵 **INFO** : Note ou observation

## Exemples

### Exemple 1 : Injection SQL

```javascript
// ❌ Code problématique
const query = `SELECT * FROM users WHERE id = ${userId}`;
db.query(query);
```

**Review** :
🔴 **CRITIQUE** - Injection SQL possible

### Exemple 2 : Gestion d'erreur

```javascript
// ❌ Code problématique
try {
  await riskyOperation();
} catch (e) {
  console.log(e);
}
```

**Review** :
🟠 **IMPORTANT** - Erreur silencieuse, logging insuffisant
```

### Créer le fichier

```bash
mkdir -p .claude/skills/code-review
cat > .claude/skills/code-review/SKILL.md << 'EOF'
# [Contenu ci-dessus]
EOF
```

## Coordonner Plusieurs Skills

La vraie puissance vient de la coordination de skills complémentaires. Voici comment procéder :

### 1. Définir des domaines clairs

Chaque skill doit avoir un domaine bien défini sans chevauchement :

```
frontend-design     → Interface utilisateur, UX, styling
backend-api        → Endpoints, business logic, middleware
database-design    → Schémas, migrations, requêtes
testing-expert     → Tests unitaires, intégration, e2e
devops-automation  → CI/CD, déploiement, monitoring
security-audit     → Vulnérabilités, best practices sécurité
```

### 2. Créer une hiérarchie de skills

Certains skills peuvent en appeler d'autres :

```markdown
# Full-Stack Architect

## Description
Coordonne les skills frontend, backend et database pour 
une architecture complète.

## Quand l'utiliser
- Démarrage d'un nouveau projet
- Refonte architecture complète
- Review architecture globale

## Coordination

### Phase 1 : Analyse
Active le **database-design** skill pour :
- Modéliser les entités
- Définir les relations
- Planifier les migrations

### Phase 2 : Backend
Active le **backend-api** skill pour :
- Concevoir les endpoints RESTful
- Implémenter la business logic
- Gérer l'authentification

### Phase 3 : Frontend
Active le **frontend-design** skill pour :
- Créer les composants UI
- Implémenter la navigation
- Gérer l'état applicatif

### Phase 4 : Testing
Active le **testing-expert** skill pour :
- Tests unitaires de chaque couche
- Tests d'intégration API
- Tests e2e du parcours utilisateur

## Checkpoints

À chaque phase, vérifier :
- ✅ Code implémenté et testé
- ✅ Documentation à jour
- ✅ Pas de code smell détecté
- ✅ Performance validée
```

### 3. Utiliser des conventions de nommage

Facilitez la coordination avec des conventions :

```markdown
## Conventions inter-skills

### Database → Backend
- Nom des modèles : PascalCase
- Nom des tables : snake_case, pluriel
- Relations : expliciter dans les deux sens

### Backend → Frontend
- Endpoints : `/api/v1/ressource`
- Réponses : toujours `{ data, error, meta }`
- Dates : ISO 8601 format

### Error Handling
- Codes HTTP standards uniquement
- Messages d'erreur structurés
- Logging uniforme : Winston format
```

### 4. Exemple de workflow coordonné

Imaginons une demande : "Ajoute une fonctionnalité de commentaires"

**Claude utilise automatiquement plusieurs skills :**

#### Étape 1 : Database Design Skill
```sql
-- Création de la table comments
CREATE TABLE comments (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  post_id INTEGER REFERENCES posts(id),
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_comments_post_id ON comments(post_id);
CREATE INDEX idx_comments_user_id ON comments(user_id);
```

#### Étape 2 : Backend API Skill
```javascript
// routes/comments.js
const express = require('express');
const router = express.Router();
const { authenticate } = require('../middleware/auth');
const { validateComment } = require('../validators/comment');

/**
 * @route   POST /api/v1/comments
 * @desc    Créer un nouveau commentaire
 * @access  Private
 */
router.post('/', authenticate, validateComment, async (req, res) => {
  try {
    const { post_id, content } = req.body;
    const user_id = req.user.id;

    const comment = await db.comments.create({
      user_id,
      post_id,
      content
    });

    res.status(201).json({
      data: comment,
      error: null,
      meta: { created_at: new Date().toISOString() }
    });
  } catch (error) {
    logger.error('Error creating comment:', error);
    res.status(500).json({
      data: null,
      error: 'Failed to create comment',
      meta: {}
    });
  }
});

module.exports = router;
```

#### Étape 3 : Frontend Design Skill
```jsx
// components/CommentForm.jsx
import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';

export const CommentForm = ({ postId, onCommentAdded }) => {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { user } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!content.trim()) {
      setError('Le commentaire ne peut pas être vide');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.post('/comments', {
        post_id: postId,
        content: content.trim()
      });

      setContent('');
      onCommentAdded(response.data);
    } catch (err) {
      setError(err.message || 'Erreur lors de l\'ajout du commentaire');
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return <p className="text-gray-600">Connectez-vous pour commenter</p>;
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="comment" className="block text-sm font-medium text-gray-700">
          Votre commentaire
        </label>
        <textarea
          id="comment"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={3}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm 
                     focus:border-blue-500 focus:ring-blue-500"
          placeholder="Partagez votre avis..."
          disabled={loading}
        />
      </div>

      {error && (
        <p className="text-sm text-red-600">{error}</p>
      )}

      <button
        type="submit"
        disabled={loading || !content.trim()}
        className="px-4 py-2 bg-blue-600 text-white rounded-md 
                   hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                   transition-colors"
      >
        {loading ? 'Envoi...' : 'Publier'}
      </button>
    </form>
  );
};
```

#### Étape 4 : Testing Expert Skill
```javascript
// __tests__/comments.test.js
describe('Comments Feature', () => {
  describe('POST /api/v1/comments', () => {
    it('should create a comment when authenticated', async () => {
      const response = await request(app)
        .post('/api/v1/comments')
        .set('Authorization', `Bearer ${validToken}`)
        .send({
          post_id: 1,
          content: 'Great post!'
        });

      expect(response.status).toBe(201);
      expect(response.body.data).toHaveProperty('id');
      expect(response.body.data.content).toBe('Great post!');
    });

    it('should reject unauthenticated requests', async () => {
      const response = await request(app)
        .post('/api/v1/comments')
        .send({
          post_id: 1,
          content: 'Great post!'
        });

      expect(response.status).toBe(401);
    });

    it('should validate comment content', async () => {
      const response = await request(app)
        .post('/api/v1/comments')
        .set('Authorization', `Bearer ${validToken}`)
        .send({
          post_id: 1,
          content: ''
        });

      expect(response.status).toBe(400);
      expect(response.body.error).toBeTruthy();
    });
  });
});
```

## Best Practices pour la Coordination

### 1. Documentation croisée

Chaque skill doit référencer les skills avec lesquels il interagit :

```markdown
## Skills connexes

- **database-design** : Pour la structure des données
- **testing-expert** : Pour valider les endpoints
- **security-audit** : Pour vérifier la sécurité
```

### 2. Interfaces claires

Définissez des contrats entre skills :

```markdown
## Contrat Backend → Frontend

### Format de réponse
\`\`\`typescript
interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  meta: {
    timestamp: string;
    requestId?: string;
    pagination?: {
      page: number;
      perPage: number;
      total: number;
    };
  };
}
\`\`\`

### Codes HTTP
- 200 : Succès
- 201 : Création réussie
- 400 : Validation échouée
- 401 : Non authentifié
- 403 : Non autorisé
- 404 : Ressource introuvable
- 500 : Erreur serveur
```

### 3. État partagé

Créez un skill de configuration partagée :

```markdown
# Project Config

## Stack technique
- Frontend : React 18 + TypeScript + Vite
- Backend : Node.js 20 + Express + PostgreSQL
- Testing : Jest + React Testing Library + Playwright
- Styling : Tailwind CSS 3

## Conventions de code
- Indentation : 2 espaces
- Quotes : Simple quotes
- Semi-colons : Toujours
- Line length : 100 caractères max

## Structure de projet
\`\`\`
src/
  components/    → Composants réutilisables
  pages/         → Pages/routes
  services/      → API calls
  hooks/         → Custom hooks
  utils/         → Fonctions utilitaires
  types/         → TypeScript types
\`\`\`
```

### 4. Priorités et conflits

Gérez les situations où plusieurs skills donnent des recommandations différentes :

```markdown
## Ordre de priorité

1. **security-audit** : La sécurité prime toujours
2. **testing-expert** : La testabilité est essentielle
3. **performance-optimizer** : Puis la performance
4. **code-style** : Enfin le style

En cas de conflit, le skill prioritaire l'emporte.
```

## Cas d'usage avancés

### Skill de migration

Coordonnez plusieurs skills pour migrer une ancienne base de code :

```markdown
# Migration Expert

## Processus

### 1. Audit (security-audit + code-review)
- Identifier les vulnérabilités
- Lister le code legacy problématique
- Prioriser les risques

### 2. Planification (full-stack-architect)
- Définir l'architecture cible
- Planifier les étapes de migration
- Estimer l'effort

### 3. Exécution par couche
- **Database** : Migrations incrémentales
- **Backend** : Refactoring par module
- **Frontend** : Composant par composant

### 4. Validation (testing-expert)
- Tests de non-régression
- Tests de performance
- Tests de sécurité

### 5. Documentation
- Documenter les changements
- Mettre à jour les runbooks
- Former l'équipe
```

### Skill de monitoring

Créez un skill qui réagit aux métriques :

```markdown
# Performance Monitor

## Déclencheurs
- Temps de réponse > 500ms
- Utilisation mémoire > 80%
- Taux d'erreur > 1%

## Actions automatiques

### Si backend lent
1. Activer **database-design** skill
   - Analyser les requêtes N+1
   - Suggérer des index
   - Proposer du caching

2. Activer **backend-api** skill
   - Optimiser les endpoints
   - Implémenter la pagination
   - Ajouter du rate limiting

### Si frontend lent
1. Activer **frontend-design** skill
   - Analyser le bundle size
   - Lazy loading des composants
   - Optimiser les re-renders
```

## Outils et automatisation

### Script de génération de skill

```bash
#!/bin/bash
# create-skill.sh

SKILL_NAME=$1
SKILL_DIR=".claude/skills/${SKILL_NAME}"

mkdir -p "${SKILL_DIR}/examples"
mkdir -p "${SKILL_DIR}/templates"

cat > "${SKILL_DIR}/SKILL.md" << EOF
# ${SKILL_NAME}

## Description
[Description du skill]

## Déclencheurs
- [Quand utiliser ce skill]

## Instructions
[Instructions détaillées]

## Exemples
[Exemples concrets]

## Skills connexes
- [Skills complémentaires]
EOF

echo "✅ Skill '${SKILL_NAME}' créé dans ${SKILL_DIR}"
```

### Template de coordination

```markdown
# [Feature Name] Workflow

## Skills impliqués
- [ ] database-design
- [ ] backend-api
- [ ] frontend-design
- [ ] testing-expert

## Étapes
1. [ ] Modélisation des données
2. [ ] Création des endpoints
3. [ ] Implémentation UI
4. [ ] Tests
5. [ ] Documentation

## Checkpoints
- Après chaque étape, valider avec code-review skill
- Avant le merge, valider avec security-audit skill
```

## Conclusion

La coordination de skills Claude Code transforme votre workflow de développement. Au lieu de répéter les mêmes instructions, vous construisez progressivement un système d'expertise personnalisé qui :

- **Accélère** le développement en appliquant automatiquement vos standards
- **Améliore** la qualité grâce à des revues systématiques
- **Documente** votre approche et vos décisions architecturales
- **Forme** les nouveaux membres de l'équipe sur vos pratiques

### Pour aller plus loin

- Créez des skills spécifiques à votre domaine métier
- Partagez vos skills avec votre équipe via un repository commun
- Versionnez vos skills comme du code
- Mesurez l'impact : temps gagné, bugs évités, qualité améliorée

Le système de skills n'est pas qu'une fonctionnalité : c'est une façon de capturer et de transmettre l'expertise, transformant Claude d'un assistant générique en un véritable membre de votre équipe.

---

**Ressources**

- [Documentation Claude Skills](https://docs.anthropic.com/claude/skills)
- [Repository de skills communautaires](https://github.com/anthropics/claude-skills)
- [Exemples de coordination avancée](https://github.com/anthropics/claude-skills-examples)
