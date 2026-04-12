---
title: "Architectures multi-agent : quand un seul agent ne suffit pas"
date: 2026-04-09
tags:
  - architecture
  - multi-agent
description: "Un guide pratique des patterns multi-agent — orchestrator-workers, pipelines, ensembles et swarms — et là où ils cassent."
---

## Un guide pratique pour programmeurs agentiques

Un seul agent avec les bons tools peut accomplir beaucoup. Mais à un moment, tu frappes un mur : la tâche est trop complexe pour un seul context window, elle exige différentes expertises à différentes étapes, ou elle a besoin de chemins d'exécution parallèles. C'est là qu'il te faut plusieurs agents qui travaillent ensemble.

Les systèmes multi-agent sont puissants. C'est aussi là que la complexité se multiplie le plus vite. Ce guide couvre quand les utiliser, comment les architecturer, et là où ils cassent.

## Pourquoi plusieurs agents ?

L'argument pour les architectures multi-agent repose sur trois contraintes :

**Les limites du context window.** Un seul agent qui gère un workflow complexe doit tenir les instructions, les définitions de tools, l'historique de conversation, les résultats intermédiaires et les documents récupérés — le tout dans un seul context window. À mesure que les tâches grossissent, le budget de contexte s'épuise. Répartir les responsabilités entre plusieurs agents fait que chacun opère avec un contexte ciblé et gérable.

**La spécialisation.** Différents sous-problèmes bénéficient de différents modèles, prompts et tools. Un agent de code a besoin de tools d'exécution et d'un modèle optimisé pour le code. Un agent de recherche a besoin de recherche web et d'un modèle fort en synthèse. Un agent de planning a besoin de profondeur de reasoning mais pas de tools du tout. Vouloir qu'un seul agent fasse tout, c'est donner une configuration médiocre à chaque sous-problème.

**Le parallélisme.** Certains sous-problèmes sont indépendants et peuvent rouler en même temps. Un seul agent exécute de manière séquentielle par nature — il génère un token à la fois. Plusieurs agents peuvent travailler en parallèle, ce qui réduit drastiquement la latence pour les tâches avec des sous-problèmes indépendants.

WHOOP opère plus de 500 agents spécialisés à travers son app — Memory, Daily Outlook, Day in Review, Activity Insights, onboarding, et des dizaines d'autres. Chacun a un rôle défini, son propre prompt et son propre ensemble de tools. Ce n'est pas de la complexité accidentelle. C'est une architecture délibérée qui permet à chaque agent d'être excellent dans une seule chose.

## Les patterns de base

### Pattern 1 : Orchestrator-Workers

Un agent (l'orchestrator) reçoit la requête de l'utilisateur, la décompose en sous-problèmes, et délègue chaque sous-problème à un worker agent spécialisé. L'orchestrator collecte les résultats et synthétise la réponse finale.

```
User request → Orchestrator → [Worker A, Worker B, Worker C] → Orchestrator → Final response
```

C'est le pattern le plus courant. Notion a reconstruit son architecture IA autour de ça — en remplaçant des chaînes de prompts spécifiques à chaque tâche par un modèle de reasoning central qui coordonne des sub-agents modulaires. L'orchestrator gère le planning et la synthèse. Les workers gèrent l'exécution.

**Quand l'utiliser :** des tâches complexes qui se décomposent naturellement en sous-problèmes (recherche + analyse + mise en forme, ou récupération de données + calcul + explication).

**Ce qui est dur :** l'orchestrator doit être assez intelligent pour bien décomposer la tâche et pour savoir quand le résultat d'un worker est assez bon. Une mauvaise décomposition mène à du travail gaspillé ou à du contexte manquant entre les sous-problèmes.

### Pattern 2 : Pipeline (handoff séquentiel)

Les agents traitent l'information en séquence, chacun raffinant ou transformant la sortie de l'étape précédente. Comme une chaîne de montage.

```
User request → Agent A (extract) → Agent B (analyze) → Agent C (format) → Final response
```

Le Bugbot original de Cursor utilisait une variation : huit instances parallèles du même agent, chacune traitant le diff de code dans un ordre différent, avec une étape de vote à la fin. C'est un hybride entre pipeline et ensemble.

**Quand l'utiliser :** des tâches avec des étapes séquentielles claires où chaque étape a des exigences différentes — extraction → validation → transformation → génération.

**Ce qui est dur :** la perte d'information entre les étapes. Chaque handoff est un point potentiel où du contexte critique se perd. Conçois ton format de communication entre agents avec soin — des données structurées avec des champs explicites, pas du texte libre.

### Pattern 3 : Débat / Ensemble

Plusieurs agents s'attaquent indépendamment au même problème, puis leurs sorties sont comparées, combinées ou soumises à un vote. Ça augmente la fiabilité au prix de la latence et du compute.

```
User request → [Agent A, Agent B, Agent C] → Aggregator → Final response
```

**Quand l'utiliser :** des décisions à haut enjeu où la précision compte plus que la vitesse — diagnostic médical, analyse légale, revue de code. Le vote majoritaire à huit passes de Bugbot était exactement ce pattern.

**Ce qui est dur :** définir comment agréger les désaccords. Le vote majoritaire est simple mais perd la nuance. Un judge agent séparé peut résoudre les conflits, mais ajoute un autre point de défaillance. Et le coût croît linéairement avec le nombre d'agents.

### Pattern 4 : Swarm autonome

Les agents spawnent dynamiquement des sub-agents selon ce qu'ils découvrent pendant l'exécution. L'orchestrator ne planifie pas tous les sous-problèmes d'avance — il s'adapte à mesure que de nouvelles informations émergent. Le modèle Composer de Cursor (basé sur Kimi K2.5) utilise Agent Swarm, où le modèle apprend par RL à décomposer dynamiquement les tâches et à dispatcher des sub-agents en parallèle.

**Quand l'utiliser :** des tâches exploratoires où la portée complète n'est pas connue d'avance — recherche, debug, investigation de données.

**Ce qui est dur :** tout. Le contrôle, l'observabilité, la gestion des coûts et la prévention des exécutions qui partent en vrille sont tous nettement plus difficiles quand la création d'agents est dynamique. Ce pattern exige un outillage mature et de solides coupe-circuits.

## La communication entre agents

La façon dont les agents se passent l'information les uns aux autres est aussi importante que ce que chaque agent fait. Trois approches, par ordre de structure croissante :

### Messages en langue naturelle

Les agents communiquent via du texte libre. Simple à implémenter, mais avec de la perte. L'agent qui reçoit doit parser du texte non structuré, ce qui peut faire manquer des détails critiques ou mal interpréter des formulations ambiguës.

Utilise ça quand : les agents gèrent des tâches intrinsèquement non structurées (écriture créative, recherche ouverte).

### Données structurées

Les agents s'échangent du JSON, du XML ou des objets typés avec des schémas définis. L'agent qui envoie produit des données structurées ; l'agent qui reçoit sait exactement quels champs attendre.

```json
{
  "task_id": "extract_metrics",
  "status": "complete",
  "results": {
    "heart_rate_avg": 54,
    "hrv_avg": 62,
    "sleep_score": 78
  },
  "confidence": 0.92,
  "sources": ["sleep_data_2024_03_15"]
}
```

Utilise ça chaque fois que des agents alimentent des étapes programmatiques en aval. La structure agit comme un contrat entre agents, rendant les défaillances explicites plutôt que silencieuses.

### State partagé / Blackboard

Tous les agents lisent et écrivent dans un objet de state partagé (parfois appelé blackboard). Chaque agent peut voir le contexte complet de ce que les autres agents ont fait et ajouter ses propres contributions.

Les Memory nuggets de WHOOP fonctionnent comme un state partagé : n'importe quel agent peut écrire une memory, et tous les agents peuvent lire les memories pertinentes. L'architecture par blocs de Notion sert un but similaire — les agents opèrent sur un graphe partagé de données structurées.

Utilise ça quand : les agents doivent être au courant du travail des autres sans communication point-à-point explicite. Le state partagé fournit la coordination sans le couplage.

## Là où les systèmes multi-agent cassent

### Les cascades de défaillances

Quand la mauvaise sortie de l'Agent A alimente l'Agent B, l'erreur s'amplifie. L'Agent B ne sait pas que l'Agent A s'est trompé — il traite l'entrée comme faisant autorité. Rendu à l'Agent C, l'erreur a été amplifiée et intégrée dans une réponse qui sonne confiante.

**Mitigation :** valide à chaque handoff. Ajoute des vérifications légères entre agents — validation de types, assertions, ou une passe rapide de LLM-as-a-judge qui flagge les résultats intermédiaires manifestement faux. Ne pars pas du principe que les agents en amont sont fiables.

### La perte de contexte

Chaque handoff entre agents est un goulot d'étranglement potentiel du contexte. L'orchestrator résume, et le résumé rate un détail critique. Le worker complète son sous-problème à la perfection — sauf qu'il n'avait pas la seule pièce d'information qui change tout.

**Mitigation :** sois explicite sur le contexte dont chaque agent a besoin. Ne compte pas sur la compréhension implicite. Inclus les métadonnées pertinentes dans les messages entre agents. Dans le doute, passe plus de contexte que ce qui semble nécessaire.

### L'explosion des coûts

Plusieurs agents, ça veut dire plusieurs appels LLM. Un pattern orchestrator-workers avec 5 workers et une étape de synthèse, ça fait au moins 7 appels LLM par requête. Si chaque worker fait du RAG et du reasoning multi-étapes, tu peux être à 20+ appels. À l'échelle de la production, ça devient cher vite.

**Mitigation :** utilise des modèles plus petits et moins chers pour les sous-problèmes simples. Tous les agents n'ont pas besoin d'un modèle frontier. Route selon la complexité de la tâche — de la même manière que Cursor utilise un modèle custom pour Tab, un 70B pour l'application de code, et des modèles frontier pour le reasoning.

### L'effondrement de l'observabilité

Quand un système multi-agent produit un mauvais résultat, tu dois savoir quel agent a échoué, ce qu'il a vu, et ce qu'il a produit. Sans logging structuré de chaque message entre agents et du reasoning de chaque agent, le debug est impossible.

**Mitigation :** logge tout. Chaque appel d'agent, chaque entrée, chaque sortie, chaque tool call. Le framework d'éval de WHOOP fournit des détails au niveau du trace pour chaque interaction d'agent — pas juste la sortie finale, mais la chaîne intermédiaire. C'est non négociable pour des systèmes multi-agent en production.

### Le surcoût de coordination

À mesure que tu ajoutes des agents, le coût de coordination grimpe. L'orchestrator dépense plus de tokens à gérer le workflow que les workers n'en dépensent sur le vrai travail. À un certain point, le surcoût dépasse le bénéfice de la spécialisation.

**Mitigation :** garde le nombre d'agents aussi petit que possible. Ne split pas en agents pour l'élégance architecturale — split seulement quand un seul agent ne peut vraiment pas gérer la tâche à cause des limites de contexte, des besoins de spécialisation ou des exigences de parallélisme. Trois agents bien conçus battent généralement dix mal conçus.

## Quand utiliser le multi-agent (et quand s'en abstenir)

**Utilise le multi-agent quand :**

- La tâche exige vraiment différents tools, modèles ou expertises à différentes étapes
- Le contexte pour la tâche complète dépasse ce qu'un seul agent peut tenir
- Des sous-problèmes indépendants peuvent être parallélisés pour gagner en latence
- Tu as besoin de fiabilité par redondance (patterns d'ensemble)

**N'utilise pas le multi-agent quand :**

- Un seul agent avec de bons tools peut gérer la tâche (la plupart des tâches)
- Tu splittes les agents pour des raisons organisationnelles plutôt que techniques
- Tu n'as pas l'infrastructure d'observabilité pour débugger les échecs multi-agent
- Le surcoût de coordination dépasse le bénéfice de la spécialisation

L'essai dans ce livre a raison : les petits agents battent les gros agents. Mais le corollaire est tout aussi vrai — un bon agent en bat trois inutiles. Ajoute des agents quand tu as une raison concrète. Retire-les quand tu peux.

## Une architecture de départ

Si tu construis ton premier système multi-agent, commence ici :

1. **Construis un seul agent qui gère la tâche complète.** Pousse-le jusqu'à ce qu'il casse — débordement de contexte, confusion sur les tools, dégradation de la qualité.
2. **Identifie le mode de défaillance.** Le contexte est-il trop gros ? L'agent peine-t-il sur un sous-problème spécifique ? La latence est-elle inacceptable ?
3. **Split seulement au point de défaillance.** Extrais le sous-problème problématique dans un worker agent spécialisé. Garde tout le reste dans l'agent principal.
4. **Ajoute une communication structurée.** Définis le contrat entre agents avec des schémas, pas du texte libre.
5. **Ajoute de l'évaluation à chaque frontière.** Teste la décomposition de l'orchestrator. Teste la sortie de chaque worker indépendamment. Teste la synthèse finale.
6. **Ajoute de l'observabilité dès le jour un.** Si tu ne peux pas tracer une défaillance à travers la chaîne d'agents, tu ne peux pas la corriger.

Cette approche te donne la simplicité d'un seul agent là où ça marche et la puissance du multi-agent là où c'est nécessaire. Ne conçois pas une architecture multi-agent. Grandis jusqu'à en avoir une.

## Pour aller plus loin

- [From Idea To Agent In Less Than Ten Minutes](https://engineering.prod.whoop.com/ai-studio) — Comment WHOOP gère plus de 500 agents spécialisés avec AI Studio
- [Notion's GPT-5 Rebuild](https://openai.com/index/notion/) — Le passage de Notion des chaînes de prompts à une architecture orchestrator-workers
- [How Kimi, Cursor, and Chroma Train Agentic Models with RL](https://www.philschmid.de/kimi-composer-context) — Le pattern Agent Swarm et le dispatch dynamique de sub-agents
- [Cursor's Bugbot Evolution](https://medium.com/data-science-collective/how-cursor-actually-works-c0702d5d91a9) — Transition de pipeline à agent pour de la revue de code en production
- [Anthropic: Building Effective Agents](https://docs.anthropic.com/en/docs/build-with-claude/agent-patterns) — Patterns et anti-patterns pour les architectures d'agents
