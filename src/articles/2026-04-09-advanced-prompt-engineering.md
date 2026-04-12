---
title: "Ingénierie de prompts avancée : au-delà des bases"
date: 2026-04-09
tags:
  - prompting
  - architecture
description: "Les patterns qui séparent les prompts qui fonctionnent en démo de ceux qui tiennent en production — gestion du contexte, outputs structurés, ingénierie few-shot et contrôle de version."
---

## Un guide pratique pour programmeurs agentiques

Tu sais déjà écrire un prompt. Tu connais les system prompts, les exemples few-shot et l'astuce qui consiste à demander au modèle de « penser étape par étape ». Ça, c'est prompt engineering 101. Ce guide traite de ce qui vient ensuite — les patterns qui séparent les prompts qui fonctionnent en démo de ceux qui tiennent en production.

## Le changement fondamental : les prompts sont du logiciel

La première chose à intégrer, c'est qu'un prompt dans un système de production n'est pas un message envoyé à un chatbot. C'est un artefact logiciel. Il a des inputs, des outputs, des dépendances et des modes de défaillance. Il devrait être versionné, testé, revu et déployé avec la même rigueur que du code applicatif.

WHOOP suit plus de 2 500 itérations de prompt à travers 41 agents en production. Cursor a rendu open source une librairie complète (Priompt) pour compiler les prompts comme des composants JSX avec des scores de priorité. Ces équipes traitent les prompts comme des artefacts d'ingénierie parce qu'elles ont appris à la dure qu'un changement désinvolte de prompt peut silencieusement dégrader un système en production.

Si tes prompts vivent dans une variable string au sein du code de ton application, sans historique de version, sans suite d'évals et sans processus de déploiement — tu n'as pas de pratique de prompt engineering. Tu as une dette.

## Des patterns structurels qui scalent

### Sépare les préoccupations dans le prompt

Un prompt bien structuré a des sections distinctes, chacune avec un but clair :

- **Rôle et contraintes :** qui est le modèle, ce qu'il peut et ne peut pas faire, quel ton utiliser.
- **Contexte :** les données dont le modèle a besoin pour répondre — documents récupérés, profil utilisateur, historique de conversation.
- **Spécification de la tâche :** ce qu'il faut faire exactement avec le contexte. Sois précis sur le format, la longueur et la structure.
- **Format d'output :** si tu as besoin de JSON, définis le schema. Si tu as besoin d'une structure spécifique, montre-la.

Mélanger ces préoccupations crée des prompts fragiles. Quand les instructions de rôle débordent sur le contexte, ou que les spécifications de tâche sont éparpillées dans le prompt, de petits changements ont des effets en cascade imprévisibles. Garde les sections distinctes, même si ça veut dire être plus verbeux.

### Utilise des formats d'output structurés

Chaque fois que ton système en aval doit parser la réponse du modèle, définis explicitement le format d'output. Pour une consommation programmatique, utilise du JSON avec un schema défini. Pour du contenu structuré, utilise des tags XML ou du markdown avec des titres cohérents.

```
Respond in the following JSON format:
{
  "answer": "your answer here",
  "confidence": "high | medium | low",
  "sources": ["list of source identifiers used"],
  "follow_up_needed": true | false
}
```

Les outputs structurés réduisent les erreurs de parsing, rendent l'évaluation plus facile (tu peux vérifier des champs individuels) et contiennent la tendance du modèle à divaguer. Beaucoup d'APIs supportent maintenant les outputs structurés nativement — sers-t'en.

### Espace négatif : dis au modèle ce qu'il ne doit PAS faire

Les modèles ont des comportements par défaut très marqués. Sans contraintes explicites, ils seront verbeux, hésitants et empressés de plaire. Les améliorations de prompt les plus percutantes sont souvent soustractives — dire au modèle ce qu'il faut éviter.

Contraintes négatives efficaces :
- « N'invente pas d'information. Si tu ne sais pas, dis-le. »
- « N'inclus pas d'avertissements ou de mises en garde sauf si spécifiquement pertinents. »
- « Ne répète pas la question avant de répondre. »
- « Si le contexte récupéré ne contient pas la réponse, dis « Je n'ai pas assez d'information » — ne devine pas. »

Ces contraintes sont particulièrement importantes dans les systèmes agentiques où l'output alimente une autre étape. Un résultat intermédiaire halluciné se propage et s'amplifie dans le pipeline.

## La gestion du contexte : le vrai problème

La compétence la plus percutante en prompt engineering avancé, ce n'est pas le travail stylistique — c'est la gestion du contexte. Le modèle ne peut utiliser que ce qui se trouve dans la context window. Faire entrer la bonne information dans cette fenêtre, dans le bon ordre, avec la bonne priorité, c'est là que les prompts de production réussissent ou échouent.

### Priorise le contexte sans pitié

Les context windows sont des budgets. Chaque token dépensé sur du contexte peu utile est un token non dépensé sur quelque chose d'utile. La librairie Priompt de Cursor rend ça explicite : chaque élément de prompt a un score de priorité, et quand le budget de tokens est dépassé, les éléments de plus basse priorité sont supprimés via une recherche binaire.

Tu n'as pas besoin de Priompt pour appliquer ce principe. Classe tes sources de contexte par importance. Place l'information la plus critique en premier (les modèles portent attention au début du contexte de façon plus fiable). Tronque par le bas, pas au hasard. Et mesure : est-ce qu'ajouter ce contexte améliore réellement tes scores d'évals, ou c'est juste du bruit ?

### Assemblage dynamique du contexte

Les prompts de production sont rarement statiques. Ils sont assemblés à l'exécution à partir de plusieurs sources : instructions système (fixes), documents récupérés (variables), profil utilisateur (variable), historique de conversation (qui grossit), résultats de tool use (dynamiques).

Conçois ton prompt comme un template avec des emplacements :

```
[SYSTEM INSTRUCTIONS - fixed, ~500 tokens]
[USER PROFILE - fetched at runtime, ~200 tokens]
[RETRIEVED CONTEXT - from RAG, top-k chunks, ~2000 tokens]
[CONVERSATION HISTORY - last N turns, ~1000 tokens]
[CURRENT QUERY - user's message]
[OUTPUT INSTRUCTIONS - format, constraints]
```

Ce pattern rend explicite d'où vient chaque morceau de contexte, quel budget il reçoit et ce qui saute en premier quand la fenêtre est serrée. Les inline tools de WHOOP vont plus loin — la récupération de données est intégrée directement dans le template du prompt via du balisage, exécutée en parallèle avant que la génération ne commence.

### Gère l'historique de conversation délibérément

Dans les conversations multi-tours, l'historique grossit à chaque échange. Les approches naïves empilent tout, finissant par pousser du contexte critique hors de la fenêtre. Des approches plus intelligentes :

- **Fenêtre glissante :** garde seulement les N derniers tours. Simple, mais perd le contexte initial.
- **Résumé :** résume périodiquement les tours plus anciens en une représentation compacte. Composer 2 de Cursor fait ça pendant l'entraînement RL — le modèle apprend quand et comment s'auto-résumer.
- **Rétention sélective :** garde les tours qui contiennent des décisions ou du contexte importants, lâche ceux qui sont purement transactionnels.
- **Extraction mémoire :** extrais les faits clés de la conversation dans un store mémoire structuré (comme WHOOP le fait avec ses memory nuggets), et injecte-les comme contexte plutôt que garder l'historique brut.

## Ingénierie few-shot

Les exemples few-shot sont souvent plus efficaces que des instructions détaillées. Le modèle apprend le format, le ton, les patterns de raisonnement et la gestion des cas limites à partir d'exemples, d'une façon que les instructions seules ne peuvent pas transmettre.

### Qualité plutôt que quantité

Deux exemples parfaits battent dix exemples médiocres. Chaque exemple devrait démontrer exactement le comportement que tu veux, y compris comment gérer les cas difficiles. Inclus au moins un exemple qui montre le modèle en train de *ne pas* faire quelque chose — refuser une mauvaise requête, dire « Je ne sais pas », ou gérer un cas limite avec grâce.

### Couvre la distribution

Tes exemples devraient représenter l'éventail des inputs que le modèle rencontrera. Si 80 % des requêtes sont de simples recherches et 20 % du raisonnement complexe, tes exemples devraient à peu près correspondre à cette distribution. Ne montre pas seulement les cas difficiles — le modèle doit aussi savoir à quoi ressemblent les cas simples.

### Utilise des exemples négatifs

Montre au modèle à quoi ressemble une mauvaise réponse et pourquoi elle est mauvaise :

```
Example (BAD response):
User: What was my heart rate during sleep?
Response: Your heart rate was probably around 60 BPM based on typical values.
Why this is bad: Uses generic data instead of the user's actual metrics.

Example (GOOD response):
User: What was my heart rate during sleep?
Response: Your average heart rate during sleep last night was 54 BPM, which is 3 BPM lower than your 30-day average.
Why this is good: Uses the user's actual data with contextual comparison.
```

Ce pattern de contraste est l'une des techniques de prompt engineering les plus efficaces. Le modèle apprend non seulement quoi faire, mais aussi ce qu'il faut éviter et pourquoi.

## Chain-of-thought et contrôle du raisonnement

### Quand utiliser chain-of-thought

Le chain-of-thought (CoT) — demander au modèle de montrer son raisonnement avant de donner une réponse — améliore la précision sur les tâches qui demandent du raisonnement en plusieurs étapes : maths, logique, planification, analyse complexe. Il n'aide pas (et peut nuire) sur les tâches de recherche simples, la classification ou l'extraction.

WHOOP l'a appris en évaluant GPT-5 : le mode de raisonnement du modèle sous-performait GPT-4.1 sur les requêtes de chat à faible latence. La surcharge du raisonnement ajoutait de la latence sans améliorer la qualité pour des questions directes. Utilise CoT délibérément, pas par défaut.

### Raisonnement structuré

Plutôt que « pense étape par étape » (qui est vague), donne au modèle une structure de raisonnement spécifique :

```
Before answering, analyze the question using these steps:
1. Identify what data is needed to answer this question
2. Check whether the provided context contains that data
3. If the data is present, formulate an answer citing specific values
4. If the data is missing, state what's missing and don't guess
```

Ça produit des chaînes de raisonnement plus cohérentes et debuggables. Tu peux évaluer chaque étape indépendamment, attrapant des défaillances de raisonnement même quand la réponse finale se trouve être correcte.

### Cache le raisonnement quand il le faut

Dans les applications face utilisateur, tu veux souvent que le modèle raisonne en interne mais montre seulement la conclusion. Utilise des tags XML ou des délimiteurs pour séparer le raisonnement de l'output :

```
<reasoning>
[Internal analysis — not shown to user]
</reasoning>
<response>
[Clean answer shown to user]
</response>
```

Parse le raisonnement dans ta couche applicative. Garde-le dans tes logs pour le debug. Ça te donne les bénéfices de précision du CoT sans le coût UX des réponses verbeuses.

## Pratiques de prompts en production

### Versionne tout

Chaque changement de prompt devrait être suivi — qui l'a changé, quand, pourquoi, et quels étaient les résultats d'évals avant et après. Utilise un système de gestion de prompts (Priompt, PromptLayer, Humanloop, ou même un repo Git avec une convention de nommage). L'objectif : tu devrais pouvoir revenir à n'importe quelle version précédente en quelques minutes.

### Teste avant de livrer

Fais tourner ta suite d'évals sur chaque changement de prompt. Compare les métriques à la baseline. Cherche les régressions sur l'ensemble du jeu de tests, pas juste des vérifications ponctuelles sur quelques exemples. La régression de l'agent Memory chez WHOOP — où un « meilleur » prompt était mesurablement pire — a été attrapée par des évals automatisées, pas par une revue manuelle.

### Traite la dette de prompt comme de la dette technique

Les prompts accumulent des scories : instructions ajoutées pour des cas limites qui ont été corrigés ailleurs par la suite, contraintes redondantes, exemples qui ne correspondent plus au comportement du modèle. Audite périodiquement tes prompts. Enlève les instructions que les évals montrent sans effet. Simplifie quand c'est possible. Un prompt plus court qui performe pareil est un meilleur prompt — il coûte moins cher, il est plus rapide et il a moins de chances d'embrouiller le modèle.

## À retenir

Le prompt engineering avancé, c'est de l'ingénierie des systèmes appliquée au langage naturel. Les compétences qui comptent ne sont pas celles de l'écriture créative — ce sont la gestion du contexte, la décomposition structurée, l'évaluation systématique et un contrôle de version discipliné. Le meilleur prompt n'est pas le plus astucieux. C'est celui qui fonctionne de façon fiable à grande échelle, qui échoue de façon prévisible et qui s'améliore de façon mesurable quand tu le modifies.

## Pour aller plus loin

- [Priompt](https://github.com/anysphere/priompt) — la librairie open source de Cursor pour compiler des prompts par priorité
- [Guide de prompt engineering d'Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — guide complet pour prompter Claude efficacement
- [From Idea To Agent In Less Than Ten Minutes](https://engineering.prod.whoop.com/ai-studio) — AI Studio de WHOOP et le pattern des inline tools
- [Guide de prompt engineering d'OpenAI](https://platform.openai.com/docs/guides/prompt-engineering) — les bonnes pratiques officielles d'OpenAI
