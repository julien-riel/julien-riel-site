---
title: "Fine-tuning vs RAG : quand enseigner au modèle et quand lui montrer la réponse"
date: 2026-04-09
tags:
  - fine-tuning
  - rag
description: "Le fine-tuning change la façon dont le modèle pense. Le RAG change ce qu'il voit. Un cadre de décision pratique pour savoir quand utiliser l'un, l'autre — ou les deux."
---

## Un guide pratique pour programmeurs agentiques

Tu as des données spécifiques à ton domaine et tu veux que ton LLM les utilise. Deux chemins s'offrent à toi. Tu peux **fine-tuner** le modèle — l'entraîner sur tes données pour que la connaissance devienne partie intégrante de ses paramètres. Ou tu peux utiliser le **RAG** — récupérer les données pertinentes au moment de la requête et les injecter dans le prompt. La plupart des équipes voient ça comme un choix binaire. Ça n'en est pas un. Ces deux approches résolvent des problèmes différents, et les meilleurs systèmes utilisent les deux.

## Ce que chacun fait vraiment

### Fine-tuning : changer la façon dont le modèle pense

Le fine-tuning prend un modèle pré-entraîné et poursuit son training sur ton dataset. Les poids du modèle changent. Tes données deviennent partie de la « mémoire » du modèle — encodées dans ses paramètres, pas passées dans le prompt.

Ce que le fine-tuning t'apporte :

- **Des patterns comportementaux.** Le modèle apprend *comment* répondre, pas seulement *quoi* répondre. Ton, format, style de raisonnement, conventions spécifiques au domaine.
- **Des connaissances implicites.** Le modèle intériorise des patterns qu'il peut appliquer à de nouvelles entrées jamais vues — de la généralisation, pas juste de la récitation.
- **Réduction de la latence.** Pas besoin d'étape de retrieval. La connaissance est dans le modèle.
- **Des prompts plus courts.** Tu n'as pas besoin de bourrer la fenêtre de contexte avec des exemples et des instructions — le modèle sait déjà.

Ce que le fine-tuning te coûte :

- **Du compute et des données d'entraînement.** Il te faut des exemples de training soignés et de haute qualité — typiquement des centaines ou des milliers. Garbage in, garbage out.
- **De l'obsolescence.** La connaissance du modèle est figée au moment du training. Quand tes données changent, tu ré-entraînes. C'est cher et lent.
- **De l'opacité.** Tu ne peux pas citer de sources. Le modèle « sait » mais ne peut pas pointer où il l'a appris.
- **L'oubli catastrophique.** Mal fait, le fine-tuning peut dégrader les capacités générales du modèle tout en améliorant sa performance sur une tâche étroite.

### RAG : changer ce que le modèle voit

Le RAG laisse le modèle intact. Au moment de la requête, tu récupères les documents pertinents dans tes données et tu les injectes dans le prompt comme contexte. Le modèle lit et répond en s'appuyant sur cette preuve externe.

Ce que le RAG t'apporte :

- **Des connaissances dynamiques.** Mets à jour les données, les réponses se mettent à jour. Aucun ré-entraînement nécessaire.
- **Des citations.** Le modèle peut pointer le document ou le passage précis qu'il a utilisé. Auditable, vérifiable.
- **Fraîcheur des données.** Les nouveaux documents sont disponibles dès qu'ils sont indexés.
- **Aucune modification du modèle.** Fonctionne avec n'importe quel modèle — propriétaire, open-source, interchangeable.

Ce que le RAG te coûte :

- **Latence du retrieval.** L'étape de recherche ajoute du temps à chaque requête.
- **Budget de fenêtre de contexte.** Les documents récupérés se disputent les tokens avec les instructions, l'historique de conversation et le reste du contexte.
- **Les échecs de retrieval.** Si la recherche retourne des documents non pertinents, le modèle génère des réponses à partir d'un mauvais contexte — potentiellement pire qu'aucun contexte.
- **Aucun changement de comportement.** Le RAG ne change pas la façon dont le modèle raisonne ni le style qu'il utilise. Il change seulement l'information disponible.

## Le cadre de décision

La question n'est pas « fine-tuning ou RAG ? ». C'est « quel problème suis-je réellement en train de résoudre ? ».

### Utilise le RAG quand le problème c'est la connaissance

Si le modèle a besoin d'accéder à des faits spécifiques, citables, qui changent dans le temps — catalogues de produits, documents de politique, données clients, textes réglementaires, bases de connaissances — le RAG est le bon outil. Le modèle n'a pas besoin d'intérioriser cette connaissance. Il doit la lire et raisonner à partir d'elle.

WHOOP Coach est un système RAG. Le LLM ne « connaît » pas ta variabilité cardiaque grâce au training. Il récupère tes données biométriques au moment de la requête, les lit et génère un coaching personnalisé. Quand tes données changent (chaque nuit, après chaque entraînement), les réponses se mettent à jour automatiquement. Fine-tuner un modèle sur les données de santé d'une seule personne serait absurde.

**Le RAG est le bon choix quand :**
- Tes données changent fréquemment (quotidien, hebdomadaire, mensuel)
- Tu dois citer des sources précises
- Les données sont par utilisateur ou par tenant (impossible de les cuire dans un seul modèle)
- La conformité exige l'auditabilité de la provenance des réponses
- Tu veux une architecture agnostique au modèle (changer de modèle sans ré-entraîner)

### Utilise le fine-tuning quand le problème c'est le comportement

Si le modèle doit raisonner différemment, écrire dans un style précis, suivre des conventions propres au domaine ou gérer des formats spécialisés — le fine-tuning change les « instincts » du modèle. Le RAG ne peut pas faire ça. Tu peux coller un guide de style dans chaque prompt, mais le modèle se battra toujours contre ses tendances par défaut.

Cursor a fine-tuné Llama-3-70B spécifiquement pour l'application de code (le modèle Fast Apply). La tâche — prendre un diff sémantique et produire le fichier complet correct — exige un pattern comportemental spécifique qui est difficile à obtenir par prompt sur un modèle généraliste. Le modèle fine-tuné atteint 1 000 tokens/seconde parce qu'il est optimisé pour cette unique tâche. Le RAG ajouterait de la latence et n'aiderait pas — le modèle doit savoir *comment* appliquer les changements de code, pas *quels* changements appliquer.

**Le fine-tuning est le bon choix quand :**
- Tu as besoin d'un style, d'un ton ou d'un format cohérent que le prompting n'arrive pas à garantir
- Le modèle a besoin de patterns de raisonnement propres au domaine (logique de diagnostic médical, conventions d'analyse juridique, règles de transformation de code)
- Tu veux réduire la taille du prompt (et donc le coût et la latence) en intériorisant des instructions répétées
- Tu as une tâche bien définie et stable qui ne change pas souvent

### Utilise les deux quand le problème est complexe

La plupart des systèmes du monde réel ont besoin des deux. Le fine-tuning gère le comportement ; le RAG gère la connaissance.

Duolingo a fine-tuné GPT-4 pour ses patterns d'interaction spécifiques (feedback gamifié, échafaudage pédagogique, la voix de Duolingo) tout en utilisant le retrieval pour accéder au contenu des leçons, aux règles de grammaire et aux données propres à l'apprenant. Le fine-tuning garantit que le modèle agit comme un tuteur Duolingo. Le retrieval garantit qu'il enseigne le bon contenu au bon apprenant.

Cursor utilise un modèle fine-tuné (Composer) pour le comportement de coding agentique et le RAG (le moteur de contexte avec embeddings et reranking) pour la connaissance de la codebase. Le fine-tuning apprend au modèle comment écrire du code, utiliser des outils et appliquer des éditions. Le RAG lui donne accès à la codebase spécifique sur laquelle il travaille.

**Utilise les deux quand :**
- Le modèle a besoin d'un comportement spécialisé (fine-tune) ET d'un accès à des données dynamiques (RAG)
- Tu construis un assistant spécifique à un domaine qui doit à la fois sonner juste et connaître les bonnes choses
- Tu veux l'efficacité des patterns intériorisés avec la fraîcheur de la connaissance récupérée

## Comparaison pratique

| Dimension | Fine-tuning | RAG |
|-----------|-------------|-----|
| **Ce qui change** | Les poids du modèle | Le contenu du prompt |
| **Fraîcheur de la connaissance** | Figée au moment du training | Aussi fraîche que l'index |
| **Coût de mise en place** | Élevé (curation de données, training) | Moyen (chunking, embedding, indexation) |
| **Coût par requête** | Plus bas (pas d'étape de retrieval) | Plus élevé (retrieval + prompts plus gros) |
| **Latence** | Plus basse | Plus élevée (aller-retour de retrieval) |
| **Citabilité** | Non — le modèle « sait » | Oui — peut citer les documents sources |
| **Changements de données** | Nécessite un ré-entraînement | Ré-indexer et servir |
| **Portabilité du modèle** | Verrouillé à un modèle | Fonctionne avec n'importe quel modèle |
| **Idéal pour** | Comportement, style, patterns de raisonnement | Faits, données, connaissance de domaine |

## Les erreurs fréquentes

### Fine-tuner sur des faits

Des équipes fine-tunent des modèles pour qu'ils « connaissent » leur documentation produit, leur FAQ ou leurs données de politique. Ça marche — jusqu'à ce que la documentation change. Alors tu ré-entraînes, ce qui prend du temps et de l'argent, et les vieilles réponses traînent en production jusqu'au déploiement du nouveau modèle. Le RAG gère les connaissances factuelles dynamiques avec bien plus d'élégance.

Fine-tune pour le comportement. RAG pour les faits. C'est l'heuristique la plus simple et elle est juste la plupart du temps.

### Du RAG sans évaluation

Des équipes montent un pipeline RAG, le testent sur une poignée de questions et l'expédient. Sans évaluation systématique de la qualité du retrieval (est-ce que tu récupères les bons chunks ?) et de la qualité de la génération (est-ce que le modèle les utilise correctement ?), tu voles à l'aveugle. WHOOP a construit un framework d'évaluation qui teste séparément le retrieval, la génération et la qualité bout en bout. Ton système RAG a besoin de la même chose.

### Le fine-tuning comme substitut à un bon prompting

Avant de fine-tuner, assure-toi d'avoir épuisé ce que le prompting peut faire. Le fine-tuning coûte cher et est irréversible (dans le sens où tu maintiens maintenant un modèle custom). Beaucoup de cas d'usage « fine-tuning » peuvent être résolus avec de meilleurs system prompts, des exemples few-shot ou des formats de sortie structurés.

La règle d'or : si tu peux décrire le comportement souhaité en langage naturel et que le modèle peut le suivre avec un bon prompting, tu n'as pas besoin de fine-tuning. Si le comportement exige des centaines d'exemples pour être démontré parce qu'il est trop subtil ou complexe à instruire, c'est là que le fine-tuning vaut son prix.

### Ignorer le terrain d'entente : le prompt caching

Beaucoup de fournisseurs d'API offrent maintenant du prompt caching — réutiliser le contexte déjà traité des requêtes précédentes quand le préfixe du prompt est identique. Ça te donne une partie des bénéfices de latence du fine-tuning (le modèle a déjà « lu » ton contexte) avec la flexibilité du RAG (le contexte peut être mis à jour). Si ton system prompt et ton contexte récupéré sont relativement stables d'une requête à l'autre, le prompt caching peut réduire significativement la latence et le coût sans aucun training.

## Un arbre de décision

```
Q: Does the model need to access specific, changing data?
├── Yes → You need RAG (at minimum)
│   Q: Does it also need specialized behavior/style?
│   ├── Yes → RAG + Fine-tuning
│   └── No → RAG + Good prompting
└── No → 
    Q: Does the model need to reason or behave differently than its defaults?
    ├── Yes → Fine-tuning (or very strong prompting first)
    └── No → You might not need either — just prompt well
```

## À retenir

Le fine-tuning et le RAG sont complémentaires, pas concurrents. Le fine-tuning change les instincts du modèle — sa façon de raisonner, d'écrire et de se comporter. Le RAG change la connaissance du modèle — ce qu'il peut référencer en répondant. La plupart des systèmes agentiques du monde réel ont besoin des deux : un modèle qui pense de la bonne façon à propos de l'information spécifique qu'il récupère au moment de la requête.

Commence par le RAG. C'est moins cher, plus rapide à itérer et ça ne te verrouille pas dans un modèle précis. Ajoute le fine-tuning quand tu as identifié un écart comportemental que le prompting n'arrive pas à combler. Et toujours — toujours — construis ton framework d'évaluation avant d'investir dans l'un ou l'autre.

## Pour aller plus loin

- [Guide du fine-tuning OpenAI](https://platform.openai.com/docs/guides/fine-tuning) — Documentation officielle sur quand et comment fine-tuner
- [WHOOP — Delivering LLM-powered Health Solutions](https://openai.com/index/whoop/) — Le RAG en production pour un coaching santé personnalisé
- [Cursor Composer 2](https://anthemcreation.com/en/artificial-intelligence/cursor-composer-2-proprietary-coding-ai-model/) — Fine-tuning + RL pour un comportement de coding spécialisé
- [L'évolution IA de Duolingo](https://openai.com/index/duolingo/) — Fine-tuning de GPT-4 pour des patterns d'interaction pédagogique combiné à du retrieval de contenu
