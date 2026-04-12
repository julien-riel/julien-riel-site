---
title: "42. La latency est un problème d'UX, pas juste d'infrastructure"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Un appel de modèle prend du temps."
---

Un appel de modèle prend du temps. Habituellement des secondes. Parfois plus. Pour un développeur qui lance un batch, ça va — tu le démarres et tu reviens plus tard. Pour un utilisateur qui attend une réponse dans une interface interactive, trois secondes, c'est long, et dix secondes, c'est cassé. Les caractéristiques de latency acceptables dans un contexte sont des deal-breakers dans un autre, et confondre les deux, c'est comme ça que tu livres un système techniquement fonctionnel que les utilisateurs abandonnent.

La réponse infrastructure à la latency, c'est l'optimisation : modèles plus petits, caching, streaming, appels parallèles. Ça compte, et tu devrais poursuivre ces pistes. Mais elles ont des limites, et la réponse la plus importante est souvent le design — façonner l'UX pour que l'attente semble plus courte, ou pour que l'utilisateur fasse quelque chose d'utile pendant que l'agent travaille.

Le streaming est l'intervention de design la plus percutante disponible. Afficher la réponse de l'agent à mesure qu'elle se génère, plutôt que d'attendre la sortie complète, change fondamentalement la perception de la latency. Une réponse de dix secondes qui stream progressivement paraît plus rapide qu'une réponse de trois secondes qui apparaît d'un coup, parce que l'utilisateur a quelque chose à lire presque immédiatement. L'expérience cognitive de l'attente est bien pire que celle de lire quelque chose qui arrive encore.

Les indicateurs de progression aident pour les opérations plus longues — pas des spinners génériques, mais des signaux spécifiques sur ce qui se passe. « Recherche dans tes documents » vaut mieux qu'un cercle qui tourne. « Rédaction d'une réponse à partir de trois sources » vaut mieux que « je réfléchis ». Ces signaux donnent aux utilisateurs un modèle mental de ce que l'agent fait, ce qui rend l'attente intentionnelle plutôt qu'opaque.

Il y a aussi une question produit sous la question infrastructure : est-ce que ça devrait être une expérience interactive tout court ? Certaines tâches d'agent sont trop longues pour faire attendre les utilisateurs de façon synchrone. Une tâche qui prend trente secondes a probablement sa place dans un workflow async — lance-la, fais autre chose, sois notifié quand c'est fini — plutôt que dans une interface de chat où l'utilisateur fixe un spinner. Choisir le mauvais modèle d'interaction crée un problème de latency qu'aucune optimisation ne résoudra complètement.

Assez rapide pour la tâche. Conçu pour l'attente. Les deux comptent.
