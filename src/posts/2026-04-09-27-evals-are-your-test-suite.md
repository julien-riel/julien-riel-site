---
title: "27. Les evals sont ta suite de tests"
date: 2026-04-09
tags:
  - building-agentic-systems
description: "Tout projet logiciel sérieux a des tests."
---

Tout projet logiciel sérieux a des tests. Les systèmes agentiques ont besoin de tests aussi — ils sont juste plus durs à écrire, ce qui est exactement pourquoi la plupart des équipes les sautent et se demandent ensuite pourquoi elles ne peuvent pas dire si un changement a rendu les choses meilleures ou pires.

La difficulté, c'est que les outputs d'agent ne sont pas toujours justes ou faux dans un sens binaire. Une fonction de code passe ses tests ou ne les passe pas. Un résumé généré capture les points clés ou ne les capture pas — mais « capture les points clés » n'est pas un prédicat que tu peux évaluer automatiquement. Cette ambiguïté est réelle, et elle pousse les équipes à lever les bras et à se fier au feeling. Le feeling ne passe pas à l'échelle.

Les évaluations — les evals — sont l'infrastructure de test pour les systèmes probabilistes. Elles consistent en un ensemble d'inputs avec des outputs connus comme bons ou des critères de qualité, une méthode pour noter les outputs de l'agent contre ces critères, et un processus pour exécuter l'eval chaque fois que quelque chose change. La notation n'a pas besoin d'être entièrement automatisée ; l'évaluation humaine est légitime et souvent nécessaire. Ce qui compte, c'est que le processus soit systématique, répétable, et qu'il s'exécute avant que tu livres.

Construire une bonne suite d'evals commence par collecter les échecs. Chaque fois que l'agent produit un mauvais output en production ou en test, cet input va dans le jeu d'eval. Au fil du temps, tu accumules une collection de cas difficiles — les inputs qui cassent les choses, les cas limites qui n'étaient pas anticipés, les scénarios où l'agent fait quelque chose de plausible mais faux. Cette collection a plus de valeur que n'importe quelle suite de tests synthétique, parce qu'elle représente la distribution réelle des façons dont ton système échoue.

Le deuxième composant, ce sont les golden outputs — des exemples de ce à quoi ressemble le bon pour une gamme représentative d'inputs. Ils définissent ton seuil de qualité de façon concrète. Quand tu changes un prompt ou fais évoluer un modèle, tu exécutes l'eval et tu vérifies combien de golden outputs tu correspondances encore. Les régressions sont visibles. Les améliorations sont mesurables.

Les équipes qui construisent des evals tôt livrent avec plus de confiance et s'améliorent plus vite. Les équipes qui ne construisent pas d'evals devinent toujours — si le nouveau modèle est meilleur, si le changement de prompt a aidé, si le système se dégrade en production.

Tu ne livrerais pas de code sans tests. Ne livre pas d'agents sans evals.
