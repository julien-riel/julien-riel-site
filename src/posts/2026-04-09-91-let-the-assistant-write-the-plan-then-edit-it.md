---
title: "91. Laissez l'assistant écrire le plan, puis éditez-le"
date: 2026-04-09
tags:
  - developer-as-user
description: "Quand tu commences un morceau de travail substantiel, demande à l'assistant d'écrire un plan d'implémentation avant d'écrire la moindre ligne de code."
---

Quand tu commences un morceau de travail substantiel, demande à l'assistant d'écrire un plan d'implémentation avant d'écrire la moindre ligne de code. Décris ce que tu essaies de construire, fournis le contexte pertinent, et demande: quelles sont les étapes, quelles sont les dépendances entre elles, quelles sont les décisions qui doivent être prises avant que l'implémentation commence?

Le plan que l'assistant produit sera imparfait. Il va manquer des contraintes spécifiques à ton codebase, faire des suppositions sur tes préférences qui ne tiendront peut-être pas, et proposer un ordonnancement qui ne correspondra peut-être pas à tes priorités. Ces imperfections sont exactement ce qui rend l'exercice utile. Éditer un mauvais plan est beaucoup plus rapide qu'en écrire un bon à partir de rien, et les imperfections révèlent les décisions que tu n'avais pas encore prises consciemment.

La conversation de planification fait aussi émerger les ambiguïtés de ta spec avant qu'elles ne deviennent des bugs dans ton code. Si tu décris une feature et que le plan de l'assistant révèle trois interprétations différentes de ce que « user settings » veut dire dans ton système, tu veux le savoir maintenant, pas après avoir implémenté la mauvaise.

Une fois que tu as édité le plan pour en faire quelque chose qui te donne confiance, il devient le document qui guide les prompts d'implémentation. Chaque étape du plan devient un prompt. Les dépendances entre étapes te disent quel contexte transporter. Les décisions que tu as prises pendant l'édition deviennent des contraintes explicites dans les prompts qui en ont besoin.

Le plan est bon marché à produire et cher à sauter. Laisse l'assistant écrire la première version.
