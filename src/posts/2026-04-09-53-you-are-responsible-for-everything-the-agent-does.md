---
title: "53. Vous êtes responsable de tout ce que fait l'agent"
date: 2026-04-09
tags:
  - agents-in-the-real-world
description: "Quand un agent fait une erreur — donne une mauvaise information, prend une action nuisible, produit un output qui porte atteinte aux intérêts d'un utilisateur — la question de la responsabilité a une réponse claire."
---

Quand un agent fait une erreur — donne une mauvaise information, prend une action nuisible, produit un output qui porte atteinte aux intérêts d'un utilisateur — la question de la responsabilité a une réponse claire. C'est toi. Pas le fournisseur du modèle, pas le framework que tu as utilisé, pas l'agent lui-même. Tu as construit le système, tu l'as déployé, tu l'as mis devant des utilisateurs. Les outputs sont les tiens.

Ce n'est pas un argument juridique, même si ça pourrait le devenir. C'est un argument de design. Les développeurs qui intériorisent la responsabilité du comportement de l'agent prennent des décisions différentes de ceux qui se sentent isolés de celle-ci. Ils bâtissent plus de validation. Ils conçoivent des défauts plus conservateurs. Ils investissent en observabilité pour pouvoir voir ce que le système fait. Ils réfléchissent soigneusement à ce qui arrive quand ça tourne mal, parce qu'ils savent que quand ça tourne mal, c'est leur problème.

La tentation de diluer la responsabilité est forte, surtout quand les agents sont vendus comme des systèmes autonomes qui prennent leurs propres décisions. L'autonomie est réelle — les agents prennent bien des décisions que tu n'as pas explicitement programmées. Mais l'autonomie dans l'exécution ne transfère pas la responsabilité pour les résultats. Tu as choisi le modèle, écrit les prompts, défini les tools, fixé le scope et décidé quand le système était prêt à être déployé. Chacune de ces décisions est la tienne.

Ça devient le plus concret dans les domaines à enjeux élevés. Un agent qui donne de l'information médicale à quelqu'un qui agit dessus. Un agent qui prend des décisions financières pour le compte d'un utilisateur. Un agent qui communique avec des clients d'une façon qui crée des obligations juridiques. Dans chaque cas, la question n'est pas de savoir si l'agent avait de bonnes intentions — c'est de savoir si les outputs étaient appropriés et si le système a été conçu avec assez de soin pour les enjeux en cause.

La posture responsable, c'est de traiter les outputs de l'agent comme tes outputs. Lis-les avec le même œil critique que tu appliquerais à n'importe quoi que tu signerais. Intègre de la revue dans le workflow pour tout ce qui a des conséquences. Sois honnête avec les utilisateurs sur ce qu'est le système et sur ce à quoi on peut ou non lui faire confiance.

L'agent agit. Tu es imputable. Conçois en conséquence.

---

## Part 5 — Mindset
