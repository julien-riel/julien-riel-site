---
title: "25. Apprends à reconnaître les patterns d'hallucination dans ton domaine"
date: 2026-04-09
tags:
  - prompting-as-engineering
description: "L'hallucination — le modèle qui génère du contenu qui sonne plausible mais n'est pas ancré dans les faits — n'est pas aléatoire."
---

L'hallucination — le modèle qui génère du contenu qui sonne plausible mais n'est pas ancré dans les faits — n'est pas aléatoire. Elle a des patterns. Les modèles hallucinent de façons prévisibles, dans des situations prévisibles, et les développeurs qui travaillent le plus efficacement avec les agents ont appris à reconnaître les patterns spécifiques à leur domaine.

Les patterns généraux sont bien documentés. Les modèles fabriquent des citations avec la bonne structure mais les mauvais détails. Ils confondent des entités qui se ressemblent sur une dimension — même nom, même domaine, même période. Ils comblent les trous de leurs connaissances avec des extrapolations qui suivent la logique du domaine mais ne sont pas réellement vraies. Ils sont plus susceptibles d'halluciner aux marges de leurs données d'entraînement — sujets de niche, événements récents, domaines hautement spécialisés où le corpus d'entraînement était mince.

Mais les patterns généraux sont moins utiles que ceux spécifiques au domaine. Un développeur qui travaille avec un agent juridique apprend que le modèle fabrique de façon fiable des citations de jurisprudence — il a la bonne cour et le bon domaine général du droit, il invente le nom de la cause et la date. Un développeur qui travaille avec un agent médical apprend que le modèle a tendance à confondre des noms de médicaments similaires et à mal énoncer les dosages d'une façon qui suit les conventions de nomenclature pharmaceutique. Un développeur qui travaille avec un agent de génération de code apprend que le modèle utilise avec confiance des fonctions de bibliothèque qui n'existent pas mais qui devraient probablement exister.

Ces patterns sont apprenables, mais seulement par l'exposition. Tu dois exécuter l'agent sur suffisamment de tâches réelles, attraper suffisamment d'échecs spécifiques, et construire une image de là où ce modèle, sur cette tâche, dans ce domaine, a tendance à se tromper. Cette connaissance ne se transfère pas proprement d'un modèle à l'autre ou d'un domaine à l'autre — elle s'acquiert localement, par système.

Le bénéfice, c'est un scepticisme ciblé qui est beaucoup plus efficace que la méfiance globale. Au lieu de tout vérifier, tu vérifies les choses qui sont susceptibles d'être fausses. Tu construis des vérifications pour les modes d'échec spécifiques que tu as appris à attendre. Tu sais quelles parties de l'output lire attentivement et quelles parties tu peux croire.

Le scepticisme général te protège des hallucinations connues. La connaissance du domaine te dit où regarder.

---

## Part 3 — Building Agentic Systems
