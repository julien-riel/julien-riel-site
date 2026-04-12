---
title: "72. La première version devrait être embarrassamment simple"
date: 2026-04-09
tags:
  - mindset
description: "Chaque principe durable en logiciel a une version de ceci en son cœur."
---

Chaque principe durable en logiciel a une version de ceci en son cœur. Commencer simple. Livrer tôt. Apprendre de l'usage réel. Le mode d'échec qu'il prévient est toujours le même : le système conçu en l'absence de preuves, bâti pour gérer des exigences qui se révèlent ne pas être réelles, complexe de façons qui coûtent du temps de maintenance sans ajouter de valeur pour l'utilisateur.

En programmation agentique, ce principe est plus important et plus souvent violé que presque partout ailleurs. L'outillage rend la complexité peu coûteuse à ajouter. Les démos de systèmes multi-agents sophistiqués font paraître inadéquates les solutions simples à un seul agent. Le domaine bouge vite et il y a une pression pour utiliser les dernières techniques, pour bâtir l'architecture qui scalera, pour résoudre des problèmes que tu n'as pas encore. Résultat : des premières versions qui sont plusieurs versions en avance sur ce que les preuves justifient.

La première version embarrassamment simple, c'est un seul agent avec un prompt minimal, un petit nombre d'outils, pas d'orchestration complexe, et une étape de revue humaine pour tout ce qui est conséquent. Elle ne gère probablement que le cas le plus courant. Elle échoue probablement sur les entrées hors de ce cas, de façons évidentes et récupérables. Elle n'impressionne probablement personne qui la voit. Elle tourne aussi, produit de vraies sorties, et génère les preuves sur lesquelles toute décision de conception ultérieure devrait s'appuyer.

Les preuves que tu obtiens de la version simple sont irremplaçables. De vrais utilisateurs interagissent avec elle de façons que tu n'avais pas anticipées. Le cas courant se révèle légèrement différent de ce que tu supposais. Les modes d'échec sont différents de ceux que tu avais anticipés. Ce que tu pensais être le problème difficile ne l'est pas, et quelque chose auquel tu n'avais pas du tout pensé l'est. Tu ne peux rien savoir de tout cela sans faire tourner le système, et la version simple tourne plus tôt, moins cher, et avec moins à défaire quand tu dois changer de direction.

Il y a aussi quelque chose de clarifiant dans la contrainte de simplicité. Quand tu t'autorises à bâtir la version complexe tout de suite, tu reportes la question difficile de ce que le système doit réellement faire. La simplicité force la réponse. Un agent veut dire un travail. Un prompt veut dire une portée claire. Une étape de revue humaine veut dire un jugement explicite sur ce qui peut et ne peut pas être confié à la machine. Les contraintes révèlent la conception.

La version embarrassamment simple n'est pas la version finale. C'est celle qui mérite le droit à la suivante.

Commence là. La sophistication viendra quand elle sera méritée.

---

---

## Partie 6 — Le développeur comme utilisateur

*Pour les développeurs qui travaillent aux côtés des assistants de code IA — Claude Code, GitHub Copilot, Cursor, et leurs successeurs.*

---

### Travailler avec l'assistant
