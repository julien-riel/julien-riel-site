---
title: "84. Montre à l'assistant à quoi ressemble du bon code dans ton codebase"
date: 2026-04-09
tags:
  - developer-as-user
description: "Les instructions abstraites produisent du code générique."
---

Les instructions abstraites produisent du code générique. « Follow our error handling conventions » produit quelque chose que l'assistant a inventé à partir de patterns communs. Coller trois exemples de comment ton codebase gère réellement les erreurs produit du code qui s'intègre. Le modèle apprend fondamentalement à partir d'exemples — donne-lui les bons.

C'est du few-shot prompting appliqué à la génération de code, et le principe est le même que dans n'importe quel context de prompting : les exemples surpassent les instructions. Tu peux passer un paragraphe à décrire tes conventions de nommage, ou tu peux coller un module bien nommé et dire « follow this style ». La seconde approche est plus rapide à écrire, plus difficile à mal interpréter, et produit un meilleur output.

Les exemples que tu choisis comptent. Un seul exemple bien choisi du codebase réel vaut plus que trois exemples synthétiques que tu as écrits pour le prompt. Le vrai code porte de l'information implicite — le niveau d'abstraction que tu favorises, la façon dont tu structures les chemins d'erreur, à quel point tu commentes, comment tu nommes les choses aux frontières d'un module. Un exemple synthétique ne peut porter que ce que tu y mets explicitement.

Il y a aussi un bénéfice de calibration. Quand tu colles un exemple et demandes à l'assistant de suivre son style, tu établis un point de référence concret pour la conversation. Si l'output dérive du style, tu peux pointer vers l'exemple et dire « more like this ». Sans l'exemple, « more like this » n'a pas de référent.

Construis une bibliothèque personnelle de bons exemples de ton codebase — les fonctions, les modules, les fichiers de test qui représentent le standard que tu vises. Ils valent plus dans un prompt que n'importe quelle description que tu pourrais écrire.
