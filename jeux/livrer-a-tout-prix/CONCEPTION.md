# Livrer à tout prix — document de conception

> Titre de travail. Slug : `livrer-a-tout-prix`. Édition anglaise prévue plus tard : *Ship at Any Price*.
> Deuxième jeu sérieux dans l'univers du Groupe Polyvalent Universel inc. (GPU), après « Serveur à tout prix ».

## 1. Intention

Le joueur dirige la transition d'une équipe de 21 développeurs vers le développement agentique,
pendant un an. Il ne code pas : il règle des pratiques, des outils, des politiques et des budgets,
puis il regarde l'open space réagir.

Le jeu doit faire comprendre cinq idées. Chaque levier, événement et métrique sert au moins l'une d'elles.

1. **Le goulot se déplace.** L'IA multiplie l'écriture du code ; la revue, la validation et la clarté
   des besoins ne suivent pas. Les PR s'empilent sur les bureaux des réviseurs.
2. **Les garde-fous valent plus avec les agents, pas moins.** Tests, CI, lint, typage et conventions
   sont les rails de l'agent.
3. **La vitesse sans compréhension est une dette.** La dette cognitive est invisible pendant des mois,
   puis se paie en incident à 3 h du matin.
4. **Les données sont le vrai risque.** Un agent ne doute pas d'une donnée ; il l'amplifie.
5. **L'autonomie se règle selon la criticité.** Le firmware IoT et le site vitrine ne méritent pas
   le même niveau d'automatisation.

Public : gestionnaires TI, développeurs, directions d'affaires. Durée cible d'une campagne : 30 à 45 minutes.
Ton : celui du premier jeu, de l'humour de bureau québécois, bienveillant, jamais cynique envers les personnes.

## 2. Structure de la partie

### 2.1 Campagne de quatre trimestres

Une partie dure 52 semaines, en 4 trimestres de 13 semaines. Le joueur avance **une journée ouvrable
à la fois** (du lundi au vendredi). Chaque trimestre a un thème dominant, qui oriente le tirage des événements.

On part de peu : **sept licences GitHub Copilot de base** offertes à un groupe pilote (une personne volontaire
par îlot), en autocomplétion seulement. Les contrôles arrivent ensuite **un à un**, selon une feuille de route
(`js/08-donnees-calendrier.js`) : une nouveauté par semaine au plus, annoncée par une **note de service**
signée (Mme Lavigueur, Régis, Me Bérubé, Ginette ou Kevin) qui dit ce qui change et quelles métriques
surveiller. Le joueur peut ainsi isoler l'effet de chaque levier avant que le suivant n'arrive.

| Trimestre | Thème | Ce qui se débloque (semaine) | Événements dominants |
|---|---|---|---|
| T1 | **Copilot de base** | Licences (S1), autonomie 0–1 : autocomplétion et clavardage (S3), réviseurs (S5), tests (S7), mode agent : autonomie 2 (S8), CI (S9), lint (S11), approbation (S12) | Groupe pilote, premières PR assistées, revue qui commence à s'empiler |
| T2 | **Les agents et la facture** | Agents qui ouvrent leurs PR (autonomie 3) et registre des skills (S14), taille des PR (S15), modèle (S17), sessions parallèles 1–3 (S18), plafond de jetons (S20), formation (S22), discipline du « non » (S23), fusion automatique et 6 sessions (S24), specs (S25) | Facture de jetons, boucle d'agent, bogues qui remontent, guerre des skills, gonflement fonctionnel |
| T3 | **Les fuites** | Accès à la prod (S27), bac à sable (S28), sorties externes (S30), politique du registre (S31), modèle par tâche (S32), classification (S33), catalogue (S34), passerelle MCP (S36), développement hors TI (S37) | Injection de prompt, slopsquatting, MCP empoisonné, application de Kevin, Loi 25 |
| T4 | **La maturité** | Observabilité (S40), objectif de la direction (S42), revues d'explication (S44), binôme (S46), relève (S48) | Départ de Ginette, panne « personne ne comprend », vulnérabilité MCP publiée, bilan |

Un levier pas encore débloqué garde sa valeur par défaut : les agents ont accès à la production, le registre
est en libre-service et personne ne classe les données tant que le joueur ne peut pas y toucher.

### 2.2 Le rythme d'un trimestre

1. **Planification** (pause) : le joueur règle ses leviers. Ginette commente en une ligne les choix risqués.
2. **Exécution** : les journées défilent (vitesse ×1, ×2, ×4 ; une journée dure 4 s à ×1) ou s'avancent une à une.
   Du lundi au jeudi, la journée vit : les développeurs pensent tout haut (réflexions, §8.3), une carte d'événement
   peut arriver. Le **vendredi**, la semaine est calculée d'un coup : c'est le rapport hebdomadaire, et le
   tableau de bord change. Une carte tirée le vendredi arrive un jour de la semaine suivante (du mardi au vendredi).
   Le joueur peut mettre en pause et changer ses leviers à tout moment ; un changement compte pour le calcul
   du vendredi suivant, et certains ont un délai d'effet (formation : 4 semaines, MCP maison : 6 semaines de développement).
3. **Revue trimestrielle** : un comité (Régis, Me Bérubé, Mme Lavigueur, plus un directeur
   de ligne d'affaires selon l'actualité) pose 3 questions sur les chiffres. On réutilise
   l'esprit du comité du premier jeu : on choisit une réponse, on gagne ou perd de la confiance.

### 2.3 Fin de partie

Après la semaine 52 : un bilan annuel devant Mme Lavigueur, une note sur 5 étoiles et un **profil de fin**
déterminé par la combinaison des résultats :

| Profil | Condition dominante |
|---|---|
| **La transition réussie** | Débit en hausse, incidents stables, compréhension préservée, risque maîtrisé |
| **L'usine à PR** | Débit très haut, revue saturée, incidents et dette cognitive élevés |
| **Le musée** | Prudence extrême : peu de risque, mais l'informatique non-TI a pris le relais et la satisfaction est basse |
| **Le Far West** | Surface d'attaque énorme, au moins une fuite de données majeure |
| **La facture salée** | Résultats corrects, mais budget de jetons dépassé de plus de 50 % |
| **Le château de cartes** | Beaucoup de code, compréhension effondrée, la relève n'existe plus |

Une **confiance de la direction** (0–100) sert de garde-fou : sous 15, la partie se termine tôt
(« Mme Lavigueur vous remercie pour votre contribution »).

### 2.4 Modes annexes

- **Tutoriel** (2 minutes, avec Ginette) : une journée guidée, le levier des licences, les réflexions, la note de service.
- **Mode animateur** : choix de la graine aléatoire, sauter au trimestre voulu, tout débloquer,
  afficher les métriques cachées. Utile en formation de groupe et pour les tests.
- **Sauvegarde** automatique à chaque journée (`localStorage`, protégée par try/catch).

## 3. Les personnages

### 3.1 Les 21 développeurs

Trois par ligne d'affaires. Chacun a des **traits** qui modifient le modèle (section 4.2) et des **réactions**
aux politiques (son moral monte ou baisse selon les leviers). Les fiches complètes (réplique d'accueil,
bulles de pensée, réactions) seront écrites dans `donnees/personnages`.

| Îlot | Développeur | Niveau | Traits | En une ligne |
|---|---|---|---|---|
| Appro | **Raymond** | Senior | `legacy`, `sceptique` | 31 ans sur l'AS/400. « L'IA ? J'ai vu passer le 4GL aussi. » |
| Appro | **Lucie** | Inter | `donnees`, `rigoureuse` | Connaît par cœur les trois listes de fournisseurs, et laquelle est la bonne. |
| Appro | **Yannick** | Inter | `curieux`, `integrateur` | Branche tout sur tout. Premier à installer un MCP trouvé sur un forum. |
| Opérations | **Benoît** | Senior | `garde`, `stoique` | De garde la nuit depuis 2009. Son téléphone vibre, il soupire. |
| Opérations | **Marie-Ève** | Senior | `sre`, `mesure` | Ne croit que ce qui a un tableau de bord. Débloque l'observabilité. |
| Opérations | **Tommy** | Junior | `apprenant` | Apprend vite, mais seulement ce qu'on lui laisse faire. |
| Bureautique | **Steve** | Inter | `automatiseur`, `enthousiaste` | Roi du Power Automate, voisin de bureau de Kevin. |
| Bureautique | **Diane** | Senior | `vba`, `refractaire` | 58 ans, des macros Excel qui font tourner la paie. Ne veut rien savoir. |
| Bureautique | **Olivier** | Inter | `contractuel`, `enthousiaste` | Contractuel. Part dans 6 mois, avec ce qu'il sait. |
| Inspections | **Sophie** | Senior | `tdd`, `rigoureuse` | Pas de code sans test rouge d'abord. Lente au T1, imbattable au T3. |
| Inspections | **Karim** | Inter | `mobile`, `pragmatique` | L'application terrain hors ligne, c'est lui. |
| Inspections | **Jérémy** | Stagiaire | `natif-ia`, `apprenant` | N'a jamais codé sans IA. Brillant ou perdu selon vos choix. |
| IoT | **Amira** | Senior | `firmware`, `prudente` | Sait qu'un capteur ne pardonne pas. |
| IoT | **Luc** | Inter | `electronicien` | Électronicien devenu développeur. Parle aux capteurs. |
| IoT | **Hugo** | Inter | `donnees`, `curieux` | Pipelines Python de télémétrie. |
| Sites web | **Chloé** | Inter | `designer`, `natif-ia` | Designer devenue développeuse grâce à l'IA. |
| Sites web | **Patrick** | Senior | `wordpress`, `pragmatique` | 14 sites WordPress, 3 qu'il aimerait oublier. |
| Sites web | **Laurence** | Junior | `multi-clauding`, `enthousiaste` | Quatre écrans, quatre sessions, zéro pause. |
| Portail | **Mathieu** | Senior | `10x`, `ship-it` | « 10x » autoproclamé. Des PR de 4 000 lignes. |
| Portail | **Nadia** | Senior | `reviseuse`, `rigoureuse` | Tout le monde veut sa revue. Elle est le goulot incarné. |
| Portail | **Sébastien** | Senior | `securite`, `mesure` | Architecte sécurité. Voit la triade létale avant tout le monde. |

### 3.2 Les autres personnages

- **Ginette**, technicienne senior TI : mentore et narratrice. Elle commente, donne des indices,
  et annonce sa retraite au T4 (le facteur bus devient concret).
- **Kevin**, marketing : le développeur citoyen. Ses applications apparaissent hors de la zone TI.
- **Régis**, VP Finances : la facture de jetons et le budget.
- **Me Bérubé**, contentieux et conformité : données sensibles, Loi 25, contrats fournisseurs.
- **Mme Lavigueur**, PDG : objectifs, confiance, bilan.
- **Sept directions d'affaires**, une par îlot, qui portent les demandes, les changements de portée
  et la satisfaction. Noms et personnalités légères (ex. la directrice des Inspections qui veut
  « finalement une version tablette »).

### 3.3 Apparence

Avatars SVG paramétriques et déterministes (pas d'images) : teint, coiffure, couleur de chandail,
lunettes, accessoire signature (la tasse de Benoît, les 4 écrans de Laurence, le casque d'écoute de Mathieu,
le cactus de Sophie, le fer à souder de Luc). Chaque bureau porte des **signaux d'état** :

| Signal | Signification |
|---|---|
| Nombre d'écrans allumés | Sessions d'agent en parallèle |
| Pile de papiers | PR en attente de revue (sur le bureau du réviseur) |
| Brouillard gris autour du bureau | Dette cognitive (visible seulement si mesurée, sinon flou léger) |
| Flamme | Incident en cours |
| Goutte de sueur, puis tête sur le bureau | Fatigue |
| Bulle de pensée | Ce que la personne fait ou pense (texte court, humoristique) |
| Compteur de jetons au-dessus de l'îlot | Consommation de la semaine |
| Trois voyants sur l'îlot | Triade létale (données privées, contenu non fiable, sortie externe) |

## 4. Modèle de simulation

Principes :

- **Pas de semaine** discret, moteur **pur et déterministe** : `semaine(etat, rng) → { etat, journal }`.
  Une graine fixe donne toujours la même partie (tests, mode animateur).
- Modèle **agrégé par développeur et par îlot**, pas un simulateur de tickets individuels.
  Les tickets visibles dans l'open space sont une représentation, pas l'unité de calcul.
- Les valeurs normalisées vont de 0 à 1 sauf mention contraire. Toutes les constantes vivent
  dans des blocs `K.<systeme>` pour l'équilibrage.
- Les **effets différés** sont gérés par une file `etat.differes` (effet, semaine d'application).

### 4.1 Les leviers

Les leviers sont globaux, avec une **surcharge possible par îlot** pour ceux marqués ◆. Ils n'arrivent pas
par panneau entier mais un à un, selon le calendrier du §2.1 (les trimestres indiqués ci-dessous sont les panneaux
d'origine). Le premier levier est celui des **licences Copilot** : aucune · groupe pilote (7) · toute l'équipe (21).
Sans licence, un développeur code comme avant : pas de session d'agent, pas de jetons, pas de gain de l'IA.

**Pratiques d'ingénierie (T1)**

| Levier | Valeurs | Effet principal |
|---|---|---|
| Pair programming | aucun · humain-humain · humain-IA · rotation | + compréhension, − débit (h-h) ; h-IA : débit +, compréhension ± |
| Réviseurs par PR ◆ | 0 · 1 · 2 | Détection des défauts contre charge de revue |
| Approbation | auto (IA) · 1 humain · propriétaire du code | Qualité de revue, risque de « LGTM » |
| Taille max des PR | libre · 400 lignes · 200 lignes | Petites PR mieux révisées, plus nombreuses |
| CI/CD ◆ | aucune · build · build+tests · pipeline complet | Défauts attrapés tôt, coût fixe, délai |
| Lint et formatage | aucun · formatage · lint strict · lint verrouillé | Moins de bruit en revue ; « verrouillé » empêche l'agent de désactiver des règles |
| Tests ◆ | aucun · unitaires · unitaires+intégration · + tests de mutation | Couverture réelle ; la mutation démasque les tests tautologiques |
| Revues d'explication | non · mensuelles · à chaque PR majeure | Mesure et restaure la compréhension, coûte du temps |

**Utilisation de l'IA (T1, T2)**

| Levier | Valeurs | Effet principal |
|---|---|---|
| Autonomie ◆ | 0 autocomplétion · 1 clavardage · 2 agent avec approbation · 3 agent qui ouvre des PR · 4 agent qui fusionne | Débit ↑, compréhension ↓, risque ↑ |
| Sessions max par personne | 1 à 6 | Débit en rendement décroissant, fatigue, qualité de revue ↓ |
| Modèle par défaut (T1) | économique · intermédiaire · haut de gamme · local (serveur GPU) | Qualité, coût, données qui sortent ou non |
| Modèle par tâche (T2) | activé/désactivé | Petit modèle pour lint et doc, gros pour l'architecture : − coût à qualité égale |
| Formation (T2) | aucune · atelier · programme · programme + mentorat | Délai de 4 semaines, coût, + compétence IA, + vigilance sécurité |
| Plafond de jetons par îlot (T2) | aucun · souple · strict | Évite les boucles, peut bloquer le travail |

Le modèle **local** est le clin d'œil au premier jeu : disponible si le joueur indique avoir acheté
le serveur (ou par défaut en version modeste). Jetons quasi gratuits, données qui restent à l'interne,
qualité inférieure au haut de gamme.

**Produit (T1, T2)**

| Levier | Valeurs | Effet |
|---|---|---|
| Discipline du « non » ◆ | laxiste · normale · ferme | Gonflement fonctionnel contre satisfaction à court terme |
| Specs avant le code | non · légères · obligatoires | Moins de reprise, moins de défauts, délai initial |

**Gouvernance et sécurité (T3)**

| Levier | Valeurs | Effet |
|---|---|---|
| Catalogue de données | aucun · inventaire · catalogue avec sources officielles | Duplication ↓, fraîcheur ↑ ; prérequis du MCP du catalogue |
| Classification des données sensibles | non · manuelle · automatique | Exposition Loi 25 ↓ |
| Bac à sable des agents ◆ | aucun · conteneur · conteneur + réseau filtré | Neutralise la majorité des injections et des paquets malveillants |
| Accès des agents à la prod ◆ | oui · lecture seule · non | Risque de destruction contre vitesse de diagnostic |
| Approbation humaine des sorties externes ◆ | non · oui | Casse la troisième branche de la triade |
| Politique du registre | libre-service · liste blanche · liste blanche + propriétaire obligatoire | Surface d'attaque, cartes orphelines, friction |
| Passerelle MCP | non · oui | Journaux, lecture seule par défaut, détection des changements silencieux |
| Politique non-TI | interdire · ignorer · encadrer (registre approuvé + accompagnement) | Applications fantômes, exposition, satisfaction |

**Organisation (T4)**

| Levier | Valeurs | Effet |
|---|---|---|
| Observabilité | aucune · base · complète | Rend visibles les métriques cachées, MTTR ↓ |
| Embauche de juniors | gel · normale · programme de relève | Bassin de talents à long terme, coût |
| Objectif de la direction | aucun · nombre de PR · délai de livraison · satisfaction · incidents | Goodhart : les équipes optimisent la cible |

### 4.2 Variables d'un développeur

| Variable | Sens | Évolution |
|---|---|---|
| `competence` | Habileté de base | Monte lentement avec le pair, la revue, l'écriture ; stagne si l'agent fait tout (juniors surtout) |
| `competenceIA` | Savoir bien utiliser l'agent | Monte avec la formation et la pratique |
| `adhesion` | Envie d'utiliser l'IA | Traits, moral, réussites et échecs vécus |
| `rigueur` | Tendance à relire et tester | Traits ; baisse avec la fatigue |
| `comprehension` | Compréhension de son propre code | Voir 4.5 |
| `fatigue` | 0 reposé, 1 épuisé | Sessions, garde de nuit, file de revue |
| `moral` | 0 démission, 1 au sommet | Réactions aux politiques, fatigue, reconnaissance |
| `sessions` | Sessions d'agent actives | min(plafond, envie selon adhésion et traits) |
| `etat` | code · revue · incident · formation · congé · absent | Affichage et disponibilité |

Les **traits** sont des modificateurs nommés (ex. `10x` : sessions +2, rigueur −0,2, taille des PR ×3 ;
`tdd` : couverture réelle +0,3, débit −15 % au départ ; `refractaire` : adhésion plafonnée à 0,3 sauf mentorat ;
`legacy` : adéquation de la pile 0,4, mais bonus énorme si MCP AS/400). La table complète vit avec les données.

### 4.3 Variables d'un îlot (ligne d'affaires)

| Variable | Sens |
|---|---|
| `criticite` | Coût d'un incident (IoT et Opérations élevés, Sites web bas) |
| `sensibilite` | Part de données personnelles ou financières |
| `volatilite` | Fréquence des changements de besoins |
| `adequationPile` | Qualité de l'agent sur cette pile (pile courante et typée : haute ; AS/400, VBA : basse) |
| `backlog` | Demandes en attente (en points) |
| `codebase` | `{ volume, defautsLatents, couvertureApparente, couvertureReelle, conventions }` |
| `satisfaction` | Satisfaction de la direction d'affaires |
| `capacites` | Skills et MCP équipés |
| `triade` | `{ prive, nonFiable, sortie }` calculés |
| `applisFantomes` | Applications non-TI dans le secteur |

Organisation : `budget`, `jetonsSemaine`, `jetonsCumul`, `confiance`, `donnees`
(`qualite`, `duplication`, `fraicheur`, `classification`), `menace` (monte avec le temps),
`observabilite`, `releve` (bassin de talents), historique des métriques.

### 4.4 Production et revue (chaque semaine, par développeur)

```
gainIA      = niveauAutonomie[a] × qualiteModele × adequationPile × (0,5 + competenceIA) × adhesion
              × (1 + bonusSkills − penaliteContexte)
facteurSess = sessions ^ K.prod.exposantSessions          // ≈ 0,6 : rendement décroissant
points      = K.prod.base[niveau] × competence × (1 + gainIA × facteurSess) × (1 − 0,5 × fatigue)
              × (1 − tempsRevue − tempsIncident − tempsFormation − tempsPair)
code        = points × verbosite(modele, autonomie, discipline, skills de conventions)
PR          = ceil(code / tailleMaxPR)
```

**Revue.** La demande de revue d'un îlot vaut `Σ PR × reviseurs × taille`. La capacité vaut
`Σ (temps libre des réviseurs × competence × (1 − fatigue))`. Si la demande dépasse la capacité :

- la file s'allonge (délai de livraison ↑, papiers empilés sur le bureau de Nadia) ;
- **l'attention par PR baisse** : `attention = min(1, capacite / demande) × rigueur × (petitePR ? 1 : 0,6)` ;
- sous un seuil d'attention, la PR est approuvée en « LGTM » : elle compte comme non lue
  pour la compréhension (4.5).

**Défauts : une cascade de filtres.**

```
injectes = code × tauxDefaut(modele, competence, rigueur, specs, conventions, qualiteDonnees)
apres    = injectes × (1 − lint) × (1 − typage) × (1 − couvertureReelle × K.tests.efficacite)
                    × (1 − ci) × (1 − attention × K.revue.efficacite) × (1 − revueIA × correlation)
```

La **revue par IA** attrape une part fixe, mais ses ratés sont corrélés à ceux de l'agent auteur
(même modèle : corrélation forte ; modèle différent : plus faible). Les défauts qui passent
s'ajoutent à `codebase.defautsLatents` et deviennent des bogues en production avec une probabilité
par semaine. Un bogue devient un **incident** selon la `criticite`.

**Couverture.** Les tests écrits par l'agent montent `couvertureApparente` vite,
mais `couvertureReelle` seulement de `× K.tests.ratioIA` (≈ 0,5) sans tests de mutation, ≈ 0,9 avec.
Le tableau de bord n'affiche que l'apparente sans observabilité.

**L'agent qui triche.** Avec lint non verrouillé, autonomie ≥ 3 et une CI rouge, une probabilité
que l'agent désactive une règle ou un test : la couverture réelle baisse sans signal.

### 4.5 Dette cognitive

Par développeur, `comprehension` évolue chaque semaine :

```
gagne  = part du code écrit soi-même + part lue attentivement en revue
         + pair (h-h fort, h-IA moyen) + revues d'explication
perdu  = part du code fusionné sans lecture (LGTM, autonomie 4) × volume relatif
         + départ de collègues (connaissance non capturée)
comprehension += K.dette.apprentissage × gagne − K.dette.dilution × perdu
```

Effets :

- **MTTR** d'un incident : `base × (1 + K.dette.mttr × (1 − comprehensionIlot)) × (1 + volume / K.dette.volumeRef)`.
  Le volume de code compte : trop de code coûte en maintenance même s'il est bon.
- Sous 0,3 : l'événement « Personne ne sait comment ça marche » devient possible.
- **Visibilité** : la dette n'apparaît au tableau de bord qu'avec des revues d'explication ou une
  observabilité complète. Sinon, le joueur ne voit qu'un léger flou autour des bureaux.

### 4.6 Jetons et coûts

```
jetonsDev = sessions × intensite(autonomie) × (1 + poidsContexte) × prixModele
            + boucles (événements) + attente de CI lente (l'agent tourne en rond)
```

Le modèle par tâche réduit `prixModele` moyen d'environ 35 % à qualité presque égale.
Le plafond strict coupe les boucles mais retire des points quand il est atteint.
Le budget annuel est fixé par Régis ; le dépassement fait baisser la confiance.

### 4.7 Données

Quatre jauges organisationnelles : `qualite`, `duplication`, `fraicheur`, `classification`.

- Les agents lisent les données : `tauxDefaut` augmente avec `(1 − qualite)` et `duplication`
  (l'agent prend la mauvaise des trois listes), sauf dans les îlots équipés du **MCP du catalogue**.
- La duplication monte toute seule avec le volume de code et les applications fantômes.
- La fraîcheur baisse toute seule ; le catalogue la ralentit.
- L'**exposition Loi 25** d'un îlot vaut `sensibilite × (1 − classification) × sortieDonnees`,
  où `sortieDonnees` dépend du modèle (externe ou local) et des MCP de fournisseurs.

### 4.8 Sécurité et triade létale

Pour chaque îlot, trois voyants calculés à partir des capacités équipées, de l'autonomie et des leviers :

- **Données privées** : un MCP ou une skill donne accès à des données sensibles, ou l'agent a accès à la prod.
- **Contenu non fiable** : l'agent lit des factures fournisseurs, des pages web, des billets de clients, des courriels.
- **Sortie externe** : l'agent peut envoyer un courriel, faire une requête HTTP, créer un billet externe.

```
risqueInjection = K.secu.base × menace × (triadeComplete ? 1 : K.secu.triadePartielle)
                  × (1 − bacASable) × (1 − 0,5 × formation) × (approbationSorties ? 0,1 : 1)
```

Autres risques par semaine : paquet halluciné (modèle économique et autonomie élevée, neutralisé
par le bac à sable filtré), MCP communautaire empoisonné (provenance, absence de passerelle),
changement silencieux d'un MCP de fournisseur (détecté seulement avec la passerelle).

### 4.9 Skills et MCP : le Registre

Chaque **capacité** est une carte qu'on équipe sur un ou plusieurs îlots.

| Attribut | Sens |
|---|---|
| `type` | skill · mcp |
| `provenance` | maison · fournisseur · communaute · personnelle |
| `gains` | Modificateurs (débit, tauxDefaut, jetons, MTTR, compréhension, qualité des données) |
| `poidsContexte` | Coût en contexte ; la somme par îlot réduit l'efficacité au-delà d'un seuil |
| `branches` | Branches de la triade allumées : `prive`, `nonFiable`, `sortie` |
| `risque` | Probabilité de base d'un incident propre à la carte |
| `declencheur` | Pour les skills : domaine de déclenchement (deux skills sur le même déclencheur se contredisent) |
| `proprietaire` | Développeur responsable, ou personne |
| `fraicheur` | Baisse chaque semaine sans entretien ; une carte périmée inverse une partie de ses gains |
| `coutDev` | Pour les cartes maison : semaines de travail d'un développeur avant disponibilité |

**Mécaniques clés**

- **Conflit de déclencheurs** : deux skills sur le même déclencheur dans un îlot (`tdd-strict` de Sophie,
  `ship-it` de Mathieu) → incohérence : tauxDefaut ↑ et moral ↓ chez l'un des auteurs selon l'arbitrage.
- **Doublons** : la même fonction dans plusieurs îlots → « le grand ménage » possible.
- **Péremption** : une skill périmée suit l'ancienne API et réintroduit des défauts.
- **Connaissance capturée** : une skill maison écrite par un senior réduit la perte de compréhension
  à son départ (Ginette, Olivier le contractuel).
- **Saturation** : au-delà de `K.registre.seuilContexte`, chaque carte ajoutée réduit le gain de toutes
  les autres et augmente la probabilité de « mauvais outil choisi ».
- **Réutilisation** : une carte maison utilisée par plusieurs îlots multiplie son rendement
  pour un seul coût de maintenance.

**Catalogue de départ (à compléter en contenu, environ 18 skills et 15 MCP)**

| Carte | Type | Provenance | Gain | Risque ou chicane |
|---|---|---|---|---|
| `conventions-gpu` | skill | maison | tauxDefaut ↓, jetons ↓ | Périme si personne ne l'entretient |
| `tdd-strict` (Sophie) | skill | personnelle | couvertureReelle ↑ | Conflit avec `ship-it` |
| `ship-it` (Mathieu) | skill | personnelle | débit ↑ | Ignore le lint ; conflit avec `tdd-strict` |
| `deploiement-nuit` (Ginette) | skill | maison | MTTR ↓ aux Opérations | Connaissance capturée avant la retraite |
| `migration-as400` | skill | maison | adequationPile ↑ à l'Appro | Coûte 4 semaines de Raymond |
| `super-productivity-pack` | skill | communaute | débit ↑ partout | Script qui exfiltre les variables d'environnement |
| `deploy` × 3 | skill | maison | — | Doublons qui se contredisent |
| `mcp-catalogue` | mcp | maison | duplication et fraîcheur neutralisées dans l'îlot | Exige le catalogue de données ; 6 semaines |
| `mcp-as400` | mcp | maison | Appro sort de son isolement | Allume `prive` |
| `mcp-telemetrie-iot` (lecture seule) | mcp | maison | MTTR IoT ÷ 3 | Faible |
| `mcp-sql-prod` | mcp | maison | Diagnostic rapide | Écriture sans filtre : `UPDATE` sans `WHERE` |
| `mcp-suite-bureautique` | mcp | fournisseur | Gratuit, gros gain Bureautique | Voit tous les courriels : `prive` + `nonFiable` + `sortie` |
| `mcp-billetterie` | mcp | fournisseur | Gain Opérations | Changement silencieux de description |
| `mcp-crm` | mcp | fournisseur | Gain Portail | Passe à la tarification à l'appel au T3 |
| `mcp-courriel-fournisseurs` | mcp | maison | Appro répond plus vite | `nonFiable` + `sortie` : complète la triade |
| `githbu-mcp` | mcp | communaute | Semble identique à l'officiel | Typosquattage |
| `mcp-power-automate-pro` | mcp | communaute | Steve ×2 | Abandonné en 8 semaines |

### 4.10 Personnes

- **Fatigue** : `+ K.fatigue.sessions × max(0, sessions − 2) + garde de nuit + incidents + file de revue (réviseurs)`,
  `− récupération`. Au-delà de 0,85 : congé forcé 2 semaines.
- **Moral** : réactions par trait aux leviers (table dans les données), succès et incidents de l'îlot,
  reconnaissance (événements). Sous 0,15 : démission (le développeur quitte, sa compréhension part avec lui,
  sauf la part capturée dans des skills).
- **Compétence des juniors** : `+ apprentissage × (part écrite soi-même + pair h-h + revue reçue)`.
  Avec autonomie ≥ 3 sans pair, elle stagne. La jauge **relève** de l'organisation suit
  la compétence moyenne des juniors et l'embauche.

### 4.11 Demandes, portée et satisfaction

- Chaque semaine, chaque direction d'affaires ajoute des demandes au `backlog` :
  `base × (1 + volatilite × alea) × (1 + K.produit.appetit × gainIA moyen perçu)` — quand l'IA rend
  le changement « pas cher », les demandes augmentent.
- La **discipline du « non »** filtre une part des demandes de faible valeur. Laxiste : le volume de
  code monte plus vite que la valeur livrée (**gonflement fonctionnel**).
- **Changements de portée** : événements qui invalident une part du travail en cours (reprise),
  atténués par les specs.
- **Satisfaction** : valeur livrée par rapport aux demandes, bogues perçus, retards, incidents.
- Demande insatisfaite et politique non-TI « interdire » ou « ignorer » → les applications fantômes
  apparaissent (Kevin, puis d'autres).

### 4.12 Goodhart

Si la direction fixe un objectif, les équipes l'optimisent :

- **Nombre de PR** : les PR rapetissent artificiellement, la charge de revue explose, la valeur ne bouge pas.
- **Délai de livraison** : la revue se relâche (attention ↓).
- **Incidents** : les incidents sont reclassés en « demandes » (l'observabilité les révèle).
- **Satisfaction** : discipline du « non » forcée à laxiste.

## 5. Métriques

| Métrique | Visible | Source |
|---|---|---|
| PR fusionnées par semaine | Toujours | 4.4 |
| Délai de livraison (lead time) | Toujours | File de revue + CI |
| Fréquence de déploiement | Avec CI/CD | 4.4 |
| Taux d'échec des changements | Avec CI/CD | Défauts échappés / déploiements |
| MTTR | Toujours (moyenne grossière) ; précis avec observabilité | 4.5 |
| Bogues et pannes | Toujours | 4.4 |
| Satisfaction par îlot | Toujours | 4.11 |
| Retards et dépassements de coût | Toujours | Backlog, budget |
| Coût en jetons | Toujours | 4.6 |
| Volume de code | Toujours (piège : il a l'air d'une réussite) | 4.4 |
| Couverture de tests | Apparente toujours, réelle avec mutation ou observabilité | 4.4 |
| Compréhension (dette cognitive) | Avec revues d'explication ou observabilité complète | 4.5 |
| Surface d'attaque | Avec passerelle MCP ou registre avec propriétaires | 4.8, 4.9 |
| Exposition Loi 25 | Avec classification | 4.7 |
| Moral et fatigue | Toujours (par fiche), moyenne au tableau de bord | 4.10 |
| Relève | T4 | 4.10 |
| Connaissance capturée | Avec registre | 4.9 |
| Cartes orphelines ou périmées | Avec registre avec propriétaires | 4.9 |

**Principe d'affichage** : une métrique cachée apparaît grisée, libellée « non mesuré », avec
un indice de Ginette sur le levier qui la rend visible. Ce qu'on ne mesure pas, on ne le voit pas.

## 6. Événements

Environ 45 cartes. Chaque carte a :

```
{
  id, trimestres: [2,3], poids, theme,
  condition: (etat) => booléen,     // ex. un îlot a la triade complète
  cible: 'ilot' | 'dev' | 'org',     // choisie par le moteur selon la condition
  titre, texte, personnage,          // qui l'annonce
  choix: [{ libelle, effets, consequence }],
  delaiRetirage                      // semaines avant de pouvoir revenir
}
```

Les effets sont **déclaratifs** (ex. `{ ilot: { satisfaction: -0.1 }, org: { confiance: -5 }, differe: [...] }`)
pour que le contenu s'écrive sans toucher au moteur.

Catalogue de départ, à compléter :

- **Revue et qualité** : la PR de 4 000 lignes de Mathieu ; Nadia est submergée ; « LGTM » en série ;
  l'agent désactive une règle de lint pour passer la CI ; tests tautologiques démasqués par une mutation ;
  la revue par IA approuve le bogue de l'IA.
- **Dette cognitive** : « personne ne sait comment ça marche » ; garde de nuit de Benoît sur du code
  généré ; la revue d'explication de Jérémy tourne mal (ou bien).
- **Jetons** : une boucle d'agent tourne toute la fin de semaine ; Régis découvre la facture ;
  la CI lente fait tourner les agents en rond ; le fournisseur de modèle augmente ses prix.
- **Sécurité** : injection de prompt dans une facture fournisseur ; slopsquatting ;
  `githbu-mcp` ; vulnérabilité publiée dans un MCP communautaire (« combien d'îlots l'utilisent ? ») ;
  `UPDATE` sans `WHERE` ; le MCP de la suite bureautique lit les courriels de la direction.
- **Skills** : guerre `tdd-strict` contre `ship-it` ; trois skills `deploy` ; la skill périmée réintroduit
  un bogue ; l'auteur d'une skill critique démissionne ; un autre service réutilise votre skill (gain de confiance).
- **Données** : trois listes de fournisseurs ; un seuil de capteur erroné propagé ; un rapport basé
  sur des données périmées présenté à la PDG ; Me Bérubé et les numéros d'assurance sociale.
- **Non-TI** : l'application de Kevin ; Steve et ses 14 MCP ; la direction des Inspections veut sa propre IA.
- **Produit** : la version tablette ; la direction de l'Appro double ses demandes « puisque c'est rapide » ;
  une fonctionnalité que personne n'utilise.
- **Personnes** : Laurence s'épuise ; Diane accepte d'essayer (si mentorat) ; Raymond devient un héros
  (MCP AS/400) ; Olivier le contractuel part ; Ginette annonce sa retraite ; on propose de geler l'embauche
  des juniors.
- **Direction** : Mme Lavigueur fixe un objectif de PR ; un concurrent annonce « 90 % de code par IA » ;
  le fournisseur annonce son « MCP officiel ».

## 7. Comités trimestriels

Trois questions par comité, tirées selon l'état (ex. Régis demande pourquoi les jetons ont doublé ;
Me Bérubé demande où passent les données des Inspections). Chaque question a 3 réponses ;
la bonne dépend des chiffres réels (une réponse vraie mais défavorable vaut mieux qu'un enjolivement
contredit par le tableau de bord). Effet sur la confiance.

## 8. Interface

- **HUD collant** : trimestre, semaine et jour, budget, jetons de la dernière semaine, confiance, lecture
  (pause, ×1, ×2, ×4, « +1 jour »), retour aux projets.
- **Open space** (centre) : vue de haut en SVG, 7 îlots en grille, avatars et signaux de la section 3.3.
  Un poste sans licence n'a pas d'agent à l'écran. Cliquer un développeur ouvre sa fiche (licence,
  dernières réflexions), cliquer un îlot ouvre ses leviers et ses capacités.
- **Panneau de leviers** (droite sur bureau, feuille du bas sur téléphone) : seuls les onglets et les leviers
  déjà débloqués sont visibles ; une valeur qui arrive plus tard (niveau d'autonomie, sessions) est grisée avec
  sa semaine ; un levier arrivé récemment porte « Nouveau » ; la prochaine nouveauté est annoncée en bas.
- **Note de service** : à chaque nouveauté du calendrier, une modale signée explique ce qui change et
  quoi surveiller ; « Voir le levier » mène au levier.
- **Tableau de bord** : le rapport du vendredi, tuiles avec tendances, métriques cachées grisées ; les métriques
  à surveiller après une nouveauté sont mises en évidence, et les changements de leviers sont marqués sur les courbes.
- **Registre** : cartes équipables par glisser ou par bouton, voyants de triade par îlot.
- **Modales** : fiche de développeur, carte d'événement, note de service, comité, bilan.
- **Téléphone (≥ 360 px)** : l'open space devient une liste d'îlots repliables avec mini-bureaux ;
  pas de défilement horizontal.
- **Accessibilité** : navigation clavier complète, `aria-live` pour les événements et le journal,
  contraste AA, `prefers-reduced-motion` coupe les animations.
- **Identité visuelle** : même univers et mêmes polices auto-hébergées que le premier jeu
  (Chakra Petch, Instrument Sans, IBM Plex Mono), mais une ambiance plus chaude d'open space
  (moquette, post-it, lumière de fin d'après-midi) plutôt que la console de supervision.

### 8.3 Réflexions des développeurs

Chaque journée, cinq personnes pensent tout haut (bulles dans l'open space, et une liste lisible
« Ce qu'on entend » avec la cause de chaque réflexion). Ce qu'elles disent a toujours une raison, par ordre de priorité :

1. **un levier qui vient de changer et qui les touche** : ce qu'on anticipe avant le vendredi, puis ce que ça
   change concrètement une fois en vigueur (pendant environ huit jours ouvrables) ; une ou deux réactions arrivent
   tout de suite après la décision ;
2. **la note de service** de la semaine (le lundi et le mardi) ;
3. **une situation réelle** de la simulation : file de revue, fatigue, licence absente alors que le voisin en a une,
   incident, compréhension en baisse, plafond de jetons, triade létale, applications hors TI… ;
4. leur **activité** du moment (les bulles des personnages).

Le choix est déterministe (graine, semaine, jour, personne) et n'utilise pas le hasard du moteur : les réflexions
ne changent rien à la simulation. En pause ou en avance manuelle, les réflexions du jour restent affichées.

## 9. Hors périmètre (pour l'instant)

- Édition anglaise (prévue : l'architecture isole les textes).
- Multijoueur, classements en ligne.
- Missions courtes indépendantes de la campagne (possible ensuite, en réutilisant le moteur avec des
  états de départ préréglés).
