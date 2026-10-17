/* 09-donnees-reflexions.js */
/* Contenu des réflexions des développeurs (lu par 33-reflexions.js).
   Une entrée est une chaîne, ou un objet :
     { texte, devs?: [ids], traits?: [traits], niveaux?: [niveaux], pilote?: bool, licence?: bool,
       effet?: bool (leviers : vrai = seulement une fois que le changement a passé un vendredi, faux = seulement avant),
       condition?: (etat, q, dev) → bool, ton?: 'plus' | 'moins' | 'neutre' }
   texte peut aussi être une fonction (etat, q, dev) → chaîne (chiffres de la semaine, nom d'un collègue).
   Les gabarits {dev} et {ilot} sont remplis par l'interface.
   - leviers[id][valeur] : ce qu'on pense d'un changement de levier qui nous touche ('*' : n'importe quelle valeur)
   - nouveautes[cle] : ce qu'on dit en lisant la note de service (cle : id de levier, 'levier:v1,v2' ou id de panneau)
   - situations : [{ id, libelle, quand(etat, q, dev, ilot) → bool | intensité, poids?, ton?, levier?, lignes: [entrées] }]
     libelle : la cause, en deux à quatre mots, affichée avec la réflexion.

   Choix éditoriaux :
   - « avant » (effet: false) : on vient de l'apprendre, on anticipe ; « après » (effet: true) : ce que ça change
     concrètement dans la journée, assez précis pour relier l'effet à sa cause (file de revue, taille des PR,
     jetons, compréhension, fatigue…) ;
   - les leviers à délai (formation, catalogue, classification, passerelle, observabilité) sont distingués par
     condition : en préparation (q.enAttente) ou en vigueur (q.levier) ;
   - toutes les conditions lisent l'état de façon défensive : elles ne lèvent jamais d'exception. */
(function (LTP) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Outils de lecture défensifs
  // ---------------------------------------------------------------------------
  function fusion(args, depart) {
    var e = {};
    for (var k in depart) e[k] = depart[k];
    for (var i = 0; i < args.length; i++) {
      var o = args[i];
      if (!o || typeof o !== 'object') continue;
      for (var c in o) if (Object.prototype.hasOwnProperty.call(o, c)) e[c] = o[c];
    }
    return e;
  }
  // Entrée générique ; av : avant l'effet (on vient de l'annoncer) ; ap : après un vendredi.
  function E(texte) { return fusion(Array.prototype.slice.call(arguments, 1), { texte: texte }); }
  function av(texte) { return fusion(Array.prototype.slice.call(arguments, 1), { texte: texte, effet: false }); }
  function ap(texte) { return fusion(Array.prototype.slice.call(arguments, 1), { texte: texte, effet: true }); }
  function qui() { return { devs: Array.prototype.slice.call(arguments) }; }
  function trait() { return { traits: Array.prototype.slice.call(arguments) }; }
  function si(f) { return { condition: f }; }

  var PLUS = { ton: 'plus' }, MOINS = { ton: 'moins' };
  var LIC = { licence: true }, SANSLIC = { licence: false };
  var PIL = { pilote: true }, NONPIL = { pilote: false };
  var JUNIORS = { niveaux: ['junior', 'stagiaire'] }, SENIORS = { niveaux: ['senior'] };

  function num(x, d) { return typeof x === 'number' && isFinite(x) ? x : d; }
  function lv(etat, q, id, ilotId) { try { return q.levier(etat, id, ilotId); } catch (e) { return undefined; } }
  function lic(etat, q, dev) { try { return !!q.aLicence(etat, dev); } catch (e) { return false; } }
  function auto(etat, q, dev) { return num(Number(lv(etat, q, 'autonomie', dev && dev.ilot)), 0); }
  function der(x) { return (x && x.derniere) || {}; }
  function aTrait(dev, t) { return !!(dev && Array.isArray(dev.traits) && dev.traits.indexOf(t) >= 0); }
  function ilotDe(etat, dev) { return (etat && etat.ilots && dev && etat.ilots[dev.ilot]) || null; }
  function sem(etat) { return num(etat && etat.semaine, 0); }
  function jour(etat) { return num(etat && etat.jour, 0); }
  function collegues(etat, q, dev) {
    try { return q.devsDe(etat, dev.ilot).filter(function (d) { return d.id !== dev.id; }); } catch (e) { return []; }
  }
  function nom(q, id) {
    try { var d = q.defDev(id); return (d && d.nom) || String(id); } catch (e) { return String(id); }
  }
  function drapeau(etat, n) { return !!(etat && etat.org && etat.org.drapeaux && etat.org.drapeaux[n]); }
  // Réviseur : a relu au moins deux PR la semaine passée, ou c'est son métier.
  function reviseur(etat, q, dev) { return num(der(dev).revues, 0) >= 2 || aTrait(dev, 'reviseur'); }
  var REV = si(reviseur);
  function pasReviseur(etat, q, dev) { return !reviseur(etat, q, dev); }
  function enVigueur(id, v) { return function (etat, q, dev) { return lv(etat, q, id, dev && dev.ilot) === v; }; }
  function prepare(id) {
    return function (etat, q) { try { return !!q.enAttente(etat, id, null); } catch (e) { return false; } };
  }
  function avecLicence(etat, q, dev) { return lic(etat, q, dev); }
  function autoMin(n) { return function (etat, q, dev) { return lic(etat, q, dev) && auto(etat, q, dev) >= n; }; }
  function et() {
    var fs = arguments;
    return function (a, b, c) { for (var i = 0; i < fs.length; i++) if (!fs[i](a, b, c)) return false; return true; };
  }
  function travailIlot(etat, id) { var t = etat && etat.travail && etat.travail.ilots; return (t && t[id]) || {}; }
  function travailDev(etat, id) { var t = etat && etat.travail && etat.travail.devs; return (t && t[id]) || {}; }
  function equipee(etat, carte, ilotId) {
    var c = etat && etat.capacites && etat.capacites[carte];
    return !!(c && Array.isArray(c.equipeeSur) && c.equipeeSur.indexOf(ilotId) >= 0);
  }
  function nbBranches(il) {
    var t = (il && il.triade) || {};
    return (t.prive ? 1 : 0) + (t.nonFiable ? 1 : 0) + (t.sortie ? 1 : 0);
  }
  function triade(etat, q, dev) { try { return !!q.triadeComplete(etat, dev.ilot); } catch (e) { return false; } }
  function incidents(etat, q, ilotId) { try { return q.incidentsActifs(etat, ilotId) || []; } catch (e) { return []; } }
  function plafondAtteint(etat, q, dev) {
    if (lv(etat, q, 'plafondJetons', dev && dev.ilot) !== 'strict') return false;
    return num(travailIlot(etat, dev.ilot).pointsPerdusPlafond, 0) > 0.2 || num(travailDev(etat, dev.id).facteurPlafond, 1) < 0.98;
  }
  function budgetRythme(etat) {
    var o = (etat && etat.org) || {}, annuel = num(o.budgetAnnuel, 0), s = sem(etat);
    if (!(annuel > 0) || s < 1) return 1;
    var depense = annuel - num(o.budget, annuel);
    return depense / (annuel * s / 52);
  }

  // ---------------------------------------------------------------------------
  // Réflexions par levier : leviers[id][valeur]
  // ---------------------------------------------------------------------------
  var L = {};

  L.licences = {
    tous: [
      av('Ma licence Copilot arrive. Je commence par quoi ?', NONPIL, PLUS),
      av('Une licence pour moi ! Je vais enfin voir de quoi parlent les pilotes.', NONPIL, PLUS),
      av('Copilot pour tout le monde. On va voir si les pilotes avaient raison.', NONPIL),
      av('Le pilote devient la norme. On va enfin pouvoir en parler entre nous.', PIL),
      av('Fini, le groupe pilote. Je perds mon statut de cobaye.', PIL),
      av('Une licence pour moi aussi ? Je n\'ai rien demandé.', trait('sceptique', 'refractaire'), MOINS),
      av('Vingt et une licences. Ça fait beaucoup de code de plus à relire.', REV),
      av('Une licence. Je vais lui montrer ce qu\'est du vrai RPG.', qui('raymond')),
      av('Une licence, enfin. Je rattrape le retard ce soir.', qui('jeremy'), PLUS),
      av('Une licence. Il me faut maintenant trois sessions de plus.', qui('laurence'), PLUS),
      av('Une licence. Premier test : est-ce qu\'il écrit le test d\'abord ?', qui('sophie')),
      av('Une licence pour le firmware ? Je l\'essaie sur les tests, pas ailleurs.', qui('amira')),
      av('Je vais lui faire écrire mes scripts de nuit. Et les relire de jour.', qui('benoit')),
      av('Ma licence. Benoît dit : lis tout ce qu\'il propose. D\'accord.', qui('tommy'), PLUS),
      av('Vingt et une licences. Je relis la politique de données du fournisseur.', qui('sebastien')),
      av('Première question pour l\'assistant : quelle liste de fournisseurs connaît-il ?', qui('lucie')),
      av('Une licence ! Il va m\'aider avec le C, j\'espère.', qui('luc'), PLUS),
      av('Une licence. Il va vouloir installer des extensions, je le sens.', qui('patrick')),
      av('Une licence avant la fin de mon contrat. Je vais livrer en double.', qui('olivier'), PLUS),
      ap('L\'autocomplétion finit mes lignes. Une fois sur deux, c\'est ce que je voulais écrire.', NONPIL),
      ap('Je pose mes questions à l\'assistant au lieu de fouiller la doc. Je gagne du temps, et je vérifie.', NONPIL, si(autoMin(1))),
      ap('Tout le monde a Copilot. Je ne suis plus la seule personne de l\'îlot à qui on demande comment ça marche.', PIL),
      ap('Depuis vendredi, il y a plus de PR dans la file : tout l\'îlot code avec l\'assistant.', REV),
      ap('L\'autocomplétion me propose du Java dans du RPG. Je l\'ai mise en sourdine.', qui('raymond')),
      ap('Il complète mes macros avec des noms de variables en anglais. Non.', qui('diane'), MOINS),
      ap('Enfin ! Je recommence à coder à ma vitesse normale.', qui('jeremy'), PLUS),
      ap('Une licence, c\'est un bon début. Il me manque juste trois sessions.', qui('laurence'), PLUS),
      ap('Je lui fais écrire le test d\'abord. Il apprend vite, le petit.', qui('sophie')),
      ap('Il écrit mes scripts Ansible. Je les relis deux fois avant la nuit.', qui('benoit')),
      ap('L\'assistant m\'explique les commandes. Benoît aussi, mais il soupire.', qui('tommy')),
      ap('Il complète les codes de fournisseurs. Avec ceux de la vieille liste.', qui('lucie')),
      ap('Je m\'en sers pour les tests. Pas pour le firmware.', qui('amira')),
      ap('Il écrit le C, je vérifie les broches. Ça marche, à deux.', qui('luc'), PLUS),
      ap('Il connaît WordPress mieux que moi. C\'est un peu inquiétant.', qui('patrick')),
      ap('Il m\'a suggéré un mot de passe en clair dans un fichier de config. Je note.', qui('sebastien'), MOINS),
      ap('Je livre plus vite. Mon successeur aura plus de code à lire.', qui('olivier')),
      ap('Plus de monde avec Copilot, plus de code dans ma pile. Toujours une seule Nadia.', qui('nadia'), MOINS),
    ],
    pilote: [
      av('On reprend ma licence ? Je venais de m\'habituer.', NONPIL, LIC, MOINS),
      av('On redevient sept. Au moins, on va pouvoir comparer.', PIL),
      ap('Sans Copilot, je retape tout. Mes doigts s\'en souviennent, lentement.', NONPIL, MOINS),
      ap('Sept pilotes et quatorze témoins : les chiffres de vendredi vont dire quelque chose.', PIL),
      ap('Sans assistant, je suis à moitié de ma vitesse. Peut-être moins.', qui('jeremy', 'laurence'), MOINS),
      ap('Le clavier, je connais. Ça me va très bien.', trait('sceptique', 'refractaire'), PLUS),
    ],
    aucune: [
      av('Plus de Copilot ? J\'avais pris le pli.', PIL, MOINS),
      av('Plus de licences du tout. Le calme revient.', trait('sceptique', 'refractaire', 'prudent'), PLUS),
      ap('Je tape la fin de mes lignes moi-même. C\'est long, une fin de ligne.', PIL, MOINS),
      ap('Plus de jetons cette semaine. Plus d\'aide non plus.'),
      ap('Sans agent, je suis à 1x. Comme vous autres.', qui('mathieu'), MOINS),
      ap('Les chiffres d\'une semaine sans IA : je les garde pour comparer.', qui('marieeve')),
      ap('Le clavier, comme avant. Je comprends tout ce que j\'écris.', trait('sceptique', 'refractaire', 'tdd'), PLUS),
    ],
  };

  L.autonomie = {
    0: [
      av('Retour à l\'autocomplétion. On ralentit, donc.', LIC),
      av('On redescend à l\'autocomplétion ? On va écrire à la main, comme en janvier.', LIC),
      av('L\'autocomplétion seulement. Mathieu va faire une crise.', qui('nadia', 'sebastien')),
      av('Le niveau d\'autonomie baisse. Pour moi, sans licence, ça ne change rien.', SANSLIC),
      ap('Autocomplétion seulement : j\'écris presque tout moi-même. Je comprends tout, aussi.', LIC),
      ap('De l\'autocomplétion. Je garde le volant, et je l\'aime bien.', LIC, trait('sceptique', 'prudent', 'firmware'), PLUS),
      ap('Moins de code dans les PR depuis vendredi. La file de revue respire.', REV),
      ap('Une suggestion par ligne. Mes quatre écrans dorment.', qui('laurence'), MOINS),
      ap('Il complète, je décide. Je décide lentement.', qui('jeremy')),
    ],
    1: [
      av('Le clavardage ! Je vais pouvoir lui poser mes questions niaiseuses.', LIC, PLUS),
      av('On va pouvoir jaser avec l\'assistant. Je prépare mes questions.', LIC),
      av('Le clavardage. Je vais lui demander d\'où il sort ses réponses.', LIC, trait('sceptique', 'prudent', 'donnees')),
      av('Clavarder avec l\'assistant ? J\'ai des questions sur la synchro hors ligne.', qui('karim')),
      av('Le clavardage, c\'est pour ceux qui ont une licence. Pas moi.', SANSLIC, MOINS),
      ap('Je demande à l\'assistant au lieu de chercher dans la doc. Je gagne une heure par jour.', LIC, PLUS),
      ap('Il m\'explique le code de 2015. Il se trompe une fois sur cinq, avec aplomb.', LIC),
      ap('Je lui demande pourquoi ça marche. Il répond. Je comprends, un peu.', qui('chloe')),
      ap('Je lui ai demandé d\'où venait un seuil. Il en a inventé un.', qui('hugo')),
      ap('Il ne sait pas ce qu\'est un sous-sol sans réseau. Je lui explique.', qui('karim')),
      ap('Je lui ai posé une question sur le RPG. Il a répondu en COBOL.', qui('raymond')),
      ap('Je lui demande ce que fait une commande avant de la lancer. Bonne habitude.', qui('tommy')),
    ],
    2: [
      av('Le mode agent : il fait les étapes, je dis oui. Ou non.', LIC),
      av('Un agent qui enchaîne les étapes. Je vais apprendre à dire non.', LIC),
      av('Le mode agent arrive dans l\'îlot. Les PR vont grossir, je le sens.', REV),
      av('Le mode agent ! Enfin quelque chose pour mes écrans.', LIC, trait('multiClauding', 'natifIA', 'enthousiaste'), PLUS),
      av('Approbation à chaque étape : je vais lire chaque étape.', qui('amira', 'sophie', 'lucie')),
      ap('L\'agent propose dix étapes, j\'en approuve dix. J\'ai lu les trois premières.', LIC),
      ap('Mes PR ont grossi d\'un tiers depuis vendredi. L\'agent aime écrire.', LIC),
      ap('Plus de code dans ma pile depuis le mode agent, et des PR plus longues.', REV, MOINS),
      ap('L\'agent fait la moitié du travail. Je comprends la moitié du code.', LIC, JUNIORS),
      ap('Mes voisins ont un agent. Moi, j\'ai mes deux mains.', SANSLIC, MOINS),
      ap('Un agent ! Qui demande la permission. Toutes les dix secondes.', qui('laurence')),
      ap('Il me demande avant de redémarrer un service. Bon agent.', qui('benoit')),
      ap('Approuver chaque étape, c\'est long. J\'approuve en lot.', qui('mathieu')),
    ],
    3: [
      av('L\'agent va ouvrir ses propres PR. Qui va les relire ?', LIC),
      av('Des PR ouvertes par un agent. Il faudra les relire comme celles d\'un inconnu.', SENIORS),
      av('L\'agent ouvre les PR ? Ma liste de PR va devenir longue.', LIC, JUNIORS),
      av('Des agents qui ouvrent des PR. J\'ai trois idées de tâches à leur donner.', qui('mathieu', 'laurence', 'olivier'), PLUS),
      av('Des agents qui ouvrent des PR. Je prépare une deuxième pile.', qui('nadia'), MOINS),
      ap('Trois PR de l\'agent ce matin. J\'en avais demandé une.', LIC),
      ap('La file de revue a doublé depuis que les agents ouvrent leurs PR.', REV, MOINS),
      ap('L\'agent écrit, j\'ouvre la PR. J\'apprends quoi, au juste ?', LIC, JUNIORS, MOINS),
      ap('Le carnet de demandes fond. La pile de revue, elle, monte.', LIC),
      ap('Les PR de l\'agent arrivent la nuit. Je les lis le matin, une à une.', trait('prudent', 'firmware', 'rigoureux')),
      ap('Mes agents ouvrent des PR pendant que je dîne. 10x, je disais.', qui('mathieu'), PLUS),
      ap('Mon agent ouvre des PR plus vite que je les lis. Je les lis quand même ?', qui('jeremy')),
      ap('L\'agent livre, je documente. Pour celui qui viendra après moi.', qui('olivier')),
    ],
    4: [
      av('Il va fusionner tout seul ? Sans personne ?', LIC, MOINS),
      av('Fusion sans regard humain. On saute la revue, alors ?', REV, MOINS),
      av('L\'agent fusionne seul. Je vais lire le journal de production tous les matins.', trait('sre', 'mesure', 'securite')),
      av('Fusion automatique ! Plus personne entre moi et la production.', qui('mathieu', 'laurence'), PLUS),
      av('Fusion automatique. Je charge mon téléphone de garde.', qui('benoit'), MOINS),
      ap('Ça a fusionné pendant que je dînais. Je lirai plus tard ce que c\'était.', LIC),
      ap('Personne n\'a lu ce qui est parti en production hier. Pas même moi.', LIC, MOINS),
      ap('Le code va en production sans moi. Quand ça brise, ce sera avec moi.', SENIORS),
      ap('Mes agents livrent tout seuls. Je ne sais plus ce qu\'il y a sur nos sites.', qui('laurence')),
      ap('Fusion automatique. Ship it, littéralement.', qui('mathieu'), PLUS),
      ap('Ma pile a fondu. Le code entre dans le Portail sans que personne le lise.', qui('nadia'), MOINS),
      ap('Un agent qui fusionne seul dans le firmware. Je ne dors plus.', trait('prudent', 'firmware'), MOINS),
      ap('Je ne reconnais plus le module que j\'ai écrit en janvier.', LIC, si(function (etat, q, dev) { return num(dev.comprehension, 1) < 0.65; })),
    ],
  };

  L.reviseurs = {
    0: [
      av('Plus de revue ? Ça va aller vite. Trop vite ?'),
      av('Plus de revue. Je ne sais pas si je dois être soulagée.', qui('nadia')),
      ap('Personne n\'a relu ma PR. Elle est en production. Bon.', MOINS),
      ap('Le délai a fondu. Les bogues de la semaine, eux, sont passés tout droit.'),
      ap('Je relis mes propres PR deux fois. Personne d\'autre ne le fera.', trait('rigoureux', 'tdd', 'prudent')),
      ap('Plus de file, plus d\'attente. Le rêve.', qui('mathieu'), PLUS),
      ap('Zéro réviseur. Mes tests sont les seuls à lire le code.', qui('sophie'), MOINS),
      ap('Personne ne relit les juniors. Ils apprennent tout seuls, et ça se voit.', SENIORS),
    ],
    1: [
      av('Un réviseur par PR. Le compromis classique.'),
      ap('Une revue par PR : la file avance, et quelqu\'un a lu.'),
      ap('Une seule revue par PR. Je respire un peu.', REV, PLUS),
      ap('Ma PR a été relue en une journée. Avec deux commentaires utiles.', JUNIORS, PLUS),
      ap('Un seul regard sur le firmware ? Je demande quand même une deuxième lecture.', qui('amira')),
      ap('Un réviseur, c\'est souvent moi. Je fais avec.', qui('nadia')),
    ],
    2: [
      av('Deux réviseurs ? Ma PR va attendre.', si(pasReviseur), MOINS),
      av('Deux réviseurs par PR. Plus de regards, plus d\'attente.'),
      av('Deux réviseurs : ma pile va doubler.', REV, MOINS),
      av('Deux réviseurs. Je réserve mes jeudis. Et mes vendredis.', qui('nadia'), MOINS),
      av('Deux réviseurs. Je vais devoir convaincre deux personnes, maintenant.', qui('mathieu'), MOINS),
      av('Deux réviseurs. Le capteur va me remercier.', qui('amira', 'luc'), PLUS),
      ap('Ma PR attend son deuxième réviseur depuis mardi.', MOINS),
      ap('Deux revues par PR : ma pile a doublé, et je la lis moins bien.', REV, MOINS),
      ap('Le deuxième réviseur a trouvé ce que le premier avait manqué. Une fois.'),
      ap('Le délai de livraison a monté depuis vendredi. Les PR attendent leur deuxième lecture.'),
      ap('Je relis le jeudi, le vendredi et le samedi. Deux réviseurs, c\'est moi et moi.', qui('nadia'), MOINS),
      ap('Deux paires d\'yeux sur ma PR. J\'ai appris deux choses.', JUNIORS, PLUS),
    ],
  };

  L.approbation = {
    auto: [
      av('L\'IA va approuver le code de l\'IA ? Pratique.'),
      av('Approbation automatique : ma PR ne va plus attendre personne.', si(pasReviseur)),
      av('Approbation par l\'IA. Je vais garder un œil sur ce qu\'elle laisse passer.', REV),
      av('Le même modèle écrit et approuve. Deux miroirs face à face.', qui('sebastien'), MOINS),
      ap('Ma PR a été approuvée en quatre secondes. Aucun commentaire. Aucun.'),
      ap('L\'approbation automatique laisse passer ce que l\'agent ne voit pas. Il ne voit pas les mêmes choses que nous.', trait('rigoureux', 'securite', 'prudent'), MOINS),
      ap('Je ne suis plus le dernier mot. Je suis devenue une suggestion.', qui('nadia'), MOINS),
      ap('Approuvé avant même que j\'aie fini mon café.', qui('mathieu'), PLUS),
      ap('La file de revue a fondu. Les questions, elles, restent sans réponse.', REV),
    ],
    humain: [
      av('Un humain a le dernier mot. Comme avant.'),
      av('Retour à l\'approbation humaine. La file va reprendre du volume.', REV),
      ap('Un humain approuve. La file reprend du poids, et les PR, du sens.'),
      ap('Je relis avant de fusionner. On a retrouvé un bogue que l\'IA avait approuvé.', REV),
      ap('Ma PR attend un humain. Je préfère ça, finalement.', JUNIORS),
    ],
    proprietaire: [
      av('Le propriétaire du code approuve. Qui est propriétaire de quoi, au juste ?'),
      ap('Toutes les PR de mon module passent par moi. Ma pile a une ombre.', SENIORS, MOINS),
      ap('Tout ce qui touche l\'AS/400 passe par moi. Ça fait du monde.', qui('raymond')),
      ap('J\'attends le propriétaire du module. Il est en réunion. Depuis mardi.', MOINS),
      ap('Le propriétaire connaît le code. Ses commentaires valent trois revues.', JUNIORS, PLUS),
      ap('Le firmware, c\'est moi qui signe. Au moins, je sais ce qui entre.', qui('amira'), PLUS),
    ],
  };

  L.taillePR = {
    libre: [
      av('Taille libre ? Les grosses PR vont revenir.', REV, MOINS),
      ap('Une PR de 3 000 lignes ce matin. Je l\'ai survolée. J\'ai honte.', REV, MOINS),
      ap('Les PR grossissent. Les revues raccourcissent.'),
      ap('Plus de limite : mes agents livrent d\'un bloc.', qui('mathieu', 'laurence'), PLUS),
    ],
    400: [
      av('400 lignes au maximum. Je vais découper.'),
      av('Une limite de 400 lignes. Mes PR vont rapetisser, et se multiplier.'),
      av('400 lignes ? Mes imports en font 300.', qui('mathieu'), MOINS),
      av('400 lignes. Je vais pouvoir lire jusqu\'au bout, pour une fois.', qui('nadia', 'amira'), PLUS),
      ap('Mes PR font 400 lignes. Elles sont lues au complet, pour une fois.', PLUS),
      ap('Des PR que je peux lire d\'un bout à l\'autre. J\'avais oublié le sentiment.', REV, PLUS),
      ap('Plus de PR, plus petites. Le délai monte un peu, les surprises baissent.'),
      ap('J\'ai découpé ma PR en dix. Dix titres à inventer.', qui('mathieu'), MOINS),
      ap('Mes agents découpent tout seuls. Ça fait beaucoup de PR.', qui('laurence')),
      ap('Une PR de 400 lignes, avec son test. Elle est passée du premier coup.', qui('sophie', 'karim'), PLUS),
    ],
    200: [
      av('200 lignes ? Je vais passer ma vie à ouvrir des PR.', MOINS),
      av('200 lignes au maximum. Ça va faire beaucoup de petites PR.'),
      av('200 lignes : je vais tout relire. Tout.', REV, PLUS),
      ap('Mes PR font 200 lignes. On me les relit en cinq minutes.', PLUS),
      ap('Quarante petites PR par jour. Chacune est lisible ; ensemble, ça fait beaucoup.', REV),
      ap('Je trouve les bogues dès la première lecture. Les petites PR, c\'est magique.', qui('nadia'), PLUS),
      ap('200 lignes. Mon import en fait 300. Je découpe l\'import ?', qui('mathieu'), MOINS),
      ap('Une idée, cinq PR. J\'ai du mal à suivre le fil de ma propre fonction.'),
    ],
  };

  L.ci = {
    aucune: [
      av('Plus de CI ? On compile chez soi, alors.', MOINS),
      ap('Un bogue de compilation en production. Je ne savais pas que c\'était possible.', MOINS),
      ap('Sans CI, je n\'ai plus aucun chiffre sur nos changements.', trait('sre', 'mesure'), MOINS),
      ap('Plus d\'attente de pipeline. Plus de filet non plus.'),
    ],
    build: [
      av('Au moins, ça va compiler.'),
      ap('La CI vérifie que ça compile. Pas que ça marche.'),
      ap('Une PR rouge ce matin : l\'agent avait importé un module qui n\'existe pas. Attrapé.', LIC),
    ],
    tests: [
      av('La CI va lancer les tests à chaque PR. Mes agents vont devoir attendre.', LIC),
      av('Les tests à chaque PR. Mes bogues vont se faire prendre avant la revue.'),
      av('La CI teste tout. Enfin, ce qu\'on a écrit comme tests.', trait('tdd', 'rigoureux', 'mesure')),
      ap('La CI a attrapé mon bogue avant la revue. Merci, la CI.', PLUS),
      ap('Mon agent attend la CI et relance en attendant. Ça fait des jetons pour rien.', si(autoMin(2)), MOINS),
      ap('L\'agent a désactivé un test pour que la CI passe. Je l\'ai vu par hasard.', si(function (etat, q, dev) { return autoMin(3)(etat, q, dev) && lv(etat, q, 'lint') !== 'verrouille'; }), MOINS),
      ap('Les PR arrivent déjà vertes. En revue, je peux parler de logique.', REV, PLUS),
      ap('Mes tests roulent à chaque PR. Je souris en dedans.', qui('sophie'), PLUS),
    ],
    complet: [
      av('Pipeline complet : on va déployer pour vrai, et souvent.'),
      av('Déploiement continu. Plus de mise en production le vendredi soir.', qui('benoit', 'patrick'), PLUS),
      av('Un pipeline complet. Ça va être plus long, mais ennuyant.', trait('sre', 'garde', 'stoique')),
      ap('Trois déploiements hier. Je ne m\'en suis presque pas rendu compte.', PLUS),
      ap('Un déploiement un mardi après-midi, et personne n\'a eu peur.', qui('benoit'), PLUS),
      ap('Plus de FTP. Mes 14 sites se déploient tout seuls.', qui('patrick'), PLUS),
      ap('Le pipeline est plus long. Mon agent tourne en rond en l\'attendant, au compteur.', si(autoMin(2))),
      ap('Je vois enfin combien de changements cassent. Le taux d\'échec a un chiffre.', trait('sre', 'mesure'), PLUS),
    ],
  };

  L.lint = {
    aucun: [
      av('Plus de lint ? Préparez-vous aux débats sur les virgules.', REV),
      ap('En revue, on se chicane sur des espaces. Encore.', MOINS),
      ap('Chaque agent a son style. Le code de l\'îlot ressemble à une courtepointe.'),
    ],
    formatage: [
      ap('Le formatage est automatique. Plus personne ne parle d\'espaces.', PLUS),
      ap('Le formateur a réécrit tout le fichier. Le diff fait 900 lignes pour rien.', REV),
    ],
    strict: [
      av('Lint strict : l\'agent va devoir se tenir droit.'),
      av('Lint strict. Les revues vont enfin parler de logique.', REV, PLUS),
      ap('Le lint attrape des erreurs bêtes avant la revue. Bien.', PLUS),
      ap('L\'agent a ajouté un commentaire pour ignorer la règle. Il est débrouillard.', si(autoMin(3)), MOINS),
      ap('Le lint refuse mon CSS. Je l\'écoute, à moitié.', qui('chloe')),
      ap('Strict, mais pas verrouillé. Je surveille les « disable » dans les PR.', qui('sebastien', 'sophie')),
    ],
    verrouille: [
      av('Verrouillé : l\'agent ne pourra plus faire taire le lint.'),
      av('Lint verrouillé. Plus de « disable » glissé en douce.', trait('securite', 'tdd', 'rigoureux'), PLUS),
      av('Lint verrouillé ? Ma PR de 4 000 lignes va rougir.', qui('mathieu'), MOINS),
      ap('L\'agent ne peut plus désactiver la règle. Il corrige son code, et ça prend plus de temps.', LIC),
      ap('Plus un seul commentaire sur le style en revue. On parle enfin de logique.', REV, PLUS),
      ap('Les règles de sécurité ne se désactivent plus. Je dors mieux.', qui('sebastien'), PLUS),
      ap('Le lint verrouillé bloque ma PR. Il a raison, en plus.', qui('mathieu')),
      ap('Mon agent a tourné en rond une heure sur une règle. Il a fini par la respecter.', si(autoMin(2))),
    ],
  };

  L.tests = {
    aucun: [
      av('Plus de tests exigés ? Je vais en écrire pareil.', trait('rigoureux', 'tdd', 'prudent')),
      av('Pas de tests. On va aller vite. Jusqu\'à quand ?'),
      ap('Pas de tests, alors on teste en production. Les clients sont patients.', MOINS),
      ap('La CI est verte parce qu\'elle n\'a rien à vérifier.', trait('sre', 'mesure', 'rigoureux')),
      ap('Un bogue est revenu. Le même qu\'en mars. Rien ne l\'empêchait de revenir.', MOINS),
    ],
    unitaires: [
      av('Des tests unitaires exigés. L\'agent va en écrire des tonnes.'),
      av('Des tests unitaires obligatoires. Je vais lire ceux de l\'agent.', trait('tdd', 'rigoureux')),
      ap('La couverture affichée monte vite. L\'agent écrit un test par fonction.'),
      ap('Ce test vérifie que la fonction retourne ce qu\'elle retourne. Bravo.', trait('tdd', 'rigoureux'), MOINS),
      ap('J\'ai écrit mes tests unitaires à la main. J\'ai compris ma fonction en les écrivant.', JUNIORS, PLUS),
    ],
    integration: [
      av('Des tests d\'intégration : on va tester les morceaux ensemble.'),
      ap('Un test d\'intégration a attrapé un bogue entre deux modules. Personne ne l\'aurait vu.', PLUS),
      ap('Je teste la synchronisation d\'un bout à l\'autre, en mode avion.', qui('karim'), PLUS),
      ap('La CI est plus lente avec les tests d\'intégration. Mais plus honnête.'),
    ],
    mutation: [
      av('Des tests de mutation ? On va savoir si nos tests mordent.'),
      av('Des tests de mutation. Ça va piquer, la couverture affichée.', trait('mesure', 'sre')),
      av('Des tests de mutation ! Je sors le gâteau.', qui('sophie'), PLUS),
      ap('Les mutants ont survécu à la moitié de nos tests. On les réécrit.'),
      ap('La couverture affichée a baissé, la vraie a monté. Le tableau de bord est moins beau, et plus vrai.', trait('mesure', 'sre', 'tdd', 'rigoureux')),
      ap('Mes tests générés ne vérifiaient… rien. Les mutants me l\'ont dit.', trait('natifIA'), MOINS),
      ap('Des mutants dans le firmware. Aucun n\'a survécu. Je suis fière.', qui('amira'), PLUS),
      ap('Mes tests de 2019 tuent 90 % des mutants. Ceux de l\'agent, 40 %.', qui('sophie')),
    ],
  };

  L.revuesExplication = {
    non: [
      ap('Plus de revues d\'explication. Je ne sais pas si c\'est un soulagement.'),
    ],
    mensuelles: [
      av('Une fois par mois, j\'explique mon code sans l\'IA. Je commence à réviser.'),
      ap('J\'ai expliqué mon module au tableau. J\'ai trouvé un bogue en parlant.', PLUS),
      ap('Expliquer mon code m\'a montré tout ce que je ne comprenais pas. Il y en avait.', JUNIORS),
      ap('J\'ai expliqué une PR de mon agent. J\'ai improvisé. Ça s\'est vu.', qui('mathieu', 'laurence'), MOINS),
      ap('Ils comprennent mieux leur code qu\'on pensait. Et moins bien qu\'ils pensaient.', SENIORS),
    ],
    majeures: [
      av('À chaque PR majeure ? Ça va prendre du temps.', MOINS),
      ap('Chaque PR majeure passe au tableau blanc. Le délai monte, la compréhension aussi.'),
      ap('Les revues sont plus longues, mais je sais enfin ce que l\'auteur voulait.', REV, PLUS),
      ap('Expliquer sans l\'agent. J\'ai transpiré. J\'ai appris.', qui('jeremy')),
      ap('Je pose une question, une seule : « Pourquoi ? » Ça suffit.', qui('sophie', 'raymond')),
    ],
  };

  L.pair = {
    aucun: [
      ap('Chacun son écran, chacun son code.'),
      ap('Plus de binôme. Personne ne me demande de m\'expliquer.', JUNIORS, MOINS),
    ],
    humain: [
      av('Un binôme humain ? On va aller moins vite, à deux.'),
      ap('À deux devant le même écran. On livre moins, on comprend tout.', PLUS),
      ap('Mon binôme m\'explique pourquoi. L\'agent m\'expliquait comment.', JUNIORS, PLUS),
      ap('Mon binôme me demande d\'expliquer chaque ligne. Toutes.', trait('dixX', 'shipIt'), MOINS),
      ap('J\'explique en codant. Je découvre ce que je sais.', SENIORS),
      ap('Benoît me laisse le clavier. Il ne dit rien, il pointe.', qui('tommy'), PLUS),
    ],
    ia: [
      av('Humain avec l\'IA : c\'est ce qu\'on fait déjà, non ?', LIC),
      ap('Je code en duo avec l\'agent. Ça va vite. Je tiens le volant, je pense.', LIC),
      ap('L\'agent est mon binôme. Il ne me pose jamais de questions.', JUNIORS, MOINS),
      ap('Un binôme avec l\'IA, sans licence ? J\'ai un binôme imaginaire.', SANSLIC, MOINS),
    ],
    rotation: [
      av('Une semaine à deux, une semaine avec l\'agent. Pourquoi pas.'),
      ap('Semaine humaine : j\'ai appris. Semaine IA : j\'ai livré.'),
      ap('La rotation me force à expliquer une semaine sur deux. Ça me garde honnête.', SENIORS),
    ],
  };

  var sessionsBeaucoup = [
    av('Six sessions ? Personne ne surveille six agents à la fois.', MOINS),
    av('Plus de sessions ! Il me faut des écrans.', trait('multiClauding', 'dixX', 'enthousiaste'), PLUS),
    ap('Je ne sais plus quelle session a écrit quoi.', si(function (etat, q, dev) { return num(dev.sessions, 0) >= 4; }), MOINS),
    ap('Quatre agents tournent. Je relis leurs PR en diagonale.', si(function (etat, q, dev) { return num(dev.sessions, 0) >= 4; })),
    ap('Les jetons ont bondi depuis vendredi. Les sessions coûtent, même quand elles attendent.', LIC),
    ap('Je garde une session. On ne m\'a pas obligé à en ouvrir six.', trait('prudent', 'sceptique', 'refractaire', 'tdd')),
    ap('Ma pile de revue a pris un étage depuis vendredi. Tout le monde a plus d\'agents.', REV, MOINS),
    ap('Six agents, un casque. Je me sens comme un chef d\'orchestre sourd.', qui('mathieu')),
  ];
  L.sessionsMax = {
    1: [
      av('Une session à la fois. On revient au calme.'),
      av('Une seule session ? Mes autres écrans vont s\'ennuyer.', trait('multiClauding', 'dixX'), MOINS),
      ap('Une session. Je finis ce que je commence.', LIC, PLUS),
      ap('Une seule session. Mes autres écrans s\'ennuient.', trait('multiClauding', 'dixX'), MOINS),
      ap('Moins de jetons, moins de bruit. Moins de code aussi.', LIC),
    ],
    2: [
      av('Deux sessions. Je vais essayer d\'en suivre deux.', LIC),
      ap('Deux agents en parallèle. Je suis encore capable de suivre.', LIC),
      ap('La deuxième session rapporte moins que la première. Mais elle rapporte.', trait('mesure', 'sre', 'pragmatique')),
    ],
    3: [
      av('Trois sessions : on va voir combien j\'en suis vraiment.', LIC),
      ap('Trois agents, trois fils. Je perds le fil du troisième.', LIC),
      ap('La troisième session rapporte moins que la deuxième. Elle fatigue autant.', LIC, MOINS),
      ap('Plus de PR dans la file depuis que chacun a trois agents.', REV, MOINS),
    ],
    4: sessionsBeaucoup, 5: sessionsBeaucoup, 6: sessionsBeaucoup,
  };

  L.modele = {
    economique: [
      av('Le modèle économique ? Régis doit être content.', LIC),
      av('Le modèle le moins cher. On va voir ce qu\'il invente.', LIC, trait('securite', 'prudent', 'rigoureux')),
      ap('L\'agent a inventé une fonction qui n\'existe pas. Avec beaucoup d\'assurance.', LIC, MOINS),
      ap('Le modèle économique écrit plus de lignes pour faire la même chose.', LIC),
      ap('Il a proposé un paquet npm qui n\'existe pas. Pas encore.', qui('sebastien'), MOINS),
      ap('Il confond les registres du microcontrôleur. Je corrige.', trait('firmware', 'electronicien'), MOINS),
      ap('Plus de petites erreurs en revue depuis le changement de modèle.', REV, MOINS),
    ],
    intermediaire: [
      ap('Retour au modèle intermédiaire. Ni génial, ni gênant.', LIC),
      ap('Le modèle de base. Il fait le travail, sans éclat.', LIC),
    ],
    haut: [
      av('Le haut de gamme ! Régis va faire une drôle de face.', LIC, PLUS),
      ap('Le haut de gamme se trompe moins. Mes PR sont plus courtes.', LIC, PLUS),
      ap('Moins d\'erreurs bêtes en revue. Les jetons coûtent plus, les relectures moins.', REV, PLUS),
      ap('Il a compris l\'architecture du premier coup. J\'ai un petit pincement.', LIC, SENIORS),
      ap('Le haut de gamme a trouvé le cas limite avant moi. Ça me vexe et ça me rassure.', qui('amira', 'sophie')),
    ],
    local: [
      av('Le modèle local, sur le serveur GPU ? Il va falloir être patient.', LIC),
      ap('Le modèle local est moins brillant. Mais il ne parle à personne.', LIC),
      ap('Les photos des inspecteurs restent chez nous. Je respire.', qui('karim')),
      ap('Ma télémétrie ne sort plus de la bâtisse. J\'aime ça.', qui('hugo'), PLUS),
      ap('Presque plus de jetons à payer. Un peu plus d\'erreurs à corriger.', LIC),
      ap('Données qui ne voyagent pas, voyant « privé » moins inquiétant.', qui('sebastien'), PLUS),
    ],
  };

  L.modeleParTache = {
    'true': [
      av('Un petit modèle pour les tâches simples ? Ça se tient.', LIC),
      ap('Le petit modèle écrit mes messages de commit. Le gros, mon architecture.', LIC),
      ap('La facture de jetons a baissé, et je n\'ai rien vu changer dans mon code.', LIC, PLUS),
      ap('Le coût par tâche a baissé du tiers. Je l\'ai mesuré, évidemment.', qui('marieeve'), PLUS),
      ap('Le petit modèle a écrit ma documentation. Elle est courte. Tant mieux.', qui('olivier', 'patrick')),
    ],
    'false': [
      ap('Un seul modèle pour tout. Plus simple, plus cher.', LIC),
    ],
  };

  var formationPrevue = [
    E('La formation commence bientôt. On m\'a réservé une demi-journée par semaine.', si(prepare('formation'))),
    E('Je suis en salle une partie de la semaine. Le travail attend.', si(prepare('formation'))),
    E('Une formation ! J\'ai déjà acheté un cahier.', qui('tommy', 'luc'), si(prepare('formation')), PLUS),
    E('Une formation sur l\'IA. Je pourrais la donner, mais bon.', qui('mathieu'), si(prepare('formation'))),
  ];
  function formationEnVigueur(v, lignes) {
    return formationPrevue.concat(lignes.map(function (x) { return fusion([x, si(enVigueur('formation', v))], {}); }));
  }
  L.formation = {
    aucune: [
      ap('Plus de formation. On apprendra à coups d\'erreurs.', MOINS),
    ],
    atelier: formationEnVigueur('atelier', [
      E('L\'atelier m\'a appris à découper mes demandes. L\'agent se trompe moins.'),
      E('Un atelier, c\'est un début. J\'ai oublié la moitié. Il reste l\'autre.'),
    ]),
    programme: formationEnVigueur('programme', [
      E('Depuis la formation, j\'écris de meilleures demandes. L\'agent se trompe moins.', LIC, PLUS),
      E('Je reconnais une injection de prompt, maintenant. Ça fait un peu peur.'),
      E('J\'applique ce qu\'on a vu : petites tâches, contexte clair, je relis tout.', LIC),
      E('En formation, on a codé sans l\'agent. J\'ai survécu.', qui('jeremy')),
    ]),
    mentorat: formationEnVigueur('mentorat', [
      E('Ginette m\'a montré comment lui parler. Il a compris ma macro. Hein.', qui('diane'), PLUS),
      E('Mon mentor me fait expliquer chaque demande avant de l\'envoyer.', JUNIORS, PLUS),
      E('J\'ai un mentoré. Je découvre tout ce que je fais sans y penser.', SENIORS),
      E('Depuis la formation, j\'écris de meilleures demandes. L\'agent se trompe moins.', LIC, PLUS),
    ]),
  };

  L.plafondJetons = {
    aucun: [
      ap('Pas de plafond de jetons. Les agents peuvent tourner toute la nuit.', LIC),
    ],
    souple: [
      av('Une alerte quand on dépasse. Je vais la recevoir, je le sens.', trait('multiClauding', 'dixX')),
      ap('Alerte de jetons à 15 h. Je regarde ce que mon agent fabrique.', LIC),
      ap('Le plafond souple ne coupe rien. Il prévient, et les boucles raccourcissent.', LIC),
      ap('Les alertes de jetons, je les compile. Une par îlot, par semaine.', qui('marieeve')),
    ],
    strict: [
      av('Un plafond strict ? Mon agent va s\'arrêter au milieu d\'une phrase.', LIC, MOINS),
      av('Un plafond strict. Plus de boucle d\'agent toute la fin de semaine.', trait('mesure', 'sre', 'stoique'), PLUS),
      ap('L\'agent s\'est arrêté au milieu de la tâche : plafond atteint. Je finis à la main.', si(plafondAtteint), MOINS),
      ap('Je découpe mes tâches pour ne pas frapper le plafond.', LIC),
      ap('Plus de boucle de fin de semaine. Plus de grosse tâche non plus.', LIC),
      ap('Mes sessions se coupent le jeudi. Le vendredi, je code à la main.', trait('multiClauding', 'dixX'), MOINS),
    ],
  };

  L.disciplineNon = {
    laxiste: [
      av('On dit oui à tout ? Le carnet va déborder.', MOINS),
      av('Oui à tout. La direction va nous aimer. Pour un temps.'),
      ap('On a dit oui à trois nouvelles fonctions. Personne ne sait qui va s\'en servir.'),
      ap('Le volume de code monte. La valeur livrée, pas autant.', trait('mesure', 'sre', 'pragmatique')),
      ap('La direction est contente cette semaine. Le carnet, lui, a doublé.'),
    ],
    normale: [
      ap('On trie les demandes comme avant. Ni oui à tout, ni non à tout.'),
    ],
    ferme: [
      av('Dire non ? La direction va bouder.'),
      av('Discipline ferme : on va enfin trier les demandes.', SENIORS, PLUS),
      ap('On a refusé deux demandes cette semaine. On a livré celles qui comptaient.', PLUS),
      ap('Moins de code à écrire, moins de code à entretenir.', SENIORS, PLUS),
      ap('La direction boude. Le carnet, lui, respire.', si(function (etat, q, dev) { var il = ilotDe(etat, dev); return !!il && num(il.satisfaction, 1) < 0.5; })),
      ap('La version montre-bracelet ? On a dit non. Poliment.', qui('karim'), PLUS),
    ],
  };

  L.specs = {
    non: [
      ap('Plus de specs. L\'agent devine ce qu\'on veut.', MOINS),
    ],
    legeres: [
      av('Une page de spec avant de coder. Je peux vivre avec ça.'),
      av('Des specs légères. L\'agent va enfin savoir quoi faire.', LIC),
      ap('Une page de spec, et l\'agent a fait la bonne chose du premier coup.', PLUS),
      ap('La spec a révélé que la direction voulait autre chose. Avant le code, pour une fois.'),
      ap('Ma spec d\'une page est devenue mes tests. Pratique.', qui('sophie'), PLUS),
    ],
    obligatoires: [
      av('Des specs obligatoires ? Le départ va être lent.', MOINS),
      ap('Les specs ralentissent le départ. Mais on ne refait plus le travail.'),
      ap('Trois pages de spec pour 400 lignes. Le monde est fou.', trait('shipIt', 'dixX'), MOINS),
      ap('J\'écris ce que je veux avant de le demander. L\'agent n\'invente presque plus.', LIC, PLUS),
    ],
  };

  L.accesProd = {
    oui: [
      av('Les agents gardent l\'écriture en production ? Courageux.', trait('securite', 'sre', 'garde')),
      ap('L\'agent peut écrire en production. Je vérifie deux fois ce qu\'il fait.'),
      ap('Un agent avec des droits d\'écriture en prod. Je relis les journaux de la base tous les matins.', qui('benoit', 'marieeve'), MOINS),
    ],
    lecture: [
      av('Lecture seule en prod. L\'agent regarde, il ne touche pas.'),
      av('Lecture seule. Plus d\'UPDATE surprise, j\'espère.', trait('securite', 'garde', 'sre'), PLUS),
      ap('L\'agent lit les journaux de prod et trouve vite. Il ne peut rien casser.', PLUS),
      ap('L\'agent diagnostique en lecture seule, je corrige. Je dors un peu mieux.', qui('benoit'), PLUS),
      ap('Lecture seule : l\'agent m\'a demandé d\'exécuter son UPDATE. J\'ai lu le WHERE.', LIC),
    ],
    non: [
      av('Plus d\'accès à la prod pour les agents ? Les diagnostics vont être longs.'),
      ap('Je copie les journaux de prod à la main pour l\'agent. Ça prend du temps.', LIC, MOINS),
      ap('Diagnostic à l\'ancienne : l\'agent n\'a plus accès. On cherche nous-mêmes.', si(function (etat, q, dev) { return incidents(etat, q, dev.ilot).length > 0; })),
      ap('Rien ne casse en prod à cause d\'un agent. Les nuits sont plus longues, par contre.', qui('benoit')),
    ],
  };

  L.bacASable = {
    aucun: [
      ap('Pas de bac à sable : l\'agent voit tout ce que ma machine voit.', trait('securite', 'prudent'), MOINS),
    ],
    conteneur: [
      av('Un conteneur pour l\'agent. Il va être à l\'étroit.'),
      ap('L\'agent travaille dans sa boîte. Ce qui casse reste dedans.', PLUS),
      ap('Le conteneur démarre lentement le matin. Moi aussi.', LIC),
      ap('Un conteneur, mais Internet est encore ouvert. Je garde un œil.', qui('sebastien')),
    ],
    filtre: [
      av('Réseau filtré : l\'agent ne parlera qu\'aux adresses permises.'),
      av('Réseau filtré ? Il me faut la liste des adresses dont mes outils ont besoin.', trait('curieux', 'integrateur', 'donnees')),
      ap('L\'agent ne peut plus joindre Internet. Il demande la permission pour chaque paquet.', LIC),
      ap('Le réseau filtré a bloqué un outil que je voulais essayer. Il avait peut-être raison.', trait('curieux', 'integrateur', 'automatiseur')),
      ap('Un paquet inventé par l\'agent ? Introuvable : le miroir interne ne le connaît pas.', LIC, PLUS),
      ap('Une injection qui veut envoyer nos données ailleurs ? Ailleurs, c\'est bloqué.', qui('sebastien'), PLUS),
    ],
  };

  L.approbationSorties = {
    'true': [
      av('Un humain confirme chaque sortie. Il va falloir cliquer.'),
      av('Approbation des sorties. Une branche de la triade de moins.', trait('securite', 'donnees'), PLUS),
      ap('J\'ai approuvé onze courriels de l\'agent ce matin. J\'en ai lu dix.', LIC),
      ap('Une requête vers l\'extérieur attendait mon clic. Je ne savais même pas qu\'il en faisait.', LIC),
      ap('Les trois voyants de l\'îlot sont allumés, mais rien ne sort sans nous.', si(triade), PLUS),
      ap('Mes flux attendent un clic à chaque envoi. Les remboursements prennent une heure de plus.', qui('steve'), MOINS),
    ],
    'false': [
      ap('Plus d\'approbation des sorties. L\'agent envoie ce qu\'il veut.', MOINS),
    ],
  };

  L.politiqueRegistre = {
    libre: [
      ap('Libre-service au registre. Tout le monde installe tout.'),
      ap('J\'ai trouvé trois MCP sur un forum. Je les essaie lundi.', trait('curieux', 'integrateur'), PLUS),
    ],
    listeBlanche: [
      av('Une liste blanche ? Plus de MCP de forum.'),
      ap('J\'ai demandé l\'ajout d\'une skill. Elle attend son approbation.'),
      ap('Je ne peux plus installer n\'importe quoi. Je lis le code des MCP, maintenant.', trait('curieux', 'integrateur', 'automatiseur')),
      ap('La liste blanche a refusé un MCP sans auteur connu. Bien.', qui('sebastien'), PLUS),
    ],
    proprietaires: [
      av('Chaque carte aura un propriétaire. Je risque d\'en adopter une.'),
      ap('Je suis propriétaire d\'une skill. Je dois la tenir à jour, maintenant.', SENIORS),
      ap('Les cartes orphelines sont apparues au registre. Il y en avait plus que je pensais.'),
      ap('Mes MCP ont chacun un nom à côté. Le mien, souvent.', qui('yannick', 'steve')),
    ],
  };

  L.passerelleMCP = {
    'true': [
      E('La passerelle MCP arrive dans deux semaines. Mes outils vont passer au péage.', si(prepare('passerelleMCP'))),
      E('Chaque appel d\'outil est journalisé. L\'agent est un peu plus lent, et beaucoup plus visible.', si(enVigueur('passerelleMCP', true))),
      E('J\'ai lu les journaux de la passerelle. Un MCP a changé sa description hier.', qui('sebastien'), si(enVigueur('passerelleMCP', true))),
      E('La passerelle a bloqué un outil apparu cette nuit. Bon réflexe.', si(enVigueur('passerelleMCP', true)), PLUS),
      E('Mes MCP passent par la passerelle. Ils ont l\'air moins fringants.', qui('yannick', 'steve'), si(enVigueur('passerelleMCP', true))),
    ],
    'false': [
      ap('Plus de passerelle. On ne saura plus qui appelle quoi.', MOINS),
    ],
  };

  L.politiqueNonTI = {
    interdire: [
      av('Interdire le développement hors TI ? Kevin va se cacher.'),
      ap('Kevin ne me montre plus ses applications. Il en fait encore.', MOINS),
      ap('Une direction m\'a demandé un « petit outil » en cachette. Je ne sais pas quoi répondre.'),
    ],
    ignorer: [
      ap('On ignore ce que fait le marketing. Le marketing aussi.'),
    ],
    encadrer: [
      av('On va accompagner Kevin. Quelqu\'un doit lui parler de sauvegardes.'),
      ap('J\'ai aidé Kevin à brancher son application sur le registre approuvé. Il était content.', qui('steve', 'patrick'), PLUS),
      ap('Les applications hors TI sortent de l\'ombre, une par une.', PLUS),
      ap('Kevin a classé ses données. Il a trouvé des NAS. Il a appelé Me Bérubé tout seul.', qui('sebastien')),
    ],
  };

  L.observabilite = {
    aucune: [
      ap('Plus d\'observabilité. On revient au flair.', MOINS),
    ],
    base: [
      E('L\'observabilité arrive dans trois semaines. Je prépare mes questions.', si(prepare('observabilite'))),
      E('Le MTTR est un vrai chiffre, maintenant. Il n\'est pas beau.', si(enVigueur('observabilite', 'base'))),
      E('Des tableaux de bord de base. Je vais peut-être dormir.', qui('benoit'), si(enVigueur('observabilite', 'base')), PLUS),
    ],
    complete: [
      E('L\'observabilité complète arrive. Ce qui était caché va se voir.', si(prepare('observabilite'))),
      E('Le tableau de bord montre la compréhension réelle. Ouch.', si(enVigueur('observabilite', 'complete')), MOINS),
      E('La couverture réelle est enfin affichée. Elle est plus basse que l\'autre.', si(enVigueur('observabilite', 'complete'))),
      E('Enfin des chiffres que je peux croire. Je les affiche dans le corridor.', qui('marieeve'), si(enVigueur('observabilite', 'complete')), PLUS),
      E('Je vois exactement où l\'incident a commencé. À 3 h, ça change tout.', qui('benoit', 'tommy'), si(enVigueur('observabilite', 'complete')), PLUS),
    ],
  };

  L.objectif = {
    aucun: [
      ap('Plus d\'objectif chiffré. On peut recommencer à faire ce qui a du sens.', PLUS),
    ],
    pr: [
      av('Un objectif en nombre de PR ? On va en faire, des PR.'),
      av('Le nombre de PR comme objectif. Je vais avoir de la lecture.', REV, MOINS),
      ap('Mes PR rapetissent pour faire monter le compteur. La valeur, elle, ne bouge pas.'),
      ap('Deux fois plus de PR à relire, pour le même travail. Merci, l\'objectif.', REV, MOINS),
      ap('Compter les PR, c\'est compter les virgules. Je le dis à qui veut l\'entendre.', qui('marieeve'), MOINS),
      ap('Objectif PR : je gagne. Je ne sais pas trop quoi.', qui('mathieu')),
    ],
    delai: [
      av('Un objectif de délai ? On va relire plus vite.'),
      ap('On relit plus vite pour faire baisser le délai. On relit moins bien, aussi.', REV, MOINS),
      ap('Le délai baisse. Les bogues, eux, montent un peu. C\'est lié.', trait('mesure', 'rigoureux')),
    ],
    satisfaction: [
      av('La satisfaction comme cible ? On va dire oui à tout.'),
      ap('On dit oui à tout pour garder la direction contente. Le carnet déborde.', MOINS),
    ],
    incidents: [
      av('Un objectif d\'incidents ? Ils vont baisser. Sur papier.'),
      ap('Un incident est devenu une « demande ». Il brûle quand même.', MOINS),
      ap('On ne déclare plus les petits incidents. Je les note dans mon carnet, au cas.', qui('benoit', 'marieeve')),
    ],
  };

  L.embauche = {
    gel: [
      av('Geler l\'embauche des juniors ? Qui va prendre la relève ?', SENIORS, MOINS),
      ap('Pas de nouveaux juniors. Personne de plus jeune que moi. Pour toujours ?', JUNIORS),
      ap('Personne derrière moi. Mes notes ne sont lues par personne.', SENIORS, MOINS),
      ap('Mon stage finit bientôt. On m\'a dit « gel ». Je comprends.', qui('jeremy'), MOINS),
    ],
    normale: [
      ap('Embauche normale. La relève arrivera à son rythme.'),
    ],
    releve: [
      av('Un programme de relève ! On va enfin transmettre.', SENIORS, PLUS),
      ap('J\'ai un junior qui me suit partout. Je lui explique pourquoi on fait les choses.', SENIORS, PLUS),
      ap('On me forme pour prendre la relève. C\'est sérieux, d\'un coup.', JUNIORS, PLUS),
      ap('Quelqu\'un apprend l\'AS/400. Volontairement. Je n\'en reviens pas.', qui('raymond'), PLUS),
    ],
  };

  var cataloguePrevu = [
    E('Le catalogue de données arrive dans un mois. Lucie a déjà son crayon.', si(prepare('catalogue'))),
    E('On m\'a demandé quelles tables je lis. Je ne les connais pas toutes.', si(prepare('catalogue'))),
  ];
  L.catalogue = {
    aucun: [
      ap('Plus de catalogue. Chacun sa liste, comme avant.', MOINS),
    ],
    inventaire: cataloguePrevu.concat([
      E('L\'inventaire montre trois copies de la même table. On s\'en doutait.', si(enVigueur('catalogue', 'inventaire'))),
      E('On sait ce qui existe. On ne sait pas encore ce qui est vrai.', qui('lucie', 'hugo'), si(enVigueur('catalogue', 'inventaire'))),
    ]),
    sources: cataloguePrevu.concat([
      E('L\'agent prend enfin la bonne liste de fournisseurs. La source officielle.', si(enVigueur('catalogue', 'sources')), PLUS),
      E('Une seule liste. Je peux mourir heureuse. Pas tout de suite.', qui('lucie'), si(enVigueur('catalogue', 'sources')), PLUS),
      E('Les seuils ont une source officielle. L\'agent cite le bon fichier.', qui('hugo', 'amira'), si(enVigueur('catalogue', 'sources')), PLUS),
    ]),
  };

  var classificationPrevue = [
    E('La classification commence dans deux semaines. On va étiqueter.', si(prepare('classification'))),
  ];
  L.classification = {
    non: [
      ap('Plus de classification. On ne saura plus ce qui sort.', MOINS),
    ],
    manuelle: classificationPrevue.concat([
      E('J\'ai étiqueté 300 colonnes à la main. Sensible, sensible, public.', si(enVigueur('classification', 'manuelle'))),
      E('Les photos des inspecteurs sont étiquetées. L\'agent ne les envoie plus dehors sans le dire.', qui('karim'), si(enVigueur('classification', 'manuelle')), PLUS),
    ]),
    auto: classificationPrevue.concat([
      E('La classification automatique a étiqueté la liste des pizzas « confidentiel ».', si(enVigueur('classification', 'auto'))),
      E('Classées automatiquement. Je vérifie les cas limites quand même.', trait('donnees', 'rigoureux'), si(enVigueur('classification', 'auto'))),
    ]),
  };

  // ---------------------------------------------------------------------------
  // Notes de service : ce qu'on dit en lisant (lundi et mardi de la semaine du déblocage)
  // ---------------------------------------------------------------------------
  var N = {
    licences: [
      E('Le groupe pilote, c\'est moi. Je vous en reparle vendredi.', PIL, PLUS),
      E('Sept licences. Je ne suis pas dans les sept.', NONPIL, MOINS),
      E('Un pilote avec des volontaires. Les volontaires sont déjà convaincus, non ?', trait('mesure', 'sre', 'tdd', 'sceptique')),
      E('Autocomplétion seulement. On commence par le plus petit pas. Sage.', SENIORS),
    ],
    autonomie: [
      E('Le clavardage, pour ceux qui ont une licence. On va pouvoir lui parler.', LIC, PLUS),
      E('Un niveau d\'autonomie par îlot. Le firmware et le site vitrine, ce n\'est pas pareil.', trait('prudent', 'firmware', 'securite')),
      E('Clavarder avec l\'assistant ? Moi, j\'ai encore mes collègues.', SANSLIC),
    ],
    reviseurs: [
      E('Combien de réviseurs ? Moi, je dis : au moins un, et qu\'il lise.', REV),
      E('Le nombre de réviseurs se règle par îlot. Le Portail va en vouloir deux.', qui('sebastien', 'nadia')),
      E('Moins de réviseurs, ma PR sort plus vite. Je dis ça de même.', trait('shipIt', 'dixX', 'enthousiaste')),
    ],
    tests: [
      E('On va parler de tests. Je sors mon cactus.', qui('sophie'), PLUS),
      E('L\'agent écrit les tests ? Qui vérifie que les tests vérifient ?', trait('rigoureux', 'tdd', 'mesure')),
      E('Des tests exigés. Je vais les faire écrire par l\'assistant.', LIC),
    ],
    'autonomie:2': [
      E('Le mode agent arrive. Il demande la permission à chaque étape, au moins.', LIC),
      E('Un agent qui enchaîne les étapes. Je vais lire chaque étape.', trait('prudent', 'rigoureux', 'firmware')),
      E('Le mode agent ! Mes écrans vont enfin servir.', trait('multiClauding', 'enthousiaste', 'natifIA'), LIC, PLUS),
      E('Le mode agent, pour les pilotes. Le reste de l\'îlot regarde.', SANSLIC, MOINS),
    ],
    ci: [
      E('Un vrai pipeline. Je prépare les chronomètres.', trait('sre', 'mesure'), PLUS),
      E('La CI va ralentir mes agents. Ils vont attendre, au compteur.', trait('dixX', 'shipIt', 'multiClauding')),
      E('Une marche à la fois. On commence par compiler.', SENIORS),
    ],
    lint: [
      E('Le lint, c\'est la revue qu\'on n\'a plus à faire.', REV, PLUS),
      E('Verrouillé ou pas ? Un agent pressé désactive ce qu\'il peut.', trait('securite', 'tdd', 'rigoureux')),
      E('Un lint pour du VBA ? Ça existe ?', qui('diane', 'raymond')),
    ],
    approbation: [
      E('Le dernier mot avant la fusion. Je veux que ce soit un humain.', trait('prudent', 'securite', 'rigoureux')),
      E('L\'IA qui approuve ? Ça irait plus vite.', trait('shipIt', 'dixX')),
      E('Propriétaire du code : ça veut dire que ma pile va grossir.', qui('nadia', 'raymond', 'amira'), MOINS),
    ],
    'autonomie:3': [
      E('Des agents qui ouvrent leurs PR. La file va déborder, je le sens.', REV, MOINS),
      E('Mon agent va ouvrir ses propres PR. Je vais le surveiller comme un stagiaire.', LIC, SENIORS),
      E('L\'agent ouvre les PR. Moi, j\'apprends quoi, si je ne les écris plus ?', JUNIORS),
    ],
    registre: [
      E('Le registre des skills ! J\'ai déjà des idées de recettes.', trait('curieux', 'integrateur', 'automatiseur'), PLUS),
      E('Une skill, c\'est du texte. Parfois avec des scripts. Qui les lit ?', qui('sebastien'), MOINS),
      E('Une skill pour garder ce qu\'on sait. Avant de partir, par exemple.', qui('olivier', 'raymond', 'diane')),
    ],
    taillePR: [
      E('Une limite de taille pour les PR. Je sais à qui on pense.', REV, PLUS),
      E('400 lignes, c\'est mes imports.', qui('mathieu'), MOINS),
      E('Des petites PR, c\'est plus de PR. Les deux à la fois.', trait('mesure', 'sre')),
    ],
    modele: [
      E('Le choix du modèle. Je regarde le coût par point livré, pas le prix du jeton.', trait('mesure', 'sre', 'pragmatique')),
      E('Un modèle local, sur le serveur GPU ? Les données resteraient ici.', trait('securite', 'donnees')),
      E('Le moins cher, Régis va le choisir. Je le connais.', SENIORS),
    ],
    sessionsMax: [
      E('Trois sessions en parallèle ! Enfin.', trait('multiClauding', 'dixX'), PLUS),
      E('Plusieurs sessions. La fatigue ne fait pas de rabais.', trait('stoique', 'prudent', 'mesure')),
      E('Trois agents en même temps ? Je n\'en surveille déjà pas un.', JUNIORS),
    ],
    plafondJetons: [
      E('Un plafond de jetons. Régis a vu le relevé, ça paraît.', LIC),
      E('Un plafond strict me couperait en plein élan.', trait('multiClauding', 'dixX'), MOINS),
      E('Un plafond souple, c\'est une alerte. Je la veux sur mon tableau de bord.', qui('marieeve')),
    ],
    formation: [
      E('Une formation ! Je réserve ma place.', trait('apprenant', 'enthousiaste'), PLUS),
      E('Le mentorat, avec Ginette ? Bon. Peut-être.', qui('diane')),
      E('Quatre semaines de préparation, et ensuite ça rapporte. On verra.', SENIORS),
    ],
    disciplineNon: [
      E('On a le droit de dire non ? Je l\'écris sur un post-it.', qui('patrick', 'karim', 'lucie'), PLUS),
      E('Dire non à la direction. On va voir qui ose.'),
      E('Le code qu\'on n\'écrit pas, on n\'a pas à l\'entretenir.', SENIORS),
    ],
    'autonomie:4': [
      E('Un agent qui fusionne seul. Pas dans mon îlot.', trait('prudent', 'firmware', 'garde', 'securite'), MOINS),
      E('Fusion automatique ! Plus besoin d\'attendre personne.', trait('shipIt', 'dixX'), PLUS),
      E('Il fusionne seul, et qui comprend ce qui est entré ?', REV),
    ],
    'sessionsMax:4,5,6': [
      E('Six sessions. Il me manque deux écrans.', qui('laurence'), PLUS),
      E('Six sessions par personne. Personne ne surveille six agents.', SENIORS, MOINS),
      E('Plus de sessions, plus de jetons. Régis va adorer.', LIC),
    ],
    specs: [
      E('Une spec, c\'est un test qui ne parle pas encore.', qui('sophie'), PLUS),
      E('Écrire avant de coder ? Comme un plan ?', JUNIORS),
      E('Une page. Pas un roman. Les directions n\'ont pas le temps.', trait('pragmatique', 'mobile')),
    ],
    accesProd: [
      E('Les agents écrivaient en prod ? Depuis janvier ?', trait('securite', 'sre', 'garde'), MOINS),
      E('Lecture seule, c\'est assez pour diagnostiquer. Je le sais, je fais la garde.', qui('benoit')),
      E('Un UPDATE sans WHERE, ça peut arriver à n\'importe qui. Surtout à un agent.', SENIORS),
    ],
    bacASable: [
      E('Un bac à sable. L\'injection reste dans la boîte.', qui('sebastien'), PLUS),
      E('Réseau filtré ? Mes outils de forum ne passeront plus.', trait('curieux', 'integrateur'), MOINS),
      E('Un conteneur pour l\'agent. Il était temps.', trait('prudent', 'securite')),
    ],
    approbationSorties: [
      E('Un humain qui confirme ce qui sort. Une branche de la triade de moins.', qui('sebastien', 'lucie')),
      E('Un clic par courriel ? Mes flux en envoient des milliers.', qui('steve'), MOINS),
      E('Les factures des fournisseurs ne pourront plus donner d\'ordres.', trait('donnees')),
    ],
    politiqueRegistre: [
      E('Une politique du registre. Mes MCP de forum sont en danger.', trait('curieux', 'integrateur', 'automatiseur'), MOINS),
      E('Un propriétaire par carte. Enfin quelqu\'un à qui poser la question.', qui('sebastien', 'marieeve'), PLUS),
      E('Qui a installé quoi ? Bonne question. Très bonne question.'),
    ],
    modeleParTache: [
      E('Un petit modèle pour la doc, un gros pour l\'architecture. Logique.', LIC),
      E('Moins un tiers sur la facture ? Je mesure avant d\'y croire.', qui('marieeve')),
      E('Le gros modèle pour le firmware, s\'il vous plaît.', qui('amira', 'luc')),
    ],
    classification: [
      E('Des étiquettes sur nos données. Je commence par les fournisseurs.', qui('lucie'), PLUS),
      E('Les photos des inspecteurs, c\'est sensible ? Oui. Oups.', qui('karim')),
      E('Les salaires sont sensibles depuis 2007. Je le sais.', qui('diane')),
    ],
    catalogue: [
      E('Un catalogue ! Les trois listes vont devoir se battre.', qui('lucie'), PLUS),
      E('La source officielle, c\'est l\'AS/400. Depuis 1994.', qui('raymond')),
      E('Une source officielle pour les seuils. Enfin.', qui('hugo')),
      E('Un catalogue de données. L\'agent va arrêter de prendre la première liste venue.', trait('donnees', 'rigoureux')),
    ],
    passerelleMCP: [
      E('Une passerelle : chaque appel d\'outil journalisé. Je vais lire ça au lit.', qui('sebastien'), PLUS),
      E('Mes MCP vont passer par un péage ?', trait('curieux', 'integrateur'), MOINS),
      E('Des journaux d\'outils. J\'en fais un tableau de bord.', qui('marieeve'), PLUS),
    ],
    politiqueNonTI: [
      E('Kevin a signé une note de service. On aura tout vu.', SENIORS),
      E('Kevin, on va t\'aider. Pour vrai.', qui('steve'), PLUS),
      E('Encadrer, c\'est mieux qu\'interdire. Interdire, ça ne marche jamais.', trait('pragmatique', 'securite')),
    ],
    observabilite: [
      E('Des métriques cachées vont apparaître. Je ne sais pas si je veux voir ça.', SENIORS),
      E('L\'observabilité ! Mes tableaux de bord vont avoir des amis.', qui('marieeve'), PLUS),
      E('Je vais savoir où ça casse avant que mon téléphone sonne.', qui('benoit', 'tommy'), PLUS),
    ],
    objectif: [
      E('Un seul chiffre pour la direction. On va tous le regarder, et seulement lui.', trait('mesure', 'sre', 'rigoureux'), MOINS),
      E('Si c\'est le nombre de PR, je gagne.', qui('mathieu'), PLUS),
      E('Quand une mesure devient une cible… Ginette me l\'a appris.', SENIORS),
    ],
    revuesExplication: [
      E('Expliquer mon code sans l\'agent ? Euh. Demain ?', trait('natifIA'), MOINS),
      E('Une revue d\'explication. J\'ai un tableau blanc et des questions.', qui('sophie', 'raymond'), PLUS),
      E('Lequel de mes agents doit expliquer ?', qui('laurence', 'mathieu')),
    ],
    pair: [
      E('Un binôme ? Je lève la main.', JUNIORS, PLUS),
      E('Deux têtes sur le même code. Je vais devoir parler.', SENIORS),
      E('Un binôme me ralentit de moitié. Au moins.', trait('dixX', 'shipIt'), MOINS),
    ],
    embauche: [
      E('La relève. Qui fera la garde quand je serai au chalet ?', qui('benoit', 'raymond')),
      E('Je vais être embauché après mon stage, ou remplacé par un agent ?', qui('jeremy'), MOINS),
      E('Un programme de relève, je signe tout de suite.', JUNIORS, PLUS),
    ],
  };

  // ---------------------------------------------------------------------------
  // Situations : ce qui se passe vraiment dans la simulation
  // ---------------------------------------------------------------------------
  function collegueAvec(etat, q, dev, f) { return collegues(etat, q, dev).filter(f)[0] || null; }
  function departsRecents(etat, dev, n) {
    var p = etat && etat.personnes, l = (p && Array.isArray(p.departs)) ? p.departs : [];
    return l.filter(function (x) { return x && x.ilot === dev.ilot && x.dev !== dev.id && sem(etat) - num(x.semaine, -99) <= n; });
  }
  function securiteRecente(etat, q) {
    var l = Array.isArray(etat && etat.incidents) ? etat.incidents : [];
    return l.filter(function (i) {
      try { return i && sem(etat) - num(i.debut, -99) <= 3 && q.estIncidentSecurite(i.cause); } catch (e) { return false; }
    });
  }

  var S = [
    { id: 'sansLicence', libelle: 'sans licence', poids: 0.8,
      quand: function (etat, q, dev) {
        return !lic(etat, q, dev) && collegues(etat, q, dev).some(function (d) { return lic(etat, q, d); });
      },
      lignes: [
        E('Mon voisin a Copilot. Moi, j\'ai mon clavier.', MOINS),
        E('Il finit ses lignes avant de les avoir commencées. Moi, je tape.'),
        E('Les pilotes parlent de leurs suggestions au dîner. Je mange mon sandwich.'),
        E('Sans licence, pas de jetons, pas de facture. Je suis le poste le moins cher de l\'îlot.'),
        E(function (etat, q, dev) {
          var p = collegueAvec(etat, q, dev, function (d) { return lic(etat, q, d); });
          return p ? nom(q, p.id) + ' a livré plus que moi cette semaine. Avec une licence, ça aide. Je dis ça de même.' : 'Le pilote a livré plus que moi cette semaine. Je dis ça de même.';
        }, si(function (etat, q, dev) {
          var p = collegueAvec(etat, q, dev, function (d) { return lic(etat, q, d); });
          return !!p && num(der(p).points, 0) > num(der(dev).points, 0) + 0.5;
        })),
        E('Pas de licence pour moi. Personne ne me l\'a demandé. Tant mieux.', trait('sceptique', 'refractaire'), PLUS),
        E('Yannick a une licence. Moi, j\'ai 31 ans d\'AS/400. On verra qui finit.', qui('raymond')),
        E('Steve a son Copilot. Moi, j\'ai mes macros. Elles marchent.', qui('diane')),
        E('Karim a une licence et pas moi. Je code comme en 2015, paraît-il.', qui('jeremy'), MOINS),
        E('Chloé a Copilot. J\'ai quatre écrans et aucun assistant. C\'est cruel.', qui('laurence'), MOINS),
        E('Mathieu a une licence. Je le sais : ses PR ont doublé.', qui('nadia'), MOINS),
        E('Karim essaie Copilot. Je regarde s\'il écrit le test d\'abord.', qui('sophie')),
        E('Yannick me demande quelle liste est la bonne. Son assistant ne le sait pas.', qui('lucie')),
        E('Marie-Ève a une licence. Moi, j\'apprends sans. Ce n\'est peut-être pas plus mal.', qui('tommy')),
        E('Marie-Ève a son assistant. Moi, j\'ai ma tasse.', qui('benoit')),
        E('Chloé a Copilot. Il connaît WordPress, paraît-il. Le pauvre.', qui('patrick')),
        E('Hugo essaie Copilot. Je regarde de loin. Prudemment.', qui('amira')),
        E('Hugo a Copilot. Moi, je parle encore aux capteurs tout seul.', qui('luc')),
        E('Steve a sa licence. Moi, je suis contractuel. On me la donnera avant mon départ ?', qui('olivier')),
        E('Je lis ce que Copilot propose à Mathieu, par-dessus son épaule.', qui('sebastien')),
      ] },

    { id: 'aucuneLicence', libelle: 'sans Copilot', poids: 0.75,
      quand: function (etat, q) { return lv(etat, q, 'licences') === 'aucune' && sem(etat) >= 1; },
      lignes: [
        E('Plus personne n\'a Copilot. L\'open space est tranquille.'),
        E('On code comme en 2022. Ça revient vite.'),
        E('Pas de licences, pas de jetons. Régis doit dormir comme un bébé.'),
        E('Le calme. Le clavier. J\'adore.', trait('sceptique', 'refractaire'), PLUS),
        E('J\'ai goûté à l\'autocomplétion. Le sevrage est dur.', PIL, MOINS),
        E('Je tape mes tests moi-même. Ils vérifient ce que je veux.', qui('sophie'), PLUS),
      ] },

    { id: 'premiersPas', libelle: 'premières semaines', poids: 0.9,
      quand: function (etat, q, dev) {
        var s = 0;
        try { s = q.semaineCourante(etat); } catch (e) { s = 99; }
        return lic(etat, q, dev) && auto(etat, q, dev) === 0 && s <= 10;
      },
      lignes: [
        E('L\'autocomplétion devine la fin de mes lignes. Des fois.'),
        E('Tab, Tab, Tab. Je redécouvre une touche.'),
        E('Il a deviné ma boucle avant moi. Il s\'est trompé sur la condition d\'arrêt.'),
        E('L\'autocomplétion, c\'est comme un collègue qui finit tes phrases. Pas toujours bien.'),
        E('Il propose un meilleur nom de variable que le mien. Ça me vexe.'),
        E('Une suggestion sur trois est bonne. Les deux autres sont drôles.'),
        E('Je mesure : une partie de mes caractères vient de l\'autocomplétion. Je note le chiffre.', qui('marieeve')),
        E('Il complète mes scripts PowerShell. Il connaît Kevin ?', qui('steve')),
        E('Il complète du Kotlin correct. Hors ligne, il ne complète rien.', qui('karim')),
        E('Il propose des noms de colonnes Pandas. La moitié existe.', qui('hugo')),
        E('Il écrit le CSS que je dessine. Je suis émue.', qui('chloe'), PLUS),
        E('De l\'autocomplétion ? Je tape plus vite que lui.', qui('mathieu')),
        E('J\'ai essayé de le brancher sur autre chose. Il ne se branche pas.', qui('yannick')),
      ] },

    { id: 'clavardage', libelle: 'clavardage', poids: 0.75,
      quand: function (etat, q, dev) { return lic(etat, q, dev) && auto(etat, q, dev) === 1; },
      lignes: [
        E('Je pose mes questions à l\'assistant. Il ne soupire jamais.'),
        E('Il m\'a expliqué une expression régulière en français. Je l\'ai presque comprise.'),
        E('Je lui colle un message d\'erreur, il me répond en trois paragraphes. Le premier suffisait.'),
        E('Je lui demande, il répond, je vérifie. Surtout, je vérifie.', SENIORS),
        E('Je pose à l\'assistant les questions que je n\'ose pas poser aux seniors.', JUNIORS),
        E('Je lui ai posé une question piège. Il est tombé dedans.', trait('sceptique', 'prudent', 'securite')),
        E('Il ne connaît pas le RPG. Je lui ai appris. Il a oublié.', qui('raymond')),
        E('Il me suggère une extension WordPress pour tout. Même pour enlever des extensions.', qui('patrick')),
        E('Je lui demande la cause d\'une alerte. Il me propose trois causes. J\'en vérifie trois.', qui('marieeve', 'benoit')),
      ] },

    { id: 'modeAgent', libelle: 'mode agent', poids: 0.75,
      quand: function (etat, q, dev) { return lic(etat, q, dev) && auto(etat, q, dev) === 2; },
      lignes: [
        E('L\'agent me demande la permission à chaque étape. Je dis oui. Souvent.'),
        E('Il a modifié six fichiers pour une virgule. Je lui ai dit non.'),
        E('Il a proposé d\'ajouter une dépendance pour additionner deux nombres. Refusé.'),
        E('J\'approuve, j\'approuve, j\'approuve. Est-ce que je lis encore ?', MOINS),
        E('Je lis chaque étape avant d\'approuver. C\'est long. C\'est bien.', trait('rigoureux', 'prudent', 'tdd')),
        E('J\'ai approuvé 40 étapes avant le dîner. Record.', qui('mathieu')),
        E('Il veut redémarrer un service. Je lui demande lequel. Il ne sait pas.', qui('benoit')),
        E('Il propose un délai dans une interruption. Non. Non. Non.', qui('amira')),
        E('Mes agents me demandent la permission en même temps. Je dis oui en chœur.', qui('laurence')),
        E('L\'agent fait le gros œuvre, je fais les finitions. Je comprends encore tout.', JUNIORS),
      ] },

    { id: 'agentPR', libelle: 'agent qui ouvre ses PR', poids: 0.8,
      quand: function (etat, q, dev) { return lic(etat, q, dev) && auto(etat, q, dev) === 3; },
      lignes: [
        E('L\'agent a ouvert trois PR cette nuit. J\'en avais demandé une.'),
        E('Je relis les PR de mon agent comme celles d\'un stagiaire très rapide.', SENIORS),
        E('Ma liste de PR ouvertes ne tient plus dans l\'écran.'),
        E('L\'agent a ouvert une PR pour corriger sa PR précédente. Il est consciencieux.'),
        E('L\'agent écrit, j\'ouvre la PR, on me la relit. J\'ai appris quoi, cette semaine ?', JUNIORS, MOINS),
        E('J\'ai écrit moins d\'un quart de mon code cette semaine. Le reste, je l\'ai lu. En partie.',
          si(function (etat, q, dev) { var w = der(dev).ecritSoiMeme; return typeof w === 'number' && w < 0.3 && num(der(dev).points, 0) > 0; })),
        E('Mon agent ouvre des PR plus vite que je les lis. Je les lis quand même ?', qui('jeremy')),
        E('Mon agent ouvre les PR, moi je les documente. Mon successeur me remerciera.', qui('olivier')),
        E('L\'agent ouvre une PR, je la ferme, il en ouvre une autre. On se comprend mal.', trait('prudent', 'firmware', 'legacy')),
      ] },

    { id: 'fusionAuto', libelle: 'fusion automatique', poids: 0.9, ton: 'moins',
      quand: function (etat, q, dev) { return lic(etat, q, dev) && auto(etat, q, dev) >= 4; },
      lignes: [
        E('Ça a fusionné pendant que je dînais. C\'était quoi, déjà ?'),
        E('Je découvre ce qui est en production en lisant le journal.'),
        E('Personne n\'a lu la PR d\'hier. Elle roule en production.'),
        E('Le firmware fusionne tout seul. Je relis après, en retenant mon souffle.', trait('prudent', 'firmware')),
        E('Ça fusionne seul le vendredi soir. Mon téléphone le sait.', qui('benoit')),
        E('Ma pile a disparu. Le code entre sans moi. Je n\'aime pas ça.', qui('nadia')),
        E('Fusion automatique : je ne parle plus à personne. Personne ne me parle non plus.', qui('mathieu'), { ton: 'neutre' }),
        E('Mes agents ont livré douze fois cette semaine. Je n\'en ai vu que trois.', qui('laurence')),
      ] },

    { id: 'multiSessions', libelle: 'plusieurs sessions', poids: 0.8,
      quand: function (etat, q, dev) { var n = num(dev.sessions, 0); return n >= 3 ? Math.min(1.3, 0.8 + n / 10) : false; },
      lignes: [
        E('Trois agents tournent. Je dirige un orchestre sans avoir lu la partition.'),
        E('Je passe d\'une session à l\'autre. Je perds le fil à chaque fois.'),
        E('Mes jetons de la semaine ont triplé. Mon débit, pas tout à fait.'),
        E('Trop d\'agents, pas assez de moi. Je le sens dans les yeux.', si(function (etat, q, dev) { return num(dev.fatigue, 0) > 0.45; }), MOINS),
        E('La session 3 a cassé ce que la session 1 avait fait. Je les présente l\'une à l\'autre.', qui('laurence')),
        E('Six agents, un casque. Je ne sais plus lequel parle.', qui('mathieu')),
        E('Une session par MCP. Logique, non ?', qui('yannick')),
        E('Trois sessions : je livre comme trois contractuels. Payé comme un.', qui('olivier')),
      ] },

    { id: 'fatigue', libelle: 'fatigue', poids: 1.0, ton: 'moins',
      quand: function (etat, q, dev) { var f = num(dev.fatigue, 0); return f > 0.6 ? Math.min(1.5, 1 + (f - 0.6) * 2) : false; },
      lignes: [
        E('Je relis la même ligne pour la troisième fois.'),
        E('Café numéro quatre. Il ne fait plus effet.'),
        E('La fatigue me fait approuver sans lire. Ça me fait peur.', REV),
        E('Vivement la fin de semaine. Elle est loin.'),
        E('Trois nuits de garde. Ma tasse a des cernes.', qui('benoit')),
        E('J\'ai relu 30 PR cette semaine. Je ne me souviens d\'aucune.', qui('nadia')),
        E('Mes quatre agents vont bien. Moi, moins.', qui('laurence')),
        E('Même un 10x a besoin de dormir. Je viens de l\'apprendre.', qui('mathieu')),
        E('Trop de sessions ouvertes. Ma tête aussi en a trop.', si(function (etat, q, dev) { return num(dev.sessions, 0) >= 3; })),
        E('Je fais des erreurs que je ne faisais pas en janvier.', SENIORS),
      ] },

    { id: 'moralBas', libelle: 'moral bas', poids: 0.9, ton: 'moins',
      quand: function (etat, q, dev) { var m = num(dev.moral, 1); return m < 0.42 ? Math.min(1.5, 1 + (0.42 - m) * 2) : false; },
      lignes: [
        E('Je ne sais plus trop pourquoi je fais ça.'),
        E('J\'ai regardé les offres d\'emploi. Juste regardé.'),
        E('Personne ne m\'a demandé mon avis sur les derniers changements.'),
        E('Je fais mes heures. Pas plus.'),
        E('Trente ans de macros, et on veut me remplacer par un robot.', qui('diane')),
        E('Si je partais, qui relirait le Portail ?', qui('nadia')),
        E('Mes tests ne comptent plus pour personne.', qui('sophie')),
        E('Je compte les années avant la retraite. Il y en a moins qu\'avant.', qui('raymond')),
        E('J\'aime coder. Je ne code plus, je surveille.', SENIORS),
      ] },

    { id: 'moralHaut', libelle: 'bon moral', poids: 0.75, ton: 'plus',
      quand: function (etat, q, dev) { return num(dev.moral, 0) > 0.8; },
      lignes: [
        E('Bonne semaine. Je ne sais pas si c\'est l\'IA ou le café.'),
        E('J\'aime ce que je fais en ce moment. Je le note, au cas.'),
        E('L\'équipe roule bien. Je touche du bois.'),
        E('Bon. Je l\'avoue : l\'agent a compris ma macro.', qui('diane')),
        E('L\'AS/400 et l\'agent s\'entendent. Je n\'en reviens pas.', qui('raymond')),
        E('J\'apprends quelque chose tous les jours. Tous les jours !', qui('tommy', 'jeremy')),
        E('Je suis à 10x et l\'équipe suit. Presque.', qui('mathieu')),
        E('Ma pile est raisonnable et mes PR aussi. Je savoure.', REV),
      ] },

    { id: 'prEnAttente', libelle: 'PR en attente', poids: 0.9, ton: 'moins', levier: 'reviseurs',
      quand: function (etat, q, dev, ilot) {
        var f = num(ilot && ilot.fileRevue, 0);
        return f >= 4 && !reviseur(etat, q, dev) && num(der(dev).pr, 0) > 0 ? Math.min(1.5, 0.9 + f / 20) : false;
      },
      lignes: [
        E('Ma PR attend une revue depuis lundi.'),
        E('J\'ai ouvert une autre PR en attendant la revue de la première.'),
        E('La file de revue avance. Lentement. Très lentement.'),
        E('Ma PR attend. Je la remets à jour sur la branche principale pour la troisième fois.'),
        E('Nadia, s\'il te plaît. Ma PR a eu des petits.', qui('mathieu')),
        E('Personne ne relit les PR du stagiaire. Je comprends, mais quand même.', qui('jeremy')),
        E('Personne ne veut relire du RPG. Ma PR vieillit avec moi.', qui('raymond')),
        E('Quatre PR en attente. Mes agents attendent aussi, au compteur.', qui('laurence')),
        E('Le code est écrit depuis mardi. Il sera livré quand quelqu\'un le lira.', SENIORS),
      ] },

    { id: 'fileRevue', libelle: 'file de revue', poids: 1.0, ton: 'moins', levier: 'reviseurs',
      quand: function (etat, q, dev, ilot) {
        var f = num(ilot && ilot.fileRevue, 0);
        return reviseur(etat, q, dev) && f > 5 ? Math.min(1.5, 0.9 + f / 20) : false;
      },
      lignes: [
        E('Encore des PR à relire. Elles poussent la nuit.'),
        E('Ma pile de revue a sa propre ombre.'),
        E('Je relis par ordre de taille. Les grosses attendent, et elles grossissent.'),
        E('Je n\'ai pas écrit une ligne de code cette semaine. J\'ai relu.'),
        E('Je commence à approuver plus vite. Trop vite.'),
        E(function (etat, q, dev) {
          var il = ilotDe(etat, dev);
          return Math.round(num(il && il.fileRevue, 0)) + ' PR dans la file. Je les ai comptées deux fois.';
        }),
        E('PR 17 de 31. Je suis encore là.', qui('nadia')),
        E('Chaque PR de firmware, je la relis au complet. La pile le sait.', qui('amira')),
        E('Tout ce qui touche l\'AS/400 passe par moi. Ça fait une pile.', qui('raymond')),
        E('Je relis les tests en premier. Ensuite, le reste. Si j\'ai le temps.', qui('sophie')),
      ] },

    { id: 'lgtm', libelle: 'approbations éclair', poids: 0.9, ton: 'moins', levier: 'taillePR',
      quand: function (etat, q, dev, ilot) {
        return num(der(dev).lgtm, 0) > 0.4 || (reviseur(etat, q, dev) && num(der(ilot).lgtm, 0) >= 1.2);
      },
      lignes: [
        E('J\'ai écrit « LGTM » sur une PR que je n\'avais pas fini de lire.'),
        E('LGTM, LGTM, LGTM. Looks good… je pense.'),
        E('La PR avait l\'air propre. Je n\'ai pas vérifié la migration.'),
        E('J\'ai approuvé sans tout lire. Je vais y repenser ce soir.', trait('rigoureux', 'tdd', 'prudent')),
        E('J\'ai approuvé trois PR en une minute. Ce n\'est pas moi, ça.', qui('nadia')),
        E('J\'approuve vite, on m\'approuve vite. Tout le monde est content.', qui('mathieu'), { ton: 'neutre' }),
        E('Des PR trop grosses pour être lues. Alors on ne les lit pas.', REV),
      ] },

    { id: 'incidentIlot', libelle: 'incident en cours', poids: 1.1, ton: 'moins',
      quand: function (etat, q, dev) { var n = incidents(etat, q, dev.ilot).length; return n > 0 ? Math.min(1.5, 1 + 0.2 * (n - 1)) : false; },
      lignes: [
        E('Ça brûle dans l\'îlot. Tout le monde regarde le même écran.'),
        E('Qui a touché à ce module en dernier ? L\'agent. Mardi.', si(autoMin(2))),
        E('On rétablit d\'abord. On comprendra après.'),
        E('Je ne comprends pas ce code. Je l\'ai pourtant approuvé.', si(function (etat, q, dev) { return num(dev.comprehension, 1) < 0.6; })),
        E('Encore un incident. Je ressors la tasse de garde.', qui('benoit')),
        E('Je chronomètre le rétablissement. Le chiffre ne sera pas beau.', qui('marieeve')),
        E('Mon premier vrai incident. Benoît me laisse taper les commandes.', qui('tommy')),
        E('Un capteur envoie n\'importe quoi. Je savais que ce jour viendrait.', qui('amira')),
        E('Le site est à terre. La campagne commence demain.', qui('patrick')),
        E('Ça sent l\'injection. Montrez-moi ce que l\'agent a lu.', qui('sebastien')),
        E('La paie ne balance pas. Je sais exactement quelle cellule.', qui('diane')),
      ] },

    { id: 'comprehensionBasse', libelle: 'compréhension basse', poids: 0.9, ton: 'moins',
      quand: function (etat, q, dev) { var c = num(dev.comprehension, 1); return c < 0.45 ? Math.min(1.5, 1 + (0.45 - c) * 3) : false; },
      lignes: [
        E('J\'ai ouvert un fichier que j\'ai fusionné en février. Je ne le reconnais pas.'),
        E('Si on me demandait d\'expliquer ce module, je dirais « l\'agent ».'),
        E('Je comprends ce que le code fait. Pas pourquoi.'),
        E('Je demande à l\'agent d\'expliquer son propre code. Il est très convaincant.', LIC),
        E('Je livre beaucoup. J\'apprends peu. Je le sens.', JUNIORS),
        E('Génial, ça marche. Pourquoi ? Aucune idée.', qui('jeremy')),
        E('Patrick m\'a demandé pourquoi ça marche. J\'ai dit « parce que ».', qui('chloe')),
        E('Mon code de janvier ? Il faut demander à mon agent de janvier.', qui('mathieu')),
        E('Il y a des modules entiers que personne ici n\'a écrits.', SENIORS),
      ] },

    { id: 'carnetVide', libelle: 'carnet presque vide', poids: 0.75,
      quand: function (etat, q, dev, ilot) { return !!ilot && sem(etat) >= 2 && num(ilot.backlog, 99) < 14; },
      lignes: [
        E('Plus grand-chose dans le carnet de demandes. Les agents tournent à vide.'),
        E('Le carnet est presque vide. On attend que la direction se réveille.'),
        E('On a livré plus vite que la direction ne demande. Ça n\'arrive jamais.'),
        E('J\'en profite pour nettoyer du vieux code. Personne ne me l\'a demandé.'),
        E('Je documente. C\'est ce qu\'on fait quand le carnet est vide.'),
        E('Mes agents n\'ont plus rien à faire. Moi non plus.', qui('laurence')),
        E('Rien au carnet ? Je mets à jour mes 14 sites, alors.', qui('patrick')),
        E('Le carnet est vide. J\'écris enfin les tests qui manquaient.', qui('sophie', 'amira')),
      ] },

    { id: 'carnetPlein', libelle: 'carnet qui déborde', poids: 0.8, ton: 'moins', levier: 'disciplineNon',
      quand: function (etat, q, dev, ilot) { var b = num(ilot && ilot.backlog, 0); return b > 110 ? Math.min(1.5, 0.6 + b / 250) : false; },
      lignes: [
        E('Le carnet de demandes déborde. Chaque idée « rapide » s\'additionne.'),
        E('La direction a encore ajouté des demandes. « Puisque c\'est rapide. »'),
        E('On ne finira jamais tout ça. Il faudrait choisir.'),
        E('Chaque fois qu\'on livre vite, la direction en demande deux autres.'),
        E('Le carnet grossit plus vite que ce qu\'on livre. Les agents n\'y changent rien.'),
        E('On dit oui à tout. Le carnet le sait.', si(function (etat, q, dev) { return lv(etat, q, 'disciplineNon', dev.ilot) === 'laxiste'; })),
        E('Francine a ajouté un tableau de bord des tableaux de bord.', qui('lucie', 'yannick', 'raymond')),
        E('La version tablette, la version montre, la version frigo. Toutes au carnet.', qui('karim')),
        E('Un 16e site au carnet. Je soupire en silence.', qui('patrick')),
        E(function (etat, q, dev) {
          var il = ilotDe(etat, dev);
          return Math.round(num(il && il.backlog, 0)) + ' points de demandes en attente. Je ne les ai pas tous lus.';
        }),
      ] },

    { id: 'plafondAtteint', libelle: 'plafond de jetons', poids: 1.0, ton: 'moins', levier: 'plafondJetons',
      quand: function (etat, q, dev) { return lic(etat, q, dev) && plafondAtteint(etat, q, dev); },
      lignes: [
        E('L\'agent s\'est arrêté net : plafond de jetons atteint.'),
        E('Mercredi, plus de jetons pour l\'îlot. Je finis à la main.'),
        E('Je découpe mes tâches pour passer sous le plafond.'),
        E('Coupée jeudi midi. Mes quatre agents boudent.', qui('laurence')),
        E('Le plafond m\'arrête avant tout le monde. C\'est presque un compliment.', qui('mathieu')),
        E('Plafond atteint. Je retrouve le plaisir d\'écrire une boucle à la main.', SENIORS, { ton: 'neutre' }),
      ] },

    { id: 'budgetFond', libelle: 'budget qui fond', poids: 0.75, ton: 'moins',
      quand: function (etat) { return sem(etat) >= 4 && budgetRythme(etat) > 1.2; },
      lignes: [
        E('On dit que Régis a imprimé la facture de jetons. En couleur.'),
        E('J\'hésite avant de relancer l\'agent. Ça coûte, paraît-il.', LIC),
        E('Le budget de l\'IA fond plus vite que prévu. On sent la pression.'),
        E('Mes sessions coûtent cher. Je le sais, Régis me l\'a dit.', si(function (etat, q, dev) { return num(dev.sessions, 0) >= 3; })),
        E('Coût par point livré : je l\'ai calculé. Régis va aimer le chiffre. Ou pas.', qui('marieeve')),
        E('Mes jetons coûtent plus cher que mon café. Ça vaut le coup.', qui('mathieu'), { ton: 'neutre' }),
      ] },

    { id: 'budgetDepasse', libelle: 'budget dépassé', poids: 0.9, ton: 'moins',
      quand: function (etat) { return num(etat && etat.org && etat.org.budget, 1) < 0; },
      lignes: [
        E('Le budget est dépassé. On parle de couper les licences.'),
        E('Régis a annulé la commande de beignes. Coïncidence ?'),
        E('On n\'a plus de budget pour l\'IA. On a encore l\'IA, par contre.'),
        E('Budget dépassé. Les contractuels partent en premier, d\'habitude.', qui('olivier')),
        E('Je ferme mes sessions inutiles. Toutes celles qui attendent.', LIC),
      ] },

    { id: 'registreEquipe', libelle: 'skills équipées', poids: 0.75,
      quand: function (etat, q, dev, ilot) { return lic(etat, q, dev) && !!ilot && Array.isArray(ilot.capacites) && ilot.capacites.length >= 1; },
      lignes: [
        E('L\'agent a chargé la skill de l\'îlot. Il fait les choses à notre façon.'),
        E('Une skill de plus au registre. L\'agent lit le mode d\'emploi avant moi.'),
        E('Grâce à la skill de l\'îlot, l\'agent arrête de réinventer nos conventions.', SENIORS),
        E('L\'agent suit une skill qui n\'est plus à jour. Il appelle l\'ancienne API.',
          si(function (etat, q, dev) {
            var il = ilotDe(etat, dev);
            return !!il && (il.capacites || []).some(function (c) { var x = etat.capacites && etat.capacites[c]; return x && num(x.fraicheur, 1) < 0.5; });
          }), MOINS),
        E('tdd-strict est équipée. L\'agent écrit le test d\'abord. Je suis fière.', qui('sophie'), si(function (etat, q, dev) { return equipee(etat, 'tdd-strict', dev.ilot); }), PLUS),
        E('ship-it est équipée. Mon agent ne pose plus de questions.', qui('mathieu'), si(function (etat, q, dev) { return equipee(etat, 'ship-it', dev.ilot); }), PLUS),
        E('L\'agent comprend les dates à sept chiffres. Je n\'en reviens pas.', qui('raymond', 'lucie'),
          si(function (etat, q, dev) { return equipee(etat, 'mcp-as400', dev.ilot) || equipee(etat, 'migration-as400', dev.ilot); }), PLUS),
        E('La procédure de nuit de Ginette est dans l\'agent. Il sait pour le serveur 3.', qui('benoit', 'tommy', 'marieeve'),
          si(function (etat, q, dev) { return equipee(etat, 'deploiement-nuit', dev.ilot); }), PLUS),
        E('J\'ai équipé l\'îlot. Tout ce que j\'ai trouvé. C\'est trop ?', qui('yannick', 'steve')),
      ] },

    { id: 'contexteCharge', libelle: 'contexte chargé', poids: 0.8, ton: 'moins',
      quand: function (etat, q, dev, ilot) { return lic(etat, q, dev) && !!ilot && Array.isArray(ilot.capacites) && ilot.capacites.length >= 6; },
      lignes: [
        E('L\'agent passe sa première minute à relire la liste de ses outils.'),
        E('Trop de skills : l\'agent a pris la mauvaise.'),
        E('Chaque carte ajoutée rend les autres un peu moins utiles. On en a combien ?'),
        E('Quatorze outils. L\'agent hésite entre deux qui font la même chose.', qui('steve', 'yannick')),
        E('Son contexte est plein avant que je lui parle.', SENIORS),
      ] },

    { id: 'skillsContradictoires', libelle: 'skills qui se contredisent', poids: 0.9, ton: 'moins',
      quand: function (etat, q, dev, ilot) { return !!ilot && !!ilot.mods && num(ilot.mods.incoherence, 0) > 0.05; },
      lignes: [
        E('Deux skills disent le contraire. L\'agent choisit selon son humeur.'),
        E('Une PR avec tests, la suivante sans. C\'est la même skill qui hésite ?'),
        E('tdd-strict et ship-it dans le même îlot. Mathieu et moi, on se regarde.', qui('sophie')),
        E('tdd-strict me ralentit. Ma skill dit l\'inverse. L\'agent est perdu.', qui('mathieu')),
        E('Deux recettes de déploiement. L\'agent a pris celle des Opérations.', REV),
      ] },

    { id: 'triadeComplete', libelle: 'triade létale', poids: 0.9, ton: 'moins',
      quand: function (etat, q, dev) { return triade(etat, q, dev); },
      lignes: [
        E('Les trois voyants de l\'îlot sont allumés. Ça ne me rassure pas.'),
        E('L\'agent lit nos données, lit des courriels et peut en envoyer. Quelqu\'un a pensé à ça ?'),
        E('Un courriel bien tourné pourrait faire sortir nos données. Je relis les sorties.'),
        E('Trois voyants allumés, mais un humain confirme chaque sortie. Ouf.',
          si(function (etat, q, dev) { return lv(etat, q, 'approbationSorties', dev.ilot) === true; }), { ton: 'neutre' }),
        E('Données privées, contenu non fiable, sortie externe. Les trois. J\'ai mal à la tête.', qui('sebastien')),
        E('L\'agent lit les factures des fournisseurs. N\'importe qui peut écrire une facture.', qui('lucie')),
        E('C\'est mon MCP qui allume le troisième voyant ? Oups.', qui('yannick', 'steve')),
      ] },

    { id: 'triadePartielle', libelle: 'deux voyants sur trois', poids: 0.75,
      quand: function (etat, q, dev, ilot) { return nbBranches(ilot) === 2; },
      lignes: [
        E('Deux voyants sur trois. Le troisième, on le surveille.'),
        E('Données privées et contenu non fiable. Il ne manque que la sortie. Ne l\'ajoutons pas.'),
        E('Il manque une branche pour la triade. Ne l\'ajoutons pas par accident.'),
        E('Deux cercles sur trois. Je garde le crayon rouge.', qui('sebastien')),
        E('Deux voyants allumés. Je mets une alerte sur le troisième.', qui('marieeve')),
        E('Le prochain MCP qu\'on équipe ici, je veux savoir ce qu\'il envoie dehors.', trait('securite', 'prudent', 'rigoureux')),
      ] },

    { id: 'applisFantomes', libelle: 'applis hors TI', poids: 0.75,
      quand: function (etat, q, dev, ilot) { return num(ilot && ilot.applisFantomes, 0) >= 1; },
      lignes: [
        E('Il y a une application hors TI dans le secteur. Personne ne sait où sont ses données.'),
        E('La direction utilise une appli faite au marketing. On l\'a appris par hasard.'),
        E('Une appli fantôme de plus. Elle a des utilisateurs, mais pas de sauvegarde.'),
        E('Personne aux TI n\'a vu le code de l\'appli que la direction utilise tous les jours.'),
        E('Kevin m\'a montré sa nouvelle appli. Elle est hébergée « sur un truc gratuit ».', qui('steve', 'olivier')),
        E('On va hériter de l\'appli de Kevin. Je le sens.', qui('patrick', 'chloe')),
        E('Une appli hors TI avec des données de clients. Je veux la voir. Maintenant.', qui('sebastien')),
      ] },

    { id: 'boguesSemaine', libelle: 'bogues de la semaine', poids: 0.9, ton: 'moins', levier: 'tests',
      quand: function (etat, q, dev, ilot) { var b = num(der(ilot).bogues, 0); return b >= 3 ? Math.min(1.5, 0.8 + b / 10) : false; },
      lignes: [
        E(function (etat, q, dev) {
          var b = Math.round(num(der(ilotDe(etat, dev)).bogues, 0));
          return b + ' bogues en production cette semaine. Ils sont passés par où ?';
        }),
        E('Le support a appelé deux fois. Ce n\'était pas pour dire merci.'),
        E('On corrige des bogues au lieu de livrer. Ça s\'appelle de la reprise.'),
        E('Personne ne relit, et ça se voit en production.', si(function (etat, q, dev) { return num(Number(lv(etat, q, 'reviseurs', dev.ilot)), 1) === 0; })),
        E('Pas de tests exigés, et des bogues qui passent. Le lien est facile à faire.', si(function (etat, q, dev) { return lv(etat, q, 'tests', dev.ilot) === 'aucun'; })),
        E('Chaque bogue a son test, maintenant. Après coup.', qui('sophie')),
        E('Des bogues ? Il y en a toujours. On corrige vite, on livre vite.', qui('mathieu'), { ton: 'neutre' }),
        E('La synchro a dupliqué des rapports. Encore. Les inspecteurs sont patients.', qui('karim')),
      ] },

    { id: 'satisfactionBasse', libelle: 'direction mécontente', poids: 0.8, ton: 'moins',
      quand: function (etat, q, dev, ilot) { var s = num(ilot && ilot.satisfaction, 1); return s < 0.4 ? Math.min(1.5, 1 + (0.4 - s) * 2) : false; },
      lignes: [
        E('La direction de l\'îlot ne nous dit plus bonjour.'),
        E('On livre, mais pas ce qu\'ils attendaient.'),
        E('La direction trouve qu\'on est lents. Le carnet lui donne raison.'),
        E('Personne ne nous a dit ce qui comptait le plus. On a deviné. Mal.'),
        E('La directrice a demandé une rencontre « pour faire le point ». Ça sent mauvais.'),
        E('On devrait leur demander ce qui compte pour eux. Pour vrai.', trait('pragmatique', 'mobile')),
        E('Nathalie veut sa version tablette. Elle le dit à tout le monde.', qui('karim', 'sophie', 'jeremy')),
        E('Francine trouve qu\'on n\'a pas changé. Elle a un peu raison.', qui('lucie', 'raymond', 'yannick')),
      ] },

    { id: 'satisfactionHaute', libelle: 'direction contente', poids: 0.75, ton: 'plus',
      quand: function (etat, q, dev, ilot) { return num(ilot && ilot.satisfaction, 0) > 0.6; },
      lignes: [
        E('La direction nous a envoyé des beignes. Des vrais.'),
        E('Ils sont contents de ce qu\'on livre. Je ne sais pas trop quoi faire de ça.'),
        E('La direction nous cite en exemple. On va devoir continuer, maintenant.'),
        E('Un merci de la direction dans un courriel. Je l\'ai imprimé.'),
        E('Le site de la campagne est sorti à temps. Personne n\'a rien remarqué. C\'est un compliment.', qui('patrick')),
        E('Les inspecteurs aiment l\'application. Même au sous-sol.', qui('karim')),
      ] },

    { id: 'confianceBasse', libelle: 'confiance en baisse', poids: 0.8, ton: 'moins',
      quand: function (etat) { return num(etat && etat.org && etat.org.confiance, 100) < 45; },
      lignes: [
        E('On dit que Mme Lavigueur n\'est pas contente. Ça descend jusqu\'ici.'),
        E('Le comité approche. Tout le monde parle à voix basse.'),
        E('J\'ai entendu « réorganisation » à la machine à café.'),
        E('Si ça va mal, je finirai mon contrat ailleurs.', qui('olivier')),
        E('Quand la direction perd confiance, on finit par perdre des licences.', LIC),
      ] },

    { id: 'confianceHaute', libelle: 'direction en confiance', poids: 0.75, ton: 'plus',
      quand: function (etat) { return num(etat && etat.org && etat.org.confiance, 0) > 85; },
      lignes: [
        E('Mme Lavigueur a parlé de nous au conseil. En bien.'),
        E('La direction nous fait confiance. On ne va pas en abuser.'),
        E('On a carte blanche, paraît-il. Je n\'aime pas les cartes blanches.', trait('prudent', 'securite', 'sceptique'), { ton: 'neutre' }),
        E('La confiance est haute, et nos chiffres sont honnêtes. Ça devrait durer.', qui('marieeve')),
      ] },

    { id: 'formation', libelle: 'formation', poids: 0.75, levier: 'formation',
      quand: function (etat, q) {
        var f = lv(etat, q, 'formation');
        return (!!f && f !== 'aucune') || prepare('formation')(etat, q);
      },
      lignes: [
        E('Une demi-journée de formation par semaine. Le travail attend.', si(prepare('formation'))),
        E('J\'ai appris à écrire une demande claire. L\'agent aussi a l\'air content.', LIC),
        E('Depuis la formation, je reconnais une injection de prompt. Ça m\'inquiète.'),
        E('J\'applique ce qu\'on a vu : petites tâches, contexte clair, relecture.', LIC),
        E('Ginette m\'a montré comment lui parler. Il m\'écoute. Hein.', qui('diane'), si(enVigueur('formation', 'mentorat'))),
        E('En formation, on a codé sans l\'agent. J\'ai survécu.', qui('jeremy')),
        E('J\'ai pris 40 pages de notes. Je les relis le soir.', qui('tommy')),
      ] },

    { id: 'lundi', libelle: 'lundi matin', poids: 0.75,
      quand: function (etat) { return jour(etat) === 0 && sem(etat) >= 1; },
      lignes: [
        E('Lundi. J\'ai regardé les chiffres de la semaine passée.'),
        E('Lundi. Café, chiffres, file de revue. Dans cet ordre.'),
        E('Lundi matin : tout le monde compare ses chiffres sans en avoir l\'air.'),
        E('Le rapport du vendredi est sorti. On en parle à la machine à café.'),
        E('Nouvelle semaine. Même file de revue.', REV),
        E(function (etat, q, dev) {
          var p = Math.round(num(der(ilotDe(etat, dev)).pointsLivres, 0));
          return 'On a livré ' + p + ' points la semaine passée. Je ne sais pas encore si c\'est bien.';
        }, si(function (etat, q, dev) { return num(der(ilotDe(etat, dev)).pointsLivres, 0) > 0.5; })),
        E(function (etat, q, dev) {
          var n = Math.round(num(der(ilotDe(etat, dev)).prFusionnees, 0));
          return n + ' PR fusionnées dans l\'îlot la semaine passée. Le chiffre préféré de la direction.';
        }, si(function (etat, q, dev) { return num(der(ilotDe(etat, dev)).prFusionnees, 0) >= 1; })),
        E('Lundi : je compare les chiffres à ceux d\'il y a un mois.', qui('marieeve')),
        E('Lundi. Nouveau record de PR, je parie.', qui('mathieu')),
      ] },

    { id: 'vendredi', libelle: 'vendredi', poids: 0.75,
      quand: function (etat) { return jour(etat) === 4; },
      lignes: [
        E('Vendredi. Les chiffres de la semaine tombent ce soir.'),
        E('Vendredi. Je finis ma PR ou je la laisse mûrir jusqu\'à lundi ?'),
        E('Vendredi après-midi. Personne ne veut être celui qui casse la production.'),
        E('On ne déploie pas le vendredi après-midi. En théorie.'),
        E('Je ferme mes sessions avant de partir. Toutes ?', LIC),
        E('Je laisse l\'agent travailler la fin de semaine ? Non. Peut-être.', si(autoMin(3))),
        E('Vendredi. Je charge mon téléphone de garde.', qui('benoit')),
        E('Vendredi : je ferme mes quatre sessions. Trois, au moins.', qui('laurence')),
      ] },

    { id: 'collegueConge', libelle: 'collègue en congé', poids: 0.85,
      quand: function (etat, q, dev) { return !!collegueAvec(etat, q, dev, function (d) { return d.etat === 'conge'; }); },
      lignes: [
        E(function (etat, q, dev) {
          var c = collegueAvec(etat, q, dev, function (d) { return d.etat === 'conge'; });
          return (c ? nom(q, c.id) : 'Un collègue') + ' est en congé. Je prends sa part en plus de la mienne.';
        }),
        E(function (etat, q, dev) {
          var c = collegueAvec(etat, q, dev, function (d) { return d.etat === 'conge'; });
          return (c ? nom(q, c.id) : 'Un collègue') + ' en congé forcé. On aurait dû le voir venir.';
        }),
        E('On est un de moins cette semaine. La file le sent.'),
        E('Benoît est en congé. C\'est moi qui ai le téléphone. Gloups.', qui('tommy'),
          si(function (etat) { return !!(etat.devs && etat.devs.benoit && etat.devs.benoit.etat === 'conge'); })),
        E(function (etat, q, dev) {
          var c = collegueAvec(etat, q, dev, function (d) { return d.etat === 'conge'; });
          return (c ? nom(q, c.id) : 'Ma collègue') + ' est en congé, et c\'est la personne qui connaît ce module. J\'attends.';
        }, SENIORS),
      ] },

    { id: 'depart', libelle: 'départ d\'un collègue', poids: 1.0, ton: 'moins',
      quand: function (etat, q, dev) { return departsRecents(etat, dev, 5).length > 0; },
      lignes: [
        E(function (etat, q, dev) {
          var d = departsRecents(etat, dev, 5)[0];
          return 'Depuis le départ de ' + (d ? nom(q, d.dev) : 'notre collègue') + ', une partie de ce code n\'a plus personne pour l\'expliquer.';
        }),
        E('Son bureau est vide. Ses PR, elles, sont encore ouvertes.'),
        E('On a hérité de son module. Pas de documentation. Juste une skill, peut-être.'),
        E(function (etat, q, dev) {
          var d = departsRecents(etat, dev, 5)[0];
          return 'Tout ce que savait ' + (d ? nom(q, d.dev) : 'notre collègue') + ' est parti avec son sac. On ne l\'avait pas écrit.';
        }, si(function (etat, q, dev) { var d = departsRecents(etat, dev, 5)[0]; return !!d && num(d.capture, 0) < 0.2; })),
        E('Olivier est parti. La feuille « NE PAS TOUCHER », c\'est moi qui l\'ai, maintenant.', qui('diane', 'steve'),
          si(function (etat, q, dev) { return departsRecents(etat, dev, 5).some(function (x) { return x.dev === 'olivier'; }); })),
      ] },

    { id: 'agentTriche', libelle: 'agent qui triche', poids: 0.9, ton: 'moins', levier: 'lint',
      quand: function (etat, q, dev) {
        return drapeau(etat, 'agentTriche_' + dev.ilot) && autoMin(3)(etat, q, dev) && lv(etat, q, 'lint') !== 'verrouille';
      },
      lignes: [
        E('L\'agent a marqué un test « skip » pour passer la CI. Je l\'ai trouvé par hasard.'),
        E('Une règle de lint désactivée dans un commit « chore ». Classique.'),
        E('La CI est verte. Je ne sais plus si c\'est une bonne nouvelle.'),
        E('Un test désactivé. Le seul qui vérifiait les taxes.', qui('sophie', 'nadia')),
        E('La couverture réelle a baissé et personne n\'a rien vu. Moi, si.', qui('marieeve')),
      ] },

    { id: 'couvertureTrompeuse', libelle: 'couverture trompeuse', poids: 0.75, levier: 'tests',
      quand: function (etat, q, dev, ilot) {
        var cb = ilot && ilot.codebase;
        return !!cb && num(cb.couvertureApparente, 0) - num(cb.couvertureReelle, 0) > 0.25;
      },
      lignes: [
        E('La couverture affichée est belle. Je ne crois pas tout ce qu\'elle dit.'),
        E(function (etat, q, dev) {
          var il = ilotDe(etat, dev), cb = (il && il.codebase) || {};
          return 'Couverture affichée : ' + Math.round(num(cb.couvertureApparente, 0) * 100) + ' %. Ce que nos tests vérifient vraiment, c\'est une autre histoire.';
        }),
        E('Nos tests passent toujours. C\'est ça qui m\'inquiète.'),
        E('L\'agent écrit des tests qui vérifient son propre code. Il est d\'accord avec lui-même.', LIC),
        E('Couverture affichée, couverture réelle : deux chiffres, un seul est vrai.', qui('sophie', 'marieeve')),
        E('Mes tests générés passent. Je ne sais pas ce qu\'ils testent.', JUNIORS),
      ] },

    { id: 'volumeGonfle', libelle: 'code qui gonfle', poids: 0.75, levier: 'disciplineNon',
      quand: function (etat, q, dev, ilot) {
        var d = null;
        try { d = q.defIlot(dev.ilot); } catch (e) { d = null; }
        var v0 = num(d && d.volumeInitial, 0), v = num(ilot && ilot.codebase && ilot.codebase.volume, 0);
        return v0 > 0 && v > v0 * 1.35;
      },
      lignes: [
        E('Le code de l\'îlot a grossi d\'un tiers depuis janvier. Les demandes, pas tant.'),
        E(function (etat, q, dev) {
          var il = ilotDe(etat, dev), d = null;
          try { d = q.defIlot(dev.ilot); } catch (e) { d = null; }
          var v0 = num(d && d.volumeInitial, 0), v = num(il && il.codebase && il.codebase.volume, 0);
          return v0 > 0 ? 'Le code de l\'îlot a grossi de ' + Math.round((v / v0 - 1) * 100) + ' % depuis janvier. Quelqu\'un va devoir le comprendre.' : 'Le code de l\'îlot grossit. Quelqu\'un va devoir le comprendre.';
        }),
        E('On entretient du code que personne n\'appelle. Je le parierais.'),
        E('Plus de code, c\'est plus de choses à comprendre. Avec le même monde.'),
        E('Quatorze sites, et beaucoup plus de code qu\'en janvier. Pour quoi ?', qui('patrick')),
        E('Je vais mesurer quel code s\'exécute vraiment. J\'ai peur du résultat.', qui('marieeve')),
      ] },

    { id: 'apprentissageEnPanne', libelle: 'apprentissage en panne', poids: 0.85, ton: 'moins', levier: 'autonomie',
      quand: function (etat, q, dev) {
        var w = der(dev).ecritSoiMeme;
        return (dev.niveau === 'junior' || dev.niveau === 'stagiaire') && autoMin(2)(etat, q, dev)
          && typeof w === 'number' && w < 0.4 && num(der(dev).points, 0) > 0;
      },
      lignes: [
        E('L\'agent écrit presque tout. Moi, je lis. Est-ce que j\'apprends ?'),
        E('J\'ai fusionné beaucoup de code cette semaine. J\'en ai écrit très peu.'),
        E('Je sais demander à l\'agent. Je ne sais pas encore le faire moi-même.'),
        E('Benoît me demande de taper les commandes moi-même. Il a raison.', qui('tommy')),
        E('Je suis super productif. Je serais incapable de le refaire sans lui.', qui('jeremy')),
        E('Mes agents codent, je coordonne. C\'est un métier, ça ?', qui('laurence')),
      ] },

    { id: 'ginetteDepartie', libelle: 'Ginette partie', poids: 0.75, ton: 'moins',
      quand: function (etat) { return drapeau(etat, 'ginetteDepartie'); },
      lignes: [
        E('Le bureau de Ginette est vide. Le serveur 3 redémarre mal le mardi, paraît-il.'),
        E('Personne ne sait où Ginette rangeait les mots de passe de 2009.'),
        E('Ginette est au chalet. Je suis le plus ancien de garde, maintenant.', qui('benoit')),
        E('J\'ai les notes de Ginette. Je les relis avant chaque nuit.', qui('tommy')),
        E('Sans Ginette, je n\'ai plus personne à écouter.', qui('diane')),
      ] },

    { id: 'debutTrimestre', libelle: 'nouveau trimestre', poids: 0.75,
      quand: function (etat, q) {
        var s = 1;
        try { s = q.semaineCourante(etat); } catch (e) { s = 1; }
        return s > 1 && (s - 1) % 13 === 0 && jour(etat) <= 1;
      },
      lignes: [
        E('Nouveau trimestre. Le comité a parlé de nous, paraît-il.'),
        E('On repart à zéro. Enfin, avec le même code.'),
        E('Nouveau trimestre, nouvelles notes de service. Je les lis toutes.', trait('mesure', 'rigoureux')),
        E('Nouveau trimestre : je fige les chiffres du précédent pour comparer.', qui('marieeve')),
        E('Nouveau trimestre, nouveaux records.', qui('mathieu')),
      ] },

    { id: 'incidentSecurite', libelle: 'incident de sécurité', poids: 0.9, ton: 'moins',
      quand: function (etat, q) { return securiteRecente(etat, q).length > 0; },
      lignes: [
        E('Il y a eu un incident de sécurité. Tout le monde change ses mots de passe.'),
        E('Me Bérubé est passé dans l\'open space. Avec un PDF.'),
        E('Je relis les MCP installés sur mon poste. Juste au cas.'),
        E('Je l\'avais dit. Je ne suis pas content d\'avoir eu raison.', qui('sebastien')),
        E('Ce n\'est pas mon MCP. J\'espère que ce n\'est pas mon MCP.', qui('yannick')),
      ] },
  ];

  LTP.donnees.reflexions = { leviers: L, nouveautes: N, situations: S };
})(globalThis.LTP = globalThis.LTP || {});
