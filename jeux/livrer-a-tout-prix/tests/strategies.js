// Stratégies scriptées d'équilibrage (ARCHITECTURE §5), au format de LTP.moteur.simuler(etat, n, strategie) :
//   { nom, leviers(etat) → [actions] (début de chaque trimestre), choix(etat, defEvenement) → idChoix,
//     comite(etat, question) → idReponse }
// Les stratégies ne proposent que des leviers débloqués et ne lisent que les données (cartes, îlots, devs) :
// elles fonctionnent avec n'importe quel contenu conforme, y compris des listes vides.

// « par-defaut » : le joueur ne touche à rien de toute la partie (premier choix, première réponse au comité).
export const NOMS_STRATEGIES = ["tout-ia", "musee", "equilibree", "economique", "aleatoire", "par-defaut"];

// Générateur déterministe propre à la stratégie (ne touche pas au hasard du moteur).
function hasardDe(graine) {
  let s = graine >>> 0 || 1;
  return () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 4294967296;
}

// Actions de leviers : seulement les leviers débloqués et les valeurs valides.
function actionsLeviers(LTP, etat, reglages, surcharges = {}) {
  const actions = [];
  const valide = (id, v) => {
    const d = LTP.donnees.leviersParId[id];
    return d && LTP.q.levierDebloque(etat, id) && d.valeurs.some((x) => x.v === v);
  };
  for (const [id, v] of Object.entries(reglages)) {
    // Pas de nouvelle action si la valeur est déjà en vigueur ou déjà programmée (leviers à délai).
    const attente = LTP.q.enAttente(etat, id, null);
    if (valide(id, v) && LTP.q.levier(etat, id) !== v && !(attente && attente.valeur === v)) {
      actions.push({ type: "levier", id, valeur: v });
    }
  }
  for (const [ilot, regl] of Object.entries(surcharges)) {
    for (const [id, v] of Object.entries(regl)) {
      const d = LTP.donnees.leviersParId[id];
      if (d && d.surchargeable && valide(id, v) && etat.ilots[ilot] && LTP.q.levier(etat, id, ilot) !== v) {
        actions.push({ type: "levier", id, valeur: v, ilot });
      }
    }
  }
  return actions;
}

// Une carte est utilisable si elle est découverte, de son trimestre, et développée si elle doit l'être.
function carteUtilisable(LTP, etat, def) {
  const c = etat.capacites[def.id];
  if (!c || c.decouverte === false) return false;
  if (def.trimestre && def.trimestre > etat.trimestre && !etat.options.toutDebloque) return false;
  if (def.disponible === "developpement") return c.disponibleSemaine != null && c.disponibleSemaine <= etat.semaine;
  return true;
}
function prerequisOk(LTP, etat, def) {
  if (!def.prerequis) return true;
  return (def.prerequis.valeurs || []).includes(LTP.q.levier(etat, def.prerequis.levier));
}
function ilotsDe(etat, def) {
  const s = (def.ilotsSuggeres || []).filter((i) => etat.ilots[i]);
  return s.length ? s : etat.ordreIlots.slice();
}

// Cartes : équiper celles qui passent le filtre, développer les cartes maison, nommer des propriétaires.
function actionsCartes(LTP, etat, { filtre, developper = false, proprietaires = false, partout = false }) {
  const actions = [];
  const seniors = (ilot) => LTP.q.devsDe(etat, ilot).filter((d) => d.niveau === "senior");
  for (const def of LTP.donnees.capacites || []) {
    if (!filtre(def)) continue;
    const c = etat.capacites[def.id];
    if (!c) continue;
    const ilots = partout ? etat.ordreIlots.slice() : ilotsDe(etat, def);
    if (carteUtilisable(LTP, etat, def)) {
      if (!prerequisOk(LTP, etat, def)) continue;
      for (const ilot of ilots) if (!c.equipeeSur.includes(ilot)) actions.push({ type: "equiper", carte: def.id, ilot });
      if (proprietaires && !c.proprietaire) {
        const dev = seniors(ilots[0])[0] || LTP.q.devsDe(etat, ilots[0])[0];
        if (dev) actions.push({ type: "proprietaire", carte: def.id, dev: dev.id });
      }
    } else if (developper && def.disponible === "developpement" && c.disponibleSemaine == null && !c.enDeveloppementPar
      && (!def.trimestre || def.trimestre <= etat.trimestre) && prerequisOk(LTP, etat, def)) {
      const dev = seniors(ilots[0])[0] || LTP.q.devsDe(etat, ilots[0])[0];
      if (dev) actions.push({ type: "developper", carte: def.id, dev: dev.id });
    }
  }
  return actions;
}

// Note un choix d'événement d'après ses effets déclaratifs et une table de préférences.
function noterChoix(LTP, choix, pref) {
  let s = 0;
  for (const e of choix.effets || []) {
    if (!e) continue;
    if (e.confiance != null) s += (pref.confiance || 0) * e.confiance;
    else if (e.budget != null) s += (pref.budget || 0) * e.budget;
    else if (e.incident) s -= (pref.incident || 0) * (e.incident.gravite || 0.5);
    else if (e.carte) {
      const def = LTP.q.defCapacite(e.carte) || {};
      if (e.op === "equiper") s += def.provenance === "communaute" ? -(pref.communaute || 0) : (pref.carte || 0);
    } else if (e.chemin && typeof e.valeur === "number") {
      const w = pref.chemins[e.chemin] || 0;
      const v = e.op === "mul" ? e.valeur - 1 : e.op === "set" ? 0 : e.valeur;
      s += w * v;
    }
  }
  return s;
}
function meilleurChoix(LTP, def, pref) {
  if (!def || !def.choix || !def.choix.length) return undefined;
  let meilleur = def.choix[0], note = -Infinity;
  for (const c of def.choix) {
    const n = noterChoix(LTP, c, pref);
    if (n > note + 1e-12) { note = n; meilleur = c; }
  }
  return meilleur.id;
}

// Réponse au comité qui obtient la meilleure évaluation selon les vrais chiffres.
function meilleureReponse(LTP, etat, question) {
  const def = (LTP.donnees.comites || []).find((c) => c.id === question.id);
  if (!def || !def.reponses || !def.reponses.length) return question.reponses && question.reponses[0] && question.reponses[0].id;
  let meilleur = def.reponses[0], note = -Infinity;
  for (const r of def.reponses) {
    let n = 0;
    try { n = Number(r.evaluer(etat, LTP.q)) || 0; } catch { n = 0; }
    if (n > note) { note = n; meilleur = r; }
  }
  return meilleur.id;
}
function premiereReponse(LTP, etat, question) {
  const def = (LTP.donnees.comites || []).find((c) => c.id === question.id);
  if (def && def.reponses && def.reponses.length) return def.reponses[0].id;
  return question.reponses && question.reponses[0] && question.reponses[0].id;
}

// Préférences de choix d'événement.
const PREF_PRUDENT = {
  confiance: 0.05, budget: 0.00001, incident: 5, communaute: 1, carte: -0.2,
  chemins: { satisfaction: 0.3, moral: 0.5, comprehension: 2, "codebase.defautsLatents": -0.1, "codebase.couvertureReelle": 2,
    fileRevue: -0.01, fatigue: -0.5, menace: -2, "donnees.qualite": 1, "donnees.classification": 1, applisFantomes: -0.5 },
};
const PREF_EQUILIBRE = {
  confiance: 0.05, budget: 0.00001, incident: 3, communaute: 0.5, carte: 0.2,
  chemins: { satisfaction: 1, moral: 1, comprehension: 2, "codebase.defautsLatents": -0.05, "codebase.couvertureReelle": 2,
    fileRevue: -0.02, fatigue: -0.8, competence: 1, competenceIA: 1, menace: -2, "donnees.qualite": 1, applisFantomes: -0.3, backlog: -0.01 },
};
const PREF_ECONOMIQUE = { ...PREF_EQUILIBRE, budget: 0.00005 };

// Autonomie selon la criticité de l'îlot.
function autonomieSelonCriticite(etat) {
  const s = {};
  for (const id of etat.ordreIlots) {
    const c = etat.ilots[id].criticite;
    s[id] = { autonomie: c >= 0.9 ? 1 : c >= 0.6 ? 2 : 3 };
  }
  return s;
}

const REGLAGES_EQUILIBRES = {
  pair: "rotation", reviseurs: 1, approbation: "humain", taillePR: 400, ci: "complet", lint: "verrouille", tests: "mutation",
  revuesExplication: "mensuelles", autonomie: 2, sessionsMax: 2, modele: "intermediaire",
  modeleParTache: true, formation: "programme", plafondJetons: "souple",
  disciplineNon: "normale", specs: "legeres",
  catalogue: "sources", classification: "auto", bacASable: "filtre", accesProd: "lecture", approbationSorties: true,
  politiqueRegistre: "listeBlanche", passerelleMCP: true, politiqueNonTI: "encadrer",
  observabilite: "complete", embauche: "releve", objectif: "aucun",
};

// Fabrique la stratégie nommée. graine : pour la stratégie aléatoire.
export function creerStrategie(LTP, nom, graine = 1) {
  switch (nom) {
    case "tout-ia":
      return {
        nom,
        leviers(etat) {
          const actions = actionsLeviers(LTP, etat, {
            autonomie: 4, sessionsMax: 6, reviseurs: 1, approbation: "humain", taillePR: "libre", ci: "aucune", lint: "aucun",
            tests: "aucun", revuesExplication: "non", pair: "ia", modele: "haut", modeleParTache: false, formation: "aucune",
            plafondJetons: "aucun", disciplineNon: "laxiste", specs: "non",
            catalogue: "aucun", classification: "non", bacASable: "aucun", accesProd: "oui", approbationSorties: false,
            politiqueRegistre: "libre", passerelleMCP: false, politiqueNonTI: "ignorer",
            observabilite: "aucune", embauche: "gel", objectif: "pr",
          });
          // Libre-service : tout ce qui est disponible, partout où c'est suggéré.
          return actions.concat(actionsCartes(LTP, etat, { filtre: () => true }));
        },
        choix: (etat, def) => (def && def.choix && def.choix[0] ? def.choix[0].id : undefined),
        comite: (etat, question) => premiereReponse(LTP, etat, question),
      };

    case "musee": {
      const prudence = {};
      for (const id of ["iot", "operations", "appro", "portail"]) prudence[id] = { autonomie: 0 };
      return {
        nom,
        leviers(etat) {
          return actionsLeviers(LTP, etat, {
            autonomie: 1, sessionsMax: 1, reviseurs: 2, approbation: "proprietaire", taillePR: 200, ci: "complet", lint: "verrouille",
            tests: "integration", revuesExplication: "mensuelles", pair: "humain", modele: "intermediaire", modeleParTache: false,
            formation: "aucune", plafondJetons: "strict", disciplineNon: "ferme", specs: "obligatoires",
            catalogue: "inventaire", classification: "manuelle", bacASable: "filtre", accesProd: "non", approbationSorties: true,
            politiqueRegistre: "proprietaires", passerelleMCP: true, politiqueNonTI: "interdire",
            observabilite: "base", embauche: "normale", objectif: "aucun",
          }, prudence);
        },
        choix: (etat, def) => meilleurChoix(LTP, def, PREF_PRUDENT),
        comite: (etat, question) => meilleureReponse(LTP, etat, question),
      };
    }

    case "equilibree":
    case "economique": {
      const economique = nom === "economique";
      const reglages = economique ? { ...REGLAGES_EQUILIBRES, modele: "economique", formation: "aucune" } : REGLAGES_EQUILIBRES;
      return {
        nom,
        leviers(etat) {
          const actions = actionsLeviers(LTP, etat, reglages, autonomieSelonCriticite(etat));
          // Catalogue maîtrisé : cartes maison et fournisseurs, pas de communautaire ; on développe et on nomme des propriétaires.
          return actions.concat(actionsCartes(LTP, etat, {
            filtre: (def) => def.provenance === "maison" || def.provenance === "fournisseur" || def.provenance === "personnelle",
            developper: true,
            proprietaires: true,
          }));
        },
        choix: (etat, def) => meilleurChoix(LTP, def, economique ? PREF_ECONOMIQUE : PREF_EQUILIBRE),
        comite: (etat, question) => meilleureReponse(LTP, etat, question),
      };
    }

    case "par-defaut": {
      // L'interface mélange l'ordre des réponses du comité : cliquer la première affichée revient
      // à répondre au hasard. Les données, elles, mettent souvent la bonne réponse en premier.
      const hasard = hasardDe(graine ^ 0x5eed);
      return {
        nom,
        leviers: () => [],
        choix: (etat, def) => (def && def.choix && def.choix[0] ? def.choix[0].id : undefined),
        comite: (etat, question) => {
          const def = (LTP.donnees.comites || []).find((c) => c.id === question.id);
          const reps = def && def.reponses ? def.reponses : question.reponses || [];
          return reps.length ? reps[Math.floor(hasard() * reps.length)].id : premiereReponse(LTP, etat, question);
        },
      };
    }

    case "aleatoire": {
      const hasard = hasardDe(graine);
      const choisir = (l) => l[Math.floor(hasard() * l.length)];
      return {
        nom,
        leviers(etat) {
          const actions = [];
          for (const l of LTP.donnees.leviers) {
            if (!LTP.q.levierDebloque(etat, l.id) || hasard() < 0.5) continue;
            actions.push({ type: "levier", id: l.id, valeur: choisir(l.valeurs).v });
            if (l.surchargeable && hasard() < 0.3) actions.push({ type: "levier", id: l.id, valeur: choisir(l.valeurs).v, ilot: choisir(etat.ordreIlots) });
          }
          for (const c of LTP.donnees.capacites || []) {
            if (hasard() < 0.25) actions.push({ type: "equiper", carte: c.id, ilot: choisir(etat.ordreIlots) });
            if (hasard() < 0.1) {
              const devs = LTP.q.devsPresents(etat);
              if (devs.length) actions.push({ type: "developper", carte: c.id, dev: choisir(devs).id });
            }
          }
          return actions;
        },
        choix: (etat, def) => (def && def.choix && def.choix.length ? choisir(def.choix).id : undefined),
        comite(etat, question) {
          const def = (LTP.donnees.comites || []).find((c) => c.id === question.id);
          if (def && def.reponses && def.reponses.length) return choisir(def.reponses).id;
          return question.reponses && question.reponses.length ? choisir(question.reponses).id : undefined;
        },
      };
    }

    default:
      throw new Error(`stratégie inconnue : ${nom}`);
  }
}

// Toutes les stratégies, par nom.
export function strategies(LTP, graine = 1) {
  const r = {};
  for (const nom of NOMS_STRATEGIES) r[nom] = creerStrategie(LTP, nom, graine);
  return r;
}

// Joue une partie complète avec la stratégie nommée et renvoie l'état final.
export function jouer(LTP, nom, graine, options = {}) {
  const etat = LTP.moteur.creer(graine, options);
  return LTP.moteur.simuler(etat, LTP.K.etat.semainesTotal + 8, creerStrategie(LTP, nom, graine)).etat;
}
