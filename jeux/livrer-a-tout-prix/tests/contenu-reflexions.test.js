// Contenu des réflexions (09-donnees-reflexions.js) et des notes de service (08-donnees-calendrier.js) :
// structure des entrées, couverture des leviers, et comportement sur des années simulées jour par jour.
import { test } from "node:test";
import assert from "node:assert/strict";
import { charger } from "./charger.js";
import { creerStrategie } from "./strategies.js";

const LTP = charger();
const R = LTP.donnees.reflexions;
const TONS = new Set(["plus", "moins", "neutre"]);
const NIVEAUX = new Set(["stagiaire", "junior", "inter", "senior"]);
const idsDevs = new Set(LTP.donnees.devs.map((d) => d.id));
const traits = new Set(Object.keys(LTP.donnees.traits));

// Leviers qui demandent un contenu riche (au moins six entrées en tout).
const IMPORTANTS = [
  "licences", "autonomie", "reviseurs", "tests", "ci", "lint", "approbation", "taillePR", "modele", "sessionsMax",
  "plafondJetons", "formation", "disciplineNon", "specs", "accesProd", "bacASable", "approbationSorties",
  "politiqueRegistre", "passerelleMCP", "politiqueNonTI", "revuesExplication", "pair", "observabilite", "objectif",
  "embauche", "catalogue", "classification", "modeleParTache",
];

// Vérifie la forme d'une entrée de contenu ; renvoie la liste des problèmes.
function problemesEntree(e, ou) {
  const p = [];
  if (typeof e === "string") return e.trim() ? p : [`${ou} : chaîne vide`];
  if (typeof e === "function") return p;
  if (!e || typeof e !== "object") return [`${ou} : entrée invalide`];
  if (!(typeof e.texte === "string" && e.texte.trim()) && typeof e.texte !== "function") p.push(`${ou} : texte manquant`);
  if (e.devs != null) {
    if (!Array.isArray(e.devs) || !e.devs.length) p.push(`${ou} : devs vide`);
    else for (const d of e.devs) if (!idsDevs.has(d)) p.push(`${ou} : dev inconnu ${d}`);
  }
  if (e.traits != null) {
    if (!Array.isArray(e.traits) || !e.traits.length) p.push(`${ou} : traits vide`);
    else for (const t of e.traits) if (!traits.has(t)) p.push(`${ou} : trait inconnu ${t}`);
  }
  if (e.niveaux != null) for (const n of e.niveaux) if (!NIVEAUX.has(n)) p.push(`${ou} : niveau inconnu ${n}`);
  for (const k of ["pilote", "licence", "effet"]) if (e[k] != null && typeof e[k] !== "boolean") p.push(`${ou} : ${k} non booléen`);
  if (e.condition != null && typeof e.condition !== "function") p.push(`${ou} : condition non fonction`);
  if (e.ton != null && !TONS.has(e.ton)) p.push(`${ou} : ton inconnu ${e.ton}`);
  const connus = new Set(["texte", "devs", "traits", "niveaux", "pilote", "licence", "effet", "condition", "ton"]);
  for (const k of Object.keys(e)) if (!connus.has(k)) p.push(`${ou} : champ inconnu ${k}`);
  return p;
}

// Une entrée ciblée doit pouvoir s'appliquer à au moins un développeur (devs, traits, niveaux, pilote compatibles).
function cibleQuelquun(e) {
  if (!e || typeof e !== "object") return true;
  return LTP.donnees.devs.some((d) => {
    if (e.devs && !e.devs.includes(d.id)) return false;
    if (e.traits && !e.traits.some((t) => d.traits.includes(t))) return false;
    if (e.niveaux && !e.niveaux.includes(d.niveau)) return false;
    if (e.pilote != null && !!d.pilote !== e.pilote) return false;
    return true;
  });
}

test("réflexions par levier : chaque levier a du contenu, avec des clés de valeur valides", () => {
  const problemes = [];
  for (const l of LTP.donnees.leviers) {
    const c = R.leviers[l.id];
    assert.ok(c && typeof c === "object", `levier sans contenu : ${l.id}`);
    const valeurs = new Set(l.valeurs.map((x) => String(x.v)));
    let total = 0, avant = 0, apres = 0, cibles = 0;
    for (const [cle, liste] of Object.entries(c)) {
      assert.ok(cle === "*" || valeurs.has(cle), `${l.id} : valeur inconnue ${cle}`);
      assert.ok(Array.isArray(liste) && liste.length, `${l.id}.${cle} : liste vide`);
      liste.forEach((e, i) => {
        problemes.push(...problemesEntree(e, `${l.id}.${cle}[${i}]`));
        if (!cibleQuelquun(e)) problemes.push(`${l.id}.${cle}[${i}] : ne vise personne`);
        total += 1;
        if (e && e.effet === false) avant += 1;
        if (e && e.effet === true) apres += 1;
        if (e && (e.devs || e.traits || e.niveaux || e.pilote != null)) cibles += 1;
      });
    }
    if (IMPORTANTS.includes(l.id)) {
      assert.ok(total >= 6, `${l.id} : ${total} entrées (6 au moins)`);
      assert.ok(cibles >= 2, `${l.id} : ${cibles} entrées ciblées`);
    }
    // Un levier sans délai a des réflexions d'avant et d'après ; un levier à délai s'appuie sur des conditions.
    if (!(l.delai > 0)) assert.ok(avant >= 1 && apres >= 1, `${l.id} : avant ${avant}, après ${apres}`);
  }
  assert.deepEqual(problemes, []);
});

test("notes de service : titre, texte, signataire connu et au moins deux réactions valides", () => {
  const signataires = new Set(["ginette", "regis", "berube", "lavigueur", "kevin"]);
  const vus = new Set();
  for (const c of LTP.donnees.calendrier) {
    const ou = `semaine ${c.semaine}`;
    assert.ok(!vus.has(c.semaine), `${ou} : deux entrées la même semaine`);
    vus.add(c.semaine);
    assert.ok(typeof c.titre === "string" && c.titre.trim().length > 3, `${ou} : titre`);
    assert.ok(typeof c.texte === "string" && c.texte.length > 60, `${ou} : texte trop court`);
    const phrases = c.texte.split(/[.!?…»](?:\s|$)/).filter((x) => x.trim()).length;
    assert.ok(phrases >= 2 && phrases <= 5, `${ou} : ${phrases} phrases`);
    assert.ok(signataires.has(c.signataire) && LTP.donnees.personnages[c.signataire], `${ou} : signataire ${c.signataire}`);
    assert.ok(Array.isArray(c.reactions) && c.reactions.length >= 2 && c.reactions.length <= 4, `${ou} : ${c.reactions && c.reactions.length} réactions`);
    const devs = new Set();
    for (const r of c.reactions) {
      assert.ok(idsDevs.has(r.dev), `${ou} : dev inconnu ${r.dev}`);
      assert.ok(!devs.has(r.dev), `${ou} : ${r.dev} réagit deux fois`);
      devs.add(r.dev);
      assert.ok(typeof r.texte === "string" && r.texte.trim(), `${ou} : réaction vide`);
      assert.ok(r.ton == null || TONS.has(r.ton), `${ou} : ton ${r.ton}`);
    }
  }
});

test("nouveautés : chaque note de service fait parler, et chaque clé correspond à une note", () => {
  const cles = new Set();
  for (const c of LTP.donnees.calendrier) {
    const l = [...(c.leviers || []), ...(c.valeurs || []).map((x) => `${x.levier}:${(x.v || []).join(",")}`), ...(c.panneaux || [])];
    for (const k of l) {
      cles.add(k);
      const liste = R.nouveautes[k];
      assert.ok(Array.isArray(liste) && liste.length >= 3, `nouveauté sans contenu : ${k}`);
      const p = [];
      liste.forEach((e, i) => p.push(...problemesEntree(e, `${k}[${i}]`)));
      assert.deepEqual(p, []);
    }
  }
  for (const k of Object.keys(R.nouveautes)) assert.ok(cles.has(k), `clé de nouveauté sans note de service : ${k}`);
});

test("situations : identifiant unique, libellé court, au moins quatre lignes dont des lignes ciblées", () => {
  assert.ok(R.situations.length >= 30, `${R.situations.length} situations`);
  const ids = new Set();
  const problemes = [];
  for (const s of R.situations) {
    assert.ok(typeof s.id === "string" && s.id && !ids.has(s.id), `id : ${s.id}`);
    ids.add(s.id);
    assert.equal(typeof s.quand, "function", `${s.id} : quand`);
    assert.ok(typeof s.libelle === "string" && s.libelle.trim(), `${s.id} : libelle`);
    const mots = s.libelle.trim().split(/[\s  ']+/).length;
    assert.ok(mots >= 1 && mots <= 5, `${s.id} : libellé de ${mots} mots`);
    assert.ok(Array.isArray(s.lignes) && s.lignes.length >= 4, `${s.id} : ${s.lignes && s.lignes.length} lignes`);
    assert.ok(s.lignes.some((e) => e && (e.devs || e.traits || e.niveaux)), `${s.id} : aucune ligne ciblée`);
    if (s.levier != null) assert.ok(LTP.donnees.leviersParId[s.levier], `${s.id} : levier ${s.levier}`);
    if (s.ton != null) assert.ok(TONS.has(s.ton), `${s.id} : ton`);
    s.lignes.forEach((e, i) => {
      problemes.push(...problemesEntree(e, `${s.id}[${i}]`));
      if (e && typeof e === "object" && e.effet != null) problemes.push(`${s.id}[${i}] : effet n'a pas de sens ici`);
    });
  }
  assert.deepEqual(problemes, []);
});

// --- Années simulées, jour par jour ---

function signature(etat, strat) {
  let n = 0;
  if (strat.suivreCalendrier !== false) {
    const s = LTP.q.semaineCourante(etat);
    n = LTP.donnees.calendrier.filter((c) => c.semaine <= s).length;
  }
  return `${etat.trimestre}:${n}`;
}

// Joue une partie jour par jour ; visiter(etat) est appelé chaque journée ouvrable, réflexions calculées.
// « plafonnee » : tout à l'IA, mais avec un plafond strict de jetons dès qu'il est débloqué.
function strategie(nom, graine) {
  if (nom !== "plafonnee") return creerStrategie(LTP, nom, graine);
  const base = creerStrategie(LTP, "tout-ia", graine);
  return {
    ...base,
    leviers(etat) {
      const a = base.leviers(etat);
      if (LTP.q.levierDebloque(etat, "plafondJetons") && LTP.q.levier(etat, "plafondJetons") !== "strict") a.push({ type: "levier", id: "plafondJetons", valeur: "strict" });
      return a;
    },
  };
}

function partie(nom, graine, visiter) {
  const strat = strategie(nom, graine);
  let e = LTP.moteur.creer(graine, {});
  const agir = (a) => { const r = LTP.moteur.action(e, a); if (!r.erreur) e = r.etat; return r; };
  let sig = null;
  const regler = () => {
    const s = signature(e, strat);
    if (s === sig) return;
    sig = s;
    for (const a of strat.leviers(e) || []) agir(a);
  };
  agir({ type: "phase", phase: "planification" });
  regler();
  agir({ type: "phase", phase: "execution" });
  for (let garde = 0; e.phase !== "fin" && garde < 600; garde++) {
    if (e.phase === "comite") {
      for (const qq of e.comite.questions) agir({ type: "comite", question: qq.id, reponse: strat.comite(e, qq) });
      agir({ type: "phase", phase: "planification" });
      regler();
      agir({ type: "phase", phase: "execution" });
      continue;
    }
    if (e.evenementEnAttente) {
      const def = LTP.q.defEvenement(e.evenementEnAttente.id);
      const r = agir({ type: "choix", choix: strat.choix(e, def) });
      if (r.erreur) agir({ type: "choix", choix: def.choix[0].id });
      continue;
    }
    if ((e.jour || 0) === 0) regler();
    LTP.systemes.reflexions.assurer(e);
    visiter(e);
    const r = LTP.moteur.jour(e);
    if (!r.avance) break;
    e = r.etat;
  }
  return e;
}

const PARTIES = [
  ["tout-ia", 1], ["tout-ia", 2], ["musee", 1], ["equilibree", 1], ["equilibree", 2], ["economique", 1],
  ["aleatoire", 1], ["aleatoire", 2], ["aleatoire", 3], ["aleatoire", 4], ["par-defaut", 1], ["plafonnee", 1],
];

// Une seule passe de simulation pour les tests qui suivent.
const bilan = (() => {
  const declenchees = {}, choisies = {}, exceptions = [], textesVides = [];
  let activiteGenerique = 0, avecSituation = 0, activiteMalgre = 0, total = 0, avecLevier = 0, choisiLevier = 0;
  const motifs = {};
  const appeler = (f, args, ou) => {
    try { return f(...args); } catch (err) { exceptions.push(`${ou} : ${err.message}`); return false; }
  };
  const verifierEntree = (e, etat, dev, ou) => {
    if (e && typeof e === "object" && typeof e.condition === "function") appeler(e.condition, [etat, LTP.q, dev], `${ou}.condition`);
    const t = e && typeof e === "object" ? e.texte : e;
    if (typeof t === "function") {
      const x = appeler(t, [etat, LTP.q, dev], `${ou}.texte`);
      if (!(typeof x === "string" && x.trim())) textesVides.push(ou);
    }
  };
  for (const [nom, graine] of PARTIES) {
    partie(nom, graine, (e) => {
      const devs = LTP.q.devsPresents(e);
      const lundi = (e.jour || 0) === 0;
      for (const d of devs) {
        const il = e.ilots[d.ilot];
        for (const s of R.situations) {
          const v = appeler(s.quand, [e, LTP.q, d, il], `situation ${s.id}`);
          if (v === true || v > 0) {
            declenchees[s.id] = (declenchees[s.id] || 0) + 1;
            s.lignes.forEach((x, i) => verifierEntree(x, e, d, `${s.id}[${i}]`));
          }
        }
        if (lundi && d.id.length % 3 === e.semaine % 3) {
          for (const [id, c] of Object.entries(R.leviers)) for (const [cle, l] of Object.entries(c)) l.forEach((x, i) => verifierEntree(x, e, d, `${id}.${cle}[${i}]`));
        }
      }
      for (const r of e.reflexions.liste) {
        total += 1;
        motifs[r.motif.split(":")[0]] = (motifs[r.motif.split(":")[0]] || 0) + 1;
        if (r.motif.startsWith("situation:")) choisies[r.motif.slice(10)] = (choisies[r.motif.slice(10)] || 0) + 1;
        if (r.immediat) continue;
        const dev = e.devs[r.dev];
        const cands = LTP.systemes.reflexions.candidats(e, dev);
        if (cands.some((c) => (c.motif === "levier" || c.motif === "reaction") && c.poids >= 2.6)) {
          avecLevier += 1;
          if (r.motif === "levier" || r.motif === "reaction") choisiLevier += 1;
        }
        if (cands.some((c) => c.motif.startsWith("situation:"))) {
          avecSituation += 1;
          if (r.motif.startsWith("activite:")) activiteMalgre += 1;
          if (r.motif.startsWith("activite:") && !((LTP.q.defDev(r.dev).bulles || {})[r.motif.slice(9)] || []).includes(r.texte)) activiteGenerique += 1;
        }
      }
    });
  }
  if (process.env.BILAN) console.log(JSON.stringify({ activiteGenerique, avecLevier, choisiLevier, declenchees, choisies, avecSituation, activiteMalgre, total, motifs, exceptions: [...new Set(exceptions)].slice(0, 20) }, null, 1));
  return { declenchees, choisies, exceptions, textesVides, avecSituation, activiteMalgre, total, motifs, avecLevier, choisiLevier };
})();

test("années simulées : aucune condition ne lève d'exception, aucun texte calculé n'est vide", () => {
  assert.deepEqual([...new Set(bilan.exceptions)], []);
  assert.deepEqual([...new Set(bilan.textesVides)], []);
  assert.ok(bilan.total > 1000, `${bilan.total} réflexions`);
});

test("années simulées : chaque situation se produit, et presque toutes sont dites au moins une fois", () => {
  const jamais = [...R.situations.map((s) => s.id).filter((id) => !bilan.declenchees[id])];
  assert.deepEqual(jamais, [], `situations jamais vraies : ${jamais.join(", ")}`);
  const muettes = R.situations.map((s) => s.id).filter((id) => !bilan.choisies[id]);
  assert.ok(muettes.length <= 2, `situations jamais choisies : ${muettes.join(", ")}`);
});

test("années simulées : peu de réflexions d'activité générique quand une situation est possible", () => {
  const part = bilan.activiteMalgre / Math.max(1, bilan.avecSituation);
  assert.ok(part <= 0.15, `${(part * 100).toFixed(1)} % d'activité (${bilan.activiteMalgre}/${bilan.avecSituation}) ; motifs : ${JSON.stringify(bilan.motifs)}`);
});
