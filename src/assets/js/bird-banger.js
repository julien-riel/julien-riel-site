/* Bird Banger — protéger un vignoble des oiseaux avec des canons effaroucheurs.
 *
 * Tout est en mètres dans le monde et en secondes de jeu. Une journée dure
 * DUREE_JOUR secondes réelles à ×1 ; le temps est donc très compressé, et
 * chaque oiseau du jeu représente une petite bande plutôt qu'un individu.
 */

// ---------- Constantes du monde ----------

const MONDE = { w: 420, h: 280 };
const VIGNE = { x0: 70, y0: 44, x1: 350, y1: 212, pasRang: 5, pasCell: 8 };
const FORET = { nord: 30, est: 388 };
const CHAI = { x: 130, y: 262, w: 26, h: 14 };
const MAISONS = [
  { x: 22, y: 84, nom: 'voisin de l’ouest' },
  { x: 404, y: 236, nom: 'voisin de l’est' },
  { x: 318, y: 264, nom: 'voisin du chemin' },
];
const DISTANCE_VOISINAGE = 125;   // m — distance minimale recommandée d'une habitation
const DUREE_JOUR = 36;            // s réelles à ×1
const DUREE_JOUR_CLAIR = 30;      // dont 30 s de jour, 6 s de nuit
const NB_JOURS = 24;              // du 1er au 24 septembre, vendange le 24 au soir
const RECOLTE_TOTALE = 30000;     // kg sur pied à la véraison
const BUDGET_DEPART = 3000;
const SEUIL_AMENDE = 100;
const AMENDE = 750;

const NB_RANGS = Math.floor((VIGNE.x1 - VIGNE.x0) / VIGNE.pasRang);
const NB_CELLULES_RANG = Math.floor((VIGNE.y1 - VIGNE.y0) / VIGNE.pasCell);
const NB_CELLULES = NB_RANGS * NB_CELLULES_RANG;
const KG_CELLULE = RECOLTE_TOTALE / NB_CELLULES;

// ---------- Les oiseaux ----------

export const ESPECES = {
  merle: {
    nom: 'Merle d’Amérique', latin: 'Turdus migratorius', couleur: '#4a4a52', ventre: '#d8763a', taille: 1,
    bande: [1, 4], vitesse: 16, mange: 0.8, sens: 0.9, seuil: 0.2, hab: 0.015, retour: 0.5, delaiRetour: [10, 22], satiete: [14, 24], sol: false,
    saison: (j) => Math.max(0.2, 1.4 - j / 24),
    desc: 'Seul ou à deux ou trois, il picore les baies une à une dès la véraison. Très craintif : presque n’importe quel bang le fait partir, mais il revient dès que le calme est revenu.',
  },
  jaseur: {
    nom: 'Jaseur d’Amérique', latin: 'Bombycilla cedrorum', couleur: '#b7925a', ventre: '#e8d28a', taille: 0.9,
    bande: [6, 14], vitesse: 19, mange: 0.9, sens: 0.6, seuil: 0.3, hab: 0.045, retour: 0.6, delaiRetour: [8, 18], satiete: [12, 20], sol: false,
    saison: (j) => 0.7 + (j > 6 && j < 18 ? 0.6 : 0),
    desc: 'Arrive en petites bandes nomades, avale les baies entières. Peu impressionné par le bruit et s’y habitue très vite : c’est l’oiseau qui use le plus rapidement l’effet d’un canon.',
  },
  etourneau: {
    nom: 'Étourneau sansonnet', latin: 'Sturnus vulgaris', couleur: '#2a2733', ventre: '#5b5878', taille: 0.95,
    bande: [14, 32], vitesse: 22, mange: 1.1, sens: 0.85, seuil: 0.25, hab: 0.022, retour: 0.5, delaiRetour: [8, 20], satiete: [14, 24], sol: false,
    saison: (j) => 0.5 + j / 10,
    desc: 'Le grand ravageur des vignobles. Les bandes grossissent tout au long de septembre, à mesure que les jeunes de l’année se rassemblent. Sensible au bruit, mais le nombre fait le dégât.',
  },
  quiscale: {
    nom: 'Quiscale bronzé', latin: 'Quiscalus quiscula', couleur: '#1c2436', ventre: '#3b5a8a', taille: 1.15,
    bande: [4, 8], vitesse: 17, mange: 1.5, sens: 0.55, seuil: 0.45, hab: 0.028, retour: 0.5, delaiRetour: [10, 24], satiete: [14, 26], sol: false,
    saison: () => 0.8,
    desc: 'Gros, hardi, au bec puissant qui ouvre les baies. Il ne bouge que pour une détonation proche et forte : un canon simple en bout de portée ne l’impressionne pas.',
  },
  corneille: {
    nom: 'Corneille d’Amérique', latin: 'Corvus brachyrhynchos', couleur: '#141414', ventre: '#2c2c2c', taille: 1.35,
    bande: [2, 5], vitesse: 15, mange: 2.0, sens: 0.75, seuil: 0.35, hab: 0.07, retour: 0.4, delaiRetour: [12, 30], satiete: [16, 28], sol: false,
    saison: () => 0.6,
    desc: 'Craintive au premier bang, mais elle apprend en quelques détonations qu’un canon qui tire toujours au même endroit et au même rythme ne fait rien. Déplacer les canons est la seule parade.',
  },
  dindon: {
    nom: 'Dindon sauvage', latin: 'Meleagris gallopavo', couleur: '#5a3b26', ventre: '#8a5e3a', taille: 2.2,
    bande: [3, 7], vitesse: 5, mange: 3.5, sens: 0.5, seuil: 0.9, hab: 0.02, retour: 0.4, delaiRetour: [24, 44], satiete: [20, 34], sol: true,
    saison: () => 0.35,
    desc: 'Arrive à pied par la lisière et vide les grappes du bas. Seule une détonation très proche ou très forte (double détonation, cartouche pyrotechnique) le fait courir vers le bois.',
  },
};
const LISTE_ESPECES = Object.keys(ESPECES);

// ---------- Les modèles de canons ----------

export const MODELES = {
  simple: {
    nom: 'Canon à propane simple', court: 'Simple', prix: 450, portee: 85, cone: 180, force: 1.0, habFacteur: 1, tirs: 1,
    desc: 'Le classique : une bouteille de propane, une chambre et un tube. Un bang à intervalle fixe dans la direction où on le pointe. Bon marché, mais les oiseaux apprennent vite son rythme.',
  },
  trepied: {
    nom: 'Canon sur trépied', court: 'Trépied', prix: 650, portee: 120, cone: 180, force: 1.0, habFacteur: 1, tirs: 1,
    desc: 'Le même canon, juché à deux mètres du sol : le son passe au-dessus du feuillage et porte beaucoup plus loin. Idéal au centre d’un grand bloc.',
  },
  rotatif: {
    nom: 'Canon rotatif', court: 'Rotatif', prix: 750, portee: 90, cone: 360, force: 1.0, habFacteur: 0.6, tirs: 1, rotatif: true,
    desc: 'La tête pivote d’un cran à chaque tir, et couvre les 360°. Le bang ne vient jamais tout à fait du même côté : les oiseaux s’y font moins vite.',
  },
  double: {
    nom: 'Canon à double détonation', court: 'Double', prix: 900, portee: 95, cone: 180, force: 1.6, habFacteur: 1.1, tirs: 2,
    desc: 'Deux détonations coup sur coup. Le seul canon fixe qui impressionne les gros oiseaux : quiscales, corneilles et dindons. Bruyant pour le voisinage.',
  },
};
export const PISTOLET = { prix: 75, cartouches: 25, prixBoite: 60, porteeTir: 100, rayon: 45, force: 2.0, habFacteur: 0.3 };
const INTERVALLE = { min: 3, max: 15, defaut: 6 };

// ---------- Utilitaires ----------

const alea = (a, b) => a + Math.random() * (b - a);
const aleaInt = (a, b) => Math.floor(alea(a, b + 1));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const angleEntre = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));

function celluleXY(i) {
  const r = Math.floor(i / NB_CELLULES_RANG), c = i % NB_CELLULES_RANG;
  return { x: VIGNE.x0 + r * VIGNE.pasRang + VIGNE.pasRang / 2, y: VIGNE.y0 + c * VIGNE.pasCell + VIGNE.pasCell / 2 };
}
function celluleA(x, y) {
  if (x < VIGNE.x0 || x >= VIGNE.x0 + NB_RANGS * VIGNE.pasRang || y < VIGNE.y0 || y >= VIGNE.y0 + NB_CELLULES_RANG * VIGNE.pasCell) return -1;
  return Math.floor((x - VIGNE.x0) / VIGNE.pasRang) * NB_CELLULES_RANG + Math.floor((y - VIGNE.y0) / VIGNE.pasCell);
}
function dansVigne(x, y) { return celluleA(x, y) >= 0; }

// ---------- État ----------

let S;
let idSuite = 1;

function etatInitial() {
  return {
    t: 0, vitesse: 1, fini: false, jourDeBut: 0,
    raisins: new Float32Array(NB_CELLULES).fill(1),
    oiseaux: [], canons: [], ondes: [], projectiles: [], retours: [], vagues: [],
    budget: BUDGET_DEPART, depenses: { materiel: 0, cartouches: 0, amendes: 0 },
    pistolet: false, cartouches: 0,
    plaintes: 0, amendes: 0,
    habGlobale: Object.fromEntries(LISTE_ESPECES.map((e) => [e, 0])),
    journal: [], stats: { tirs: 0, effrayes: 0, mangeKg: 0, parEspece: Object.fromEntries(LISTE_ESPECES.map((e) => [e, 0])) },
    selection: null, mode: 'observer', modeleAPlacer: null,
    dernierBang: null,
  };
}

export const jour = () => Math.min(NB_JOURS, Math.floor(S.t / DUREE_JOUR) + 1);
const fracJour = () => (S.t % DUREE_JOUR) / DUREE_JOUR;
export const estJour = () => (S.t % DUREE_JOUR) < DUREE_JOUR_CLAIR;
export function heure() {
  const f = S.t % DUREE_JOUR;
  const h = f < DUREE_JOUR_CLAIR ? 6 + 14 * f / DUREE_JOUR_CLAIR : 20 + 10 * (f - DUREE_JOUR_CLAIR) / (DUREE_JOUR - DUREE_JOUR_CLAIR);
  const hh = Math.floor(h) % 24, mm = Math.floor((h % 1) * 60);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}
/* Le sucre monte de 17 à 24,5 °Brix au fil du mois. Sous 20 °Brix, le raisin est vert :
   il part en jus, payé une misère ; au-dessus, le prix monte jusqu'à la pleine maturité. */
const BRIX_DEPART = 17, BRIX_MUR = 24.5;
export const BRIX_SEUIL = 20;
const PRIX_VERT = 250, PRIX_SEUIL = 900, PRIX_MUR = 1650;   // $/t
export const brix = () => Math.min(BRIX_MUR, BRIX_DEPART + (BRIX_MUR - BRIX_DEPART) * (jour() - 1 + fracJour()) / (NB_JOURS - 1));
export const prixPourBrix = (b) => Math.round(b < BRIX_SEUIL
  ? PRIX_VERT + (b - BRIX_DEPART) / (BRIX_SEUIL - BRIX_DEPART) * (PRIX_SEUIL - PRIX_VERT)
  : PRIX_SEUIL + (b - BRIX_SEUIL) / (BRIX_MUR - BRIX_SEUIL) * (PRIX_MUR - PRIX_SEUIL));
export const prixTonne = () => prixPourBrix(brix());
/* Jours à attendre avant que le raisin passe le seuil de maturité. */
export const joursAvantMaturite = () => Math.max(0, Math.ceil((BRIX_SEUIL - brix()) / ((BRIX_MUR - BRIX_DEPART) / (NB_JOURS - 1))));
export const recolteKg = () => { let s = 0; for (let i = 0; i < NB_CELLULES; i++) s += S.raisins[i]; return s * KG_CELLULE; };

function noter(texte, classe) {
  S.journal.unshift({ t: S.t, jour: jour(), heure: heure(), texte, classe });
  if (S.journal.length > 40) S.journal.length = 40;
}

// ---------- Oiseaux ----------

function pointLisiere(sol) {
  // Les bandes arrivent par le bois au nord ou à l'est ; les dindons à pied par le nord.
  if (sol || Math.random() < 0.6) return { x: alea(60, 380), y: alea(4, FORET.nord - 4) };
  return { x: alea(FORET.est + 4, MONDE.w - 4), y: alea(30, 230) };
}

function choisirCellule(pres, rayon, eviter) {
  // Cellule avec des raisins, proche de `pres`, loin du dernier bang si possible.
  let meilleure = -1, score = -Infinity;
  for (let k = 0; k < 40; k++) {
    const i = Math.floor(Math.random() * NB_CELLULES);
    if (S.raisins[i] < 0.08) continue;
    const p = celluleXY(i);
    let s = S.raisins[i] * 2 - (pres ? dist(p, pres) / (rayon || 60) : 0);
    if (eviter && dist(p, eviter) < 70) s -= 3;
    if (s > score) { score = s; meilleure = i; }
  }
  return meilleure;
}

export function apparaitre(espece, n, origine, satiete) {
  const E = ESPECES[espece];
  const nb = n || aleaInt(E.bande[0], E.bande[1]);
  const o = origine || pointLisiere(E.sol);
  const cibleI = choisirCellule(E.sol ? { x: o.x, y: VIGNE.y0 + 20 } : null, 80, S.dernierBang && S.t - S.dernierBang.t < 20 ? S.dernierBang : null);
  const cible = cibleI >= 0 ? celluleXY(cibleI) : { x: MONDE.w / 2, y: MONDE.h / 2 };
  const bande = idSuite++;
  for (let k = 0; k < nb; k++) {
    const ang = Math.random() * Math.PI * 2, r = Math.random() * (E.sol ? 6 : 14);
    S.oiseaux.push({
      id: idSuite++, espece, bande, x: o.x + Math.cos(ang) * r, y: o.y + Math.sin(ang) * r,
      etat: 'vol', cible: cibleI, cx: cible.x + Math.cos(ang) * r, cy: cible.y + Math.sin(ang) * r,
      vx: 0, vy: 0, phase: Math.random() * 6.28, timer: 0, satiete: satiete || alea(E.satiete[0], E.satiete[1]), fuite: null,
    });
  }
  return nb;
}

function planifierJour(j) {
  // Les vagues de la journée, plus nombreuses à mesure que le sucre monte et que les bandes grossissent.
  const n = Math.round(3 + j * 0.5 + Math.random());
  S.vagues = [];
  for (let k = 0; k < n; k++) {
    const u = Math.random();
    const f = u < 0.4 ? Math.random() * 0.3 : u < 0.65 ? 0.3 + Math.random() * 0.4 : 0.7 + Math.random() * 0.28;
    S.vagues.push((j - 1) * DUREE_JOUR + f * DUREE_JOUR_CLAIR);
  }
  S.vagues.sort((a, b) => a - b);
}

function especeDuJour(j) {
  const poids = LISTE_ESPECES.map((e) => ESPECES[e].saison(j));
  let r = Math.random() * poids.reduce((a, b) => a + b, 0);
  for (let i = 0; i < poids.length; i++) { r -= poids[i]; if (r <= 0) return LISTE_ESPECES[i]; }
  return 'etourneau';
}

function majOiseau(o, dt) {
  const E = ESPECES[o.espece];
  o.phase += dt * (E.sol ? 6 : 18);
  if (o.etat === 'vol' || o.etat === 'fuite') {
    let tx, ty, v = E.vitesse;
    if (o.etat === 'fuite') { tx = o.fuite.x; ty = o.fuite.y; v *= E.sol ? 2.2 : 1.7; }
    else { tx = o.cx; ty = o.cy; }
    const dx = tx - o.x, dy = ty - o.y, d = Math.hypot(dx, dy);
    if (d < 1.5) {
      if (o.etat === 'fuite') { o.etat = 'parti'; return; }
      o.etat = 'mange'; o.vx = o.vy = 0; o.timer = 0;
      return;
    }
    const pas = Math.min(d, v * dt);
    o.vx = dx / d * v; o.vy = dy / d * v;
    o.x += dx / d * pas; o.y += dy / d * pas;
    if (o.etat === 'fuite') { o.timer += dt; if (o.timer > 12) o.etat = 'parti'; }
    return;
  }
  if (o.etat === 'mange') {
    o.timer += dt; o.satiete -= dt;
    const i = o.cible;
    if (i >= 0 && S.raisins[i] > 0) {
      const q = Math.min(S.raisins[i], E.mange * dt / KG_CELLULE);
      S.raisins[i] -= q;
      S.stats.mangeKg += q * KG_CELLULE; S.stats.parEspece[o.espece] += q * KG_CELLULE;
    }
    if (o.satiete <= 0) { partir(o); return; }
    if (i < 0 || S.raisins[i] < 0.05 || o.timer > alea(3, 6)) {
      const j = choisirCellule(o, E.sol ? 20 : 30, null);
      if (j < 0) { partir(o); return; }
      const p = celluleXY(j);
      o.cible = j; o.cx = p.x + alea(-2, 2); o.cy = p.y + alea(-3, 3); o.etat = 'vol'; o.timer = 0;
    }
  }
}

function partir(o) {
  o.etat = 'fuite'; o.timer = 0;
  o.fuite = ESPECES[o.espece].sol ? { x: o.x, y: 6 } : pointLisiere(false);
}

function effrayer(o, origine) {
  const E = ESPECES[o.espece];
  o.etat = 'fuite'; o.timer = 0;
  // On s'éloigne du bang, vers la lisière la plus opposée.
  const ang = Math.atan2(o.y - origine.y, o.x - origine.x);
  let fx = o.x + Math.cos(ang) * 400, fy = o.y + Math.sin(ang) * 400;
  if (E.sol) { fx = o.x + Math.cos(ang) * 60; fy = 6; }
  o.fuite = { x: clamp(fx, -20, MONDE.w + 20), y: clamp(fy, -20, MONDE.h + 20) };
  S.stats.effrayes++;
  if (Math.random() < E.retour && o.satiete > 3) S.retours.push({ t: S.t + alea(E.delaiRetour[0], E.delaiRetour[1]), espece: o.espece, satiete: o.satiete * 0.8 });
}

// ---------- Canons ----------

export function placer(modele, x, y, gratuit) {
  const M = MODELES[modele];
  if (!M) return null;
  if (!gratuit && S.budget < M.prix) { noter(`Budget insuffisant pour un ${M.nom.toLowerCase()} (${M.prix} $).`, 'alerte'); return null; }
  if (!positionValide(x, y)) { noter('Impossible de poser un canon ici.', 'alerte'); return null; }
  const c = {
    id: idSuite++, modele, x, y, dir: -Math.PI / 2, intervalle: INTERVALLE.defaut, aleatoire: false,
    prochain: S.t + alea(1, 4), hab: Object.fromEntries(LISTE_ESPECES.map((e) => [e, 0])), tirs: 0, pose: S.t, dernierDeplacement: S.t,
  };
  if (!gratuit) { S.budget -= M.prix; S.depenses.materiel += M.prix; }
  S.canons.push(c);
  const m = MAISONS.find((h) => dist(h, c) < DISTANCE_VOISINAGE);
  noter(`${M.nom} posé${m ? `, à moins de ${DISTANCE_VOISINAGE} m du ${m.nom}` : ''}.`, m ? 'alerte' : '');
  return c;
}

export function positionValide(x, y) {
  if (x < 4 || y < FORET.nord || x > FORET.est || y > MONDE.h - 4) return false;
  if (Math.abs(x - CHAI.x) < CHAI.w / 2 + 3 && Math.abs(y - CHAI.y) < CHAI.h / 2 + 3) return false;
  return !MAISONS.some((h) => dist(h, { x, y }) < 12);
}

export function deplacer(c, x, y) {
  if (!positionValide(x, y)) return false;
  const d = Math.hypot(c.x - x, c.y - y);
  c.x = x; c.y = y;
  if (d > 25) {
    for (const e of LISTE_ESPECES) c.hab[e] *= 0.25;
    c.dernierDeplacement = S.t;
    noter(`${MODELES[c.modele].nom} déplacé de ${Math.round(d)} m : les oiseaux devront s’y réhabituer.`, 'bon');
  }
  return true;
}

export function retirer(c) {
  const i = S.canons.indexOf(c);
  if (i < 0) return;
  S.canons.splice(i, 1);
  const rev = Math.round(MODELES[c.modele].prix / 2);
  S.budget += rev; S.depenses.materiel -= rev;
  if (S.selection === c) S.selection = null;
  noter(`${MODELES[c.modele].nom} retiré et revendu ${rev} $.`);
}

function bang(origine, portee, cone, dir, force, habFacteur, source) {
  S.stats.tirs++;
  S.dernierBang = { x: origine.x, y: origine.y, t: S.t };
  S.ondes.push({ x: origine.x, y: origine.y, r: portee, cone, dir, t: 0, force });
  const presentes = new Set();
  const fuyards = [];
  let effrayes = 0;
  for (const o of S.oiseaux) {
    if (o.etat === 'fuite' || o.etat === 'parti') continue;
    const d = dist(o, origine);
    if (d > portee) continue;
    if (cone < 360 && angleEntre(Math.atan2(o.y - origine.y, o.x - origine.x), dir) > (cone / 2) * Math.PI / 180) continue;
    const E = ESPECES[o.espece];
    presentes.add(o.espece);
    const attenuation = d < portee * 0.45 ? 1 : 1 - 0.65 * (d - portee * 0.45) / (portee * 0.55);
    const f = force * attenuation;
    if (f < E.seuil) continue;
    const hab = Math.min(0.95, (source ? source.hab[o.espece] : 0) + S.habGlobale[o.espece]);
    const p = E.sens * Math.min(1.4, f) * (1 - hab);
    if (Math.random() < p) { effrayer(o, origine); effrayes++; fuyards.push(o); }
  }
  // Contagion : quand une partie de la bande décolle, les voisins suivent.
  for (const f of fuyards) {
    for (const o of S.oiseaux) {
      if (o.bande !== f.bande || o.etat === 'fuite' || o.etat === 'parti' || dist(o, f) > 30) continue;
      if (Math.random() < 0.5) { effrayer(o, origine); effrayes++; }
    }
  }
  // Accoutumance : les espèces présentes apprennent, surtout si le canon est prévisible.
  for (const e of presentes) {
    const E = ESPECES[e];
    let inc = E.hab * habFacteur;
    if (source && source.aleatoire) inc *= 0.55;
    if (source) source.hab[e] = Math.min(0.95, source.hab[e] + inc);
    S.habGlobale[e] = Math.min(0.25, S.habGlobale[e] + inc * 0.08);
  }
  // Voisinage.
  for (const h of MAISONS) {
    const d = dist(h, origine);
    if (d < DISTANCE_VOISINAGE) S.plaintes += (1 - d / DISTANCE_VOISINAGE) * 3.2 * force;
  }
  if (S.plaintes >= SEUIL_AMENDE) {
    S.plaintes = 45; S.amendes++; S.budget -= AMENDE; S.depenses.amendes += AMENDE;
    noter(`Plainte au conseil municipal : amende de ${AMENDE} $. Éloignez les canons des maisons.`, 'alerte');
  }
  return effrayes;
}

export function tirer(c) {
  const M = MODELES[c.modele];
  if (M.rotatif) c.dir += Math.PI / 4;
  c.tirs++;
  let n = bang(c, M.portee, M.cone, c.dir, M.force, M.habFacteur, c);
  if (M.tirs > 1) S.ondes.push({ x: c.x, y: c.y, r: M.portee, cone: M.cone, dir: c.dir, t: -0.25, force: M.force });
  c.prochain = S.t + c.intervalle * (c.aleatoire ? alea(0.5, 1.5) : 1);
  return n;
}

// ---------- Pistolet ----------

export function acheterPistolet() {
  if (S.pistolet) return true;
  if (S.budget < PISTOLET.prix) { noter('Budget insuffisant pour le pistolet.', 'alerte'); return false; }
  S.budget -= PISTOLET.prix; S.depenses.materiel += PISTOLET.prix; S.pistolet = true;
  S.cartouches += PISTOLET.cartouches;
  noter(`Pistolet lance-cartouches acheté, avec une première boîte de ${PISTOLET.cartouches} cartouches.`);
  return true;
}
export function acheterCartouches() {
  if (!S.pistolet) return acheterPistolet();
  if (S.budget < PISTOLET.prixBoite) { noter('Budget insuffisant pour une boîte de cartouches.', 'alerte'); return false; }
  S.budget -= PISTOLET.prixBoite; S.depenses.cartouches += PISTOLET.prixBoite; S.cartouches += PISTOLET.cartouches;
  noter(`Boîte de ${PISTOLET.cartouches} cartouches achetée.`);
  return true;
}
export function tirerCartouche(x, y) {
  if (!S.pistolet || S.cartouches <= 0) { noter('Plus de cartouches : achetez une boîte.', 'alerte'); return false; }
  if (!estJour()) { noter('Pas de tir la nuit : le règlement municipal l’interdit.', 'alerte'); return false; }
  S.cartouches--;
  const dx = x - CHAI.x, dy = y - (CHAI.y - 10), d = Math.hypot(dx, dy);
  const p = Math.min(d, PISTOLET.porteeTir);
  S.projectiles.push({ x: CHAI.x, y: CHAI.y - 10, cx: CHAI.x + dx / d * p, cy: CHAI.y - 10 + dy / d * p, v: 75, trace: [] });
  return true;
}

// ---------- Boucle de simulation ----------

function pas(dt) {
  if (S.fini) return;
  const jAvant = jour();
  S.t += dt;
  const j = jour();
  if (j !== jAvant || S.t === dt) {
    if (S.t === dt) noter('1er septembre : la véraison est passée, les baies se colorent. Les premiers merles rôdent.', 'bon');
    else noter(`${j} septembre. Sucre : ${brix().toFixed(1)} °Brix, payé ${prixTonne()} $/t${brix() < BRIX_SEUIL ? ' (raisin vert)' : ''}. Récolte sur pied : ${(recolteKg() / 1000).toFixed(1)} t.`);
    if (brix() >= BRIX_SEUIL && brix() - (BRIX_MUR - BRIX_DEPART) / (NB_JOURS - 1) < BRIX_SEUIL) noter(`${BRIX_SEUIL} °Brix : le raisin est mûr pour le vin. Chaque jour de plus vaut maintenant de l’argent — et des oiseaux.`, 'bon');
    planifierJour(j);
    for (const c of S.canons) for (const e of LISTE_ESPECES) c.hab[e] = Math.max(0, c.hab[e] - 0.03);
    if (j === 8) noter('Les bandes d’étourneaux grossissent : les jeunes de l’année se rassemblent.', 'alerte');
    if (j === NB_JOURS) noter('Dernier jour avant la vendange. Tenez bon.', 'alerte');
  }
  if (S.t >= NB_JOURS * DUREE_JOUR) { vendanger(); return; }
  // Nuit : rien ne vole, les canons se taisent, le voisinage se calme.
  S.plaintes = Math.max(0, S.plaintes - 6 / DUREE_JOUR * dt);
  if (estJour()) {
    while (S.vagues.length && S.vagues[0] <= S.t) { S.vagues.shift(); apparaitre(especeDuJour(j)); }
    for (let i = S.retours.length - 1; i >= 0; i--) {
      if (S.retours[i].t <= S.t) { apparaitre(S.retours[i].espece, 1, null, S.retours[i].satiete); S.retours.splice(i, 1); }
    }
    for (const c of S.canons) if (S.t >= c.prochain) tirer(c);
  } else {
    // À la tombée de la nuit, tout le monde rentre au bois.
    for (const o of S.oiseaux) if (o.etat === 'mange' || o.etat === 'vol') partir(o);
    for (const c of S.canons) if (c.prochain < S.t + 1) c.prochain = S.t + 1;
  }
  for (const o of S.oiseaux) majOiseau(o, dt);
  S.oiseaux = S.oiseaux.filter((o) => o.etat !== 'parti');
  for (const p of S.projectiles) {
    const dx = p.cx - p.x, dy = p.cy - p.y, d = Math.hypot(dx, dy);
    const s = Math.min(d, p.v * dt);
    p.x += dx / d * s; p.y += dy / d * s;
    p.trace.push({ x: p.x, y: p.y }); if (p.trace.length > 8) p.trace.shift();
    if (d - s < 0.5) { p.fini = true; bang(p, PISTOLET.rayon, 360, 0, PISTOLET.force, PISTOLET.habFacteur, null); }
  }
  S.projectiles = S.projectiles.filter((p) => !p.fini);
  for (const w of S.ondes) w.t += dt;
  S.ondes = S.ondes.filter((w) => w.t < 0.9);
}

export function avancer(secondes) {
  const h = 1 / 30;
  for (let t = 0; t < secondes && !S.fini; t += h) pas(h);
}

export function vendanger() {
  if (S.fini) return S.bilan;
  S.fini = true; S.vitesse = 0;
  const kg = recolteKg(), part = kg / RECOLTE_TOTALE, valeur = kg / 1000 * prixTonne();
  const dep = S.depenses.materiel + S.depenses.cartouches + S.depenses.amendes;
  // La note compare le net à ce qu'aurait rapporté une pleine vendange, mûre, sans un sou de protection :
  // rentrer tout le raisin ne vaut rien s'il est vert, et une saison de canons doit payer.
  const ideal = RECOLTE_TOTALE / 1000 * PRIX_MUR;
  const rendement = Math.max(0, valeur - dep) / ideal;
  const vert = brix() < BRIX_SEUIL;
  let note = rendement >= 0.85 && S.amendes === 0 ? 'A' : rendement >= 0.72 ? 'B' : rendement >= 0.58 ? 'C' : rendement >= 0.4 ? 'D' : 'E';
  if (vert && note < 'D') note = 'D';
  const immaturite = kg / 1000 * (PRIX_MUR - prixTonne());
  S.bilan = { jour: jour(), brix: brix(), kg, part, prixTonne: prixTonne(), valeur, depenses: dep, net: valeur - dep, rendement, vert, immaturite, note, amendes: S.amendes, tirs: S.stats.tirs, effrayes: S.stats.effrayes, mangeKg: S.stats.mangeKg, parEspece: { ...S.stats.parEspece } };
  noter(`Vendange le ${jour()} septembre à ${brix().toFixed(1)} °Brix : ${(kg / 1000).toFixed(1)} t rentrées.`, 'bon');
  return S.bilan;
}

export function recommencer() {
  for (const d of document.querySelectorAll('dialog[open]')) d.close();
  S = etatInitial();
  window.birdBanger.etat = S;
  planifierJour(1);
  pas(1 / 30);
}

// ---------- Rendu ----------

const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const vue = { s: 1, ox: 0, oy: 0, w: 0, h: 0 };
let souris = null, glisse = null, dpr = 1;
let afficherPortees = true;

function redimensionner() {
  const r = canvas.parentElement.getBoundingClientRect();
  dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
  canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
  vue.w = r.width; vue.h = r.height;
  vue.s = Math.min(r.width / MONDE.w, r.height / MONDE.h);
  vue.ox = (r.width - MONDE.w * vue.s) / 2; vue.oy = (r.height - MONDE.h * vue.s) / 2;
}
const versMonde = (px, py) => ({ x: (px - vue.ox) / vue.s, y: (py - vue.oy) / vue.s });
export const ecran = (p) => ({ x: vue.ox + p.x * vue.s, y: vue.oy + p.y * vue.s });

// Fond statique (sol, rangs, bois, bâtiments), redessiné seulement quand la taille change.
let fond = null, fondCle = '';
function dessinerFond() {
  const cle = canvas.width + 'x' + canvas.height;
  if (fond && fondCle === cle) return;
  fondCle = cle;
  fond = document.createElement('canvas'); fond.width = canvas.width; fond.height = canvas.height;
  const g = fond.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#0f1410'; g.fillRect(0, 0, vue.w, vue.h);
  g.translate(vue.ox, vue.oy); g.scale(vue.s, vue.s);
  // Prairie sèche.
  g.fillStyle = '#b9a96f'; g.fillRect(0, 0, MONDE.w, MONDE.h);
  // Bruit léger d'herbe.
  let graine = 7;
  const rnd = () => { graine = (graine * 16807) % 2147483647; return graine / 2147483647; };
  for (let k = 0; k < 2200; k++) {
    g.fillStyle = rnd() < 0.5 ? 'rgba(90,110,40,.18)' : 'rgba(255,240,200,.16)';
    g.fillRect(rnd() * MONDE.w, rnd() * MONDE.h, 1.2, 1.2);
  }
  // Chemin au sud.
  g.fillStyle = '#8b7d63'; g.fillRect(0, MONDE.h - 14, MONDE.w, 14);
  g.strokeStyle = 'rgba(255,255,255,.18)'; g.setLineDash([6, 8]); g.lineWidth = .6;
  g.beginPath(); g.moveTo(0, MONDE.h - 7); g.lineTo(MONDE.w, MONDE.h - 7); g.stroke(); g.setLineDash([]);
  // Allées entre les rangs (sol nu) puis feuillage.
  g.fillStyle = '#a8955f';
  g.fillRect(VIGNE.x0 - 4, VIGNE.y0 - 4, NB_RANGS * VIGNE.pasRang + 8, NB_CELLULES_RANG * VIGNE.pasCell + 8);
  // Bois : nord et est.
  const arbre = (x, y, r) => {
    g.fillStyle = 'rgba(20,40,18,.35)'; g.beginPath(); g.arc(x + 1.2, y + 1.4, r, 0, 6.29); g.fill();
    const t = rnd();
    g.fillStyle = t < 0.3 ? '#2f5a2b' : t < 0.6 ? '#3c6d31' : t < 0.85 ? '#4a7a36' : '#7f6a2a';
    g.beginPath(); g.arc(x, y, r, 0, 6.29); g.fill();
    g.fillStyle = 'rgba(255,255,220,.13)'; g.beginPath(); g.arc(x - r * .3, y - r * .3, r * .45, 0, 6.29); g.fill();
  };
  g.fillStyle = '#3a5a2c'; g.fillRect(0, 0, MONDE.w, FORET.nord - 2); g.fillRect(FORET.est + 2, 0, MONDE.w - FORET.est, MONDE.h - 14);
  for (let k = 0; k < 260; k++) arbre(rnd() * MONDE.w, rnd() * (FORET.nord + 2) - 3, 3 + rnd() * 4);
  for (let k = 0; k < 130; k++) arbre(FORET.est + rnd() * (MONDE.w - FORET.est) + 1, rnd() * (MONDE.h - 14), 3 + rnd() * 4);
  // Bâtiments.
  const maison = (x, y, w, h, toit, mur, nom) => {
    g.fillStyle = 'rgba(0,0,0,.25)'; g.fillRect(x - w / 2 + 1.5, y - h / 2 + 1.5, w, h);
    g.fillStyle = mur; g.fillRect(x - w / 2, y - h / 2, w, h);
    g.fillStyle = toit; g.fillRect(x - w / 2, y - h / 2, w, h / 2);
    g.fillStyle = 'rgba(0,0,0,.18)'; g.fillRect(x - w / 2, y - 0.6, w, 1.2);
    if (nom) { g.fillStyle = '#2a2418'; g.font = '600 4.2px Outfit, sans-serif'; g.textAlign = 'center'; g.fillText(nom, x, y + h / 2 + 5); }
  };
  for (const h of MAISONS) maison(h.x, h.y, 12, 10, '#7a3f34', '#d8cdb5', 'voisin');
  maison(CHAI.x, CHAI.y, CHAI.w, CHAI.h, '#5b5f66', '#c9bfa8', 'chai');
  g.fillStyle = '#e8d9b0'; g.beginPath(); g.arc(CHAI.x, CHAI.y - 10, 1.6, 0, 6.29); g.fill();
}

function couleurCellule(v) {
  // Plein de raisins : vert profond piqué de violet. Vide : vert-jaune fatigué.
  const r = Math.round(70 + (1 - v) * 60), gg = Math.round(120 - (1 - v) * 18), b = Math.round(55 + v * 30);
  return `rgb(${r},${gg},${b})`;
}

function dessiner() {
  dessinerFond();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(fond, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.translate(vue.ox, vue.oy); ctx.scale(vue.s, vue.s);

  // Rangs de vigne.
  for (let i = 0; i < NB_CELLULES; i++) {
    const r = Math.floor(i / NB_CELLULES_RANG), c = i % NB_CELLULES_RANG;
    const x = VIGNE.x0 + r * VIGNE.pasRang + 1, y = VIGNE.y0 + c * VIGNE.pasCell;
    const v = S.raisins[i];
    ctx.fillStyle = couleurCellule(v);
    ctx.fillRect(x, y, VIGNE.pasRang - 2, VIGNE.pasCell - 0.6);
    if (v > 0.15) {
      ctx.fillStyle = `rgba(88,40,110,${0.25 + v * 0.55})`;
      ctx.fillRect(x + 0.5, y + 1.2, VIGNE.pasRang - 3, 1.2);
      if (v > 0.55) ctx.fillRect(x + 0.5, y + 4.6, VIGNE.pasRang - 3, 1.2);
    }
  }

  // Zones sensibles et portées en mode placement.
  const placement = S.mode === 'placer' && S.modeleAPlacer;
  if (placement) {
    for (const h of MAISONS) {
      ctx.fillStyle = 'rgba(226,106,85,.10)'; ctx.strokeStyle = 'rgba(226,106,85,.55)'; ctx.lineWidth = .6; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(h.x, h.y, DISTANCE_VOISINAGE, 0, 6.29); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  // Canons et leurs portées.
  for (const c of S.canons) {
    const M = MODELES[c.modele];
    const sel = S.selection === c;
    if (afficherPortees || sel) dessinerCone(c.x, c.y, M.portee, M.cone, c.dir, sel ? 0.22 : 0.09, habMax(c));
    dessinerCanon(c.x, c.y, c.dir, M, sel, habMax(c));
  }
  if (placement && souris) {
    const M = MODELES[S.modeleAPlacer];
    const ok = positionValide(souris.x, souris.y) && S.budget >= M.prix;
    dessinerCone(souris.x, souris.y, M.portee, M.cone, -Math.PI / 2, 0.2, 0, ok ? null : '#e26a55');
    ctx.globalAlpha = 0.8; dessinerCanon(souris.x, souris.y, -Math.PI / 2, M, true, 0); ctx.globalAlpha = 1;
  }
  if (S.mode === 'pistolet' && souris) {
    const dx = souris.x - CHAI.x, dy = souris.y - (CHAI.y - 10), d = Math.hypot(dx, dy), p = Math.min(d, PISTOLET.porteeTir);
    const tx = CHAI.x + dx / d * p, ty = CHAI.y - 10 + dy / d * p;
    ctx.strokeStyle = 'rgba(230,184,76,.55)'; ctx.setLineDash([2, 3]); ctx.lineWidth = .7;
    ctx.beginPath(); ctx.moveTo(CHAI.x, CHAI.y - 10); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(230,184,76,.7)'; ctx.beginPath(); ctx.arc(tx, ty, PISTOLET.rayon, 0, 6.29); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx - 4, ty); ctx.lineTo(tx + 4, ty); ctx.moveTo(tx, ty - 4); ctx.lineTo(tx, ty + 4); ctx.stroke();
  }

  // Ondes de choc.
  for (const w of S.ondes) {
    if (w.t < 0) continue;
    const k = w.t / 0.9;
    ctx.globalAlpha = (1 - k) * 0.9;
    ctx.strokeStyle = w.force >= 1.6 ? '#ffd27a' : '#fff1c9'; ctx.lineWidth = 1.6 - k;
    const r = w.r * (0.15 + 0.85 * k);
    ctx.beginPath();
    if (w.cone >= 360) ctx.arc(w.x, w.y, r, 0, 6.29);
    else { const a = w.cone / 2 * Math.PI / 180; ctx.arc(w.x, w.y, r, w.dir - a, w.dir + a); }
    ctx.stroke();
    if (k < 0.15) { ctx.fillStyle = '#fff6d8'; ctx.beginPath(); ctx.arc(w.x, w.y, 3 + k * 20, 0, 6.29); ctx.fill(); }
    ctx.globalAlpha = 1;
  }

  // Projectiles.
  for (const p of S.projectiles) {
    ctx.strokeStyle = 'rgba(255,200,120,.6)'; ctx.lineWidth = .8; ctx.beginPath();
    p.trace.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y))); ctx.stroke();
    ctx.fillStyle = '#ffe9a8'; ctx.beginPath(); ctx.arc(p.x, p.y, 1.4, 0, 6.29); ctx.fill();
  }

  // Oiseaux : les terrestres d'abord, puis les ombres, puis les volants.
  for (const o of S.oiseaux) if (ESPECES[o.espece].sol) dessinerOiseau(o, false);
  for (const o of S.oiseaux) if (!ESPECES[o.espece].sol && o.etat !== 'mange') { ctx.fillStyle = 'rgba(0,0,0,.2)'; ctx.beginPath(); ctx.ellipse(o.x + 2.5, o.y + 3, 2, 1, 0, 0, 6.29); ctx.fill(); }
  for (const o of S.oiseaux) if (!ESPECES[o.espece].sol) dessinerOiseau(o, o.etat !== 'mange');

  // Nuit.
  if (!estJour()) {
    const f = (S.t % DUREE_JOUR - DUREE_JOUR_CLAIR) / (DUREE_JOUR - DUREE_JOUR_CLAIR);
    const a = Math.sin(f * Math.PI) * 0.72;
    ctx.fillStyle = `rgba(8,12,40,${a})`; ctx.fillRect(-vue.ox / vue.s, -vue.oy / vue.s, vue.w / vue.s, vue.h / vue.s);
  }
}

function habMax(c) { return Math.max(...LISTE_ESPECES.map((e) => c.hab[e])); }

function dessinerCone(x, y, portee, cone, dir, alpha, hab, couleur) {
  const col = couleur || (hab > 0.6 ? '#e26a55' : hab > 0.3 ? '#e6b84c' : '#8fd08a');
  ctx.fillStyle = col; ctx.globalAlpha = alpha;
  ctx.beginPath();
  if (cone >= 360) ctx.arc(x, y, portee, 0, 6.29);
  else { const a = cone / 2 * Math.PI / 180; ctx.moveTo(x, y); ctx.arc(x, y, portee, dir - a, dir + a); ctx.closePath(); }
  ctx.fill();
  ctx.globalAlpha = Math.min(1, alpha * 3); ctx.strokeStyle = col; ctx.lineWidth = .5; ctx.stroke();
  ctx.globalAlpha = 1;
}

function dessinerCanon(x, y, dir, M, sel, hab) {
  ctx.save(); ctx.translate(x, y);
  ctx.fillStyle = 'rgba(0,0,0,.3)'; ctx.beginPath(); ctx.ellipse(1, 1.5, 4.5, 3, 0, 0, 6.29); ctx.fill();
  if (M.portee > 100) { ctx.strokeStyle = '#3a3a3a'; ctx.lineWidth = .8; for (let k = 0; k < 3; k++) { const a = k * 2.09 + 0.5; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(a) * 5, Math.sin(a) * 5); ctx.stroke(); } }
  ctx.fillStyle = '#c9c9c9'; ctx.beginPath(); ctx.ellipse(0, 0, 3.2, 2.2, 0, 0, 6.29); ctx.fill();
  ctx.strokeStyle = '#2b2b2b'; ctx.lineWidth = .5; ctx.stroke();
  ctx.fillStyle = '#d9d9d9'; ctx.beginPath(); ctx.arc(-2.2, 0, 1.6, 0, 6.29); ctx.fill(); // bouteille
  ctx.rotate(dir);
  ctx.fillStyle = '#3d3d3d'; ctx.fillRect(0, -1.1, M.tirs > 1 ? 11 : 9, 2.2);
  if (M.tirs > 1) ctx.fillRect(0, -2.3, 9, 1.2);
  ctx.fillStyle = '#111'; ctx.fillRect(8, -1.4, 1.5, 2.8);
  ctx.restore();
  if (sel) { ctx.strokeStyle = '#fff'; ctx.lineWidth = .7; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.29); ctx.stroke(); ctx.setLineDash([]); }
  if (hab > 0.3) { ctx.fillStyle = hab > 0.6 ? '#e26a55' : '#e6b84c'; ctx.beginPath(); ctx.arc(x + 5, y - 5, 1.8, 0, 6.29); ctx.fill(); }
}

function dessinerOiseau(o, enVol) {
  const E = ESPECES[o.espece];
  const t = E.taille * (enVol ? 1.2 : 1);
  ctx.save(); ctx.translate(o.x, o.y);
  const ang = (o.vx || o.vy) ? Math.atan2(o.vy, o.vx) : (o.phase % 6.28) - 3.14;
  ctx.rotate(ang);
  if (E.sol) {
    ctx.fillStyle = 'rgba(0,0,0,.25)'; ctx.beginPath(); ctx.ellipse(0.6, 0.8, 3.2 * t / 2.2 * 2, 2 * t / 2.2 * 2, 0, 0, 6.29); ctx.fill();
    ctx.fillStyle = E.couleur; ctx.beginPath(); ctx.ellipse(0, 0, 3 * t / 2.2 * 2, 1.9 * t / 2.2 * 2, 0, 0, 6.29); ctx.fill();
    ctx.fillStyle = E.ventre; ctx.beginPath(); ctx.ellipse(-2.5, 0, 2, 2.6, 0, 0, 6.29); ctx.fill(); // queue en éventail
    ctx.fillStyle = '#c94a4a'; ctx.beginPath(); ctx.arc(3.4, 0, 0.8, 0, 6.29); ctx.fill();
    ctx.restore(); return;
  }
  const bat = enVol ? Math.sin(o.phase) : 0.15;
  ctx.fillStyle = E.couleur;
  // Ailes.
  ctx.beginPath(); ctx.moveTo(0.3 * t, 0); ctx.lineTo(-1.2 * t, -3.2 * t * (0.35 + Math.abs(bat) * 0.75)); ctx.lineTo(-1.6 * t, 0); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(0.3 * t, 0); ctx.lineTo(-1.2 * t, 3.2 * t * (0.35 + Math.abs(bat) * 0.75)); ctx.lineTo(-1.6 * t, 0); ctx.closePath(); ctx.fill();
  // Corps, ventre, tête.
  ctx.beginPath(); ctx.ellipse(0, 0, 2 * t, 0.9 * t, 0, 0, 6.29); ctx.fill();
  ctx.fillStyle = E.ventre; ctx.beginPath(); ctx.ellipse(0.2 * t, 0.25 * t, 1.1 * t, 0.45 * t, 0, 0, 6.29); ctx.fill();
  ctx.fillStyle = E.couleur; ctx.beginPath(); ctx.arc(1.9 * t, 0, 0.7 * t, 0, 6.29); ctx.fill();
  ctx.fillStyle = '#e0c060'; ctx.beginPath(); ctx.moveTo(2.5 * t, -0.3 * t); ctx.lineTo(3.3 * t, 0); ctx.lineTo(2.5 * t, 0.3 * t); ctx.closePath(); ctx.fill();
  ctx.restore();
}

// ---------- Son ----------

let audio = null, sonActif = true;
function sonBang(force) {
  if (!sonActif) return;
  try {
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === 'suspended') audio.resume();
    const t0 = audio.currentTime;
    const n = audio.sampleRate * 0.35, buf = audio.createBuffer(1, n, audio.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 3);
    const src = audio.createBufferSource(); src.buffer = buf;
    const lp = audio.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 900;
    const g = audio.createGain(); g.gain.setValueAtTime(0.5 * Math.min(1.3, force), t0); g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.4);
    const osc = audio.createOscillator(); osc.type = 'sine'; osc.frequency.setValueAtTime(110, t0); osc.frequency.exponentialRampToValueAtTime(35, t0 + 0.25);
    const g2 = audio.createGain(); g2.gain.setValueAtTime(0.35, t0); g2.gain.exponentialRampToValueAtTime(0.001, t0 + 0.3);
    src.connect(lp).connect(g).connect(audio.destination); osc.connect(g2).connect(audio.destination);
    src.start(t0); osc.start(t0); osc.stop(t0 + 0.3);
  } catch (e) { /* pas de son */ }
}

// ---------- Interface ----------

const $ = (id) => document.getElementById(id);
const fmtKg = (kg) => kg >= 1000 ? `${(kg / 1000).toFixed(1)} t` : `${Math.round(kg)} kg`;
const fmtDollars = (v) => `${Math.round(v).toLocaleString('fr-CA')} $`;

function majInterface() {
  $('valJour').textContent = `${jour()} septembre`;
  $('valHeure').textContent = estJour() ? heure() : `${heure()} · nuit`;
  $('valBrix').textContent = `${brix().toFixed(1)} °Brix${brix() < BRIX_SEUIL ? ' · vert' : ''}`;
  $('valPrix').textContent = `${prixTonne()} $/t`;
  $('valPrix').className = brix() < BRIX_SEUIL ? 'alerte' : '';
  $('valPrix').textContent = `${prixTonne()} $/t`;
  const kg = recolteKg(), part = kg / RECOLTE_TOTALE;
  $('valRecolte').textContent = fmtKg(kg);
  $('valPart').textContent = `${Math.round(part * 100)} %`;
  $('barreRecolte').style.width = `${part * 100}%`;
  $('barreRecolte').style.background = part > 0.8 ? '' : part > 0.6 ? 'linear-gradient(90deg,#b47b2a,#e6b84c)' : 'linear-gradient(90deg,#8a3a2a,#e26a55)';
  $('valValeur').textContent = fmtDollars(kg / 1000 * prixTonne());
  $('valMange').textContent = fmtKg(S.stats.mangeKg);
  $('valBudget').textContent = fmtDollars(S.budget);
  $('valBudget').classList.toggle('alerte', S.budget < 0);
  $('valDepenses').textContent = fmtDollars(S.depenses.materiel + S.depenses.cartouches + S.depenses.amendes);
  $('valCartouches').textContent = S.pistolet ? `${S.cartouches}` : 'pas de pistolet';
  const comptes = Object.fromEntries(LISTE_ESPECES.map((e) => [e, 0]));
  for (const o of S.oiseaux) if (o.etat === 'mange' || o.etat === 'vol') comptes[o.espece]++;
  let total = 0;
  for (const e of LISTE_ESPECES) { const el = $('nb-' + e); if (el) el.textContent = comptes[e]; total += comptes[e]; el.parentElement.classList.toggle('present', comptes[e] > 0); }
  $('valOiseaux').textContent = total;
  $('barrePlaintes').style.width = `${Math.min(100, S.plaintes)}%`;
  $('valPlaintes').textContent = S.plaintes < 25 ? 'calme' : S.plaintes < 55 ? 'on grogne' : S.plaintes < 85 ? 'on se plaint' : 'la mairie s’en mêle';
  $('valPlaintes').classList.toggle('alerte', S.plaintes >= 55);
  $('valTirs').textContent = S.stats.tirs;
  $('valEffrayes').textContent = S.stats.effrayes;
  for (const b of document.querySelectorAll('[data-vitesse]')) b.classList.toggle('actif', Number(b.dataset.vitesse) === S.vitesse);
  $('btnPistolet').hidden = !S.pistolet;
  $('btnPistolet').classList.toggle('actif', S.mode === 'pistolet');
  $('btnPistolet').textContent = `🔫 Pistolet · ${S.cartouches}`;
  for (const b of document.querySelectorAll('[data-modele]')) {
    const M = MODELES[b.dataset.modele];
    b.classList.toggle('actif', S.mode === 'placer' && S.modeleAPlacer === b.dataset.modele);
    b.disabled = S.budget < M.prix;
  }
  $('btnAcheterPistolet').textContent = S.pistolet ? `Boîte de ${PISTOLET.cartouches} cartouches · ${PISTOLET.prixBoite} $` : `Pistolet + ${PISTOLET.cartouches} cartouches · ${PISTOLET.prix} $`;
  $('btnAcheterPistolet').disabled = S.budget < (S.pistolet ? PISTOLET.prixBoite : PISTOLET.prix);
  majJournal();
  majFiche();
  $('astuce').hidden = S.canons.length > 0 || S.t > 60;
  const nuit = !estJour();
  $('valHeure').classList.toggle('nuit', nuit);
}

let journalCle = '';
function majJournal() {
  const cle = S.journal.length ? S.journal[0].t + ':' + S.journal.length : '';
  if (cle === journalCle) return; journalCle = cle;
  $('journal').innerHTML = S.journal.slice(0, 14).map((l) => `<li class="${l.classe || ''}"><time>${l.jour} sept. ${l.heure}</time>${l.texte}</li>`).join('');
}

function majFiche() {
  const c = S.selection;
  $('ficheCanon').hidden = !c;
  $('boutique').hidden = !!c;
  if (!c) return;
  const M = MODELES[c.modele];
  $('ficheTitre').textContent = M.nom;
  $('ficheModele').textContent = `${M.cone >= 360 ? '360°' : `cône de ${M.cone}°`} · portée ${M.portee} m · ${M.tirs > 1 ? 'deux détonations' : 'une détonation'}`;
  $('ficheTirs').textContent = c.tirs;
  const m = MAISONS.map((h) => dist(h, c)).sort((a, b) => a - b)[0];
  $('ficheVoisin').textContent = `${Math.round(m)} m`;
  $('ficheVoisin').classList.toggle('alerte', m < DISTANCE_VOISINAGE);
  if (document.activeElement !== $('intervalle')) $('intervalle').value = c.intervalle;
  $('valIntervalle').textContent = `${c.intervalle} s${c.aleatoire ? ' ± 50 %' : ''}`;
  $('aleatoire').checked = c.aleatoire;
  $('ligneDirection').hidden = M.cone >= 360;
  if (document.activeElement !== $('direction')) $('direction').value = Math.round(((c.dir * 180 / Math.PI) % 360 + 360) % 360);
  $('habituation').innerHTML = LISTE_ESPECES.map((e) => {
    const h = Math.min(1, c.hab[e] + S.habGlobale[e]);
    return `<div class="hab"><span>${ESPECES[e].nom}</span><i><b style="width:${h * 100}%;background:${h > 0.6 ? '#e26a55' : h > 0.3 ? '#e6b84c' : '#8fd08a'}"></b></i><em>${Math.round(h * 100)} %</em></div>`;
  }).join('');
  const hm = habMax(c);
  $('ficheConseil').textContent = hm > 0.6 ? 'Les oiseaux ne l’écoutent plus : déplacez-le d’au moins 25 m, ou changez son rythme.' : hm > 0.3 ? 'L’effet s’émousse. Un intervalle aléatoire ralentit l’accoutumance.' : 'Il fait encore son effet.';
}

function afficherBilan(b) {
  $('bilanNote').textContent = b.note;
  $('bilanNote').dataset.note = b.note;
  $('bilanTitre').textContent = b.vert ? 'Vendangé vert.' : b.rendement >= 0.85 ? 'Belle vendange.' : b.rendement >= 0.58 ? 'Vendange honorable.' : b.part >= 0.45 ? 'Vendange amputée.' : 'Les oiseaux ont vendangé avant vous.';
  const pire = LISTE_ESPECES.slice().sort((a, z) => b.parEspece[z] - b.parEspece[a])[0];
  $('bilanTable').innerHTML = `
    <dt>Vendangé le</dt><dd>${b.jour} septembre à ${b.brix.toFixed(1)} °Brix</dd>
    <dt>Récolte rentrée</dt><dd class="grand">${fmtKg(b.kg)} · ${Math.round(b.part * 100)} %</dd>
    <dt>Mangée par les oiseaux</dt><dd>${fmtKg(b.mangeKg)}${b.mangeKg > 0 ? `, surtout par ${ESPECES[pire].nom.toLowerCase().replace(/^./, (s) => s)}s` : ''}</dd>
    <dt>Valeur (${b.prixTonne} $/t${b.vert ? ', prix du raisin de jus' : ''})</dt><dd>${fmtDollars(b.valeur)}</dd>
    ${b.immaturite > 0 ? `<dt>Manque à gagner, raisin pas mûr</dt><dd>${fmtDollars(b.immaturite)}</dd>` : ''}
    <dt>Dépenses d’effarouchement</dt><dd>${fmtDollars(b.depenses)}${b.amendes ? ` dont ${b.amendes} amende${b.amendes > 1 ? 's' : ''}` : ''}</dd>
    <dt>Net</dt><dd class="grand">${fmtDollars(b.net)}</dd>
    <dt>Par rapport à une pleine vendange mûre</dt><dd>${Math.round(b.rendement * 100)} %</dd>
    <dt>Détonations</dt><dd>${b.tirs}, ${b.effrayes} oiseaux effrayés</dd>`;
  $('bilanMorale').textContent = b.vert ? `À ${b.brix.toFixed(1)} °Brix, ce raisin fait du jus, pas du vin : il est payé ${b.prixTonne} $ la tonne au lieu de ${PRIX_MUR}. Une saison de canons sert justement à pouvoir attendre les 20 °Brix.`
    : b.note === 'A' ? 'Canons déplacés à temps, intervalles variés, voisins épargnés : c’est exactement ce que fait un bon vigneron.'
    : b.note === 'B' ? 'Bon travail. Pour faire mieux : déplacer les canons dès que l’accoutumance monte, et garder quelques cartouches pour les gros oiseaux.'
    : b.note === 'C' ? 'Les canons ont fait leur part, mais les oiseaux ont appris leur rythme. Un canon qui ne bouge jamais devient un bruit de fond.'
    : 'Dans la vraie vie, un vignoble aussi exposé se couvre de filets. Les canons ne suffisent pas seuls : ils gagnent du temps, pas la guerre.';
  $('bilan').showModal();
}

// ---------- Événements ----------

function positionSouris(ev) {
  const r = canvas.getBoundingClientRect();
  return versMonde(ev.clientX - r.left, ev.clientY - r.top);
}
function canonSous(p) {
  let meilleur = null, dm = 9 / Math.max(0.6, vue.s / 3);
  for (const c of S.canons) { const d = dist(c, p); if (d < dm) { dm = d; meilleur = c; } }
  return meilleur;
}

function brancher() {
  window.addEventListener('resize', redimensionner);
  canvas.addEventListener('pointermove', (ev) => {
    souris = positionSouris(ev);
    if (glisse) {
      glisse.bouge = glisse.bouge || dist(glisse.depart, souris) > 2;
      if (glisse.bouge) { glisse.canon.x = clamp(souris.x, 4, MONDE.w - 4); glisse.canon.y = clamp(souris.y, 4, MONDE.h - 4); }
    }
    canvas.style.cursor = S.mode === 'placer' || S.mode === 'pistolet' ? 'crosshair' : canonSous(souris) ? 'grab' : 'default';
  });
  canvas.addEventListener('pointerleave', () => { souris = null; });
  canvas.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    const p = positionSouris(ev); souris = p;
    if (!audio) sonBang(0.001);
    if (S.fini) return;
    if (S.mode === 'placer' && S.modeleAPlacer) {
      const c = placer(S.modeleAPlacer, p.x, p.y);
      if (c) { S.selection = c; if (S.budget < MODELES[S.modeleAPlacer].prix) { S.mode = 'observer'; S.modeleAPlacer = null; } }
      return;
    }
    if (S.mode === 'pistolet') { tirerCartouche(p.x, p.y); return; }
    const c = canonSous(p);
    if (c) { glisse = { canon: c, depart: { x: c.x, y: c.y }, origine: { x: c.x, y: c.y }, bouge: false }; canvas.setPointerCapture(ev.pointerId); S.selection = c; }
    else S.selection = null;
  });
  canvas.addEventListener('pointerup', (ev) => {
    if (!glisse) return;
    const g = glisse; glisse = null;
    if (g.bouge) {
      const x = g.canon.x, y = g.canon.y;
      g.canon.x = g.origine.x; g.canon.y = g.origine.y;
      if (!deplacer(g.canon, x, y)) noter('On ne pose pas un canon dans le bois, sur le chemin ni sur une maison.', 'alerte');
    }
    try { canvas.releasePointerCapture(ev.pointerId); } catch (e) { /* rien */ }
  });
  canvas.addEventListener('contextmenu', (ev) => { ev.preventDefault(); S.mode = 'observer'; S.modeleAPlacer = null; });

  for (const b of document.querySelectorAll('[data-vitesse]')) b.addEventListener('click', () => { if (!S.fini) S.vitesse = Number(b.dataset.vitesse); });
  for (const b of document.querySelectorAll('[data-modele]')) b.addEventListener('click', () => {
    if (S.mode === 'placer' && S.modeleAPlacer === b.dataset.modele) { S.mode = 'observer'; S.modeleAPlacer = null; }
    else { S.mode = 'placer'; S.modeleAPlacer = b.dataset.modele; S.selection = null; }
    majInterface();
  });
  $('btnAcheterPistolet').addEventListener('click', () => { acheterCartouches(); majInterface(); });
  $('btnPistolet').addEventListener('click', () => { S.mode = S.mode === 'pistolet' ? 'observer' : 'pistolet'; S.modeleAPlacer = null; S.selection = null; majInterface(); });
  $('btnFermerFiche').addEventListener('click', () => { S.selection = null; majInterface(); });
  $('btnRetirer').addEventListener('click', () => { if (S.selection) retirer(S.selection); majInterface(); });
  $('intervalle').addEventListener('input', () => { if (S.selection) { S.selection.intervalle = Number($('intervalle').value); majFiche(); } });
  $('aleatoire').addEventListener('change', () => { if (S.selection) { S.selection.aleatoire = $('aleatoire').checked; majFiche(); } });
  $('direction').addEventListener('input', () => { if (S.selection) S.selection.dir = Number($('direction').value) * Math.PI / 180; });
  $('chkPortees').addEventListener('change', () => { afficherPortees = $('chkPortees').checked; });
  $('btnSon').addEventListener('click', () => { sonActif = !sonActif; $('btnSon').textContent = sonActif ? '🔊 Son' : '🔇 Muet'; $('btnSon').classList.toggle('actif', sonActif); });
  $('btnAide').addEventListener('click', () => $('aide').showModal());
  $('btnVendanger').addEventListener('click', () => {
    const j = joursAvantMaturite();
    $('confirmeVendangeDetail').textContent = brix() < BRIX_SEUIL
      ? `À ${brix().toFixed(1)} °Brix, votre raisin est vert : il partira en jus à ${prixTonne()} $ la tonne au lieu de ${PRIX_MUR} à maturité. Encore ${j} jour${j > 1 ? 's' : ''} avant les 20 °Brix qui font un vin.`
      : `À ${brix().toFixed(1)} °Brix, le raisin est payé ${prixTonne()} $ la tonne ; il vaudra ${PRIX_MUR} $ le 24 septembre, si les oiseaux en laissent.`;
    $('confirmeVendange').showModal();
  });
  $('btnConfirmerVendange').addEventListener('click', () => { $('confirmeVendange').close(); afficherBilan(vendanger()); majInterface(); });
  $('btnReset').addEventListener('click', () => { recommencer(); majInterface(); });
  $('btnRejouer').addEventListener('click', () => { $('bilan').close(); recommencer(); majInterface(); });
  for (const b of document.querySelectorAll('[data-espece-aide]')) b.addEventListener('click', () => { $('aide').showModal(); $('aide-' + b.dataset.especeAide)?.scrollIntoView({ block: 'start' }); });
  document.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, select, textarea') || document.querySelector('dialog[open]')) return;
    if (ev.key === ' ') { ev.preventDefault(); S.vitesse = S.vitesse ? 0 : 1; }
    else if (ev.key === '1') S.vitesse = 1; else if (ev.key === '2') S.vitesse = 2; else if (ev.key === '3') S.vitesse = 4;
    else if (ev.key === 'Escape') { S.mode = 'observer'; S.modeleAPlacer = null; S.selection = null; }
    else if (ev.key === 'h' || ev.key === '?') $('btnAide').click();
    else if (ev.key === 'p' && S.pistolet) $('btnPistolet').click();
    else if (ev.key === 'Delete' && S.selection) retirer(S.selection);
    majInterface();
  });
}

// ---------- Boucle ----------

let dernier = 0, accum = 0, tirsVus = 0;
function boucle(ts) {
  const dtReel = Math.min(0.1, (ts - dernier) / 1000 || 0); dernier = ts;
  if (!S.fini && S.vitesse > 0) {
    accum += dtReel * S.vitesse;
    const h = 1 / 30; let n = 0;
    while (accum >= h && n < 12) { pas(h); accum -= h; n++; }
    if (accum > h) accum = 0;
  }
  if (S.stats.tirs !== tirsVus) { const f = S.ondes.length ? S.ondes[S.ondes.length - 1].force : 1; sonBang(f); tirsVus = S.stats.tirs; }
  dessiner();
  majInterface();
  if (S.fini && S.bilan && !$('bilan').open && !S.bilanAffiche) { S.bilanAffiche = true; afficherBilan(S.bilan); }
  requestAnimationFrame(boucle);
}

window.birdBanger = {
  get etat() { return S; }, set etat(v) { S = v; },
  avancer, apparaitre, placer, tirer, deplacer, retirer, vendanger, recommencer, tirerCartouche, acheterPistolet, acheterCartouches,
  recolteKg, jour, heure, brix, prixTonne, prixPourBrix, joursAvantMaturite, BRIX_SEUIL, ecran, MODELES, ESPECES, PISTOLET, MONDE, VIGNE, MAISONS, CHAI, RECOLTE_TOTALE, NB_JOURS, DUREE_JOUR,
};

S = etatInitial();
planifierJour(1);
redimensionner();
brancher();
pas(1 / 30);
majInterface();
requestAnimationFrame(boucle);
