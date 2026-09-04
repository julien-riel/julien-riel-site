/* ======================================================================
   Fourmilière — simulateur 2D d'une colonie de Lasius niger vue en coupe.
   Aucune dépendance. Les distances sont en millimètres, le temps simulé
   en minutes. Deux horloges cohabitent : le temps « biologique » (âge,
   développement du couvain, saisons) qui suit la vitesse choisie, et le
   temps « visuel » des déplacements, plafonné pour rester lisible.
   ====================================================================== */

/* ---------- Constantes ---------- */

const MM = 2.5;                       // taille d'une cellule de terrain (mm)
const GW = 240, GH = 160;             // grille : 600 × 400 mm
const LARG = GW * MM, HAUT = GH * MM;
const SOL = 40;                       // rangée du niveau du sol naturel
const Y_SOL = SOL * MM;
const AIR = 0, TERRE = 1, GALERIE = 2, ROCHE = 3, MEUBLE = 4;

const MIN_PAR_SEC = 5;                // minutes simulées par seconde réelle à ×1
const VITESSES = [0, 1, 8, 32];
const VISUEL_MAX = 4;                 // au-delà, seul le temps biologique accélère
const JOUR = 1440;
const INF = 32767;

const CAP_JABOT = 0.5;                // µL, jabot social d'une ouvrière
const CAP_REINE = 1.5;
const POP_MAX = 1500;                 // plafond d'ouvrières du simulateur
const CLE = 'fourmiliere-v1';

const MOIS = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'];
const JOURS_MOIS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

const ROLES = {
  vestibule: 'Vestibule',
  couvain: 'Chambre à couvain',
  reine: 'Chambre royale',
  repos: 'Chambre de repos',
  hivernage: "Chambre d'hivernage",
};

/* ---------- Utilitaires ---------- */

function mulberry(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const R = Math.random;
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp = (a, b, t) => a + (b - a) * t;
function gauss(rng = R) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const entre = (a, b) => a + R() * (b - a);
const choix = (liste) => liste[Math.floor(R() * liste.length)];
const fmt = (n, d = 1) => Number(n).toLocaleString('fr-CA', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtEntier = (n) => Math.round(n).toLocaleString('fr-CA');
const idx = (c, r) => r * GW + c;
const cellDe = (x, y) => idx(clamp(Math.floor(x / MM), 0, GW - 1), clamp(Math.floor(y / MM), 0, GH - 1));

/* ---------- État global ---------- */

let S = null;                         // tout l'état simulé (sauvegardé)
let champs = { entree: null, front: null, fondation: null, chambres: [] };
let champsSales = true;
let derniereRecalc = -10;

const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const cam = { x: LARG / 2, y: 150, z: 2.2, zMin: 1, zMax: 12 };
const ui = {
  vitesse: 1, mode: 'observer', pheromone: true, plan: false, temperature: false,
  suivre: false, selection: null, astuce: true,
};
let texSol = null, texGal = null;     // textures hors écran (terrain, galeries)
let dpr = 1;

/* ======================================================================
   Environnement : calendrier, soleil, températures
   ====================================================================== */

function dateSim() {
  const jours = Math.floor(S.minutes / JOUR);
  let doy = S.doy0 + jours;
  const an = S.an0 + Math.floor(doy / 365);
  doy %= 365;
  let m = 0, d = doy;
  while (d >= JOURS_MOIS[m]) { d -= JOURS_MOIS[m]; m++; }
  return { an, doy, mois: m, jour: d + 1, heure: (S.minutes % JOUR) / 60, jours };
}
const facteurSaison = (doy) => -Math.cos(2 * Math.PI * (doy - 20) / 365);   // −1 en janvier, +1 fin juillet
function soleil(doy) {
  const s = facteurSaison(doy);
  return { lever: 7 - 2.2 * s, coucher: 17 + 3.5 * s };
}
function luminosite(doy, h) {
  const { lever, coucher } = soleil(doy);
  return clamp((h - lever + 0.6) / 1.4, 0, 1) * clamp((coucher - h + 0.6) / 1.4, 0, 1);
}
function tempAir(doy, h) {
  const s = facteurSaison(doy);
  return 9 + 13 * s + (5 + 2 * s) * -Math.cos(2 * Math.PI * (h - 4) / 24);
}
// Température du sol à une profondeur donnée : l'amplitude journalière s'amortit
// en quelques centimètres, la vague saisonnière prend du retard avec la profondeur,
// et la butte exposée au soleil chauffe les premiers centimètres l'après-midi.
function tempSol(profMm, doy, h) {
  const d = Math.max(0, profMm) / 10;
  const s = -Math.cos(2 * Math.PI * (doy - 20 - d * 0.6) / 365);
  const amort = Math.exp(-d / 7);
  const solaire = 7 * luminosite(doy, h) * (0.5 + 0.5 * facteurSaison(doy)) * Math.exp(-d / 6);
  return 9 + 13 * s + (5 + 2 * s) * amort * -Math.cos(2 * Math.PI * (h - 4 - d * 0.7) / 24) + solaire;
}
// Vitesse de développement du couvain selon la température (Q10 ≈ 2, arrêt sous 10 °C).
function facteurDeveloppement(T) {
  if (T < 10) return 0;
  if (T < 14) return (T - 10) / 4 * 0.35;
  if (T > 34) return Math.max(0, 1.6 - (T - 34) * 0.3);
  return Math.min(1.6, Math.pow(2, (T - 25) / 10));
}
function nomSaison(doy) {
  if (doy < 79 || doy >= 355) return 'hiver';
  if (doy < 172) return 'printemps';
  if (doy < 266) return 'été';
  return 'automne';
}

/* ======================================================================
   Monde : terrain, plan du nid, butte, champs de distance
   ====================================================================== */

function genererMonde(graine) {
  const rng = mulberry(graine);
  const grille = new Uint8Array(GW * GH);
  for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) grille[idx(c, r)] = r < SOL ? AIR : TERRE;
  const plan = new Int8Array(GW * GH).fill(-1);
  const cellChambre = new Int8Array(GW * GH).fill(-1);

  const colEntree = Math.round(GW / 2 + (rng() - 0.5) * 30);
  const puitsX = new Int16Array(GH);
  let x = colEntree;
  for (let r = 0; r < GH; r++) {
    if (r > SOL + 2 && rng() < 0.22) {
      const vers = x > colEntree + 5 ? -1 : x < colEntree - 5 ? 1 : (rng() < 0.5 ? -1 : 1);
      x += vers;
    }
    puitsX[r] = x;
  }

  const gabarit = [
    { prof: 40, role: 'vestibule', seuil: 0 },
    { prof: 62, role: 'couvain', seuil: 4 },
    { prof: 105, role: 'reine', seuil: 0 },
    { prof: 84, role: 'couvain', seuil: 15 },
    { prof: 130, role: 'repos', seuil: 40 },
    { prof: 155, role: 'couvain', seuil: 80 },
    { prof: 180, role: 'repos', seuil: 140 },
    { prof: 205, role: 'couvain', seuil: 220 },
    { prof: 230, role: 'repos', seuil: 320 },
    { prof: 268, role: 'hivernage', seuil: 450 },
    { prof: 250, role: 'couvain', seuil: 600 },
    { prof: 288, role: 'repos', seuil: 800 },
  ];
  const chambres = gabarit.map((g, i) => {
    const cote = i % 2 === 0 ? 1 : -1;
    const rowC = SOL + Math.round(g.prof / MM);
    const rx = 9 + rng() * 6, ry = 4.5 + rng() * 2.5;
    const ecart = 22 + rng() * 40;
    const cx = clamp(puitsX[rowC] * MM + cote * ecart, rx + 12, LARG - rx - 12);
    return { i, cx, cy: rowC * MM + MM / 2, rx, ry, role: g.role, seuil: g.seuil, cote, total: 0, creusees: 0, pret: false, presents: [], temp: 15 };
  });

  const marquer = (c, r, i) => {
    if (c < 0 || c >= GW || r < SOL || r >= GH) return;
    const k = idx(c, r);
    if (plan[k] < 0 || chambres[i].seuil < chambres[plan[k]].seuil) plan[k] = i;
  };
  for (const ch of chambres) {
    const rowC = Math.floor(ch.cy / MM);
    for (let r = SOL; r <= rowC; r++) { marquer(puitsX[r], r, ch.i); marquer(puitsX[r] + 1, r, ch.i); }
    // galerie d'accès, avec un léger flottement vertical
    const colC = Math.round(ch.cx / MM);
    let c = puitsX[rowC] + (ch.cote > 0 ? 2 : -1);
    let r = rowC;
    let n = Math.abs(colC - c);
    while (c !== colC && n-- > 0) {
      if (Math.abs(colC - c) > 4 && rng() < 0.18) r += rng() < 0.5 ? -1 : 1;
      r = clamp(r, rowC - 2, rowC + 2);
      if (Math.abs(colC - c) <= 4) r += Math.sign(rowC - r);
      marquer(c, r, ch.i); marquer(c, r + 1, ch.i);
      c += Math.sign(colC - c);
    }
    for (let rr = SOL; rr < GH; rr++) for (let cc = 0; cc < GW; cc++) {
      const dx = (cc * MM + MM / 2 - ch.cx) / ch.rx, dy = (rr * MM + MM / 2 - ch.cy) / ch.ry;
      const e = dx * dx + dy * dy;
      if (e <= 1) { marquer(cc, rr, ch.i); ch.total++; }
      if (e <= 1.35 && cellChambre[idx(cc, rr)] < 0) cellChambre[idx(cc, rr)] = ch.i;
    }
  }

  // Pierres : la terre n'est pas homogène, et les galeries doivent les contourner.
  for (let essai = 0; essai < 14; essai++) {
    const px = rng() * LARG, py = Y_SOL + 25 + rng() * (HAUT - Y_SOL - 45);
    const rx = 4 + rng() * 8, ry = rx * (0.5 + rng() * 0.5);
    let libre = true;
    const cellules = [];
    for (let rr = SOL; rr < GH && libre; rr++) for (let cc = 0; cc < GW; cc++) {
      const dx = (cc * MM + MM / 2 - px) / (rx + 2 * MM), dy = (rr * MM + MM / 2 - py) / (ry + 2 * MM);
      if (dx * dx + dy * dy <= 1) {
        if (plan[idx(cc, rr)] >= 0) { libre = false; break; }
        const dx2 = (cc * MM + MM / 2 - px) / rx, dy2 = (rr * MM + MM / 2 - py) / ry;
        if (dx2 * dx2 + dy2 * dy2 <= 1) cellules.push(idx(cc, rr));
      }
    }
    if (libre) for (const k of cellules) grille[k] = ROCHE;
  }

  const xPlante = rng() < 0.5 ? 40 + rng() * 60 : LARG - 40 - rng() * 60;
  const xDepotoir = clamp((colEntree + 1) * MM + (xPlante < LARG / 2 ? 1 : -1) * (110 + rng() * 60), 30, LARG - 30);
  const herbes = [];
  for (let hx = 6; hx < LARG; hx += 5 + rng() * 9) herbes.push({ x: hx, h: 4 + rng() * 9, pente: (rng() - 0.5) * 0.9 });
  const pierresSurface = [{ x: rng() * LARG, rx: 5 + rng() * 6, ry: 3 + rng() * 3 }, { x: rng() * LARG, rx: 4 + rng() * 5, ry: 2.5 + rng() * 2 }];

  return {
    graine, grille, plan, cellChambre, chambres, puitsX, colEntree, xEntree: (colEntree + 1) * MM,
    rowEntree: SOL, hauteurSol: new Float32Array(GW).fill(Y_SOL), pheromone: new Float32Array(GW),
    xPlante, xDepotoir, herbes, pierresSurface,
  };
}

function hauteur(x) {
  const c = clamp(x / MM - 0.5, 0, GW - 1.001);
  const i = Math.floor(c), t = c - i;
  return lerp(S.hauteurSol[i], S.hauteurSol[Math.min(GW - 1, i + 1)], t);
}
function penteSurface(x) {
  return Math.atan2(hauteur(x + 3) - hauteur(x - 3), 6);
}

function relaxerButte(a, b) {
  const h = S.hauteurSol, maxDiff = MM * Math.tan(35 * Math.PI / 180);
  for (let it = 0; it < 8; it++) {
    for (let c = Math.max(0, a); c < Math.min(GW - 1, b); c++) {
      const d = h[c + 1] - h[c];      // > 0 : la colonne c est plus haute
      if (d > maxDiff) { const m = (d - maxDiff) * 0.35; h[c] += m; h[c + 1] -= m; }
      else if (-d > maxDiff) { const m = (-d - maxDiff) * 0.35; h[c] -= m; h[c + 1] += m; }
    }
  }
  for (let c = Math.max(0, a); c <= Math.min(GW - 1, b); c++) h[c] = Math.max(12, h[c]);
}

function majCellulesButte(a, b) {
  const g = S.grille;
  for (let c = Math.max(0, a); c <= Math.min(GW - 1, b); c++) {
    const rTop = Math.max(0, Math.ceil(S.hauteurSol[c] / MM - 0.5));
    const entree = c === S.colEntree || c === S.colEntree + 1;
    for (let r = rTop; r < SOL; r++) {
      const k = idx(c, r);
      if (g[k] === AIR) g[k] = entree ? GALERIE : MEUBLE;
    }
    if (entree) {
      let r = 0;
      while (r < GH && g[idx(c, r)] !== GALERIE) r++;
      if (r < S.rowEntree) { S.rowEntree = r; champsSales = true; }
    }
  }
}

// Un déblai déposé en surface : une cellule de terre foisonnée, répartie sur trois colonnes.
function deposerDeblai(xmm) {
  const c0 = clamp(Math.round(xmm / MM - 0.5), 1, GW - 2);
  const total = MM * 1.25;
  S.hauteurSol[c0 - 1] -= total * 0.25;
  S.hauteurSol[c0] -= total * 0.5;
  S.hauteurSol[c0 + 1] -= total * 0.25;
  relaxerButte(c0 - 10, c0 + 10);
  majCellulesButte(c0 - 12, c0 + 12);
}
function pointDepotDeblai() {
  const cote = R() < 0.5 ? -1 : 1;
  return clamp(S.xEntree + cote * (9 + Math.abs(gauss()) * 30), 8, LARG - 8);
}

function creusable(k) {
  const p = S.plan[k];
  if (p < 0) return false;
  const g = S.grille[k];
  return (g === TERRE || g === MEUBLE) && S.chambres[p].seuil <= S.nbOuvrieres;
}
function creuserCellule(k) {
  const g = S.grille[k];
  if (g !== TERRE && g !== MEUBLE) return false;
  S.grille[k] = GALERIE;
  peindreGalerie(k);
  const p = S.plan[k];
  if (p >= 0) {
    const ch = S.chambres[p];
    const c = k % GW, r = (k - c) / GW;
    const dx = (c * MM + MM / 2 - ch.cx) / ch.rx, dy = (r * MM + MM / 2 - ch.cy) / ch.ry;
    if (dx * dx + dy * dy <= 1) {
      ch.creusees++;
      if (!ch.pret && ch.creusees / ch.total >= 0.75) {
        ch.pret = true;
        if (S.minutes > 10) journal(`${ROLES[ch.role]} achevée à ${fmt((ch.cy - Y_SOL) / 10, 0)} cm de profondeur.`);
      }
    }
  }
  champsSales = true;
  return true;
}

function calculerChamp(sources) {
  const d = new Int16Array(GW * GH).fill(INF);
  const file = new Int32Array(GW * GH);
  const g = S.grille;
  let h = 0, t = 0;
  for (const s of sources) if (g[s] === GALERIE && d[s] !== 0) { d[s] = 0; file[t++] = s; }
  while (h < t) {
    const i = file[h++];
    const c = i % GW, dd = d[i] + 1;
    if (c > 0 && g[i - 1] === GALERIE && d[i - 1] > dd) { d[i - 1] = dd; file[t++] = i - 1; }
    if (c < GW - 1 && g[i + 1] === GALERIE && d[i + 1] > dd) { d[i + 1] = dd; file[t++] = i + 1; }
    if (i >= GW && g[i - GW] === GALERIE && d[i - GW] > dd) { d[i - GW] = dd; file[t++] = i - GW; }
    if (i < GW * (GH - 1) && g[i + GW] === GALERIE && d[i + GW] > dd) { d[i + GW] = dd; file[t++] = i + GW; }
  }
  return d;
}

function recalculerChamps() {
  champsSales = false;
  const g = S.grille;
  champs.entree = calculerChamp([idx(S.colEntree, S.rowEntree), idx(S.colEntree + 1, S.rowEntree)]);
  champs.chambres = S.chambres.map((ch) => {
    const k = cellDe(ch.cx, ch.cy);
    if (g[k] === GALERIE) return calculerChamp([k]);
    // centre pas encore creusé : partir de n'importe quelle cellule ouverte de la chambre
    const src = [];
    for (let r = Math.floor((ch.cy - ch.ry) / MM); r <= Math.ceil((ch.cy + ch.ry) / MM); r++)
      for (let c = Math.floor((ch.cx - ch.rx) / MM); c <= Math.ceil((ch.cx + ch.rx) / MM); c++)
        if (c >= 0 && c < GW && r >= 0 && r < GH && g[idx(c, r)] === GALERIE && S.cellChambre[idx(c, r)] === ch.i) src.push(idx(c, r));
    return calculerChamp(src);
  });
  const front = [];
  const fond = [];
  for (let k = 0; k < GW * GH; k++) {
    if (g[k] !== GALERIE) continue;
    const c = k % GW;
    const voisins = [c > 0 ? k - 1 : -1, c < GW - 1 ? k + 1 : -1, k >= GW ? k - GW : -1, k < GW * (GH - 1) ? k + GW : -1];
    for (const v of voisins) {
      if (v < 0) continue;
      if (creusable(v)) { front.push(k); break; }
    }
    if (S.fondation) for (const v of voisins) {
      if (v >= 0 && S.fondation[v] && (g[v] === TERRE || g[v] === MEUBLE)) { fond.push(k); break; }
    }
  }
  S.nbFront = front.length;
  champs.front = calculerChamp(front);
  champs.fondation = S.fondation ? calculerChamp(fond) : null;
}

function voisinCreusable(k, fondation) {
  const c = k % GW;
  const voisins = [c > 0 ? k - 1 : -1, c < GW - 1 ? k + 1 : -1, k >= GW ? k - GW : -1, k < GW * (GH - 1) ? k + GW : -1];
  const ok = [];
  for (const v of voisins) {
    if (v < 0) continue;
    if (fondation ? (S.fondation[v] && (S.grille[v] === TERRE || S.grille[v] === MEUBLE)) : creusable(v)) ok.push(v);
  }
  return ok.length ? choix(ok) : -1;
}

function chambreDe(f) { return S.cellChambre[cellDe(f.x, f.y)]; }
function chambresPar(role) { return S.chambres.filter((ch) => ch.role === role && ch.pret); }
function chambrePlusProfondePrete() {
  let best = null;
  for (const ch of S.chambres) if (ch.pret && (!best || ch.cy > best.cy)) best = ch;
  return best || S.chambres[2];
}
function spotChambre(ch, serre = 0.7) {
  for (let i = 0; i < 10; i++) {
    const a = R() * Math.PI * 2, rr = Math.sqrt(R()) * serre;
    const x = ch.cx + Math.cos(a) * rr * ch.rx, y = ch.cy + Math.sin(a) * rr * ch.ry;
    if (S.grille[cellDe(x, y)] === GALERIE) return { x, y };
  }
  return { x: ch.cx, y: ch.cy };
}
function spotCouvain(ch) {
  for (let i = 0; i < 10; i++) {
    const x = ch.cx + gauss() * ch.rx * 0.32, y = ch.cy + gauss() * ch.ry * 0.35;
    const dx = (x - ch.cx) / ch.rx, dy = (y - ch.cy) / ch.ry;
    if (dx * dx + dy * dy < 0.8 && S.grille[cellDe(x, y)] === GALERIE) return { x, y };
  }
  return { x: ch.cx, y: ch.cy };
}

/* ======================================================================
   Entités
   ====================================================================== */

let prochainId = 1;

function creerFourmi(caste, opts = {}) {
  const f = {
    id: prochainId++, caste, tache: opts.tache || (caste === 'reine' ? 'reine' : caste === 'ouvriere' ? 'nourrice' : 'sexue'),
    etat: 'repos', x: opts.x ?? LARG / 2, y: opts.y ?? HAUT / 2, cap: R() * Math.PI * 2,
    taille: opts.taille ?? (caste === 'reine' || caste === 'gyne' ? 8.5 : caste === 'male' ? 4 : opts.nanitique ? entre(2.6, 3.2) : entre(3.2, 4.8)),
    age: opts.age ?? 0, longevite: opts.longevite ?? (caste === 'ouvriere' ? Math.max(60, 240 + gauss() * 80) : caste === 'reine' ? 365 * 15 : 60),
    jabot: opts.jabot ?? CAP_JABOT * 0.5, prot: opts.prot ?? 0.3, energie: opts.energie ?? 0.9, faim: 0,
    porte: null, mode: 'nid', cible: null, prochain: null, apres: null, minuterie: 0, partenaire: null,
    chambreMaison: null, sorties: 0, rapporte: 0, nanitique: !!opts.nanitique, callow: opts.callow ?? 0,
    dir: 1, xCible: null, souvenir: null, grimpe: 0, phase: R() * 10, lat: (R() - 0.5) * 1.2,
    ne: opts.ne ?? S.minutes, oeufs: 0, reserves: opts.reserves ?? 0,
  };
  if (caste === 'reine') f.jabot = opts.jabot ?? CAP_REINE * 0.6;
  S.fourmis.push(f);
  return f;
}

function creerCouvain(stade, x, y, ch, opts = {}) {
  const b = {
    id: prochainId++, stade, x, y, dev: opts.dev ?? 0, sucre: opts.sucre ?? 0.7, prot: opts.prot ?? 0.5,
    chambre: ch ? ch.i : -1, porteur: null, sexue: opts.sexue || null, jeune: 0, repas: null, pret: false,
  };
  S.couvain.push(b);
  return b;
}

function creerObjet(type, x, y, opts = {}) {
  const o = { id: prochainId++, type, x, y, mode: opts.mode || 'nid', chambre: opts.chambre ?? -1, masse: opts.masse ?? 1, porteur: null, age: 0, taille: opts.taille ?? 4 };
  S.objets.push(o);
  return o;
}

function creerSource(type, x, opts = {}) {
  const s = { id: prochainId++, type, x, reserve: opts.reserve ?? 0, max: opts.max ?? 0, masse: opts.masse ?? 0, decouverte: false, age: 0, nom: opts.nom || '' };
  S.sources.push(s);
  return s;
}

function journal(texte) {
  const d = dateSim();
  S.journal.push({ t: S.minutes, texte, date: `${d.jour} ${MOIS[d.mois].slice(0, 4)}${MOIS[d.mois].length > 4 ? '.' : ''} an ${d.an}` });
  if (S.journal.length > 60) S.journal.shift();
  S.journalVersion = (S.journalVersion || 0) + 1;
}

/* ======================================================================
   Scénarios
   ====================================================================== */

function nouvellePartie(scenario) {
  const graine = Math.floor(R() * 1e9);
  const monde = genererMonde(graine);
  S = Object.assign(monde, {
    scenario, minutes: 0, doy0: 0, an0: 1, fourmis: [], couvain: [], objets: [], sources: [], journal: [],
    nbOuvrieres: 0, hiver: false, fondation: null, fondationScellee: false, claustration: false,
    reine: null, chambreCouvain: null, chambreNymphes: null, vol: null, compteurs: { eclosions: 0, morts: 0, oeufs: 0 },
    derniereHeure: 0, nbFront: 0, tAir: 15, tNid: 15, activite: 0, tickMeteo: 0, prochainInsecte: 24 + R() * 24,
    sexuesProduits: false, alertes: {}, nbCreuseurs: 0, journalVersion: Math.floor(R() * 1e9),
  });
  prochainId = 1;
  ui.selection = null; ui.suivre = false;

  creerSource('miellat', S.xPlante, { reserve: 20, max: 30, nom: 'une colonie de pucerons' });

  if (scenario === 'fondation') {
    S.doy0 = 200; S.an0 = 1;         // 20 juillet : lendemain du vol nuptial
    const reine = creerFourmi('reine', { x: S.xEntree + 30, y: hauteur(S.xEntree + 30) - 1.5, age: 365 + 20, reserves: 1, jabot: CAP_REINE * 0.9, prot: 0.9 });
    reine.mode = 'surface'; reine.dir = -1; reine.etat = 'fondation_marche';
    S.reine = reine;
    // Elle creuse elle-même un puits jusqu'à sa loge, un peu plus petite que la future chambre royale.
    const fond = new Uint8Array(GW * GH);
    const chR = S.chambres[2];
    const rowC = Math.floor(chR.cy / MM);
    for (let r = SOL; r <= rowC; r++) { fond[idx(S.puitsX[r], r)] = 1; fond[idx(S.puitsX[r] + 1, r)] = 1; }
    const colC = Math.round(chR.cx / MM);
    let c = S.puitsX[rowC] + (chR.cote > 0 ? 2 : -1);
    while (c !== colC) { fond[idx(c, rowC)] = 1; fond[idx(c, rowC + 1)] = 1; c += Math.sign(colC - c); }
    for (let rr = rowC - 3; rr <= rowC + 3; rr++) for (let cc = colC - 4; cc <= colC + 4; cc++) {
      const dx = (cc * MM + MM / 2 - chR.cx) / 6, dy = (rr * MM + MM / 2 - chR.cy) / 4;
      if (dx * dx + dy * dy <= 1 && S.plan[idx(cc, rr)] >= 0) fond[idx(cc, rr)] = 1;
    }
    S.fondation = fond;
    journal('Une jeune reine fécondée se pose après son vol nuptial. Elle a arraché ses ailes : elle ne volera plus jamais.');
  } else {
    S.doy0 = 160; S.an0 = 3;         // 10 juin, troisième printemps de la colonie
    for (const ch of S.chambres) if (ch.seuil <= 80) {
      for (let k = 0; k < GW * GH; k++) if (S.plan[k] === ch.i) { S.grille[k] = GALERIE; }
    }
    for (const ch of S.chambres) {
      ch.creusees = 0;
      for (let rr = SOL; rr < GH; rr++) for (let cc = 0; cc < GW; cc++) {
        const dx = (cc * MM + MM / 2 - ch.cx) / ch.rx, dy = (rr * MM + MM / 2 - ch.cy) / ch.ry;
        if (dx * dx + dy * dy <= 1 && S.grille[idx(cc, rr)] === GALERIE) ch.creusees++;
      }
      ch.pret = ch.creusees / ch.total >= 0.75;
    }
    let nCreusees = 0;
    for (let k = 0; k < GW * GH; k++) if (S.grille[k] === GALERIE) nCreusees++;
    for (let i = 0; i < nCreusees; i++) deposerDeblai(pointDepotDeblai());

    const chR = S.chambres[2], chV = S.chambres[0];
    const reine = creerFourmi('reine', Object.assign(spotChambre(chR, 0.3), { age: 365 * 3 + 300, prot: 0.8 }));
    S.reine = reine;
    S.nbOuvrieres = 250;
    S.chambreCouvain = S.chambres[1];
    for (let i = 0; i < 250; i++) {
      const age = Math.max(1, Math.abs(gauss()) * 90 + R() * 30);
      const ch = i < 100 ? S.chambres[1] : i < 175 ? chV : choix([S.chambres[3], S.chambres[4], S.chambres[1]]);
      creerFourmi('ouvriere', Object.assign(spotChambre(ch), { age, longevite: Math.max(age + 20, 240 + gauss() * 80), jabot: CAP_JABOT * (0.3 + R() * 0.6) }));
    }
    for (let i = 0; i < 60; i++) creerCouvain('oeuf', 0, 0, S.chambres[1], { dev: R() });
    for (let i = 0; i < 50; i++) creerCouvain('larve', 0, 0, S.chambres[1], { dev: R() });
    for (let i = 0; i < 40; i++) creerCouvain('nymphe', 0, 0, S.chambres[1], { dev: R() });
    for (const b of S.couvain) Object.assign(b, spotCouvain(S.chambres[1]));
    for (let i = 0; i < 3; i++) { const p = spotCouvain(S.chambres[1]); creerObjet('proie', p.x, p.y, { chambre: 1, masse: 0.8 }); }
    journal('Colonie établie : une reine, environ 250 ouvrières et 150 œufs, larves et nymphes. Troisième printemps.');
  }
  repartirTaches();
  choisirChambresCouvain();
  champsSales = true;
  recalculerChamps();
  peindreTerrain();
  cam.x = S.xEntree; cam.y = scenario === 'fondation' ? Y_SOL + 20 : Y_SOL + 70;
  sauvegardeEnAttente = true;
}

/* ======================================================================
   Régulation de la colonie (une fois par heure simulée)
   ====================================================================== */

function repartirTaches() {
  const ouvrieres = S.fourmis.filter((f) => f.caste === 'ouvriere');
  S.nbOuvrieres = ouvrieres.length;
  if (!ouvrieres.length) return;
  const couvain = S.couvain.length;
  const jabotMoyen = ouvrieres.reduce((a, f) => a + f.jabot, 0) / ouvrieres.length / CAP_JABOT;
  const fNourrices = clamp(0.22 + 0.45 * couvain / ouvrieres.length, 0.2, 0.55);
  const fFourrageuses = clamp(0.22 + 0.35 * (1 - jabotMoyen), 0.15, 0.45);
  ouvrieres.sort((a, b) => a.age - b.age);
  const nN = Math.round(ouvrieres.length * fNourrices);
  const nF = Math.round(ouvrieres.length * fFourrageuses);
  ouvrieres.forEach((f, i) => {
    const voulue = i < nN ? 'nourrice' : i >= ouvrieres.length - nF ? 'fourrageuse' : 'entretien';
    if (f.tache !== voulue && f.mode === 'nid' && !f.porte && (f.etat === 'repos' || f.etat === 'hiberne')) f.tache = voulue;
    else if (f.tache !== voulue && !f.tacheVoulue) f.tacheVoulue = voulue;
    if (f.tache === voulue) f.tacheVoulue = null;
  });
}

function choisirChambresCouvain() {
  const d = dateSim();
  const candidates = S.chambres.filter((ch) => ch.pret && (ch.role === 'couvain' || ch.role === 'reine' || ch.role === 'hivernage'));
  for (const ch of S.chambres) ch.temp = tempSol(ch.cy - Y_SOL, d.doy, d.heure);
  if (!candidates.length) { S.chambreCouvain = S.chambres[2]; S.chambreNymphes = null; return; }
  if (S.hiver) {
    const h = chambrePlusProfondePrete();
    S.chambreCouvain = h; S.chambreNymphes = null; return;
  }
  const optimum = 26;
  const tri = candidates.slice().sort((a, b) => Math.abs(a.temp - optimum) - Math.abs(b.temp - optimum));
  const meilleure = tri[0];
  const actuelle = S.chambreCouvain && S.chambres[S.chambreCouvain.i];
  // hystérésis : on ne déménage le couvain que si le gain dépasse 1,5 °C
  if (!actuelle || !actuelle.pret || Math.abs(actuelle.temp - optimum) - Math.abs(meilleure.temp - optimum) > 1.5) S.chambreCouvain = meilleure;
  const seconde = tri.find((ch) => ch.i !== S.chambreCouvain.i && ch.role === 'couvain');
  S.chambreNymphes = S.couvain.length > 90 && seconde ? seconde : null;
}

function chambreCible(b) {
  if (b.stade === 'nymphe' && S.chambreNymphes) return S.chambreNymphes;
  return S.chambreCouvain || S.chambres[2];
}

function tickHeure() {
  const d = dateSim();
  const mult = VITESSES[ui.vitesse] / Math.min(VITESSES[ui.vitesse] || 1, VISUEL_MAX);
  const dtJ = 1 / 24;               // un tick = une heure biologique
  const dtL = dtJ / mult;           // temps logistique (faim, consommation)
  S.tAir = tempAir(d.doy, d.heure);
  S.tNid = tempSol(S.reine ? S.reine.y - Y_SOL : 100, d.doy, d.heure);
  S.activite = clamp((S.tAir - 8) / 10, 0, 1) * (luminosite(d.doy, d.heure) > 0.2 ? 1 : 0.45);

  // Hivernage, avec hystérésis
  if (!S.hiver && S.tNid < 8 && !S.claustration) { S.hiver = true; journal(`Le sol du nid passe sous 8 °C : la colonie entre en diapause hivernale. Plus de ponte, plus de sorties ; les larves attendront le printemps.`); }
  else if (S.hiver && S.tNid > 10.5) { S.hiver = false; journal(`Le sol se réchauffe : fin de la diapause. Les ouvrières reprennent les sorties et la reine recommencera bientôt à pondre.`); }

  repartirTaches();
  choisirChambresCouvain();
  const fT = facteurDeveloppement(S.tNid);

  // Ponte
  const reine = S.reine;
  if (reine && !S.hiver) {
    let taux;
    if (S.fondation) taux = 0;      // elle creuse encore sa loge
    else if (S.claustration) taux = reine.oeufs < 14 && reine.reserves > 0.25 ? 1.3 : 0;
    else if (S.scenario === 'fondation' && S.nbOuvrieres < 30) taux = 3 + S.nbOuvrieres * 0.4;
    else taux = 24;
    taux *= fT * clamp(reine.jabot / (CAP_REINE * 0.25), 0.1, 1) * (reine.prot > 0.1 ? 1 : 0.15) * Math.max(0, 1 - S.nbOuvrieres / POP_MAX);
    if (S.couvain.length > S.nbOuvrieres * 1.2 + 20) taux *= 0.3;
    reine.oeufsEnAttente = (reine.oeufsEnAttente || 0) + taux * dtJ;
    if (S.claustration) reine.reserves = Math.max(0, reine.reserves - 0.004 * dtJ);
  }

  // Vieillissement, faim, mort
  for (const f of S.fourmis) {
    f.age += dtJ;
    if (f.callow > 0) f.callow = Math.max(0, f.callow - dtJ);
    const conso = (f.caste === 'reine' ? 0.25 : f.tache === 'fourrageuse' ? 0.4 : 0.32) * dtL;
    f.energie -= conso;
    if (f.energie < 0.6 && f.jabot > 0.02) { const q = Math.min(f.jabot, 0.05); f.jabot -= q; f.energie = Math.min(1, f.energie + q * 5); }
    if (f.caste === 'reine' && S.claustration && f.jabot < CAP_REINE * 0.3 && f.reserves > 0) { const q = Math.min(f.reserves, 0.03); f.reserves -= q; f.jabot += q * 4; }
    if (f.energie <= 0) { f.energie = 0; f.faim += dtJ; } else f.faim = 0;
    if (f.age > f.longevite || f.faim > 3) mourir(f, f.faim > 3 ? 'faim' : 'age');
    if (f.tache === 'sexue' && f.age > 80) mourir(f, 'age');
  }

  // Couvain
  for (const b of S.couvain.slice()) {
    if (b.porteur) continue;
    b.jeune += dtJ;
    const fTb = facteurDeveloppement(S.chambres[b.chambre] ? S.chambres[b.chambre].temp : S.tNid);
    if (b.stade === 'oeuf') {
      b.dev += dtJ / 10 * fTb;
      if (b.dev >= 1) { b.stade = 'larve'; b.dev = 0; b.sucre = 0.6; b.prot = 0.4; }
    } else if (b.stade === 'larve') {
      b.sucre = Math.max(0, b.sucre - 0.5 * dtL);
      b.prot = Math.max(0, b.prot - 0.22 * dtL);
      if (b.repas) {
        if (b.repas.porteur || b.repas.masse <= 0) b.repas = null;
        else { const q = Math.min(b.repas.masse, 0.17 * dtL * 24); b.repas.masse -= q; b.prot = Math.min(1, b.prot + q * 0.45); if (b.repas.masse <= 0.01) { retirerObjet(b.repas); b.repas = null; } }
      }
      const nutrition = clamp(b.sucre * 1.6, 0, 1) * clamp(b.prot * 1.8 + 0.25, 0, 1);
      if (!S.hiver) b.dev += dtJ / (b.sexue ? 16 : 12) * fTb * nutrition;
      if (b.sucre <= 0 && b.prot <= 0) { b.jeuneFaim = (b.jeuneFaim || 0) + dtJ; if (b.jeuneFaim > 6) { retirerCouvain(b); S.compteurs.morts++; alerte('larves', `Des larves meurent de faim. Les ouvrières les mangeront : rien ne se perd dans une fourmilière.`); continue; } }
      else b.jeuneFaim = 0;
      if (b.dev >= 1) { b.stade = 'nymphe'; b.dev = 0; b.repas = null; }
    } else if (b.stade === 'nymphe') {
      b.dev += dtJ / 12 * fTb;
      if (b.dev >= 1) { b.pret = true; b.attente = (b.attente || 0) + dtJ; if (b.attente > 2) eclore(b, null); }
    }
  }

  // Objets : les cadavres et proies laissés dehors se décomposent
  for (const o of S.objets.slice()) {
    if (o.porteur) continue;
    o.age += dtJ;
    if (o.mode === 'surface' && o.age > 5) retirerObjet(o);
    if (o.type === 'proie' && o.mode === 'nid' && o.age > 6) retirerObjet(o);
  }
  for (const s of S.sources.slice()) {
    s.age += dtJ;
    if (s.type === 'miellat') s.reserve = Math.min(s.max, s.reserve + 2 * dtL * 24 * (S.tAir > 10 ? 1 : 0.1));
    else if (s.type === 'insecte' && (s.masse <= 0.05 || s.age > 4)) { S.sources.splice(S.sources.indexOf(s), 1); }
    else if (s.type === 'sucre' && (s.reserve <= 0.05 || s.age > 3)) { S.sources.splice(S.sources.indexOf(s), 1); }
  }
  // Un insecte mort tombe parfois près du nid
  S.prochainInsecte -= 1;
  if (S.prochainInsecte <= 0) {
    S.prochainInsecte = 30 + R() * 40;
    if (S.activite > 0.3 && S.sources.filter((s) => s.type === 'insecte').length < 2) {
      let x; do { x = 20 + R() * (LARG - 40); } while (Math.abs(x - S.xEntree) < 25);
      const nom = choix(['une mouche morte', 'un petit coléoptère', 'une chenille tombée d’une feuille', 'un moucheron']);
      creerSource('insecte', x, { masse: 6 + R() * 8, nom });
    }
  }

  // Sexués : une colonie mûre en élève au printemps, ils s'envolent l'été
  if (!S.sexuesProduits && S.nbOuvrieres >= 400 && d.doy >= 95 && d.doy <= 135 && S.reine) { S.sexuesProduits = true; S.sexuesAnnee = d.an; journal(`La colonie est assez forte pour élever des sexués : une partie des œufs de ce printemps donnera des reines ailées (gynes) et des mâles.`); }
  if (S.sexuesProduits && S.sexuesAnnee !== d.an && d.doy > 250) S.sexuesProduits = false;
  const ailes = S.fourmis.filter((f) => f.tache === 'sexue' && f.etat !== 'vol_nuptial');
  if (ailes.length && !S.vol && d.doy >= 195 && d.doy <= 240 && d.heure >= 14 && d.heure <= 19 && (S.tAir > 23 || d.doy > 235)) {
    S.vol = { t: S.minutes };
    const g = ailes.filter((f) => f.caste === 'gyne').length, m = ailes.length - g;
    journal(`Vol nuptial ! ${g} gyne${g > 1 ? 's' : ''} et ${m} mâle${m > 1 ? 's' : ''} quittent le nid par une fin d'après-midi chaude et lourde. Les mâles mourront en quelques jours ; les femelles fécondées perdront leurs ailes et tenteront chacune de fonder une colonie.`);
    for (const f of ailes) { f.tacheVoulue = null; f.etat = 'repos'; f.minuterie = 0; f.volPret = true; }
  }
  if (S.vol && S.minutes - S.vol.t > JOUR) S.vol = null;

  // Alertes de pédagogie douce
  if (S.nbOuvrieres > 0 && S.objets.filter((o) => o.type === 'proie' && o.mode === 'nid').length === 0 && S.couvain.some((b) => b.stade === 'larve' && b.prot < 0.15)) alerte('proteines', `Les larves manquent de protéines : sans proies, elles grandissent lentement et la reine pond moins. Déposez un insecte mort en surface pour voir le recrutement.`);
}

function alerte(cle, texte) {
  const dernier = S.alertes[cle] || -1e9;
  if (S.minutes - dernier < JOUR * 5) return;
  S.alertes[cle] = S.minutes;
  journal(texte);
}

function retirerCouvain(b) {
  const i = S.couvain.indexOf(b);
  if (i >= 0) S.couvain.splice(i, 1);
  if (b.porteur) { b.porteur.porte = null; }
}
function retirerObjet(o) {
  const i = S.objets.indexOf(o);
  if (i >= 0) S.objets.splice(i, 1);
  if (o.porteur) o.porteur.porte = null;
}

function eclore(b, nourrice) {
  retirerCouvain(b);
  S.compteurs.eclosions++;
  const ch = S.chambres[b.chambre] || S.chambreCouvain;
  if (b.sexue) {
    const f = creerFourmi(b.sexue, { x: b.x, y: b.y, callow: 2, tache: 'sexue' });
    f.chambreMaison = chambresPar('repos')[0] || S.chambres[0];
    return f;
  }
  const nanitique = S.scenario === 'fondation' && S.compteurs.eclosions <= 12 && S.nbOuvrieres < 12;
  const f = creerFourmi('ouvriere', { x: b.x, y: b.y, callow: 2, nanitique, jabot: CAP_JABOT * 0.35 });
  f.chambreMaison = ch;
  S.nbOuvrieres++;
  if (S.compteurs.eclosions === 1 && S.scenario === 'fondation') journal(`Première ouvrière ! Une nanitique, plus petite que la normale : la reine n'avait que ses réserves pour l'élever. Elle ouvrira le nid vers la surface.`);
  if (S.claustration && S.nbOuvrieres >= 1) { S.claustration = false; journal(`La claustration prend fin : les premières ouvrières vont creuser vers la surface et rapporter de la nourriture. La reine ne fera plus que pondre.`); }
  return f;
}

function mourir(f, cause) {
  liberer(f);
  const i = S.fourmis.indexOf(f);
  if (i >= 0) S.fourmis.splice(i, 1);
  if (f.porte) lacher(f, true);
  S.compteurs.morts++;
  if (f.caste === 'ouvriere') S.nbOuvrieres = Math.max(0, S.nbOuvrieres - 1);
  if (ui.selection === f.id) { ui.selection = null; ui.suivre = false; }
  if (f.caste === 'reine') {
    S.reine = null;
    journal(cause === 'faim' ? `La reine est morte de faim. Sans elle, plus d'œufs fécondés : la colonie est condamnée à s'éteindre avec ses dernières ouvrières.` : `La reine est morte de vieillesse. La colonie n'a aucun moyen de la remplacer : elle s'éteindra lentement.`);
  }
  if (cause === 'disparue') return;
  if (f.mode === 'surface') creerObjet('cadavre', f.x, f.y, { mode: 'surface', taille: f.taille });
  else creerObjet('cadavre', f.x, f.y, { mode: 'nid', chambre: chambreDe(f), taille: f.taille });
}

function liberer(f) {
  if (f.partenaire) {
    const p = f.partenaire;
    if (p.partenaire === f) { p.partenaire = null; if (p.etat === 'trophallaxie' || p.etat === 'attend') entrerEtat(p, 'repos', 1); }
    f.partenaire = null;
  }
}

/* ======================================================================
   Comportement : machine à états par fourmi
   ====================================================================== */

function entrerEtat(f, etat, duree = 0) {
  f.etat = etat;
  f.minuterie = duree;
  f.cible = null; f.prochain = null;
}

// Aller quelque part dans le nid : par un champ de distance, puis en ligne droite.
function viser(f, cible, apres) {
  f.cible = cible;
  f.apres = apres;
  f.prochain = null;
  f.etat = 'deplacement';
  if (cible.chambre !== undefined && cible.chambre >= 0) {
    const ci = cellDe(f.x, f.y);
    if (S.cellChambre[ci] === cible.chambre) cible.libre = true;
  }
}
function viserChambre(f, ch, apres, serre = 0.7) {
  const p = spotChambre(ch, serre);
  viser(f, { champ: champs.chambres[ch.i], chambre: ch.i, x: p.x, y: p.y }, apres);
}
function viserPoint(f, x, y, apres) {
  const ch = S.cellChambre[cellDe(x, y)];
  viser(f, { champ: ch >= 0 ? champs.chambres[ch] : null, chambre: ch, x, y, libre: ch < 0 || ch === chambreDe(f) }, apres);
}
function viserSurface(f, x, apres) {
  f.xCible = x; f.apres = apres; f.etat = 'deplacement';
  f.dir = x > f.x ? 1 : -1;
}

const vitesseDe = (f) => {
  let v = f.caste === 'reine' ? 5 : f.caste === 'gyne' ? 7 : f.mode === 'surface' ? 15 : 9.5;
  if (f.porte) v *= 0.72;
  if (f.callow > 0) v *= 0.7;
  if (f.nanitique) v *= 0.85;
  return v;
};

function orienter(f, dx, dy) {
  const a = Math.atan2(dy, dx);
  let d = a - f.cap;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  f.cap += d * 0.35;
}

function choisirVoisin(ci, champ) {
  const c = ci % GW, r = (ci - c) / GW;
  const g = S.grille;
  let best = -1, bd = champ[ci], candidats = [];
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    if (!dr && !dc) continue;
    const nc = c + dc, nr = r + dr;
    if (nc < 0 || nc >= GW || nr < 0 || nr >= GH) continue;
    const k = idx(nc, nr);
    if (g[k] !== GALERIE) continue;
    if (dr && dc && (g[idx(c + dc, r)] !== GALERIE || g[idx(c, r + dr)] !== GALERIE)) continue;
    if (champ[k] < bd) { bd = champ[k]; candidats = [k]; }
    else if (champ[k] === bd && champ[k] < champ[ci]) candidats.push(k);
  }
  if (candidats.length) best = choix(candidats);
  if (best < 0) return null;
  const bc = best % GW, br = (best - bc) / GW;
  return { x: bc * MM + MM / 2 + (R() - 0.5) * 1.2, y: br * MM + MM / 2 + (R() - 0.5) * 1.2 };
}

function assainir(f) {
  const ci = cellDe(f.x, f.y);
  if (S.grille[ci] === GALERIE) return true;
  const c = ci % GW, r = (ci - c) / GW;
  for (let ray = 1; ray <= 6; ray++) for (let dr = -ray; dr <= ray; dr++) for (let dc = -ray; dc <= ray; dc++) {
    const nc = c + dc, nr = r + dr;
    if (nc < 0 || nc >= GW || nr < 0 || nr >= GH) continue;
    if (S.grille[idx(nc, nr)] === GALERIE) { f.x = nc * MM + MM / 2; f.y = nr * MM + MM / 2; return true; }
  }
  const ch = S.chambres[2];
  const p = spotChambre(ch); f.x = p.x; f.y = p.y;
  return false;
}

function arriver(f) {
  const apres = f.apres;
  f.cible = null; f.prochain = null; f.xCible = null; f.apres = null;
  if (apres) f.etat = apres; else entrerEtat(f, 'repos', 1);
  f.minuterie = 0;
  f.nouveau = true;
}

function deplacer(f, dtV) {
  const v = vitesseDe(f) * dtV;
  if (f.mode === 'surface') {
    if (f.xCible === null || f.xCible === undefined) return;
    const dx = f.xCible - f.x;
    if (Math.abs(dx) <= v + 0.3) { f.x = f.xCible; arriver(f); return; }
    f.dir = Math.sign(dx);
    f.x += f.dir * v;
    f.y = hauteur(f.x) - f.taille * 0.12;
    return;
  }
  const c = f.cible;
  if (!c) { arriver(f); return; }
  const ci = cellDe(f.x, f.y);
  if (!c.libre) {
    const champ = c.champ;
    if (!champ) { c.libre = true; }
    else {
      const d = champ[ci];
      if (d === 0 || (c.chambre >= 0 && S.cellChambre[ci] === c.chambre && d < INF)) {
        if (c.x === undefined) { arriver(f); return; }
        c.libre = true;
      } else if (d >= INF) {
        if (!assainir(f)) { arriver(f); return; }
        f.echecs = (f.echecs || 0) + 1;
        if (f.echecs > 3 || champ[cellDe(f.x, f.y)] >= INF) { f.echecs = 0; f.apres = null; arriver(f); return; }
        return;
      } else {
        if (!f.prochain || Math.hypot(f.prochain.x - f.x, f.prochain.y - f.y) < 0.7) {
          f.prochain = choisirVoisin(ci, champ);
          if (!f.prochain) { arriver(f); return; }
        }
        const dx = f.prochain.x - f.x, dy = f.prochain.y - f.y, dd = Math.hypot(dx, dy) || 1;
        const pas = Math.min(v, dd);
        f.x += dx / dd * pas; f.y += dy / dd * pas;
        orienter(f, dx, dy);
        return;
      }
    }
  }
  const dx = c.x - f.x, dy = c.y - f.y, dd = Math.hypot(dx, dy);
  if (dd < 0.8) { arriver(f); return; }
  const pas = Math.min(v, dd);
  const nx = f.x + dx / dd * pas, ny = f.y + dy / dd * pas;
  if (S.grille[cellDe(nx, ny)] === GALERIE) { f.x = nx; f.y = ny; orienter(f, dx, dy); }
  else { f.blocage = (f.blocage || 0) + 1; if (f.blocage > 20) { f.blocage = 0; arriver(f); } }
}

/* ---------- Manipulation d'objets et de couvain ---------- */

function prendre(f, chose, type) {
  if (chose.porteur) return false;
  chose.porteur = f;
  f.porte = { type, ref: chose };
  return true;
}
function lacher(f, ici = false) {
  const p = f.porte;
  if (!p) return;
  f.porte = null;
  if (p.type === 'deblai') { if (f.mode === 'surface') deposerDeblai(f.x); return; }
  const ref = p.ref;
  if (!ref) return;
  ref.porteur = null;
  ref.x = f.x + Math.cos(f.cap) * f.taille * 0.35;
  ref.y = f.y + Math.sin(f.cap) * f.taille * 0.35;
  if (p.type === 'couvain') { ref.chambre = chambreDe(f); }
  else { ref.mode = f.mode; ref.chambre = f.mode === 'nid' ? chambreDe(f) : -1; if (f.mode === 'surface') ref.y = hauteur(ref.x) - 0.8; }
}

/* ---------- Décisions ---------- */

function decider(f) {
  f.creuseur = false;
  if (f.tacheVoulue && f.mode === 'nid' && !f.porte) { f.tache = f.tacheVoulue; f.tacheVoulue = null; }
  if (f.caste === 'reine') return deciderReine(f);
  if (f.tache === 'sexue') return deciderSexue(f);
  if (f.mode === 'surface') return deciderSurface(f);

  if (f.porte) return rangerCeQuOnPorte(f);

  if (S.hiver) {
    const h = chambrePlusProfondePrete();
    if (chambreDe(f) !== h.i) return viserChambre(f, h, 'hiberne');
    return entrerEtat(f, 'hiberne', 20 + R() * 40);
  }

  // Faim : solliciter une congénère
  if (f.jabot < CAP_JABOT * 0.2 && f.tache !== 'fourrageuse') {
    const ici = chambreDe(f);
    const donneuse = ici >= 0 && S.chambres[ici].presents.some((a) => a !== f && a.jabot > CAP_JABOT * 0.55 && (a.etat === 'offre' || a.etat === 'repos' || a.etat === 'decharge'));
    if (donneuse || (f.attenteFaim || 0) < 2) { f.attenteFaim = (f.attenteFaim || 0) + 1; return entrerEtat(f, 'sollicite', 12); }
    f.attenteFaim = 0;
    return viserChambre(f, S.chambres[0].pret ? S.chambres[0] : S.chambres[2], 'sollicite');
  }

  if (f.tache === 'nourrice') return deciderNourrice(f);
  if (f.tache === 'entretien') return deciderEntretien(f);
  return deciderFourrageuse(f);
}

function rangerCeQuOnPorte(f) {
  const p = f.porte;
  if (p.type === 'deblai' || p.type === 'cadavre') {
    // Pas de sortie (loge scellée, puits bouché) : on tasse dans un coin
    if (!champs.entree || champs.entree[cellDe(f.x, f.y)] >= INF) { lacher(f); if (p.type === 'deblai') return entrerEtat(f, 'repos', 2); return entrerEtat(f, 'repos', 4); }
    return viser(f, { champ: champs.entree }, 'sortie');
  }
  if (p.type === 'couvain') { const ch = chambreCible(p.ref); const s = spotCouvain(ch); return viser(f, { champ: champs.chambres[ch.i], chambre: ch.i, x: s.x, y: s.y }, 'depose'); }
  if (p.type === 'proie') {
    const ch = S.chambreCouvain || S.chambres[2];
    const larve = S.couvain.find((b) => b.stade === 'larve' && b.chambre === ch.i && !b.porteur && !b.repas);
    if (larve) return viser(f, { champ: champs.chambres[ch.i], chambre: ch.i, x: larve.x, y: larve.y }, 'depose');
    const s = spotCouvain(ch);
    return viser(f, { champ: champs.chambres[ch.i], chambre: ch.i, x: s.x, y: s.y }, 'depose');
  }
  lacher(f);
  return entrerEtat(f, 'repos', 2);
}

function deciderNourrice(f) {
  const ch = S.chambreCouvain || S.chambres[2];
  const ici = chambreDe(f);
  f.chambreMaison = ch;
  const r = R();

  // Le couvain qui n'est pas au bon endroit : œufs pondus chez la reine, migration thermique
  const aDeplacer = S.couvain.filter((b) => !b.porteur && b.chambre !== chambreCible(b).i && b.chambre >= 0);
  if (aDeplacer.length && r < 0.7) {
    const b = aDeplacer.reduce((m, b) => (b.chambre === ici ? -1 : 0) < (m.chambre === ici ? -1 : 0) ? b : m, aDeplacer[Math.floor(R() * aDeplacer.length)]);
    return viserPoint(f, b.x, b.y, 'prendCouvain'), (f.viseeCouvain = b);
  }
  const nymphePrete = S.couvain.find((b) => b.stade === 'nymphe' && b.pret && !b.porteur && !b.assistee);
  if (nymphePrete) { nymphePrete.assistee = true; f.viseeCouvain = nymphePrete; return viserPoint(f, nymphePrete.x, nymphePrete.y, 'aideEclosion'); }

  const larvesFaim = S.couvain.filter((b) => b.stade === 'larve' && !b.porteur && b.sucre < 0.45);
  if (larvesFaim.length && f.jabot > CAP_JABOT * 0.22) {
    const b = choix(larvesFaim); f.viseeCouvain = b;
    return viserPoint(f, b.x, b.y, 'nourritLarve');
  }
  const larvesProt = S.couvain.filter((b) => b.stade === 'larve' && !b.porteur && b.prot < 0.4 && !b.repas);
  const proies = S.objets.filter((o) => o.type === 'proie' && o.mode === 'nid' && !o.porteur && !S.couvain.some((b) => b.repas === o));
  if (larvesProt.length && proies.length && r < 0.8) {
    const o = proies[0]; f.viseeObjet = o;
    return viserPoint(f, o.x, o.y, 'prendProie');
  }
  if (S.reine && S.reine.jabot < CAP_REINE * 0.5 && f.jabot > CAP_JABOT * 0.35 && r < 0.6) {
    return viserPoint(f, S.reine.x + Math.cos(S.reine.cap) * 6, S.reine.y + Math.sin(S.reine.cap) * 6, 'nourritReine');
  }
  if (ici !== ch.i) return viserChambre(f, ch, 'repos');
  if (f.jabot > CAP_JABOT * 0.6 && r < 0.25) return entrerEtat(f, 'offre', 15);
  if (r < 0.12) return entrerEtat(f, 'toilettage', 3 + R() * 3);
  if (r < 0.3) { const oeuf = S.couvain.find((b) => b.stade === 'oeuf' && !b.porteur && b.chambre === ici); if (oeuf) { f.viseeCouvain = oeuf; return viserPoint(f, oeuf.x, oeuf.y, 'lecheOeufs'); } }
  if (r < 0.5) { const s = spotChambre(ch); return viser(f, { champ: champs.chambres[ch.i], chambre: ch.i, x: s.x, y: s.y }, 'repos'); }
  return entrerEtat(f, 'repos', 3 + R() * 8);
}

function deciderEntretien(f) {
  const ici = chambreDe(f);
  const r = R();
  const cadavre = S.objets.find((o) => o.type === 'cadavre' && o.mode === 'nid' && !o.porteur);
  if (cadavre && r < 0.75) { f.viseeObjet = cadavre; return viserPoint(f, cadavre.x, cadavre.y, 'prendCadavre'); }

  const maxCreuseurs = Math.max(2, Math.round(S.nbOuvrieres * 0.08));
  if (S.nbFront > 0 && S.nbCreuseurs < maxCreuseurs && S.tNid > 10 && r < 0.7) {
    S.nbCreuseurs++;
    f.creuseur = true;
    return viser(f, { champ: champs.front }, 'chercheFront');
  }
  if (f.jabot > CAP_JABOT * 0.6 && r < 0.6) {
    const ch = S.chambreCouvain || S.chambres[2];
    if (ici !== ch.i) return viserChambre(f, ch, 'offre');
    return entrerEtat(f, 'offre', 20);
  }
  if (f.jabot < CAP_JABOT * 0.45 && r < 0.5) {
    const v = S.chambres[0].pret ? S.chambres[0] : S.chambres[2];
    if (ici !== v.i) return viserChambre(f, v, 'sollicite');
    return entrerEtat(f, 'sollicite', 15);
  }
  if (r < 0.1) return entrerEtat(f, 'toilettage', 3 + R() * 3);
  if (r < 0.35) {
    const repos = chambresPar('repos');
    const ch = repos.length && R() < 0.6 ? choix(repos) : (S.chambres[0].pret ? S.chambres[0] : S.chambres[2]);
    return viserChambre(f, ch, 'repos');
  }
  return entrerEtat(f, 'repos', 4 + R() * 10);
}

function deciderFourrageuse(f) {
  const ici = chambreDe(f);
  const v = S.chambres[0].pret ? S.chambres[0] : S.chambres[2];
  if (f.jabot > CAP_JABOT * 0.55) {
    if (f.decharges >= 3) {
      // Personne ne prend sa récolte : elle nourrit elle-même une larve ou la reine
      f.decharges = 0;
      const larve = S.couvain.find((b) => b.stade === 'larve' && !b.porteur && b.sucre < 0.7);
      if (larve) { f.viseeCouvain = larve; return viserPoint(f, larve.x, larve.y, 'nourritLarve'); }
      if (S.reine && S.reine.jabot < CAP_REINE * 0.8) return viserPoint(f, S.reine.x + Math.cos(S.reine.cap) * 6, S.reine.y + Math.sin(S.reine.cap) * 6, 'nourritReine');
      f.energie = 1; f.jabot = CAP_JABOT * 0.3;
    }
    if (f.decharges >= 2) { f.decharges = 0; const ch = S.chambreCouvain || S.chambres[2]; if (ici !== ch.i) return viserChambre(f, ch, 'decharge'); return entrerEtat(f, 'decharge', 25); }
    if (ici !== v.i) return viserChambre(f, v, 'decharge');
    return entrerEtat(f, 'decharge', 25);
  }
  if (f.jabot < CAP_JABOT * 0.12 && S.activite < 0.2) { if (ici !== v.i) return viserChambre(f, v, 'sollicite'); return entrerEtat(f, 'sollicite', 15); }
  if (S.activite > R() * 0.9 && S.rowEntree < GH && champs.entree[cellDe(f.x, f.y)] < INF) {
    return viser(f, { champ: champs.entree }, 'sortie');
  }
  if (ici !== v.i && R() < 0.5) return viserChambre(f, v, 'repos');
  return entrerEtat(f, 'repos', 5 + R() * 15);
}

function deciderSurface(f) {
  if (f.caste === 'reine') return deciderReine(f);
  if (f.porte) {
    if (f.porte.type === 'deblai') return viserSurface(f, pointDepotDeblai(), 'deposeDeblai');
    if (f.porte.type === 'cadavre') return viserSurface(f, clamp(S.xDepotoir + gauss() * 10, 5, LARG - 5), 'deposeCadavre');
    return viserSurface(f, S.xEntree, 'entree');
  }
  if (f.tache !== 'fourrageuse' || S.hiver) return viserSurface(f, S.xEntree, 'entree');
  if (f.jabot > CAP_JABOT * 0.55) return retourNid(f, true);
  // Fidélité au site : une fourrageuse retourne là où elle a déjà trouvé à manger
  if (f.souvenir !== null && S.sources.some((s) => Math.abs(s.x - f.souvenir) < 6)) { f.suitPiste = false; return viserSurface(f, f.souvenir + (R() - 0.5) * 4, 'arriveSource'); }
  f.souvenir = null;
  // Sinon, une piste de phéromone près de l'entrée ?
  const cE = Math.round(S.xEntree / MM);
  let g = 0, d = 0;
  for (let i = 3; i < 10; i++) { g += S.pheromone[clamp(cE - i, 0, GW - 1)]; d += S.pheromone[clamp(cE + i, 0, GW - 1)]; }
  if (Math.max(g, d) > 0.8 && R() < 0.85) {
    f.dir = d > g ? 1 : -1; f.suitPiste = true;
    return entrerEtat(f, 'suit_piste', 60);
  }
  f.dir = R() < 0.5 ? -1 : 1; f.suitPiste = false;
  return entrerEtat(f, 'exploration', 25 + R() * 45);
}

function retourNid(f, avecNourriture) {
  f.depose = avecNourriture;
  return viserSurface(f, S.xEntree, 'entree');
}

function deciderReine(f) {
  if (S.scenario === 'fondation' && f.mode === 'surface') return viserSurface(f, S.xEntree, 'fondation_entre');
  if (f.porte) return rangerCeQuOnPorte(f);
  const chR = S.chambres[2];
  if (S.fondation && !S.fondationScellee) {
    // Elle creuse elle-même sa loge
    if (champs.fondation && champs.fondation[cellDe(f.x, f.y)] < INF) return viser(f, { champ: champs.fondation }, 'chercheFondation');
    return entrerEtat(f, 'repos', 3);
  }
  if (S.hiver) { const h = chambrePlusProfondePrete(); if (chambreDe(f) !== h.i) return viserChambre(f, h, 'hiberne', 0.3); return entrerEtat(f, 'hiberne', 30); }
  const maison = S.hiver ? chambrePlusProfondePrete() : chR;
  if (chambreDe(f) !== maison.i && champs.chambres[maison.i][cellDe(f.x, f.y)] < INF) return viserChambre(f, maison, 'repos', 0.35);
  if ((f.oeufsEnAttente || 0) >= 1) return entrerEtat(f, 'pond', 4);
  // Seule (fondation) : elle fait aussi office de nourrice
  if (S.nbOuvrieres === 0) {
    const larve = S.couvain.find((b) => b.stade === 'larve' && b.sucre < 0.5);
    if (larve && f.reserves > 0.05 && R() < 0.8) { f.viseeCouvain = larve; return viserPoint(f, larve.x, larve.y, 'nourritLarve'); }
    const nymphe = S.couvain.find((b) => b.stade === 'nymphe' && b.pret && !b.assistee);
    if (nymphe) { nymphe.assistee = true; f.viseeCouvain = nymphe; return viserPoint(f, nymphe.x, nymphe.y, 'aideEclosion'); }
  }
  const r = R();
  if (r < 0.1) return entrerEtat(f, 'toilettage', 4);
  if (r < 0.2) { const s = spotChambre(maison, 0.4); return viser(f, { champ: champs.chambres[maison.i], chambre: maison.i, x: s.x, y: s.y }, 'repos'); }
  return entrerEtat(f, 'repos', 6 + R() * 12);
}

function deciderSexue(f) {
  if (f.mode === 'surface') { entrerEtat(f, 'vol_nuptial', 6); return; }
  if (f.volPret) return viser(f, { champ: champs.entree }, 'sortie');
  const maison = f.chambreMaison && S.chambres[f.chambreMaison.i].pret ? S.chambres[f.chambreMaison.i] : (chambresPar('repos')[0] || S.chambres[0]);
  if (f.jabot < CAP_JABOT * 0.3) return entrerEtat(f, 'sollicite', 15);
  if (chambreDe(f) !== maison.i) return viserChambre(f, maison, 'repos');
  if (R() < 0.15) return entrerEtat(f, 'toilettage', 4);
  return entrerEtat(f, 'attente', 10 + R() * 20);
}

/* ---------- Exécution des états ---------- */

function penser(f, dtV, dtL) {
  if (f.etat === 'deplacement') { deplacer(f, dtV); return; }
  if (f.nouveau) { f.nouveau = false; return executerArrivee(f); }
  f.minuterie -= dtV;

  switch (f.etat) {
    case 'repos': case 'hiberne': case 'attente':
      if (f.etat === 'hiberne' && S.hiver && R() < 0.002) { f.cap += (R() - 0.5); }
      if (f.minuterie <= 0) decider(f);
      return;
    case 'toilettage': case 'lecheOeufs':
      if (f.minuterie <= 0) decider(f);
      return;
    case 'sollicite':
      if (f.minuterie <= 0) { decider(f); }
      return;
    case 'offre': case 'decharge':
      if (f.minuterie <= 0) { if (f.etat === 'decharge') f.decharges = (f.decharges || 0) + 1; decider(f); }
      return;
    case 'attend':
      if (f.minuterie <= 0) { liberer(f); decider(f); }
      return;
    case 'trophallaxie':
      if (f.minuterie <= 0) {
        const p = f.partenaire;
        if (p && p.partenaire === f) {
          const don = f.donneuse ? f : p, rec = f.donneuse ? p : f;
          const cap = rec.caste === 'reine' ? CAP_REINE : CAP_JABOT;
          const garde = don.tache === 'fourrageuse' ? CAP_JABOT * 0.25 : CAP_JABOT * 0.3;
          const q = Math.max(0, Math.min(don.jabot - garde, cap - rec.jabot));
          don.jabot -= q; rec.jabot += q;
          if (rec.caste === 'reine') { rec.prot = Math.min(1, rec.prot + don.prot * 0.5); don.prot *= 0.5; }
          else { const qp = Math.min(don.prot * 0.4, 1 - rec.prot); rec.prot += qp; don.prot -= qp; }
          if (don.tache === 'fourrageuse') don.rapporte += q;
          p.partenaire = null; f.partenaire = null;
          entrerEtat(p, 'repos', 0.5); entrerEtat(f, 'repos', 0.5);
          if (don.etat !== 'repos') return;
          if (don.jabot > garde + 0.05 && (don.tache === 'fourrageuse' || don.tache === 'entretien')) entrerEtat(don, don.tache === 'fourrageuse' ? 'decharge' : 'offre', 15);
        } else { f.partenaire = null; entrerEtat(f, 'repos', 0.5); }
      }
      return;
    case 'creuse':
      if (f.minuterie <= 0) {
        if (f.creuseCell >= 0 && creuserCellule(f.creuseCell)) {
          if (f.caste === 'reine') {
            // La reine avance dans la cellule qu'elle vient d'ouvrir et repousse la terre derrière elle
            const c = f.creuseCell % GW, r = (f.creuseCell - c) / GW;
            f.x = c * MM + MM / 2; f.y = r * MM + MM / 2;
            const suivante = S.fondation ? voisinCreusable(f.creuseCell, true) : -1;
            f.creuseCell = -1;
            if (suivante >= 0) {
              const sc = suivante % GW, sr = (suivante - sc) / GW;
              f.cap = Math.atan2(sr * MM + MM / 2 - f.y, sc * MM + MM / 2 - f.x);
              f.creuseCell = suivante;
              return entrerEtat(f, 'creuse', 1.2);
            }
            derniereRecalc = -10;
            if (S.fondation) verifierFondation(f);
            return decider(f);
          }
          f.porte = { type: 'deblai' };
        }
        f.creuseCell = -1;
        if (f.creuseur) { f.creuseur = false; S.nbCreuseurs = Math.max(0, S.nbCreuseurs - 1); }
        decider(f);
      }
      return;
    case 'nourritLarve': case 'nourritReine': case 'aideEclosion': case 'pond':
      if (f.minuterie <= 0) finirAction(f);
      return;
    case 'exploration': case 'suit_piste': case 'cherche':
      surfaceExplorer(f, dtV);
      return;
    case 'recolte':
      f.grimpe = Math.min(1, f.grimpe + dtV * 0.5);
      if (f.minuterie <= 0) {
        const s = f.source;
        if (s && S.sources.includes(s)) {
          if (s.type === 'insecte') { const m = Math.min(s.masse, 1); s.masse -= m; f.porte = { type: 'proie', ref: creerObjet('proie', f.x, f.y, { mode: 'surface', masse: m }) }; f.porte.ref.porteur = f; }
          else { const q = Math.min(s.reserve, CAP_JABOT - f.jabot); s.reserve -= q; f.jabot += q; f.energie = 1; }
          if (!s.decouverte) { s.decouverte = true; journal(s.type === 'miellat' ? `Une fourrageuse découvre ${s.nom} sur la plante : du miellat sucré, à volonté. Elle rentre en déposant une piste de phéromone.` : `Une fourrageuse trouve ${s.nom} et rentre marquer une piste pour recruter des congénères.`); }
          f.sorties++;
          f.souvenir = s.x;
        }
        f.grimpe = 0; f.source = null;
        return retourNid(f, true);
      }
      return;
    case 'vol_nuptial':
      f.y -= dtV * 25; f.x += Math.sin(f.phase + S.minutes) * dtV * 8;
      if (f.y < -20) mourir(f, 'disparue');
      return;
    default:
      if (f.minuterie <= 0) decider(f);
  }
}

function executerArrivee(f) {
  switch (f.etat) {
    case 'sortie': {
      if (f.mode === 'nid') {
        f.mode = 'surface'; f.x = S.xEntree; f.y = hauteur(f.x) - 1;
        if (f.tache === 'fourrageuse' && !f.porte && R() < 0.004) { f.destin = true; }
      }
      return deciderSurface(f);
    }
    case 'entree': case 'fondation_entre': {
      if (f.destin) { f.destin = false; }
      if (f.etat === 'fondation_entre') {
        // Premier coup de mandibules dans le sol, depuis la surface
        f.mode = 'nid'; f.x = S.xEntree; f.y = SOL * MM - 1.5; f.cap = Math.PI / 2;
        f.creuseCell = idx(S.puitsX[SOL], SOL);
        return entrerEtat(f, 'creuse', 3);
      }
      f.mode = 'nid'; f.x = S.xEntree; f.y = S.rowEntree * MM + MM / 2 + 1;
      if (f.tache === 'fourrageuse' && f.depose) { f.depose = false; }
      return decider(f);
    }
    case 'deposeDeblai': lacher(f); return viserSurface(f, S.xEntree, 'entree');
    case 'deposeCadavre': lacher(f); return viserSurface(f, S.xEntree, 'entree');
    case 'depose': {
      const p = f.porte;
      lacher(f);
      if (p && p.type === 'proie') { const larve = S.couvain.find((b) => b.stade === 'larve' && !b.repas && Math.hypot(b.x - p.ref.x, b.y - p.ref.y) < 4); if (larve) { larve.repas = p.ref; f.prot = Math.min(1, f.prot + 0.15); } }
      return entrerEtat(f, 'repos', 1 + R() * 2);
    }
    case 'prendCouvain': {
      const b = f.viseeCouvain; f.viseeCouvain = null;
      if (b && S.couvain.includes(b) && !b.porteur && Math.hypot(b.x - f.x, b.y - f.y) < 6) { prendre(f, b, 'couvain'); return rangerCeQuOnPorte(f); }
      return decider(f);
    }
    case 'prendProie': case 'prendCadavre': {
      const o = f.viseeObjet; f.viseeObjet = null;
      if (o && S.objets.includes(o) && !o.porteur && Math.hypot(o.x - f.x, o.y - f.y) < 6) { prendre(f, o, o.type); if (o.type === 'proie') f.prot = Math.min(1, f.prot + 0.1); return rangerCeQuOnPorte(f); }
      return decider(f);
    }
    case 'nourritLarve': case 'nourritReine': case 'aideEclosion': case 'lecheOeufs':
      f.minuterie = f.etat === 'aideEclosion' ? 5 : f.etat === 'lecheOeufs' ? 4 : 3.5;
      return;
    case 'chercheFront': case 'chercheFondation': {
      const k = voisinCreusable(cellDe(f.x, f.y), f.etat === 'chercheFondation');
      if (k >= 0) { f.creuseCell = k; const c = k % GW, r = (k - c) / GW; orienter(f, c * MM + MM / 2 - f.x, r * MM + MM / 2 - f.y); f.cap = Math.atan2(r * MM + MM / 2 - f.y, c * MM + MM / 2 - f.x); return entrerEtat(f, 'creuse', f.caste === 'reine' ? 2.5 : 3 + R() * 3); }
      if (f.creuseur) { f.creuseur = false; S.nbCreuseurs = Math.max(0, S.nbCreuseurs - 1); }
      return entrerEtat(f, 'repos', 1);
    }
    case 'arriveSource': {
      const s = S.sources.find((s) => Math.abs(s.x - f.x) < 7);
      if (s) return commencerRecolte(f, s);
      f.souvenir = null; f.dir = R() < 0.5 ? -1 : 1;
      return entrerEtat(f, 'cherche', 15);
    }
    case 'sollicite': case 'offre': case 'decharge': case 'repos': case 'hiberne': case 'attente':
      f.minuterie = f.etat === 'repos' ? 3 + R() * 8 : f.etat === 'hiberne' ? 30 : f.etat === 'attente' ? 15 : f.etat === 'decharge' ? 25 : 15;
      return;
    case 'trophallaxie':
      f.minuterie = 3.5;
      if (f.partenaire && f.partenaire.partenaire === f) { f.partenaire.etat = 'trophallaxie'; f.partenaire.minuterie = 3.5; f.cap = Math.atan2(f.partenaire.y - f.y, f.partenaire.x - f.x); f.partenaire.cap = f.cap + Math.PI; }
      else { f.partenaire = null; entrerEtat(f, 'repos', 0.5); }
      return;
    case 'pond': f.minuterie = 4; return;
    case 'vol_nuptial': f.minuterie = 6; return;
    default:
      return decider(f);
  }
}

function finirAction(f) {
  const b = f.viseeCouvain; f.viseeCouvain = null;
  switch (f.etat) {
    case 'nourritLarve':
      if (b && S.couvain.includes(b) && b.stade === 'larve') {
        if (f.caste === 'reine' && S.claustration) { f.reserves = Math.max(0, f.reserves - 0.005); b.sucre = Math.min(1, b.sucre + 0.4); b.prot = Math.min(1, b.prot + 0.3); }
        else { const q = Math.min(f.jabot - CAP_JABOT * 0.1, 0.1); if (q > 0) { f.jabot -= q; b.sucre = Math.min(1, b.sucre + q * 4); } }
      }
      break;
    case 'nourritReine':
      if (S.reine && Math.hypot(S.reine.x - f.x, S.reine.y - f.y) < 12) {
        const q = Math.max(0, Math.min(f.jabot - CAP_JABOT * 0.2, CAP_REINE - S.reine.jabot, 0.2));
        f.jabot -= q; S.reine.jabot += q;
        const qp = Math.min(f.prot * 0.5, 1 - S.reine.prot); S.reine.prot += qp; f.prot -= qp;
        S.reine.cap = Math.atan2(f.y - S.reine.y, f.x - S.reine.x);
      }
      break;
    case 'aideEclosion':
      if (b && S.couvain.includes(b) && b.stade === 'nymphe') eclore(b, f);
      break;
    case 'pond': {
      f.oeufsEnAttente = Math.max(0, (f.oeufsEnAttente || 0) - 1);
      f.oeufs++; S.compteurs.oeufs++;
      f.prot = Math.max(0, f.prot - 0.012);
      const d = dateSim();
      let sexue = null;
      if (S.sexuesProduits && d.doy >= 95 && d.doy <= 135 && R() < 0.12) sexue = R() < 0.5 ? 'gyne' : 'male';
      creerCouvain('oeuf', f.x - Math.cos(f.cap) * f.taille * 0.45, f.y - Math.sin(f.cap) * f.taille * 0.45, S.chambres[chambreDe(f)] || S.chambres[2], { sexue });
      if (f.oeufs === 1) journal(S.scenario === 'fondation' ? `La reine pond son premier œuf, enfermée dans sa loge. Elle ne mangera pas avant l'arrivée de ses premières ouvrières : elle vit sur ses graisses et ses muscles alaires, qui se résorbent.` : `Premier œuf de la saison observé.`);
      break;
    }
  }
  decider(f);
}

function verifierFondation(f) {
  let reste = 0;
  for (let k = 0; k < GW * GH; k++) if (S.fondation[k] && S.grille[k] !== GALERIE) reste++;
  if (reste > 0) return;
  // Loge terminée : elle referme le puits derrière elle. La claustration commence.
  for (let r = S.rowEntree; r <= SOL + 10; r++) {
    for (const c of [S.puitsX[r], S.puitsX[r] + 1]) { const k = idx(c, r); if (S.grille[k] === GALERIE) { S.grille[k] = TERRE; peindreTerre(k); } }
  }
  S.fondationScellee = true; S.claustration = true; S.fondation = null;
  champsSales = true;
  const chR = S.chambres[2];
  const p = spotChambre(chR, 0.3); f.x = p.x; f.y = p.y; assainir(f);
  journal(`Sa loge est creusée et le puits rebouché : c'est la fondation claustrale. Pendant des semaines, la reine restera enfermée sans manger, à pondre et à élever seule sa première génération.`);
}

function commencerRecolte(f, s) {
  f.source = s;
  f.x = s.x + (R() - 0.5) * 3;
  f.grimpe = 0;
  entrerEtat(f, 'recolte', s.type === 'insecte' ? 8 + R() * 6 : s.type === 'miellat' ? 14 + R() * 12 : 8 + R() * 6);
}

function surfaceExplorer(f, dtV) {
  const v = vitesseDe(f) * dtV;
  const c = clamp(Math.round(f.x / MM - 0.5), 0, GW - 1);
  // Une source à portée ?
  for (const s of S.sources) {
    if (Math.abs(s.x - f.x) < 5 && (s.type !== 'insecte' || s.masse > 0.05) && (s.type === 'insecte' || s.reserve > 0.05)) return commencerRecolte(f, s);
  }
  if (f.etat === 'suit_piste') {
    // Suivre la piste en s'éloignant du nid ; quand elle s'estompe, chercher autour
    const devant = S.pheromone[clamp(c + f.dir * 3, 0, GW - 1)] + S.pheromone[clamp(c + f.dir * 6, 0, GW - 1)];
    if (devant < 0.25) { f.etat = 'cherche'; f.minuterie = 12 + R() * 10; f.centre = f.x; }
  } else if (f.etat === 'cherche') {
    if (R() < 0.05 * dtV * 10) f.dir = -f.dir;
    if (f.centre !== undefined && Math.abs(f.x - f.centre) > 30) f.dir = Math.sign(f.centre - f.x);
  } else {
    if (R() < 0.012 * dtV * 10) f.dir = -f.dir;
  }
  if (f.x < 6) f.dir = 1;
  if (f.x > LARG - 6) f.dir = -1;
  f.x += f.dir * v;
  f.y = hauteur(f.x) - f.taille * 0.12;
  if (f.destin && Math.abs(f.x - S.xEntree) > 80 && R() < 0.05) { journal(`Une fourrageuse n'est pas rentrée. Dehors, araignées, oiseaux et carabes guettent : sortir est le travail le plus dangereux, réservé aux plus âgées.`); mourir(f, 'disparue'); return; }
  if (f.minuterie <= 0) { f.centre = undefined; return retourNid(f, false); }
}

// Rencontres : les fourmis qui offrent et celles qui sollicitent se trouvent dans la même chambre.
function apparier() {
  for (const ch of S.chambres) {
    if (ch.presents.length < 2) continue;
    const offres = [], demandes = [];
    for (const f of ch.presents) {
      if (f.partenaire) continue;
      if ((f.etat === 'offre' || f.etat === 'decharge') && f.jabot > CAP_JABOT * 0.4) offres.push(f);
      else if (f.etat === 'sollicite') demandes.push(f);
      // Une ouvrière au repos accepte volontiers une goutte quand on la lui propose
      else if ((f.etat === 'repos' || f.etat === 'attente') && f.caste !== 'reine' && f.jabot < CAP_JABOT * 0.45) demandes.push(f);
    }
    while (offres.length && demandes.length) {
      const don = offres.pop(), rec = demandes.pop();
      don.partenaire = rec; rec.partenaire = don;
      don.donneuse = true; rec.donneuse = false;
      entrerEtat(don, 'attend', 12);
      const a = Math.atan2(rec.y - don.y, rec.x - don.x);
      viserPoint(rec, don.x + Math.cos(a) * (don.taille * 0.5 + rec.taille * 0.5 + 0.3), don.y + Math.sin(a) * (don.taille * 0.5 + rec.taille * 0.5 + 0.3), 'trophallaxie');
      rec.cible.libre = true;
    }
  }
}

/* ======================================================================
   Simulation : une image
   ====================================================================== */

let derniere = performance.now(), accHud = 0, accSauvegarde = 0, sauvegardeEnAttente = false;

function simuler(dtV, dtS, dtL) {
  // Phéromone : évaporation en temps visuel
  const ev = Math.exp(-dtV / 45);
  const ph = S.pheromone;
  for (let i = 0; i < GW; i++) ph[i] *= ev;
  for (const f of S.fourmis) {
    if (f.mode === 'surface' && f.tache === 'fourrageuse' && f.etat === 'deplacement' && f.depose && (f.jabot > CAP_JABOT * 0.5 || (f.porte && f.porte.type === 'proie'))) {
      ph[clamp(Math.round(f.x / MM - 0.5), 0, GW - 1)] += dtV * 3;
    }
  }
  S.nbCreuseurs = 0;
  for (const f of S.fourmis) if (f.creuseur) S.nbCreuseurs++;

  // Heures biologiques
  S.derniereHeure += dtS;
  let n = 0;
  while (S.derniereHeure >= 60 && n++ < 50) { S.derniereHeure -= 60; tickHeure(); }

  if (champsSales && performance.now() - derniereRecalc > 700) { derniereRecalc = performance.now(); recalculerChamps(); }

  for (const ch of S.chambres) ch.presents.length = 0;
  for (const f of S.fourmis) {
    if (f.mode === 'nid') { const c = S.cellChambre[cellDe(f.x, f.y)]; if (c >= 0) S.chambres[c].presents.push(f); }
    f.phase += dtV * (f.etat === 'deplacement' || f.etat === 'exploration' || f.etat === 'suit_piste' || f.etat === 'cherche' ? 14 : 0);
  }
  apparier();
  for (const f of S.fourmis.slice()) {
    penser(f, dtV, dtL);
    if (f.porte && f.porte.ref) { f.porte.ref.x = f.x + Math.cos(f.cap) * f.taille * 0.5; f.porte.ref.y = f.y + Math.sin(f.cap) * f.taille * 0.5; }
  }
}

function boucle(t) {
  const dtReel = Math.min(0.05, (t - derniere) / 1000);
  derniere = t;
  const v = VITESSES[ui.vitesse];
  if (v > 0 && S) {
    const dtV = dtReel * Math.min(v, VISUEL_MAX);
    const dtS = dtReel * v * MIN_PAR_SEC;
    const mult = v / Math.min(v, VISUEL_MAX);
    S.minutes += dtS;
    simuler(dtV, dtS, dtS / mult);
    sauvegardeEnAttente = true;
  }
  camera(dtReel);
  dessiner();
  accHud += dtReel;
  if (accHud > 0.25) { accHud = 0; rafraichirHud(); }
  accSauvegarde += dtReel;
  if (accSauvegarde > 5 && sauvegardeEnAttente) { accSauvegarde = 0; sauver(); }
  requestAnimationFrame(boucle);
}

/* ======================================================================
   Rendu
   ====================================================================== */

function peindreTerrain() {
  const E = 4;                      // pixels de texture par cellule
  texSol = document.createElement('canvas'); texSol.width = GW * E; texSol.height = GH * E;
  texGal = document.createElement('canvas'); texGal.width = GW * E; texGal.height = GH * E;
  const t = texSol.getContext('2d');
  const rng = mulberry(S.graine + 7);
  for (let r = 0; r < GH; r++) {
    const p = clamp((r - SOL) / (GH - SOL), 0, 1);
    for (let c = 0; c < GW; c++) {
      const k = idx(c, r);
      const g = S.grille[k];
      const bruit = (rng() - 0.5) * 14;
      if (g === ROCHE) { const v = 96 + bruit; t.fillStyle = `rgb(${v},${v - 2},${v - 6})`; }
      else {
        // Horizon A sombre et humifère en surface, plus ocre et minéral en profondeur
        const rr = lerp(84, 118, p) + bruit, gg = lerp(58, 78, p) + bruit * 0.9, bb = lerp(38, 46, p) + bruit * 0.6;
        t.fillStyle = `rgb(${rr},${gg},${bb})`;
      }
      t.fillRect(c * E, r * E, E, E);
      if (g !== ROCHE && rng() < 0.05) { t.fillStyle = rng() < 0.5 ? 'rgba(30,20,12,.55)' : 'rgba(190,160,120,.45)'; t.fillRect(c * E + rng() * 3, r * E + rng() * 3, 1.5, 1.5); }
    }
  }
  // cellules déjà creusées
  for (let k = 0; k < GW * GH; k++) if (S.grille[k] === GALERIE) peindreGalerie(k);
}
function peindreGalerie(k) {
  if (!texGal) return;
  const E = 4, c = k % GW, r = (k - c) / GW;
  const g = texGal.getContext('2d');
  g.fillStyle = '#9a7a57';
  g.fillRect(c * E - 0.5, r * E - 0.5, E + 1, E + 1);
}
function peindreTerre(k) {
  if (!texGal) return;
  const E = 4, c = k % GW, r = (k - c) / GW;
  const g = texGal.getContext('2d');
  g.clearRect(c * E, r * E, E, E);
}

function couleurCiel(lum) {
  const nuit = [12, 16, 34], aube = [232, 150, 96], jour = [138, 190, 230];
  const bas = lum < 0.5 ? nuit.map((v, i) => lerp(v, aube[i], lum * 2)) : aube.map((v, i) => lerp(v, jour[i], (lum - 0.5) * 2));
  const hautC = lum < 0.5 ? nuit.map((v, i) => lerp(v, [110, 130, 190][i], lum * 2)) : [110, 130, 190].map((v, i) => lerp(v, [80, 150, 220][i], (lum - 0.5) * 2));
  return { bas: `rgb(${bas.map(Math.round)})`, haut: `rgb(${hautC.map(Math.round)})` };
}

function versEcran(x, y) { return { x: (x - cam.x) * cam.z + canvas.width / 2 / dpr, y: (y - cam.y) * cam.z + canvas.height / 2 / dpr }; }
function versMonde(sx, sy) { return { x: (sx - canvas.width / 2 / dpr) / cam.z + cam.x, y: (sy - canvas.height / 2 / dpr) / cam.z + cam.y }; }

function camera(dt) {
  const w = canvas.width / dpr, h = canvas.height / dpr;
  cam.zMin = Math.min(w / LARG, h / HAUT) * 0.98;
  cam.z = clamp(cam.z, cam.zMin, cam.zMax);
  if (ui.suivre && ui.selection) {
    const f = S.fourmis.find((a) => a.id === ui.selection);
    if (f) { cam.x = lerp(cam.x, f.x, Math.min(1, dt * 6)); cam.y = lerp(cam.y, f.y, Math.min(1, dt * 6)); }
  }
  const demiL = w / 2 / cam.z, demiH = h / 2 / cam.z;
  if (demiL * 2 >= LARG) cam.x = LARG / 2; else cam.x = clamp(cam.x, demiL, LARG - demiL);
  if (demiH * 2 >= HAUT) cam.y = HAUT / 2; else cam.y = clamp(cam.y, demiH, HAUT - demiH);
}

function dessiner() {
  if (!S) return;
  const w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const d = dateSim();
  const lum = luminosite(d.doy, d.heure);
  const ciel = couleurCiel(lum);
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, ciel.haut); grad.addColorStop(1, ciel.bas);
  ctx.fillStyle = grad; ctx.fillRect(0, 0, w, h);

  // hors du monde : fond neutre
  ctx.fillStyle = '#12100d';
  const o = versEcran(0, 0), o2 = versEcran(LARG, HAUT);
  if (o.x > 0) ctx.fillRect(0, 0, o.x, h);
  if (o2.x < w) ctx.fillRect(o2.x, 0, w - o2.x, h);
  if (o.y > 0) ctx.fillRect(0, 0, w, o.y);
  if (o2.y < h) ctx.fillRect(0, o2.y, w, h - o2.y);

  ctx.save();
  ctx.translate(w / 2 - cam.x * cam.z, h / 2 - cam.y * cam.z);
  ctx.scale(cam.z, cam.z);

  // Soleil ou lune
  {
    const { lever, coucher } = soleil(d.doy);
    const t = (d.heure - lever) / (coucher - lever);
    if (t > -0.05 && t < 1.05) {
      const sx = 30 + t * (LARG - 60), sy = Y_SOL - 20 - Math.sin(clamp(t, 0, 1) * Math.PI) * 70;
      ctx.fillStyle = `rgba(255, 236, 170, ${0.9 * lum + 0.1})`;
      ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2); ctx.fill();
    } else {
      const tn = ((d.heure - coucher + 24) % 24) / ((lever - coucher + 24) % 24);
      const mx = 30 + tn * (LARG - 60), my = Y_SOL - 20 - Math.sin(clamp(tn, 0, 1) * Math.PI) * 60;
      ctx.fillStyle = 'rgba(230, 232, 240, .9)';
      ctx.beginPath(); ctx.arc(mx, my, 6, 0, Math.PI * 2); ctx.fill();
    }
  }
  // Nuit : le ciel s'assombrit, on voit quelques étoiles
  if (lum < 0.3) {
    ctx.fillStyle = `rgba(255,255,255,${(0.3 - lum) * 2})`;
    const rng = mulberry(S.graine + 3);
    for (let i = 0; i < 40; i++) { ctx.fillRect(rng() * LARG, rng() * (Y_SOL - 30), 0.7, 0.7); }
  }

  // Plante à pucerons (derrière le sol pour les racines)
  dessinerPlante(lum);

  // Sol : polygone sous la ligne de surface, texture à l'intérieur
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(0, HAUT);
  for (let c = 0; c < GW; c++) ctx.lineTo(c * MM + MM / 2, S.hauteurSol[c]);
  ctx.lineTo(LARG, HAUT); ctx.closePath();
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(texSol, 0, 0, LARG, HAUT);
  // terre meuble de la butte : plus claire et plus grumeleuse
  ctx.fillStyle = 'rgba(160,120,80,.32)';
  ctx.beginPath(); ctx.moveTo(0, Y_SOL);
  for (let c = 0; c < GW; c++) ctx.lineTo(c * MM + MM / 2, S.hauteurSol[c]);
  ctx.lineTo(LARG, Y_SOL); ctx.closePath(); ctx.fill();
  ctx.drawImage(texGal, 0, 0, LARG, HAUT);
  ctx.restore();

  // ligne de surface
  ctx.strokeStyle = lum > 0.5 ? 'rgba(80,110,50,.7)' : 'rgba(40,50,30,.7)';
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  for (let c = 0; c < GW; c++) { const x = c * MM + MM / 2, y = S.hauteurSol[c]; if (c) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
  ctx.stroke();

  if (ui.temperature) dessinerTemperatures(d);
  if (ui.plan) dessinerPlan();

  dessinerSurface(lum);
  if (ui.pheromone) dessinerPheromone();

  // Couvain et objets
  for (const b of S.couvain) if (!b.porteur) dessinerCouvain(b);
  for (const ob of S.objets) if (!ob.porteur) dessinerObjet(ob);
  for (const f of S.fourmis) dessinerFourmi(f);
  for (const f of S.fourmis) if (f.porte && f.porte.ref) { const ref = f.porte.ref; if (f.porte.type === 'couvain') dessinerCouvain(ref); else dessinerObjet(ref); }
  for (const f of S.fourmis) if (f.porte && f.porte.type === 'deblai') { ctx.fillStyle = '#a8804f'; ctx.beginPath(); ctx.ellipse(f.x + Math.cos(f.cap) * f.taille * 0.55, f.y + Math.sin(f.cap) * f.taille * 0.55, 1.1, 0.9, 0, 0, Math.PI * 2); ctx.fill(); }

  // Sélection
  if (ui.selection) {
    const f = S.fourmis.find((a) => a.id === ui.selection);
    if (f) {
      ctx.strokeStyle = 'rgba(229,165,75,.95)'; ctx.lineWidth = 1.2 / cam.z * 1.5;
      ctx.beginPath(); ctx.arc(f.x, f.y, f.taille * 0.75 + 1.5, 0, Math.PI * 2); ctx.stroke();
    }
  }
  ctx.restore();

  // Nuit : voile bleuté sur la partie aérienne uniquement
  if (lum < 1) {
    const ySol = versEcran(0, Y_SOL).y;
    ctx.fillStyle = `rgba(10,14,40,${(1 - lum) * 0.25})`;
    ctx.fillRect(0, 0, w, Math.max(0, ySol));
  }
}

function dessinerPlante(lum) {
  const x = S.xPlante, y0 = hauteur(x);
  ctx.save();
  ctx.strokeStyle = lum > 0.4 ? '#4f7a2c' : '#2f4a1c'; ctx.lineWidth = 1.4; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(x, y0 + 1); ctx.quadraticCurveTo(x + 3, y0 - 30, x - 2, y0 - 62); ctx.stroke();
  ctx.fillStyle = lum > 0.4 ? '#5c8f34' : '#365522';
  const feuilles = [[-14, -28, -0.9], [12, -40, 0.7], [-10, -50, -1.1], [9, -18, 0.6]];
  for (const [dx, dy, a] of feuilles) {
    ctx.save(); ctx.translate(x + dx * 0.5, y0 + dy); ctx.rotate(a);
    ctx.beginPath(); ctx.ellipse(0, 0, 9, 3.2, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
  // pucerons sur la tige et le dessous des feuilles
  ctx.fillStyle = lum > 0.4 ? '#9ed36a' : '#5f8a3a';
  const rng = mulberry(S.graine + 11);
  for (let i = 0; i < 9; i++) { const t = 0.3 + rng() * 0.6; const px = x + 3 * (1 - t) * t * 4 - 2 * t + (rng() - 0.5) * 2, py = y0 - t * 62; ctx.beginPath(); ctx.ellipse(px, py, 0.9, 0.65, rng(), 0, Math.PI * 2); ctx.fill(); }
  // racines
  ctx.strokeStyle = 'rgba(200,180,140,.35)'; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(x, y0); ctx.quadraticCurveTo(x - 6, y0 + 14, x - 12, y0 + 26); ctx.moveTo(x, y0); ctx.quadraticCurveTo(x + 5, y0 + 12, x + 9, y0 + 22); ctx.stroke();
  ctx.restore();
}

function dessinerSurface(lum) {
  // herbe là où la butte n'a pas recouvert le sol
  ctx.strokeStyle = lum > 0.4 ? 'rgba(96,140,60,.9)' : 'rgba(50,75,35,.9)'; ctx.lineWidth = 0.5; ctx.lineCap = 'round';
  ctx.beginPath();
  for (const hb of S.herbes) {
    const y = hauteur(hb.x);
    if (y < Y_SOL - 1.5) continue;
    ctx.moveTo(hb.x, y + 0.5); ctx.lineTo(hb.x + hb.pente * hb.h * 0.6, y - hb.h);
    ctx.moveTo(hb.x + 1.2, y + 0.5); ctx.lineTo(hb.x + 1.2 + (hb.pente - 0.4) * hb.h * 0.5, y - hb.h * 0.7);
  }
  ctx.stroke();
  for (const p of S.pierresSurface) {
    const y = hauteur(p.x);
    ctx.fillStyle = '#6e6a63'; ctx.beginPath(); ctx.ellipse(p.x, y + p.ry * 0.3, p.rx, p.ry, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,.12)'; ctx.beginPath(); ctx.ellipse(p.x - p.rx * 0.25, y - p.ry * 0.2, p.rx * 0.45, p.ry * 0.3, -0.4, 0, Math.PI * 2); ctx.fill();
  }
  // sources de nourriture
  for (const s of S.sources) {
    const y = hauteur(s.x);
    if (s.type === 'sucre') {
      const rr = 2 + 3 * Math.sqrt(s.reserve / 12);
      ctx.fillStyle = 'rgba(255,240,200,.75)'; ctx.beginPath(); ctx.ellipse(s.x, y - rr * 0.35, rr, rr * 0.5, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,.5)'; ctx.beginPath(); ctx.ellipse(s.x - rr * 0.3, y - rr * 0.55, rr * 0.3, rr * 0.15, 0, 0, Math.PI * 2); ctx.fill();
    } else if (s.type === 'insecte') {
      const L = 3 + 4 * Math.sqrt(s.masse / 14);
      ctx.save(); ctx.translate(s.x, y - L * 0.18); ctx.rotate(0.1);
      ctx.fillStyle = '#3b2c22'; ctx.beginPath(); ctx.ellipse(0, 0, L * 0.5, L * 0.22, 0, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(200,210,230,.45)'; ctx.lineWidth = 0.4; ctx.beginPath(); ctx.ellipse(L * 0.15, -L * 0.15, L * 0.45, L * 0.14, 0.4, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = '#3b2c22'; ctx.beginPath(); for (let i = -1; i <= 1; i++) { ctx.moveTo(i * L * 0.15, L * 0.1); ctx.lineTo(i * L * 0.2 + L * 0.25, -L * 0.4); } ctx.stroke();
      ctx.restore();
    }
  }
}

function dessinerPheromone() {
  const ph = S.pheromone;
  for (let c = 0; c < GW; c++) {
    if (ph[c] < 0.05) continue;
    const a = Math.min(0.9, ph[c] / 1.5);
    const y = S.hauteurSol[c];
    ctx.fillStyle = `rgba(140,220,255,${a * 0.55})`;
    ctx.fillRect(c * MM, y - 0.4, MM + 0.2, 1.1);
  }
}

function dessinerTemperatures(d) {
  // Une échelle verticale sur le bord gauche : la température du sol selon la profondeur
  const L = Math.max(8, 22 / cam.z);
  const police = Math.max(2.4, 11 / cam.z);
  ctx.save();
  for (let y = Y_SOL; y < HAUT; y += 5) {
    const T = tempSol(y + 2.5 - Y_SOL, d.doy, d.heure);
    const t = clamp(T / 30, 0, 1);
    ctx.fillStyle = `hsl(${lerp(225, 15, t)}, 75%, ${lerp(45, 55, t)}%)`;
    ctx.fillRect(0, y, L, 5.2);
  }
  ctx.fillStyle = 'rgba(255,255,255,.95)';
  ctx.font = `${police}px ui-monospace, monospace`; ctx.textAlign = 'left';
  for (let y = Y_SOL; y < HAUT; y += 50) {
    const T = tempSol(y - Y_SOL, d.doy, d.heure);
    ctx.fillText(`${fmt((y - Y_SOL) / 10, 0)} cm · ${fmt(T, 1)} °C`, L + 2, y + police * 1.1);
  }
  ctx.restore();
}

function dessinerPlan() {
  ctx.save();
  ctx.font = `${Math.max(2.4, 10 / cam.z)}px Outfit, system-ui, sans-serif`; ctx.textAlign = 'center';
  for (const ch of S.chambres) {
    const ouverte = ch.seuil <= S.nbOuvrieres;
    ctx.setLineDash(ch.pret ? [] : [1.2, 1]);
    ctx.strokeStyle = ch.pret ? 'rgba(229,165,75,.75)' : ouverte ? 'rgba(229,165,75,.45)' : 'rgba(255,255,255,.22)';
    ctx.lineWidth = 0.4;
    ctx.beginPath(); ctx.ellipse(ch.cx, ch.cy, ch.rx + 1.5, ch.ry + 1.5, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = ch.pret ? 'rgba(255,240,215,.95)' : 'rgba(255,255,255,.55)';
    let lib = ROLES[ch.role];
    if (S.chambreCouvain && S.chambreCouvain.i === ch.i) lib += ' · couvain ici';
    else if (S.chambreNymphes && S.chambreNymphes.i === ch.i) lib += ' · nymphes ici';
    if (!ouverte) lib += ` (dès ${ch.seuil} ouvrières)`;
    const police = Math.max(2.4, 10 / cam.z);
    ctx.fillText(lib, ch.cx, ch.cy - ch.ry - police);
    ctx.fillStyle = 'rgba(255,255,255,.6)';
    ctx.fillText(`${fmt(ch.temp, 1)} °C`, ch.cx, ch.cy + ch.ry + police * 1.7);
  }
  ctx.restore();
}

function dessinerCouvain(b) {
  ctx.save();
  ctx.translate(b.x, b.y);
  if (b.stade === 'oeuf') {
    ctx.fillStyle = '#f4efe2';
    ctx.beginPath(); ctx.ellipse(0, 0, 0.55, 0.35, b.id % 7 * 0.5, 0, Math.PI * 2); ctx.fill();
  } else if (b.stade === 'larve') {
    const L = (1.2 + 2.8 * b.dev) * (b.sexue ? 1.5 : 1);
    ctx.rotate(b.id % 11 * 0.6);
    ctx.fillStyle = b.sucre < 0.25 ? '#e2d5bd' : '#f6efe0';
    ctx.beginPath(); ctx.moveTo(-L * 0.5, 0); ctx.quadraticCurveTo(0, -L * 0.45, L * 0.5, 0); ctx.quadraticCurveTo(0, L * 0.05, -L * 0.5, 0); ctx.fill();
    ctx.strokeStyle = 'rgba(120,100,70,.35)'; ctx.lineWidth = 0.15;
    ctx.beginPath(); for (let i = 1; i < 5; i++) { const x = -L * 0.5 + L * i / 5; ctx.moveTo(x, -L * 0.14); ctx.lineTo(x, L * 0.02); } ctx.stroke();
  } else {
    const L = 3.6 * (b.sexue ? 1.6 : 1);
    ctx.rotate(b.id % 9 * 0.7);
    ctx.fillStyle = b.pret ? '#d9c39a' : '#e8d6b0';
    ctx.beginPath(); ctx.ellipse(0, 0, L * 0.5, L * 0.24, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(150,120,80,.5)'; ctx.lineWidth = 0.15;
    ctx.beginPath(); ctx.ellipse(0, 0, L * 0.5, L * 0.24, 0, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.restore();
}

function dessinerObjet(o) {
  ctx.save();
  ctx.translate(o.x, o.y);
  if (o.type === 'proie') {
    const L = 1 + 1.6 * Math.sqrt(Math.max(0, o.masse));
    ctx.fillStyle = '#4a352a';
    ctx.beginPath(); ctx.ellipse(0, 0, L * 0.6, L * 0.4, o.id % 5 * 0.7, 0, Math.PI * 2); ctx.fill();
  } else if (o.type === 'cadavre') {
    ctx.rotate(o.id % 7 * 0.9);
    ctx.fillStyle = '#5b5148';
    const L = o.taille || 4;
    ctx.beginPath(); ctx.ellipse(-L * 0.28, 0, L * 0.22, L * 0.14, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(L * 0.05, 0, L * 0.15, L * 0.09, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(L * 0.32, 0, L * 0.12, L * 0.1, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#5b5148'; ctx.lineWidth = 0.18;
    ctx.beginPath(); for (let i = -1; i <= 1; i++) { ctx.moveTo(i * L * 0.12, 0); ctx.lineTo(i * L * 0.12 + L * 0.1, -L * 0.28); ctx.moveTo(i * L * 0.12, 0); ctx.lineTo(i * L * 0.12 + L * 0.1, L * 0.28); } ctx.stroke();
  }
  ctx.restore();
}

function dessinerFourmi(f) {
  const L = f.taille;
  const z = cam.z;
  ctx.save();
  ctx.translate(f.x, f.y);
  let cap = f.cap;
  if (f.mode === 'surface' && f.etat !== 'vol_nuptial') {
    if (f.etat === 'recolte') { cap = f.grimpe > 0 ? -Math.PI / 2 : cap; ctx.translate(0, -f.grimpe * 40); }
    else cap = f.dir > 0 ? penteSurface(f.x) : Math.PI + penteSurface(f.x);
  }
  ctx.rotate(cap);
  const callow = f.callow > 0 ? clamp(f.callow / 2, 0, 1) : 0;
  const base = f.caste === 'reine' ? '#1a120d' : f.caste === 'male' ? '#26201c' : '#2b1b12';
  const corps = callow ? `rgb(${lerp(43, 170, callow)},${lerp(27, 130, callow)},${lerp(18, 90, callow)})` : base;
  const marche = f.etat === 'deplacement' || f.etat === 'exploration' || f.etat === 'suit_piste' || f.etat === 'cherche';

  if (z >= 2.6) {
    // pattes : trois paires, alternance de tripode
    ctx.strokeStyle = corps; ctx.lineWidth = L * 0.045; ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < 3; i++) {
      const ax = L * (0.12 - i * 0.11);
      for (const s of [-1, 1]) {
        const ph = marche ? Math.sin(f.phase + (i + (s > 0 ? 1 : 0)) * Math.PI) * 0.18 : 0;
        const kx = ax + L * (0.14 - i * 0.14) + ph * L, ky = s * L * 0.22;
        const px = kx + L * (0.06 - i * 0.1) + ph * L * 0.6, py = s * L * 0.36;
        ctx.moveTo(ax, s * L * 0.05); ctx.lineTo(kx, ky); ctx.lineTo(px, py);
      }
    }
    // antennes coudées
    ctx.moveTo(L * 0.4, -L * 0.03); ctx.lineTo(L * 0.52, -L * 0.14); ctx.lineTo(L * 0.66, -L * 0.1);
    ctx.moveTo(L * 0.4, L * 0.03); ctx.lineTo(L * 0.52, L * 0.14); ctx.lineTo(L * 0.66, L * 0.1);
    ctx.stroke();
  }
  ctx.fillStyle = corps;
  const gasterL = f.caste === 'reine' || f.caste === 'gyne' ? 0.3 : 0.24, gasterW = f.caste === 'reine' || f.caste === 'gyne' ? 0.2 : 0.15;
  // gastre
  ctx.beginPath(); ctx.ellipse(-L * 0.28, 0, L * gasterL, L * gasterW, 0, 0, Math.PI * 2); ctx.fill();
  // pétiole
  ctx.beginPath(); ctx.ellipse(-L * 0.04, 0, L * 0.05, L * 0.04, 0, 0, Math.PI * 2); ctx.fill();
  // thorax
  ctx.beginPath(); ctx.ellipse(L * 0.12, 0, L * 0.16, L * (f.caste === 'reine' || f.caste === 'gyne' ? 0.12 : 0.085), 0, 0, Math.PI * 2); ctx.fill();
  // tête
  ctx.beginPath(); ctx.ellipse(L * 0.36, 0, L * 0.13, L * 0.11, 0, 0, Math.PI * 2); ctx.fill();
  if (z >= 3.5) {
    // reflets du gastre : les segments
    ctx.strokeStyle = 'rgba(255,255,255,.14)'; ctx.lineWidth = L * 0.02;
    ctx.beginPath(); for (let i = 1; i <= 3; i++) { const x = -L * 0.28 + L * gasterL * (0.6 - i * 0.35); ctx.moveTo(x, -L * gasterW * 0.9); ctx.lineTo(x, L * gasterW * 0.9); } ctx.stroke();
    // mandibules
    ctx.strokeStyle = corps; ctx.lineWidth = L * 0.035;
    ctx.beginPath(); ctx.moveTo(L * 0.46, -L * 0.05); ctx.lineTo(L * 0.55, -L * 0.02); ctx.moveTo(L * 0.46, L * 0.05); ctx.lineTo(L * 0.55, L * 0.02); ctx.stroke();
  }
  if (f.caste === 'gyne' || f.caste === 'male') {
    ctx.fillStyle = 'rgba(210,225,240,.42)';
    ctx.beginPath(); ctx.ellipse(-L * 0.25, -L * 0.1, L * 0.5, L * 0.12, -0.15, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(-L * 0.25, L * 0.1, L * 0.5, L * 0.12, 0.15, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

/* ======================================================================
   Interface : panneaux, fiche, commandes
   ====================================================================== */

const $ = (id) => document.getElementById(id);

const ETATS = {
  repos: ['Au repos', "Une ouvrière est inactive une bonne moitié du temps. Ce n'est pas de la paresse : la colonie garde une réserve de main-d'œuvre prête à réagir à une découverte, un éboulement ou une intrusion."],
  hiberne: ['En diapause hivernale', "Le sol est trop froid. Serrée contre ses congénères au fond du nid, elle vit au ralenti sur ses réserves. Aucune sortie, aucune ponte jusqu'au printemps."],
  attente: ["En attente du vol nuptial", "Nourrie par les ouvrières, elle attend une fin d'après-midi chaude et humide d'été pour s'envoler et s'accoupler en plein ciel."],
  toilettage: ['Toilettage', "Elle nettoie ses antennes avec le peigne de ses pattes avant. Des antennes propres, c'est un odorat fiable : presque toute la vie sociale des fourmis passe par des odeurs."],
  lecheOeufs: ['Lèche les œufs', "Les nourrices lèchent sans cesse les œufs et les regroupent en paquets. Leur salive a un effet antiseptique : sans ce soin, les moisissures les envahiraient."],
  sollicite: ['Sollicite de la nourriture', "Elle tapote les antennes des congénères qu'elle croise : c'est la demande rituelle de trophallaxie. Une fourmi bien nourrie lui régurgitera une goutte de son jabot social."],
  offre: ['Offre de la nourriture', "Son jabot est plein. Elle propose une goutte régurgitée à qui la sollicite : c'est ainsi que la nourriture des fourrageuses parvient jusqu'aux larves et à la reine, de bouche en bouche."],
  decharge: ['Décharge sa récolte', "Rentrée avec le jabot plein, elle cherche une ouvrière d'intérieur à qui transférer sa récolte, pour repartir au plus vite. Plus la file d'attente est longue, moins la source vaut la peine."],
  attend: ['Trophallaxie (donne)', "Elle régurgite une goutte de son jabot social pour une congénère. Le jabot est un estomac collectif : ce qu'il contient appartient à la colonie."],
  trophallaxie: ['Trophallaxie', "Bouche à bouche, l'une régurgite une goutte de nourriture liquide, l'autre l'absorbe. Ces échanges répartissent la nourriture dans toute la colonie et diffusent aussi l'odeur commune du nid."],
  creuse: ['Creuse', "Avec ses mandibules, elle détache une boulette de terre qu'elle ira déposer dehors. Le nid grandit avec la colonie : plus d'ouvrières, plus de chambres."],
  nourritLarve: ['Nourrit une larve', "Elle régurgite une goutte sucrée pour une larve. Les larves sont les seules à digérer les solides : elles sont, en quelque sorte, l'estomac de la colonie."],
  nourritReine: ['Nourrit la reine', "La reine ne se nourrit plus seule : ses ouvrières l'alimentent par trophallaxie. Bien nourrie en protéines, elle pond davantage."],
  aideEclosion: ["Aide à l'éclosion", "Chez Lasius, la nymphe est enfermée dans un cocon de soie. Les ouvrières l'ouvrent avec leurs mandibules pour en sortir la jeune adulte, encore pâle et molle."],
  pond: ['Pond un œuf', "La reine dépose un œuf minuscule, fécondé avec le sperme stocké depuis son unique vol nuptial. Les nourrices l'emporteront au paquet d'œufs."],
  exploration: ['Explore', "Sans piste à suivre, elle part au hasard, en zigzag, la tête basse. Elle garde toujours le cap du nid grâce au soleil et à son compteur de pas."],
  suit_piste: ['Suit une piste', "Elle suit une piste de phéromone déposée par des congénères revenues chargées. Plus la source est bonne, plus la piste est renforcée."],
  cherche: ['Cherche autour', "La piste s'estompe ici : la source doit être proche. Elle fouille les alentours."],
  recolte: ['Récolte', "Elle se remplit le jabot. Sur la plante, elle caresse les pucerons avec ses antennes pour qu'ils libèrent une goutte de miellat : les fourmis « traient » leurs pucerons et les protègent des coccinelles."],
  deplacement: ['Se déplace', "Elle se dirige vers son prochain objectif en suivant les galeries, guidée par l'odeur et le toucher : sous terre, ses yeux ne lui servent à rien."],
  chercheFront: ['Cherche un front de taille', "Elle rejoint un endroit où le plan du nid prévoit d'agrandir une galerie ou une chambre."],
  chercheFondation: ['Creuse sa loge', "La jeune reine creuse seule un puits et une petite loge, à quelques centimètres sous la surface."],
  fondation_marche: ['Cherche un site', "Ailes arrachées, la reine cherche un sol meuble où creuser sa loge de fondation."],
  vol_nuptial: ['Vol nuptial', "Elle s'envole pour s'accoupler en plein ciel. Une gyne s'accouple une fois pour toute sa vie ; un mâle meurt dans les jours qui suivent."],
};

const SAVIEZ = {
  nourrice: [
    "Les jeunes ouvrières restent au nid : leurs glandes nourricières sont actives et leur cuticule encore tendre. C'est le polyéthisme d'âge.",
    "Une nourrice déplace le couvain plusieurs fois par jour pour le garder autour de 25 °C : près de la surface l'après-midi, plus bas la nuit.",
    "Les larves de Lasius niger tissent un cocon avant de se nymphoser : ces « œufs de fourmi » vendus pour les poissons sont en fait des cocons.",
  ],
  entretien: [
    "Les ouvrières emportent les mortes loin du nid : c'est la nécrophorèse, déclenchée par l'odeur d'acide oléique des cadavres.",
    "Une fourmilière de Lasius niger peut déplacer plusieurs kilos de terre par an, une boulette à la fois.",
    "Le nid est un thermostat : la butte capte le soleil et les chambres profondes gardent une température stable.",
  ],
  fourrageuse: [
    "Sortir est réservé aux ouvrières les plus âgées : c'est la tâche la plus dangereuse, et la colonie perd ainsi le moins d'espérance de vie.",
    "Une fourrageuse retrouve le nid par intégration de trajet : elle compte ses pas et mesure l'angle du soleil.",
    "Lasius niger élève des pucerons comme du bétail : elle les traie pour leur miellat, les défend, et en emporte parfois dans le nid pour l'hiver.",
    "La piste de phéromone s'évapore en moins d'une heure : une source épuisée cesse d'être visitée d'elle-même.",
  ],
  reine: [
    "Une reine Lasius niger a vécu 28 ans et 9 mois en captivité : un record chez les insectes.",
    "Elle ne s'accouple qu'une fois, lors de son vol nuptial, et garde le sperme toute sa vie dans sa spermathèque.",
    "Un œuf fécondé donne une femelle (ouvrière ou reine) ; un œuf non fécondé donne un mâle. C'est l'haplodiploïdie.",
  ],
  sexue: [
    "Les mâles n'ont pas de père : ils naissent d'œufs non fécondés et n'ont qu'un jeu de chromosomes.",
    "Le vol nuptial est synchronisé sur des kilomètres : toutes les colonies d'une région lâchent leurs sexués le même après-midi orageux.",
  ],
};

const TACHES = { nourrice: 'Nourrice', entretien: "Ouvrière d'entretien", fourrageuse: 'Fourrageuse', reine: 'Reine', sexue: 'Sexué ailé' };

function nomFourmi(f) {
  if (f.caste === 'reine') return 'La reine';
  if (f.caste === 'gyne') return `Gyne nº ${f.id}`;
  if (f.caste === 'male') return `Mâle nº ${f.id}`;
  return `Ouvrière nº ${f.id}`;
}
function ouSeTrouve(f) {
  if (f.mode === 'surface') return f.etat === 'recolte' && f.source ? (f.source.type === 'miellat' ? 'sur la plante' : 'sur une proie') : `en surface, à ${fmt(Math.abs(f.x - S.xEntree) / 10, 0)} cm de l'entrée`;
  const ch = chambreDe(f);
  const prof = fmt((f.y - Y_SOL) / 10, 1);
  if (ch >= 0) return `${ROLES[S.chambres[ch].role].toLowerCase()}, ${prof} cm`;
  return `galerie, ${prof} cm`;
}
function ageTexte(jours) {
  if (jours < 1) return `${fmt(jours * 24, 0)} h`;
  if (jours < 365) return `${fmt(jours, 0)} j`;
  const a = Math.floor(jours / 365), m = Math.floor((jours % 365) / 30);
  return `${a} an${a > 1 ? 's' : ''}${m ? ` ${m} mois` : ''}`;
}

let journalAffiche = -1, selectionAffichee = null, indexSaviez = 0;

function rafraichirHud() {
  if (!S) return;
  const d = dateSim();
  $('valDate').textContent = `${d.jour} ${MOIS[d.mois]} · an ${d.an}`;
  const hh = Math.floor(d.heure), mm = Math.floor((d.heure - hh) * 60);
  $('valHeure').textContent = `${hh} h ${String(mm).padStart(2, '0')}`;
  const lum = luminosite(d.doy, d.heure);
  $('valSaison').textContent = `${nomSaison(d.doy)} · ${lum > 0.5 ? 'jour' : lum > 0.1 ? 'crépuscule' : 'nuit'}`;
  $('valTAir').textContent = `${fmt(S.tAir, 1)} °C`;
  $('valTNid').textContent = `${fmt(S.tNid, 1)} °C`;
  $('valPhase').textContent = S.hiver ? 'Diapause hivernale' : S.claustration ? 'Fondation claustrale' : S.activite > 0.5 ? 'Activité normale' : S.activite > 0.1 ? 'Sorties réduites (froid ou nuit)' : 'Repli au nid';

  const ouv = S.fourmis.filter((f) => f.caste === 'ouvriere');
  const n = { nourrice: 0, entretien: 0, fourrageuse: 0 };
  for (const f of ouv) n[f.tache] = (n[f.tache] || 0) + 1;
  $('valOuvrieres').textContent = fmtEntier(ouv.length);
  $('valNourrices').textContent = fmtEntier(n.nourrice);
  $('valEntretien').textContent = fmtEntier(n.entretien);
  $('valFourrageuses').textContent = fmtEntier(n.fourrageuse);
  const tot = Math.max(1, ouv.length);
  $('segNourrices').style.width = `${n.nourrice / tot * 100}%`;
  $('segEntretien').style.width = `${n.entretien / tot * 100}%`;
  $('segFourrageuses').style.width = `${n.fourrageuse / tot * 100}%`;
  const c = { oeuf: 0, larve: 0, nymphe: 0 };
  for (const b of S.couvain) c[b.stade]++;
  $('valOeufs').textContent = fmtEntier(c.oeuf);
  $('valLarves').textContent = fmtEntier(c.larve);
  $('valNymphes').textContent = fmtEntier(c.nymphe);
  const sex = S.fourmis.filter((f) => f.tache === 'sexue').length;
  $('ligneSexues').hidden = sex === 0;
  $('valSexues').textContent = fmtEntier(sex);
  $('valReine').textContent = S.reine ? `présente · ${ageTexte(S.reine.age)}` : 'morte';
  $('valReine').classList.toggle('alerte', !S.reine);

  const nid = S.fourmis.filter((f) => f.caste === 'ouvriere');
  const jab = nid.length ? nid.reduce((a, f) => a + f.jabot, 0) / nid.length / CAP_JABOT : 0;
  $('valSucre').textContent = `${fmt(jab * 100, 0)} %`;
  $('barreSucre').style.width = `${jab * 100}%`;
  const proies = S.objets.filter((o) => o.type === 'proie' && o.mode === 'nid').reduce((a, o) => a + o.masse, 0);
  $('valProies').textContent = `${fmt(proies, 1)} mg`;
  const pisteMax = Math.max(...S.pheromone);
  $('valPiste').textContent = pisteMax > 0.8 ? 'active' : pisteMax > 0.2 ? 'faible' : 'aucune';
  $('valEclosions').textContent = fmtEntier(S.compteurs.eclosions);
  $('valMorts').textContent = fmtEntier(S.compteurs.morts);

  if (journalAffiche !== S.journalVersion) {
    journalAffiche = S.journalVersion;
    const ul = $('journal');
    ul.innerHTML = '';
    for (const e of S.journal.slice().reverse().slice(0, 30)) {
      const li = document.createElement('li');
      const time = document.createElement('time'); time.textContent = e.date;
      const span = document.createElement('span'); span.textContent = e.texte;
      li.append(time, span); ul.appendChild(li);
    }
  }
  rafraichirFiche();
}

function rafraichirFiche() {
  const f = ui.selection ? S.fourmis.find((a) => a.id === ui.selection) : null;
  const vide = $('ficheVide'), contenu = $('ficheContenu');
  if (!f) {
    if (ui.selection) { ui.selection = null; ui.suivre = false; }
    vide.hidden = false; contenu.hidden = true; selectionAffichee = null;
    return;
  }
  vide.hidden = true; contenu.hidden = false;
  if (selectionAffichee !== f.id) { selectionAffichee = f.id; indexSaviez = Math.floor(R() * 10); }
  $('fTitre').textContent = nomFourmi(f);
  const badges = [];
  badges.push(f.caste === 'reine' ? 'reine fécondée' : f.caste === 'gyne' ? 'future reine' : f.caste === 'male' ? 'mâle' : TACHES[f.tache]);
  if (f.tacheVoulue) badges.push(`bientôt ${TACHES[f.tacheVoulue].toLowerCase()}`);
  if (f.nanitique) badges.push('nanitique');
  if (f.callow > 0) badges.push('imago fraîche');
  if (f.caste === 'reine' && S.claustration) badges.push('claustrée');
  $('fBadges').innerHTML = badges.map((b) => `<span>${b}</span>`).join('');
  const et = ETATS[f.etat] || ETATS.repos;
  let etatLib = et[0], expl = et[1];
  if (f.etat === 'deplacement') {
    const ap = f.apres;
    const vers = { sortie: 'vers la sortie', entree: "vers l'entrée du nid", depose: f.porte ? (f.porte.type === 'couvain' ? 'transporte du couvain vers la chambre la mieux tempérée' : 'apporte une proie aux larves') : 'range quelque chose', deposeDeblai: 'va déposer sa boulette de terre dehors', deposeCadavre: 'emporte une morte au dépotoir', chercheFront: 'va agrandir le nid', prendCouvain: 'va chercher du couvain à déplacer', prendProie: 'va chercher un morceau de proie', prendCadavre: 'va chercher une morte à évacuer', nourritLarve: 'va nourrir une larve', nourritReine: 'va nourrir la reine', aideEclosion: 'va ouvrir un cocon', hiberne: 'descend hiverner', arriveSource: 'retourne à une source connue', trophallaxie: 'rejoint une congénère pour un échange', decharge: 'rentre décharger sa récolte', sollicite: 'va quémander à manger', offre: 'va distribuer sa récolte', chercheFondation: 'creuse sa loge' };
    if (ap && vers[ap]) etatLib = `Se déplace : ${vers[ap]}`;
    if (f.mode === 'surface' && f.depose) { etatLib = 'Rentre chargée, en marquant la piste'; expl = "Chargée, elle rentre en ligne droite et dépose une traînée de phéromone avec l'extrémité de son abdomen. D'autres fourrageuses suivront cette piste : c'est le recrutement de masse."; }
    else if (f.mode === 'surface' && f.porte && f.porte.type === 'deblai') { expl = "Elle emporte une boulette de terre pour la déposer dehors. L'accumulation forme la butte, ou le cratère, typique des nids de Lasius."; }
  }
  if (f.etat === 'creuse' && f.caste === 'reine') { etatLib = 'Creuse sa loge'; expl = ETATS.chercheFondation[1]; }
  if (f.caste === 'reine' && f.etat === 'repos' && S.claustration) { expl = "Enfermée dans sa loge, elle attend. Elle ne mangera pas avant l'éclosion de ses premières ouvrières et vit sur ses réserves : ses graisses et ses muscles alaires, désormais inutiles, qui se résorbent."; }
  $('fEtat').textContent = etatLib;
  $('fExplication').textContent = expl;

  const lignes = [];
  lignes.push(['Âge', f.caste === 'reine' ? `${ageTexte(f.age)} (fécondée depuis ${ageTexte(Math.max(0, f.age - 365))})` : ageTexte(f.age)]);
  if (f.caste === 'ouvriere') lignes.push(['Espérance de vie', `~${ageTexte(f.longevite)}`]);
  lignes.push(['Taille', `${fmt(f.taille, 1)} mm`]);
  const cap = f.caste === 'reine' ? CAP_REINE : CAP_JABOT;
  lignes.push(['Jabot social', `${fmt(f.jabot, 2)} µL · ${fmt(f.jabot / cap * 100, 0)} %`]);
  lignes.push(['Énergie', `${fmt(f.energie * 100, 0)} %${f.energie <= 0 ? ' · affamée' : ''}`]);
  if (f.caste === 'reine') {
    lignes.push(['Œufs pondus', fmtEntier(f.oeufs)]);
    lignes.push(['Protéines', `${fmt(f.prot * 100, 0)} %`]);
    if (S.claustration || S.scenario === 'fondation' && f.reserves > 0) lignes.push(['Réserves corporelles', `${fmt(f.reserves * 100, 0)} %`]);
    lignes.push(['Spermathèque', 'un seul accouplement, pour la vie']);
  }
  const porte = !f.porte ? 'rien' : f.porte.type === 'deblai' ? 'une boulette de terre' : f.porte.type === 'proie' ? `un morceau de proie (${fmt(f.porte.ref.masse, 1)} mg)` : f.porte.type === 'cadavre' ? 'une congénère morte' : f.porte.ref.stade === 'oeuf' ? 'un œuf' : f.porte.ref.stade === 'larve' ? 'une larve' : 'un cocon';
  lignes.push(['Porte', porte]);
  lignes.push(['Position', ouSeTrouve(f)]);
  if (f.tache === 'fourrageuse' || f.sorties > 0) { lignes.push(['Sorties', fmtEntier(f.sorties)]); lignes.push(['Nourriture rapportée', `${fmt(f.rapporte, 1)} µL`]); }
  $('fTable').innerHTML = lignes.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
  const faits = SAVIEZ[f.caste === 'reine' ? 'reine' : f.tache] || SAVIEZ.nourrice;
  $('fSaviez').textContent = faits[indexSaviez % faits.length];
  $('btnSuivre').textContent = ui.suivre ? '◉ Ne plus suivre' : '◎ Suivre à la caméra';
  $('btnSuivre').classList.toggle('actif', ui.suivre);
}

function selectionner(f) {
  ui.selection = f ? f.id : null;
  if (!f) ui.suivre = false;
  if (f && ui.astuce) { ui.astuce = false; $('astuce').hidden = true; }
  rafraichirFiche();
}

function fourmiSous(sx, sy) {
  const m = versMonde(sx, sy);
  const rayon = Math.max(2.5, 10 / cam.z);
  let best = null, bd = Infinity;
  for (const f of S.fourmis) {
    let fy = f.y;
    if (f.mode === 'surface' && f.etat === 'recolte') fy -= f.grimpe * 40;
    const d = Math.hypot(f.x - m.x, fy - m.y) - f.taille * 0.3 - (f.caste === 'reine' ? 1 : 0);
    if (d < rayon && d < bd) { bd = d; best = f; }
  }
  return best;
}

function brancherCommandes() {
  // vitesse
  document.querySelectorAll('[data-vitesse]').forEach((b) => b.addEventListener('click', () => reglerVitesse(+b.dataset.vitesse)));
  document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => reglerMode(b.dataset.mode)));
  $('chkPheromone').addEventListener('change', (e) => { ui.pheromone = e.target.checked; });
  $('chkPlan').addEventListener('change', (e) => { ui.plan = e.target.checked; });
  $('chkTemp').addEventListener('change', (e) => { ui.temperature = e.target.checked; });
  $('btnZoomPlus').addEventListener('click', () => zoomer(1.5));
  $('btnZoomMoins').addEventListener('click', () => zoomer(1 / 1.5));
  $('btnRecentrer').addEventListener('click', () => { ui.suivre = false; cam.x = S.xEntree; cam.y = Y_SOL + 60; cam.z = 2.2; rafraichirFiche(); });
  $('btnSuivre').addEventListener('click', () => { ui.suivre = !ui.suivre; rafraichirFiche(); });
  $('btnFermerFiche').addEventListener('click', () => selectionner(null));
  $('btnReset').addEventListener('click', () => { if (confirm('Recommencer la simulation ? La colonie actuelle sera perdue.')) { localStorage.removeItem(CLE); nouvellePartie($('scenario').value); rafraichirHud(); } });
  $('scenario').addEventListener('change', (e) => { localStorage.removeItem(CLE); nouvellePartie(e.target.value); rafraichirHud(); });
  const aide = $('aide');
  const ouvrirAide = () => { aide.showModal(); };
  $('btnAide').addEventListener('click', ouvrirAide);
  aide.addEventListener('click', (e) => { if (e.target === aide) aide.close(); });

  // Clavier
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' || aide.open) return;
    if (e.code === 'Space') { e.preventDefault(); reglerVitesse(ui.vitesse === 0 ? (ui.derniereVitesse || 1) : 0); }
    else if (e.key === '1') reglerVitesse(1); else if (e.key === '2') reglerVitesse(2); else if (e.key === '3') reglerVitesse(3);
    else if (e.key === 'f' || e.key === 'F') { if (ui.selection) { ui.suivre = !ui.suivre; rafraichirFiche(); } }
    else if (e.key === 'Escape') { if (ui.mode !== 'observer') reglerMode('observer'); else selectionner(null); }
    else if (e.key === '+' || e.key === '=') zoomer(1.4); else if (e.key === '-') zoomer(1 / 1.4);
    else if (e.key === 'h' || e.key === 'H' || e.key === '?') ouvrirAide();
  });

  // Souris et toucher : glisser pour déplacer, molette et pincement pour zoomer, clic pour sélectionner
  const pointeurs = new Map();
  let glisse = null, pinceDist = 0;
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointeurs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointeurs.size === 1) glisse = { x: e.clientX, y: e.clientY, cx: cam.x, cy: cam.y, bouge: false };
    else if (pointeurs.size === 2) { const [a, b] = [...pointeurs.values()]; pinceDist = Math.hypot(a.x - b.x, a.y - b.y); glisse = null; }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!pointeurs.has(e.pointerId)) return;
    pointeurs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointeurs.size === 2) {
      const [a, b] = [...pointeurs.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinceDist > 0) zoomer(d / pinceDist, (a.x + b.x) / 2, (a.y + b.y) / 2);
      pinceDist = d;
      return;
    }
    if (!glisse) return;
    const dx = e.clientX - glisse.x, dy = e.clientY - glisse.y;
    if (!glisse.bouge && Math.hypot(dx, dy) > 5) { glisse.bouge = true; ui.suivre = false; }
    if (glisse.bouge) { cam.x = glisse.cx - dx / cam.z; cam.y = glisse.cy - dy / cam.z; }
  });
  const finPointeur = (e) => {
    if (!pointeurs.has(e.pointerId)) return;
    pointeurs.delete(e.pointerId);
    if (glisse && !glisse.bouge && e.type === 'pointerup') clic(e);
    if (pointeurs.size === 0) glisse = null;
  };
  canvas.addEventListener('pointerup', finPointeur);
  canvas.addEventListener('pointercancel', finPointeur);
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); const r = canvas.getBoundingClientRect(); zoomer(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  canvas.addEventListener('dblclick', (e) => { const r = canvas.getBoundingClientRect(); zoomer(1.8, e.clientX - r.left, e.clientY - r.top); });
}

function clic(e) {
  const r = canvas.getBoundingClientRect();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (ui.mode !== 'observer') {
    const m = versMonde(sx, sy);
    if (m.x > 5 && m.x < LARG - 5 && m.y < hauteur(m.x) + 6 && Math.abs(m.x - S.xEntree) > 10) {
      if (ui.mode === 'sucre') { creerSource('sucre', m.x, { reserve: 12, max: 12, nom: 'une goutte de sucre' }); journal(`Une goutte d'eau sucrée est déposée à ${fmt(Math.abs(m.x - S.xEntree) / 10, 0)} cm de l'entrée.`); }
      else { creerSource('insecte', m.x, { masse: 10, nom: 'un insecte mort déposé là' }); journal(`Un insecte mort est déposé à ${fmt(Math.abs(m.x - S.xEntree) / 10, 0)} cm de l'entrée. Les larves ont besoin de ces protéines.`); }
      return;
    }
  }
  const f = fourmiSous(sx, sy);
  selectionner(f);
}

function zoomer(facteur, sx, sy) {
  const w = canvas.width / dpr, h = canvas.height / dpr;
  if (sx === undefined) { sx = w / 2; sy = h / 2; }
  const avant = versMonde(sx, sy);
  cam.z = clamp(cam.z * facteur, cam.zMin, cam.zMax);
  const apres = versMonde(sx, sy);
  cam.x += avant.x - apres.x; cam.y += avant.y - apres.y;
}

function reglerVitesse(v) {
  if (v !== 0) ui.derniereVitesse = v;
  ui.vitesse = v;
  document.querySelectorAll('[data-vitesse]').forEach((b) => b.classList.toggle('actif', +b.dataset.vitesse === v));
  $('etatVitesse').textContent = v === 0 ? 'En pause' : v === 1 ? '1 s = 5 min' : v === 2 ? '1 s = 40 min' : '1 s ≈ 2 h 40';
}
function reglerMode(m) {
  ui.mode = m;
  document.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('actif', b.dataset.mode === m));
  canvas.classList.toggle('depot', m !== 'observer');
}

function redimensionner() {
  dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  canvas.style.width = `${r.width}px`; canvas.style.height = `${r.height}px`;
}

/* ======================================================================
   Sauvegarde
   ====================================================================== */

function rle(arr) {
  const out = [];
  let cur = arr[0], n = 0;
  for (const v of arr) { if (v === cur) n++; else { out.push(`${n}:${cur}`); cur = v; n = 1; } }
  out.push(`${n}:${cur}`);
  return out.join(',');
}
function unrle(s, taille) {
  const arr = new Uint8Array(taille);
  let i = 0;
  for (const part of s.split(',')) { const [n, v] = part.split(':').map(Number); arr.fill(v, i, i + n); i += n; }
  return arr;
}

const CHAMPS_FOURMI = ['id', 'caste', 'tache', 'x', 'y', 'cap', 'taille', 'age', 'longevite', 'jabot', 'prot', 'energie', 'faim', 'mode', 'chambreMaisonI', 'sorties', 'rapporte', 'nanitique', 'callow', 'souvenir', 'ne', 'oeufs', 'reserves', 'oeufsEnAttente', 'volPret', 'tacheVoulue'];

function sauver() {
  if (!S) return;
  sauvegardeEnAttente = false;
  try {
    const s = {
      v: 1, scenario: S.scenario, graine: S.graine, minutes: S.minutes, doy0: S.doy0, an0: S.an0,
      grille: rle(S.grille), hauteurSol: Array.from(S.hauteurSol, (v) => Math.round(v * 100) / 100),
      pheromone: Array.from(S.pheromone, (v) => Math.round(v * 100) / 100),
      fondation: S.fondation ? rle(S.fondation) : null, fondationScellee: S.fondationScellee, claustration: S.claustration,
      hiver: S.hiver, compteurs: S.compteurs, journal: S.journal, journalVersion: S.journalVersion, sexuesProduits: S.sexuesProduits, sexuesAnnee: S.sexuesAnnee,
      alertes: S.alertes, prochainInsecte: S.prochainInsecte, derniereHeure: S.derniereHeure, prochainId,
      reineId: S.reine ? S.reine.id : null,
      fourmis: S.fourmis.map((f) => { const o = {}; for (const k of CHAMPS_FOURMI) if (f[k] !== undefined && f[k] !== null && f[k] !== false && f[k] !== 0) o[k] = f[k]; if (f.chambreMaison) o.chambreMaisonI = f.chambreMaison.i; return o; }),
      couvain: S.couvain.map((b) => ({ id: b.id, stade: b.stade, x: b.porteur ? b.porteur.x : b.x, y: b.porteur ? b.porteur.y : b.y, dev: b.dev, sucre: b.sucre, prot: b.prot, chambre: b.porteur ? S.cellChambre[cellDe(b.porteur.x, b.porteur.y)] : b.chambre, sexue: b.sexue, jeune: b.jeune, pret: b.pret })),
      objets: S.objets.map((o) => ({ id: o.id, type: o.type, x: o.porteur ? o.porteur.x : o.x, y: o.porteur ? o.porteur.y : o.y, mode: o.porteur ? o.porteur.mode : o.mode, chambre: o.porteur ? S.cellChambre[cellDe(o.porteur.x, o.porteur.y)] : o.chambre, masse: o.masse, age: o.age, taille: o.taille })),
      sources: S.sources.map((s) => ({ id: s.id, type: s.type, x: s.x, reserve: s.reserve, max: s.max, masse: s.masse, decouverte: s.decouverte, age: s.age, nom: s.nom })),
      cam: { x: cam.x, y: cam.y, z: cam.z }, ui: { vitesse: ui.vitesse, pheromone: ui.pheromone, plan: ui.plan, temperature: ui.temperature },
    };
    localStorage.setItem(CLE, JSON.stringify(s));
  } catch (e) { /* stockage indisponible : on continue sans */ }
}

function charger() {
  let s;
  try { s = JSON.parse(localStorage.getItem(CLE)); } catch (e) { return false; }
  if (!s || s.v !== 1) return false;
  try {
    const monde = genererMonde(s.graine);
    S = Object.assign(monde, {
      scenario: s.scenario, minutes: s.minutes, doy0: s.doy0, an0: s.an0, fourmis: [], couvain: [], objets: [], sources: [], journal: s.journal || [],
      journalVersion: s.journalVersion || 1, nbOuvrieres: 0, hiver: !!s.hiver, fondation: s.fondation ? unrle(s.fondation, GW * GH) : null,
      fondationScellee: !!s.fondationScellee, claustration: !!s.claustration, reine: null, chambreCouvain: null, chambreNymphes: null, vol: null,
      compteurs: s.compteurs || { eclosions: 0, morts: 0, oeufs: 0 }, derniereHeure: s.derniereHeure || 0, nbFront: 0, tAir: 15, tNid: 15, activite: 0,
      prochainInsecte: s.prochainInsecte || 30, sexuesProduits: !!s.sexuesProduits, sexuesAnnee: s.sexuesAnnee, alertes: s.alertes || {}, nbCreuseurs: 0,
    });
    S.grille = unrle(s.grille, GW * GH);
    S.hauteurSol = Float32Array.from(s.hauteurSol);
    S.pheromone = Float32Array.from(s.pheromone);
    // rangée d'entrée et chambres creusées
    let r = 0; while (r < GH && S.grille[idx(S.colEntree, r)] !== GALERIE) r++;
    S.rowEntree = Math.min(r, SOL);
    for (const ch of S.chambres) {
      ch.creusees = 0;
      for (let rr = SOL; rr < GH; rr++) for (let cc = 0; cc < GW; cc++) {
        const dx = (cc * MM + MM / 2 - ch.cx) / ch.rx, dy = (rr * MM + MM / 2 - ch.cy) / ch.ry;
        if (dx * dx + dy * dy <= 1 && S.grille[idx(cc, rr)] === GALERIE) ch.creusees++;
      }
      ch.pret = ch.creusees / ch.total >= 0.75;
    }
    prochainId = 1;
    for (const o of s.fourmis) {
      const f = creerFourmi(o.caste, { x: o.x, y: o.y, age: o.age || 0, longevite: o.longevite, jabot: o.jabot || 0, prot: o.prot || 0, energie: o.energie || 0, nanitique: !!o.nanitique, callow: o.callow || 0, taille: o.taille, ne: o.ne || 0, reserves: o.reserves || 0, tache: o.tache });
      f.id = o.id; f.cap = o.cap || 0; f.faim = o.faim || 0; f.mode = o.mode || 'nid'; f.sorties = o.sorties || 0; f.rapporte = o.rapporte || 0; f.souvenir = o.souvenir ?? null;
      f.oeufs = o.oeufs || 0; f.oeufsEnAttente = o.oeufsEnAttente || 0; f.volPret = !!o.volPret; f.tacheVoulue = o.tacheVoulue || null;
      if (o.chambreMaisonI !== undefined) f.chambreMaison = S.chambres[o.chambreMaisonI];
      if (f.mode === 'surface') { f.y = hauteur(f.x) - 1; }
      if (f.id === s.reineId) S.reine = f;
      if (f.caste === 'ouvriere') S.nbOuvrieres++;
    }
    for (const o of s.couvain) { const b = creerCouvain(o.stade, o.x, o.y, S.chambres[o.chambre], { dev: o.dev, sucre: o.sucre, prot: o.prot, sexue: o.sexue }); b.id = o.id; b.jeune = o.jeune || 0; b.pret = !!o.pret; }
    for (const o of s.objets) { const ob = creerObjet(o.type, o.x, o.y, { mode: o.mode, chambre: o.chambre, masse: o.masse, taille: o.taille }); ob.id = o.id; ob.age = o.age || 0; }
    for (const o of s.sources) { const src = creerSource(o.type, o.x, { reserve: o.reserve, max: o.max, masse: o.masse, nom: o.nom }); src.id = o.id; src.decouverte = !!o.decouverte; src.age = o.age || 0; }
    prochainId = Math.max(prochainId, s.prochainId || 1);
    if (s.cam) { cam.x = s.cam.x; cam.y = s.cam.y; cam.z = s.cam.z; }
    if (s.ui) { ui.pheromone = s.ui.pheromone !== false; ui.plan = !!s.ui.plan; ui.temperature = !!s.ui.temperature; }
    $('chkPheromone').checked = ui.pheromone; $('chkPlan').checked = ui.plan; $('chkTemp').checked = ui.temperature;
    $('scenario').value = S.scenario;
    repartirTaches();
    choisirChambresCouvain();
    champsSales = true;
    recalculerChamps();
    peindreTerrain();
    for (const f of S.fourmis) if (f.mode === 'nid') assainir(f);
    tickHeure();
    return true;
  } catch (e) {
    S = null;
    return false;
  }
}

/* ======================================================================
   Démarrage
   ====================================================================== */

function init() {
  redimensionner();
  new ResizeObserver(redimensionner).observe(canvas.parentElement);
  brancherCommandes();
  if (!charger()) nouvellePartie($('scenario').value);
  reglerVitesse(1);
  reglerMode('observer');
  tickHeure();
  rafraichirHud();
  window.addEventListener('pagehide', sauver);
  requestAnimationFrame(boucle);
}

// Petit accès pour les tests automatisés
window.fourmiliere = {
  get etat() { return S; },
  get cam() { return cam; },
  ecran(f) { const r = canvas.getBoundingClientRect(); const p = versEcran(f.x, f.y); return { x: p.x + r.left, y: p.y + r.top }; },
  selectionner(id) { selectionner(S.fourmis.find((f) => f.id === id) || null); rafraichirHud(); },
  vitesse(v) { reglerVitesse(v); },
};

init();
