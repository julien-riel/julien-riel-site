/* Simulateur de pelle mécanique en 3D, vue depuis la cabine.
   Un terrain en carte de hauteurs, une mini-pelle de la classe 0–2 t,
   deux manettes ISO. Tout est en mètres, radians et secondes. */

import * as THREE from 'three';

/* ---------- Utilitaires ---------- */

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const gauss = (d2, sigma) => Math.exp(-d2 / (2 * sigma * sigma));
const lisse = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const fmt = (v, dec = 2) => v.toFixed(dec).replace('.', ',');
const hash = (k) => { const s = Math.sin(k * 12.9898 + 78.233) * 43758.5453; return s - Math.floor(s); };

/* ---------- Constantes : la machine ---------- */

const M = {
  chenilleL: 1.70, chenilleH: 0.38, voie: 0.45,
  fleche: 1.55, balancier: 0.92, godet: 0.42, godetLarg: 0.42,
  capacite: 0.045,                       // m³ foisonnés : un godet de 45 L
  tourelle: new THREE.Vector3(-0.05, 0.52, 0),
  pivot: new THREE.Vector3(0.42, 0.30, 0.22),   // pied de flèche, à droite de la cabine
  oeil: new THREE.Vector3(-0.18, 1.24, -0.24),  // l'opérateur, assis à gauche, le buste penché vers l'avant
};
const LIM = { fleche: [-0.75, 1.15], balancier: [-2.75, -0.30], godet: [-2.75, 0.20] };
const VIT = { fleche: 0.70, balancier: 1.00, godet: 1.40, rotation: 0.95, chenille: 0.9, virage: 0.7 };

/* ---------- Constantes : le terrain ---------- */

const N = 128, DX = 0.15;
const L = (N - 1) * DX;                  // 19,05 m de côté
const DEMI = L / 2;
const ROC = -2.2;
const FOISONNEMENT = 1.25;
const PENTE_MEUBLE = Math.tan(34 * Math.PI / 180) * DX;
const G = 9.81;
const LS_ETAT = 'jr-pelle3d-etat-v1';
const LS_RECORDS = 'jr-pelle3d-records-v1';

const CHANTIERS = {
  bac: {
    nom: 'Bac à sable',
    consigne: 'Aucun objectif : creusez, pivotez, empilez. Un tas de terre meuble vous attend devant, un peu à droite.',
    depart: { x: -1.5, z: 0, cap: 0 },
    sol: () => 0,
    meuble: (x, z) => 1.0 * gauss((x - 3.2) ** 2 + (z - 1.2) ** 2, 0.85),
    zone: null,
  },
  fosse: {
    nom: 'Fosse',
    consigne: 'Creusez le carré balisé jusqu\'au <b>fond marqué</b>, à 55 cm. Le déblai va où vous voulez, sauf dans la fosse.',
    depart: { x: -0.6, z: 0, cap: 0 },
    sol: () => 0,
    meuble: () => 0,
    zone: { x1: 2.3, x2: 3.3, z1: -0.5, z2: 0.5 }, cible: -0.55, mode: 'creuser', tol: 0.05,
  },
  remblai: {
    nom: 'Remblayage',
    consigne: 'Une fosse est ouverte devant vous : comblez-la jusqu\'au <b>niveau du terrain</b> avec le tas sur votre gauche.',
    depart: { x: -0.6, z: 0, cap: 0 },
    sol: (x, z) => -0.55 * lisse((x - 2.3 + 0.25) / 0.25) * lisse((3.3 + 0.25 - x) / 0.25) * lisse((z + 0.5 + 0.25) / 0.25) * lisse((0.5 + 0.25 - z) / 0.25),
    meuble: (x, z) => 0.7 * gauss((x - 0.9) ** 2 + (z + 2.6) ** 2, 0.8),
    fosse: { x1: 2.0, x2: 3.6, z1: -0.8, z2: 0.8 },
    zone: { x1: 2.3, x2: 3.3, z1: -0.5, z2: 0.5 }, cible: -0.08, mode: 'remplir', tol: 0.10,
  },
};

/* ---------- État ---------- */

const sol = new Float32Array(N * N);
const meuble = new Float32Array(N * N);
const vierge = new Uint8Array(N * N);
const bruit = new Float32Array(N * N);
for (let k = 0; k < N * N; k++) bruit[k] = (hash(k) - 0.5) * 0.09;

let chantier = 'bac';
const etat = {
  x: 0, z: 0, y: 0, cap: 0, tangage: 0, roulis: 0,
  rotation: 0, fleche: 0.45, balancier: -1.6, godet: -1.4,
  charge: 0, volume: 0, chrono: 0, chronoActif: false, termine: false,
  regardLacet: 0, regardTangage: -0.30,
};
const axes = { rotation: 0, balancier: 0, fleche: 0, godet: 0, chenille: 0, virage: 0 };
const clavier = {};
let records = {};
let terrainSale = true;
let sauvegardeEnAttente = false;

/* ---------- Terrain : données ---------- */

const cellX = (j) => j * DX - DEMI;
const cellZ = (i) => i * DX - DEMI;
const idxJ = (x) => clamp(Math.round((x + DEMI) / DX), 0, N - 1);
const idxI = (z) => clamp(Math.round((z + DEMI) / DX), 0, N - 1);

function hauteurEn(x, z) {
  const fj = clamp((x + DEMI) / DX, 0, N - 1), fi = clamp((z + DEMI) / DX, 0, N - 1);
  const j = Math.floor(fj), i = Math.floor(fi);
  const j1 = Math.min(j + 1, N - 1), i1 = Math.min(i + 1, N - 1);
  const tx = fj - j, tz = fi - i;
  const a = lerp(sol[i * N + j], sol[i * N + j1], tx);
  const b = lerp(sol[i1 * N + j], sol[i1 * N + j1], tx);
  return lerp(a, b, tz);
}

function genererTerrain(cle) {
  const c = CHANTIERS[cle];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const k = i * N + j, x = cellX(j), z = cellZ(i);
      const base = c.sol(x, z) + 0.03 * Math.sin(x * 0.9) * Math.cos(z * 0.7);
      const tas = c.meuble(x, z);
      sol[k] = base + tas;
      meuble[k] = tas;
      let v = tas < 0.004;
      if (c.fosse && x > c.fosse.x1 && x < c.fosse.x2 && z > c.fosse.z1 && z < c.fosse.z2) v = false;
      vierge[k] = v ? 1 : 0;
    }
  }
  terrainSale = true;
}

function dansZone(c, x, z) {
  const zn = c.zone;
  return x >= zn.x1 && x <= zn.x2 && z >= zn.z1 && z <= zn.z2;
}
function celluleOk(c, k) {
  if (c.mode === 'creuser') return sol[k] <= c.cible + c.tol;
  if (c.mode === 'remplir') return sol[k] >= c.cible - c.tol;
  return Math.abs(sol[k] - c.cible) <= c.tol;
}

/* Coupe la cellule k jusqu'à y ; renvoie le volume foisonné enlevé. */
function couperCellule(k, y, maxVol) {
  const plancher = Math.max(y, ROC);
  if (sol[k] <= plancher || maxVol <= 0) return 0;
  let d = sol[k] - plancher;
  const partMeuble = Math.min(d, meuble[k]);
  let vol = (partMeuble + (d - partMeuble) * FOISONNEMENT) * DX * DX;
  if (vol > maxVol) { d *= maxVol / vol; vol = maxVol; }
  sol[k] -= d;
  meuble[k] -= Math.min(d, meuble[k]);
  vierge[k] = 0;
  terrainSale = true;
  return vol;
}

/* Dépose un volume foisonné autour de (x, z). */
function deposer(x, z, vol) {
  const rayon = 0.24;
  const j0 = idxJ(x - rayon), j1 = idxJ(x + rayon), i0 = idxI(z - rayon), i1 = idxI(z + rayon);
  let total = 0;
  for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
    const w = 1 - Math.hypot(cellX(j) - x, cellZ(i) - z) / rayon;
    if (w > 0) total += w;
  }
  if (total <= 0) return;
  for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
    const w = 1 - Math.hypot(cellX(j) - x, cellZ(i) - z) / rayon;
    if (w <= 0) continue;
    const k = i * N + j, dh = (vol * w / total) / (DX * DX);
    sol[k] += dh; meuble[k] += dh; vierge[k] = 0;
  }
  terrainSale = true;
}

/* Talus naturel : seule la terre meuble glisse, vers ses quatre voisines. */
let sens = 0;
function tasser() {
  sens = (sens + 1) & 3;
  const inv = sens & 1;
  if (sens < 2) {
    for (let i = 0; i < N; i++) {
      if (!inv) for (let j = 0; j < N - 1; j++) glisser(i * N + j, i * N + j + 1);
      else for (let j = N - 2; j >= 0; j--) glisser(i * N + j, i * N + j + 1);
    }
  } else {
    for (let j = 0; j < N; j++) {
      if (!inv) for (let i = 0; i < N - 1; i++) glisser(i * N + j, (i + 1) * N + j);
      else for (let i = N - 2; i >= 0; i--) glisser(i * N + j, (i + 1) * N + j);
    }
  }
}
function glisser(a, b) {
  const d = sol[a] - sol[b];
  if (d > PENTE_MEUBLE && meuble[a] > 0) {
    const q = Math.min((d - PENTE_MEUBLE) * 0.5, meuble[a]);
    sol[a] -= q; meuble[a] -= q; sol[b] += q; meuble[b] += q; vierge[b] = 0; terrainSale = true;
  } else if (-d > PENTE_MEUBLE && meuble[b] > 0) {
    const q = Math.min((-d - PENTE_MEUBLE) * 0.5, meuble[b]);
    sol[b] -= q; meuble[b] -= q; sol[a] += q; meuble[a] += q; vierge[a] = 0; terrainSale = true;
  }
}

/* ---------- Scène ---------- */

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9cc0e6);
scene.fog = new THREE.Fog(0x9cc0e6, 28, 90);

const camera = new THREE.PerspectiveCamera(72, 1, 0.05, 140);

scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x6b5a3a, 0.95));
const soleil = new THREE.DirectionalLight(0xffe2b8, 1.7);
soleil.castShadow = true;
soleil.shadow.mapSize.set(1024, 1024);
soleil.shadow.camera.left = -7; soleil.shadow.camera.right = 7;
soleil.shadow.camera.top = 7; soleil.shadow.camera.bottom = -7;
soleil.shadow.camera.near = 1; soleil.shadow.camera.far = 40;
soleil.shadow.bias = -0.0015;
scene.add(soleil, soleil.target);

/* Terrain : une grille dont on rafraîchit les hauteurs et les couleurs. */
const geoTerrain = new THREE.BufferGeometry();
{
  const pos = new Float32Array(N * N * 3), col = new Float32Array(N * N * 3), idx = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const k = i * N + j;
    pos[k * 3] = cellX(j); pos[k * 3 + 1] = 0; pos[k * 3 + 2] = cellZ(i);
  }
  for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) {
    const a = i * N + j, b = a + 1, c = a + N, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  geoTerrain.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geoTerrain.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geoTerrain.setIndex(idx);
}
const terrain = new THREE.Mesh(geoTerrain, new THREE.MeshLambertMaterial({ vertexColors: true }));
terrain.receiveShadow = true;
scene.add(terrain);

/* Au-delà du chantier : une plaine, des conifères. */
const plaine = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshLambertMaterial({ color: 0x6a8f3f }));
plaine.rotation.x = -Math.PI / 2;
plaine.position.y = -0.02;
plaine.receiveShadow = true;
scene.add(plaine);
{
  const nb = 70;
  const troncs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.12, 0.16, 1, 6), new THREE.MeshLambertMaterial({ color: 0x4b3a28 }), nb);
  const cimes = new THREE.InstancedMesh(new THREE.ConeGeometry(1, 1, 7), new THREE.MeshLambertMaterial({ color: 0x2f5a35 }), nb);
  const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
  for (let n = 0; n < nb; n++) {
    const ang = hash(n * 7 + 1) * Math.PI * 2, r = DEMI + 2 + hash(n * 7 + 2) * 22;
    const h = 3 + hash(n * 7 + 3) * 4, x = Math.cos(ang) * r, z = Math.sin(ang) * r;
    p.set(x, 0.5, z); s.set(1, 1, 1); m.compose(p, q, s); troncs.setMatrixAt(n, m);
    p.set(x, 1 + h / 2, z); s.set(h * 0.32, h, h * 0.32); m.compose(p, q, s); cimes.setMatrixAt(n, m);
  }
  troncs.castShadow = cimes.castShadow = true;
  scene.add(troncs, cimes);
}

/* Balisage de la zone de travail. */
const balisage = new THREE.Group();
scene.add(balisage);

const COULEURS = {
  herbe: new THREE.Color(0x6f9a3c), terre: new THREE.Color(0x6d4d31), meuble: new THREE.Color(0xb48a56),
  ok: new THREE.Color(0x55cf7c), ko: new THREE.Color(0xff5f4d),
};
const tmpC = new THREE.Color();
function rafraichirTerrain() {
  const pos = geoTerrain.attributes.position.array, col = geoTerrain.attributes.color.array;
  const c = CHANTIERS[chantier];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const k = i * N + j;
    pos[k * 3 + 1] = sol[k];
    const base = vierge[k] ? COULEURS.herbe : (meuble[k] > 0.004 ? COULEURS.meuble : COULEURS.terre);
    tmpC.copy(base).offsetHSL(0, 0, bruit[k]);
    if (c.zone && dansZone(c, cellX(j), cellZ(i))) tmpC.lerp(celluleOk(c, k) ? COULEURS.ok : COULEURS.ko, 0.35);
    col[k * 3] = tmpC.r; col[k * 3 + 1] = tmpC.g; col[k * 3 + 2] = tmpC.b;
  }
  geoTerrain.attributes.position.needsUpdate = true;
  geoTerrain.attributes.color.needsUpdate = true;
  geoTerrain.computeVertexNormals();
  terrainSale = false;
}

function construireBalisage() {
  balisage.clear();
  const c = CHANTIERS[chantier];
  if (!c.zone) return;
  const { x1, x2, z1, z2 } = c.zone;
  const matPiquet = new THREE.MeshLambertMaterial({ color: 0xffd24a });
  const matTete = new THREE.MeshLambertMaterial({ color: 0xff4d3d });
  for (const [x, z] of [[x1, z1], [x1, z2], [x2, z1], [x2, z2]]) {
    const h = hauteurEn(x, z);
    const piquet = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.0, 0.05), matPiquet);
    piquet.position.set(x, h + 0.42, z);
    piquet.castShadow = true;
    const tete = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.18, 0.06), matTete);
    tete.position.set(x, h + 0.85, z);
    balisage.add(piquet, tete);
  }
  /* Ruban entre les piquets et cadre au niveau visé. */
  const ruban = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x1, 0.8, z1), new THREE.Vector3(x2, 0.8, z1), new THREE.Vector3(x2, 0.8, z1), new THREE.Vector3(x2, 0.8, z2),
      new THREE.Vector3(x2, 0.8, z2), new THREE.Vector3(x1, 0.8, z2), new THREE.Vector3(x1, 0.8, z2), new THREE.Vector3(x1, 0.8, z1),
    ]), new THREE.LineBasicMaterial({ color: 0xffd24a }));
  balisage.add(ruban);
  const fond = new THREE.Mesh(new THREE.PlaneGeometry(x2 - x1, z2 - z1),
    new THREE.MeshBasicMaterial({ color: 0xffe07a, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
  fond.rotation.x = -Math.PI / 2;
  fond.position.set((x1 + x2) / 2, c.cible + 0.01, (z1 + z2) / 2);
  balisage.add(fond);
}

/* ---------- La machine ---------- */

const ORANGE = 0xf27a1f;
const matOrange = new THREE.MeshLambertMaterial({ color: ORANGE });
const matSombre = new THREE.MeshLambertMaterial({ color: 0x23262a });
const matGris = new THREE.MeshLambertMaterial({ color: 0x8e959b });
const matAcier = new THREE.MeshLambertMaterial({ color: 0xc7ccd1 });
const matNoir = new THREE.MeshLambertMaterial({ color: 0x15181b });

function boite(w, h, d, mat, x, y, z, parent) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  parent.add(m);
  return m;
}

const chassis = new THREE.Group();          // position et cap
const assiette = new THREE.Group();         // tangage et roulis
chassis.add(assiette);
scene.add(chassis);

/* Train de roulement. */
for (const zc of [-M.voie, M.voie]) {
  boite(M.chenilleL, M.chenilleH, 0.3, matSombre, 0, M.chenilleH / 2, zc, assiette);
  boite(M.chenilleL - 0.3, 0.1, 0.34, matNoir, 0, M.chenilleH / 2, zc, assiette);
}
boite(1.2, 0.14, 0.7, matSombre, 0, M.chenilleH + 0.07, 0, assiette);
boite(0.12, 0.3, 1.25, matOrange, 0.98, 0.42, 0, assiette);   // lame, relevée
boite(0.5, 0.06, 0.06, matSombre, 0.75, 0.32, 0.35, assiette);
boite(0.5, 0.06, 0.06, matSombre, 0.75, 0.32, -0.35, assiette);

/* Tourelle. */
const tourelle = new THREE.Group();
tourelle.position.copy(M.tourelle);
assiette.add(tourelle);
boite(0.8, 0.08, 0.8, matNoir, 0, 0.0, 0, tourelle);                 // couronne
boite(0.62, 0.5, 1.0, matOrange, -0.42, 0.28, 0, tourelle);          // caisson moteur, contrepoids
boite(0.55, 0.3, 0.5, matOrange, 0.16, 0.18, 0.25, tourelle);        // côté droit, réservoir, bas pour dégager la vue
boite(0.06, 0.16, 0.06, matSombre, -0.45, 0.6, 0.35, tourelle);      // échappement
/* Cabine, à gauche. */
boite(1.0, 0.06, 0.62, matNoir, -0.05, 0.53, -0.18, tourelle);       // plancher
boite(0.3, 0.1, 0.42, matSombre, -0.28, 0.63, -0.18, tourelle);      // assise
boite(0.08, 0.5, 0.42, matSombre, -0.46, 0.9, -0.18, tourelle);      // dossier
boite(0.22, 0.05, 0.5, matNoir, 0.14, 0.59, -0.18, tourelle);        // tablier bas : le sol reste visible jusqu'aux chenilles
for (const dz of [-0.12, 0.12]) boite(0.03, 0.2, 0.03, matGris, -0.04, 0.72, -0.18 + dz, tourelle);  // manettes, au bout des accoudoirs
for (const [dx, dz] of [[-0.52, -0.5], [-0.52, 0.12]]) boite(0.05, 1.1, 0.05, matNoir, dx, 1.08, dz, tourelle);       // montants arrière
for (const [dx, dz] of [[0.5, -0.5], [0.5, 0.14]]) boite(0.035, 1.1, 0.035, matNoir, dx, 1.08, dz, tourelle);       // montants avant, fins et loin de l'œil
boite(1.1, 0.06, 0.7, matNoir, -0.01, 1.63, -0.19, tourelle);        // toit
boite(1.1, 0.03, 0.7, matOrange, -0.01, 1.68, -0.19, tourelle);

/* Équipement. */
const fleche = new THREE.Group();
fleche.position.copy(M.pivot);
tourelle.add(fleche);
boite(M.fleche, 0.16, 0.13, matOrange, M.fleche / 2, 0.0, 0, fleche);
boite(M.fleche * 0.55, 0.1, 0.13, matOrange, M.fleche * 0.42, 0.11, 0, fleche);   // dos cintré
const balancier = new THREE.Group();
balancier.position.set(M.fleche, 0, 0);
fleche.add(balancier);
boite(M.balancier + 0.2, 0.12, 0.1, matOrange, M.balancier / 2 - 0.1, 0, 0, balancier);
boite(0.16, 0.16, 0.12, matOrange, -0.12, 0.08, 0, balancier);
const godet = new THREE.Group();
godet.position.set(M.balancier, 0, 0);
balancier.add(godet);
{
  const forme = new THREE.Shape();
  const pts = [[0, 0], [0.02, 0.12], [0.12, 0.22], [0.29, 0.22], [0.40, 0.08], [0.42, 0]];
  forme.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) forme.lineTo(pts[i][0], pts[i][1]);
  forme.lineTo(0, 0);
  const geo = new THREE.ExtrudeGeometry(forme, { depth: M.godetLarg, bevelEnabled: false });
  geo.translate(0, 0, -M.godetLarg / 2);
  const coque = new THREE.Mesh(geo, matAcier);
  coque.castShadow = true;
  godet.add(coque);
  for (const dz of [-0.14, 0, 0.14]) boite(0.1, 0.03, 0.04, matGris, 0.46, 0.02, dz, godet);   // dents
  boite(0.14, 0.12, 0.2, matGris, 0.04, -0.08, 0, godet);                                       // oreille
}
const terreGodet = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.2, M.godetLarg - 0.04), new THREE.MeshLambertMaterial({ color: 0x8a6440 }));
terreGodet.position.set(0.19, 0.12, 0);
godet.add(terreGodet);

/* Vérins : corps + tige, orientés chaque image entre deux points du monde. */
const geoCyl = new THREE.CylinderGeometry(1, 1, 1, 10);
function creerVerin(rCorps, rTige) {
  const corps = new THREE.Mesh(geoCyl, new THREE.MeshLambertMaterial({ color: 0x9e4406 }));
  const tige = new THREE.Mesh(geoCyl, matAcier);
  corps.castShadow = true;
  scene.add(corps, tige);
  return { corps, tige, rCorps, rTige };
}
const verins = {
  fleche: creerVerin(0.045, 0.02),
  balancier: creerVerin(0.04, 0.018),
  godet: creerVerin(0.035, 0.016),
};
const HAUT = new THREE.Vector3(0, 1, 0);
const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vDir = new THREE.Vector3();
function orienterVerin(v, a, b, longueurCorps) {
  vDir.subVectors(b, a);
  const Lg = vDir.length();
  vDir.normalize();
  const lc = Math.min(longueurCorps, Lg - 0.05);
  v.corps.position.copy(a).addScaledVector(vDir, lc / 2);
  v.corps.quaternion.setFromUnitVectors(HAUT, vDir);
  v.corps.scale.set(v.rCorps, lc, v.rCorps);
  v.tige.position.copy(a).addScaledVector(vDir, Lg / 2);
  v.tige.quaternion.setFromUnitVectors(HAUT, vDir);
  v.tige.scale.set(v.rTige, Lg, v.rTige);
}

/* Caméra : à la place de l'opérateur, tournée vers l'avant. */
const tete = new THREE.Group();
tete.position.copy(M.oeil);
tourelle.add(tete);
tete.add(camera);

/* Mottes qui tombent du godet. */
const MAX_MOTTES = 220;
const mottes = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 6, 5), new THREE.MeshLambertMaterial({ color: 0x8a6440 }), MAX_MOTTES);
mottes.count = 0;
scene.add(mottes);
const particules = [];
const mTmp = new THREE.Matrix4(), pTmp = new THREE.Vector3(), qTmp = new THREE.Quaternion(), sTmp = new THREE.Vector3();

/* ---------- Cinématique et physique ---------- */

function appliquerPose() {
  chassis.position.set(etat.x, etat.y, etat.z);
  chassis.rotation.set(0, etat.cap, 0);
  assiette.rotation.set(etat.roulis, 0, etat.tangage, 'ZXY');
  tourelle.rotation.y = etat.rotation;
  fleche.rotation.z = etat.fleche;
  balancier.rotation.z = etat.balancier;
  godet.rotation.z = etat.godet;
  tete.rotation.set(0, -Math.PI / 2 + etat.regardLacet, 0);
  camera.rotation.set(etat.regardTangage, 0, 0);
  const f = etat.charge / M.capacite;
  terreGodet.visible = f > 0.02;
  terreGodet.scale.y = Math.max(0.05, f);
  terreGodet.position.y = 0.22 - 0.1 * f;
  scene.updateMatrixWorld(true);

  /* Vérins, calculés dans le monde. */
  const pA = tourelle.localToWorld(vA.set(0.25, 0.08, M.pivot.z));
  const pB = fleche.localToWorld(vB.set(M.fleche * 0.48, 0.14, 0));
  orienterVerin(verins.fleche, pA, pB, 0.42);
  const pC = fleche.localToWorld(vA.set(M.fleche * 0.62, 0.17, 0));
  const pD = balancier.localToWorld(vB.set(-0.2, 0.1, 0));
  orienterVerin(verins.balancier, pC, pD, 0.36);
  const pE = balancier.localToWorld(vA.set(0.12, 0.09, 0));
  const pF = godet.localToWorld(vB.set(-0.1, -0.14, 0));
  orienterVerin(verins.godet, pE, pF, 0.3);
}

/* La machine s'assied sur le relief sous ses chenilles. */
const ECHANT = [];
for (let u = -1; u <= 1; u += 0.5) for (let v = -1; v <= 1; v += 1) ECHANT.push([u * M.chenilleL / 2, v * (M.voie + 0.1)]);
function asseoir() {
  const c = Math.cos(etat.cap), s = Math.sin(etat.cap);
  let suu = 0, svv = 0, suh = 0, svh = 0, sh = 0;
  const hs = [];
  for (const [u, v] of ECHANT) {
    const x = etat.x + u * c + v * s, z = etat.z - u * s + v * c;
    const h = hauteurEn(x, z);
    hs.push(h); sh += h; suu += u * u; svv += v * v; suh += u * h; svh += v * h;
  }
  const moy = sh / ECHANT.length;
  let a = 0, b = 0;
  for (let n = 0; n < ECHANT.length; n++) { a += ECHANT[n][0] * (hs[n] - moy); b += ECHANT[n][1] * (hs[n] - moy); }
  a = clamp(a / suu, -0.6, 0.6); b = clamp(b / svv, -0.6, 0.6);
  let appui = -Infinity;
  for (let n = 0; n < ECHANT.length; n++) appui = Math.max(appui, hs[n] - a * ECHANT[n][0] - b * ECHANT[n][1]);
  etat.y = lerp(etat.y, appui, 0.2);
  etat.tangage = lerp(etat.tangage, Math.atan(a), 0.15);
  etat.roulis = lerp(etat.roulis, Math.atan(b), 0.15);
}

function actionner(dt) {
  const k = (plus, moins) => (clavier[plus] ? 1 : 0) - (clavier[moins] ? 1 : 0);
  const rot = axes.rotation || k('KeyA', 'KeyD');
  const bal = axes.balancier || k('KeyW', 'KeyS');
  const fl = axes.fleche || k('KeyI', 'KeyK');
  const go = axes.godet || k('KeyL', 'KeyJ');
  const ch = axes.chenille || k('ArrowUp', 'ArrowDown');
  const vi = axes.virage || k('ArrowLeft', 'ArrowRight');

  etat.rotation += rot * VIT.rotation * dt;
  etat.balancier = clamp(etat.balancier + bal * VIT.balancier * dt, LIM.balancier[0], LIM.balancier[1]);
  etat.fleche = clamp(etat.fleche - fl * VIT.fleche * dt, LIM.fleche[0], LIM.fleche[1]);
  etat.godet = clamp(etat.godet + go * VIT.godet * dt, LIM.godet[0], LIM.godet[1]);
  if (ch) {
    etat.x = clamp(etat.x + Math.cos(etat.cap) * ch * VIT.chenille * dt, -DEMI + 1.6, DEMI - 1.6);
    etat.z = clamp(etat.z - Math.sin(etat.cap) * ch * VIT.chenille * dt, -DEMI + 1.6, DEMI - 1.6);
  }
  etat.cap += vi * VIT.virage * dt;
  if ((rot || bal || fl || go || ch || vi) && !etat.termine) etat.chronoActif = true;
}

const cellulesVues = new Set();
function creuser() {
  const reste = M.capacite - etat.charge;
  if (reste <= 1e-6) return;
  let pris = 0;
  cellulesVues.clear();
  for (let n = -2; n <= 2; n++) {
    const p = godet.localToWorld(vA.set(M.godet, 0, n * 0.1));
    const j0 = idxJ(p.x - 0.09), j1 = idxJ(p.x + 0.09), i0 = idxI(p.z - 0.09), i1 = idxI(p.z + 0.09);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const k = i * N + j;
      if (cellulesVues.has(k)) continue;
      cellulesVues.add(k);
      if (sol[k] > p.y) pris += couperCellule(k, p.y, reste - pris);
    }
  }
  if (pris > 0) etat.charge += pris;
}

function vidanger(dt) {
  if (etat.charge <= 0) return;
  const n = vDir.set(0, -1, 0).transformDirection(godet.matrixWorld);
  const fuite = clamp((0.35 - n.y) / 0.7, 0, 1);
  if (fuite <= 0) return;
  const vol = Math.min(etat.charge, 0.07 * fuite * dt);
  etat.charge -= vol;
  const bouche = godet.localToWorld(vA.set(0.21, -0.06, 0));
  const nb = 1 + (Math.random() < fuite ? 1 : 0);
  for (let i = 0; i < nb && particules.length < MAX_MOTTES; i++) {
    particules.push({
      x: bouche.x + (Math.random() - 0.5) * 0.25, y: bouche.y, z: bouche.z + (Math.random() - 0.5) * 0.3,
      vx: n.x * 0.5 + (Math.random() - 0.5) * 0.4, vy: n.y * 0.4 + Math.random() * 0.2, vz: n.z * 0.5 + (Math.random() - 0.5) * 0.4,
      vol: vol / nb, r: 0.035 + Math.random() * 0.03,
    });
  }
}

function animerMottes(dt) {
  for (let i = particules.length - 1; i >= 0; i--) {
    const p = particules[i];
    p.vy -= G * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
    if (p.y <= hauteurEn(p.x, p.z) + p.r * 0.4) {
      deposer(clamp(p.x, -DEMI + 0.3, DEMI - 0.3), clamp(p.z, -DEMI + 0.3, DEMI - 0.3), p.vol);
      etat.volume += p.vol;
      sauvegardeEnAttente = true;
      particules.splice(i, 1);
    }
  }
  mottes.count = particules.length;
  for (let i = 0; i < particules.length; i++) {
    const p = particules[i];
    pTmp.set(p.x, p.y, p.z); sTmp.set(p.r, p.r, p.r);
    mTmp.compose(pTmp, qTmp, sTmp);
    mottes.setMatrixAt(i, mTmp);
  }
  mottes.instanceMatrix.needsUpdate = true;
}

/* ---------- Avancement ---------- */

function avancement() {
  const c = CHANTIERS[chantier];
  if (!c.zone) return null;
  let ok = 0, tot = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (!dansZone(c, cellX(j), cellZ(i))) continue;
    tot++;
    if (celluleOk(c, i * N + j)) ok++;
  }
  return tot ? ok / tot : 0;
}

function verifierFin() {
  if (etat.termine) return;
  const p = avancement();
  if (p !== null && p >= 0.98) {
    etat.termine = true;
    etat.chronoActif = false;
    const t = etat.chrono, ancien = records[chantier];
    let texte = `Terminé en ${fmtChrono(t)}, ${fmt(etat.volume)} m³ déplacés.`;
    if (!ancien || t < ancien) {
      records[chantier] = t;
      sauverRecords();
      texte += ancien ? ' Nouveau record !' : ' Premier chrono enregistré.';
    } else texte += ` Record : ${fmtChrono(ancien)}.`;
    ui.banniereTexte.textContent = texte;
    ui.banniere.hidden = false;
    sauvegardeEnAttente = true;
  }
}

/* ---------- Interface ---------- */

const ui = {};
for (const id of ['chantier', 'btnReset', 'btnAide', 'btnRecentrer', 'aide', 'titreChantier', 'consigne', 'valAvancement', 'barreAvancement',
  'valChrono', 'valGodet', 'barreGodet', 'pisteGodet', 'valProfondeur', 'valVolume', 'banniere', 'banniereTexte', 'btnBanniere', 'chargement']) {
  ui[id] = document.getElementById(id);
}

function fmtChrono(s) {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${r < 10 ? '0' : ''}${r}`;
}

let dernierAvancement = null;
function rafraichirHud() {
  const c = CHANTIERS[chantier];
  const p = avancement();
  if (p !== dernierAvancement) { dernierAvancement = p; if (c.zone) terrainSale = true; }
  ui.valAvancement.textContent = p === null ? '—' : `${etat.termine ? 100 : Math.min(99, Math.floor(p * 100))} %`;
  ui.barreAvancement.style.width = p === null ? '0%' : `${(etat.termine ? 1 : p) * 100}%`;
  ui.valChrono.textContent = c.zone ? fmtChrono(etat.chrono) : '—';
  const f = etat.charge / M.capacite;
  ui.valGodet.textContent = `${Math.round(etat.charge * 1000)} L`;
  ui.barreGodet.style.width = `${f * 100}%`;
  ui.pisteGodet.classList.toggle('plein', f > 0.97);
  const dents = godet.localToWorld(vA.set(M.godet, 0, 0));
  const prof = dents.y - hauteurEn(etat.x, etat.z);
  ui.valProfondeur.textContent = `${prof < 0 ? '−' : '+'}${fmt(Math.abs(prof))} m`;
  ui.valProfondeur.classList.toggle('alerte', prof < -0.2);
  ui.valVolume.textContent = `${fmt(etat.volume)} m³`;
}

function afficherChantier() {
  const c = CHANTIERS[chantier];
  ui.titreChantier.textContent = c.nom;
  ui.consigne.innerHTML = c.consigne;
  ui.chantier.value = chantier;
}

/* ---------- Manettes et clavier ---------- */

function brancherManette(el, surChangement) {
  const pouce = el.querySelector('.pouce');
  const R = () => el.clientWidth * 0.34;
  let id = null;
  const poser = (e) => {
    const r = el.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    const d = Math.hypot(dx, dy), rr = R();
    const f = d > rr ? rr / d : 1;
    const px = dx * f, py = dy * f;
    pouce.style.transform = `translate(${px}px, ${py}px)`;
    const nx = px / rr, ny = -py / rr;
    const zone = (v) => (Math.abs(v) < 0.12 ? 0 : (v - Math.sign(v) * 0.12) / 0.88);
    surChangement(zone(nx), zone(ny));
  };
  el.addEventListener('pointerdown', (e) => {
    if (id !== null) return;
    id = e.pointerId;
    el.setPointerCapture(id);
    el.classList.add('active');
    poser(e);
    e.preventDefault();
  });
  el.addEventListener('pointermove', (e) => { if (e.pointerId === id) poser(e); });
  const lacher = (e) => {
    if (e.pointerId !== id) return;
    id = null;
    el.classList.remove('active');
    pouce.style.transform = '';
    surChangement(0, 0);
  };
  el.addEventListener('pointerup', lacher);
  el.addEventListener('pointercancel', lacher);
}

function brancherCommandes() {
  brancherManette(document.getElementById('manetteG'), (x, y) => { axes.rotation = -x; axes.balancier = y; });
  brancherManette(document.getElementById('manetteD'), (x, y) => { axes.fleche = y; axes.godet = x; });

  for (const b of document.querySelectorAll('.pedale[data-action]')) {
    const a = b.dataset.action;
    const val = { avancer: ['chenille', 1], reculer: ['chenille', -1], tournerG: ['virage', 1], tournerD: ['virage', -1] }[a];
    const debut = (e) => { e.preventDefault(); b.setPointerCapture?.(e.pointerId); axes[val[0]] = val[1]; b.classList.add('actif'); };
    const fin = () => { if (axes[val[0]] === val[1]) axes[val[0]] = 0; b.classList.remove('actif'); };
    b.addEventListener('pointerdown', debut);
    b.addEventListener('pointerup', fin);
    b.addEventListener('pointercancel', fin);
    b.addEventListener('lostpointercapture', fin);
    b.addEventListener('keydown', (e) => { if (e.code === 'Space' || e.code === 'Enter') { e.preventDefault(); axes[val[0]] = val[1]; b.classList.add('actif'); } });
    b.addEventListener('keyup', (e) => { if (e.code === 'Space' || e.code === 'Enter') fin(); });
  }

  /* Regard : glisser sur la vue. */
  let regardId = null, rx = 0, ry = 0;
  canvas.addEventListener('pointerdown', (e) => {
    if (regardId !== null) return;
    regardId = e.pointerId; rx = e.clientX; ry = e.clientY;
    canvas.setPointerCapture(regardId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerId !== regardId) return;
    etat.regardLacet = clamp(etat.regardLacet - (e.clientX - rx) * 0.0045, -1.7, 1.7);
    etat.regardTangage = clamp(etat.regardTangage - (e.clientY - ry) * 0.0045, -1.0, 0.6);
    rx = e.clientX; ry = e.clientY;
  });
  const finRegard = (e) => { if (e.pointerId === regardId) regardId = null; };
  canvas.addEventListener('pointerup', finRegard);
  canvas.addEventListener('pointercancel', finRegard);
  ui.btnRecentrer.addEventListener('click', () => { etat.regardLacet = 0; etat.regardTangage = -0.30; });

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || ui.aide.open) return;
    if (e.code === 'KeyH' || e.key === '?') { ui.aide.showModal(); return; }
    if (e.code.startsWith('Arrow') || /^Key[ADWSIKJL]$/.test(e.code)) { clavier[e.code] = true; e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => { delete clavier[e.code]; });
  window.addEventListener('blur', () => { for (const k of Object.keys(clavier)) delete clavier[k]; });

  ui.chantier.addEventListener('change', () => { demarrerChantier(ui.chantier.value); ui.chantier.blur(); });
  ui.btnReset.addEventListener('click', () => { demarrerChantier(chantier); ui.btnReset.blur(); });
  ui.btnAide.addEventListener('click', () => ui.aide.showModal());
  ui.btnBanniere.addEventListener('click', () => { ui.banniere.hidden = true; });
}

/* ---------- Persistance ---------- */

function sauver() {
  sauvegardeEnAttente = false;
  try {
    const arr = (t) => Array.from(t, (v) => Math.round(v * 1000) / 1000);
    localStorage.setItem(LS_ETAT, JSON.stringify({
      chantier, sol: arr(sol), meuble: arr(meuble), vierge: Array.from(vierge),
      machine: { x: etat.x, z: etat.z, cap: etat.cap, rotation: etat.rotation, fleche: etat.fleche, balancier: etat.balancier, godet: etat.godet },
      charge: etat.charge, volume: etat.volume, chrono: etat.chrono, termine: etat.termine,
    }));
  } catch (err) { /* stockage indisponible : on continue sans sauvegarde */ }
}
function sauverRecords() {
  try { localStorage.setItem(LS_RECORDS, JSON.stringify(records)); } catch (err) { /* idem */ }
}

function charger() {
  try {
    const r = JSON.parse(localStorage.getItem(LS_RECORDS) || '{}');
    if (r && typeof r === 'object') for (const [k, v] of Object.entries(r)) if (CHANTIERS[k] && Number.isFinite(v) && v > 0) records[k] = v;
  } catch (err) { records = {}; }
  try {
    const d = JSON.parse(localStorage.getItem(LS_ETAT) || 'null');
    if (!d || !CHANTIERS[d.chantier] || !Array.isArray(d.sol) || d.sol.length !== N * N || !Array.isArray(d.meuble) || d.meuble.length !== N * N) return false;
    chantier = d.chantier;
    for (let k = 0; k < N * N; k++) {
      const s = Number(d.sol[k]), m = Number(d.meuble[k]);
      if (!Number.isFinite(s) || !Number.isFinite(m)) return false;
      sol[k] = clamp(s, ROC, 6);
      meuble[k] = clamp(m, 0, Math.max(0, sol[k] - ROC));
      vierge[k] = Array.isArray(d.vierge) && d.vierge[k] ? 1 : 0;
    }
    const m = d.machine || {}, dep = CHANTIERS[chantier].depart;
    const num = (v, def, a, b) => (Number.isFinite(v) ? clamp(v, a, b) : def);
    etat.x = num(m.x, dep.x, -DEMI + 1.6, DEMI - 1.6);
    etat.z = num(m.z, dep.z, -DEMI + 1.6, DEMI - 1.6);
    etat.cap = num(m.cap, dep.cap, -1e4, 1e4);
    etat.rotation = num(m.rotation, 0, -1e4, 1e4);
    etat.fleche = num(m.fleche, 0.45, LIM.fleche[0], LIM.fleche[1]);
    etat.balancier = num(m.balancier, -1.6, LIM.balancier[0], LIM.balancier[1]);
    etat.godet = num(m.godet, -1.4, LIM.godet[0], LIM.godet[1]);
    etat.charge = num(d.charge, 0, 0, M.capacite);
    etat.volume = num(d.volume, 0, 0, 1e6);
    etat.chrono = num(d.chrono, 0, 0, 1e7);
    etat.termine = !!d.termine;
    etat.chronoActif = false;
    etat.y = hauteurEn(etat.x, etat.z);
    terrainSale = true;
    return true;
  } catch (err) {
    console.warn('Chantier sauvegardé illisible, chantier par défaut utilisé.', err);
    return false;
  }
}

/* ---------- Chantier ---------- */

function demarrerChantier(cle) {
  chantier = cle;
  genererTerrain(cle);
  particules.length = 0;
  const dep = CHANTIERS[cle].depart;
  Object.assign(etat, {
    x: dep.x, z: dep.z, cap: dep.cap, tangage: 0, roulis: 0,
    rotation: 0, fleche: 0.45, balancier: -1.6, godet: -1.4,
    charge: 0, volume: 0, chrono: 0, chronoActif: false, termine: false,
    regardLacet: 0, regardTangage: -0.30,
  });
  etat.y = hauteurEn(etat.x, etat.z);
  dernierAvancement = null;
  ui.banniere.hidden = true;
  afficherChantier();
  construireBalisage();
  rafraichirTerrain();
  appliquerPose();
  rafraichirHud();
  sauver();
}

/* ---------- Boucle ---------- */

function redimensionner() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w < h ? 80 : 68;
  camera.updateProjectionMatrix();
}

let derniere = performance.now();
let accHud = 0, accSauvegarde = 0, premiereImage = true;

function boucle(t) {
  const dt = Math.min(0.05, (t - derniere) / 1000);
  derniere = t;

  actionner(dt);
  asseoir();
  appliquerPose();
  creuser();
  vidanger(dt);
  animerMottes(dt);
  tasser();
  if (etat.chronoActif) etat.chrono += dt;
  if (terrainSale) { rafraichirTerrain(); sauvegardeEnAttente = true; }

  soleil.position.set(etat.x - 6, 12, etat.z + 5);
  soleil.target.position.set(etat.x, 0, etat.z);

  renderer.render(scene, camera);
  if (premiereImage) { premiereImage = false; ui.chargement.remove(); }

  accHud += dt;
  if (accHud > 0.1) { accHud = 0; rafraichirHud(); verifierFin(); }
  accSauvegarde += dt;
  if (accSauvegarde > 4 && sauvegardeEnAttente) { accSauvegarde = 0; sauver(); }

  requestAnimationFrame(boucle);
}

/* ---------- Démarrage ---------- */

function init() {
  redimensionner();
  window.addEventListener('resize', redimensionner);
  brancherCommandes();
  if (!charger()) demarrerChantier('bac');
  else { afficherChantier(); construireBalisage(); rafraichirTerrain(); appliquerPose(); }
  rafraichirHud();
  window.addEventListener('pagehide', sauver);
  requestAnimationFrame(boucle);
}

init();
