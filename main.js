// =============================================================
// 기타 코드 운지 교정기 — Phase 0~1
//   Phase 0: 웹캠 띄우기 (getUserMedia + 비디오/캔버스 동기화)
//   Phase 1: MediaPipe Hand Landmarker 실시간 손 추적 오버레이
//
// 핵심 좌표 규약 (Phase 2~에서도 그대로 재사용):
//   - MediaPipe 랜드마크는 정규화 좌표(0~1)로 들어온다.
//   - 화면 표시 좌표는 toDisplay()가 단일하게 책임진다.
//   - 거울 모드(MIRROR)일 때 비디오는 CSS로 좌우 반전되므로,
//     toDisplay()도 x를 (1-x)로 뒤집어 시각적으로 정확히 일치시킨다.
//   - 캔버스 자체는 CSS 반전하지 않으므로 텍스트가 정상으로 보인다.
// =============================================================

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

// MediaPipe CDN 자원
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// 손가락 끝 랜드마크 인덱스 → 기타 운지 손가락 번호(라벨)
const FINGERTIPS = {
  4: "T",  // 엄지(Thumb)
  8: "1",  // 검지(Index)
  12: "2", // 중지(Middle)
  16: "3", // 약지(Ring)
  20: "4", // 새끼(Pinky)
};

// Phase 3: 운지로 매핑할 손끝(엄지는 보통 넥 뒤라 제외)
const FRET_FINGERS = [
  { idx: 8, finger: "1" },
  { idx: 12, finger: "2" },
  { idx: 16, finger: "3" },
  { idx: 20, finger: "4" },
];
const FINGER_NAMES = { "1": "검지", "2": "중지", "3": "약지", "4": "새끼", T: "엄지" };

// Phase 4: 코드 운지 데이터
//  줄 1=얇은 고음E … 6=굵은 저음E. fret: 0=개방, 'x'=뮤트, 양수=프렛. finger: '1'~'4'/null.
const CHORDS = {
  C: { name: "C", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 3, finger: "3" },
    { string: 4, fret: 2, finger: "2" },
    { string: 3, fret: 0, finger: null },
    { string: 2, fret: 1, finger: "1" },
    { string: 1, fret: 0, finger: null },
  ] },
  G: { name: "G", voicing: [
    { string: 6, fret: 3, finger: "2" },
    { string: 5, fret: 2, finger: "1" },
    { string: 4, fret: 0, finger: null },
    { string: 3, fret: 0, finger: null },
    { string: 2, fret: 0, finger: null },
    { string: 1, fret: 3, finger: "3" },
  ] },
  D: { name: "D", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: "x", finger: null },
    { string: 4, fret: 0, finger: null },
    { string: 3, fret: 2, finger: "1" },
    { string: 2, fret: 3, finger: "3" },
    { string: 1, fret: 2, finger: "2" },
  ] },
  A: { name: "A", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 0, finger: null },
    { string: 4, fret: 2, finger: "1" },
    { string: 3, fret: 2, finger: "2" },
    { string: 2, fret: 2, finger: "3" },
    { string: 1, fret: 0, finger: null },
  ] },
  E: { name: "E", voicing: [
    { string: 6, fret: 0, finger: null },
    { string: 5, fret: 2, finger: "2" },
    { string: 4, fret: 2, finger: "3" },
    { string: 3, fret: 1, finger: "1" },
    { string: 2, fret: 0, finger: null },
    { string: 1, fret: 0, finger: null },
  ] },
  Am: { name: "Am", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 0, finger: null },
    { string: 4, fret: 2, finger: "2" },
    { string: 3, fret: 2, finger: "3" },
    { string: 2, fret: 1, finger: "1" },
    { string: 1, fret: 0, finger: null },
  ] },
  Em: { name: "Em", voicing: [
    { string: 6, fret: 0, finger: null },
    { string: 5, fret: 2, finger: "2" },
    { string: 4, fret: 2, finger: "3" },
    { string: 3, fret: 0, finger: null },
    { string: 2, fret: 0, finger: null },
    { string: 1, fret: 0, finger: null },
  ] },
  Dm: { name: "Dm", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: "x", finger: null },
    { string: 4, fret: 0, finger: null },
    { string: 3, fret: 2, finger: "2" },
    { string: 2, fret: 3, finger: "3" },
    { string: 1, fret: 1, finger: "1" },
  ] },
  // ── 세븐스 ──
  A7: { name: "A7", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 0, finger: null },
    { string: 4, fret: 2, finger: "2" },
    { string: 3, fret: 0, finger: null },
    { string: 2, fret: 2, finger: "3" },
    { string: 1, fret: 0, finger: null },
  ] },
  D7: { name: "D7", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: "x", finger: null },
    { string: 4, fret: 0, finger: null },
    { string: 3, fret: 2, finger: "2" },
    { string: 2, fret: 1, finger: "1" },
    { string: 1, fret: 2, finger: "3" },
  ] },
  E7: { name: "E7", voicing: [
    { string: 6, fret: 0, finger: null },
    { string: 5, fret: 2, finger: "2" },
    { string: 4, fret: 0, finger: null },
    { string: 3, fret: 1, finger: "1" },
    { string: 2, fret: 0, finger: null },
    { string: 1, fret: 0, finger: null },
  ] },
  G7: { name: "G7", voicing: [
    { string: 6, fret: 3, finger: "3" },
    { string: 5, fret: 2, finger: "2" },
    { string: 4, fret: 0, finger: null },
    { string: 3, fret: 0, finger: null },
    { string: 2, fret: 0, finger: null },
    { string: 1, fret: 1, finger: "1" },
  ] },
  C7: { name: "C7", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 3, finger: "3" },
    { string: 4, fret: 2, finger: "2" },
    { string: 3, fret: 3, finger: "4" },
    { string: 2, fret: 1, finger: "1" },
    { string: 1, fret: 0, finger: null },
  ] },
  // ── 기타 ──
  Cadd9: { name: "Cadd9", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 3, finger: "2" },
    { string: 4, fret: 2, finger: "1" },
    { string: 3, fret: 0, finger: null },
    { string: 2, fret: 3, finger: "3" },
    { string: 1, fret: 0, finger: null },
  ] },
  // ── 바레(고급): 검지가 한 프렛의 여러 줄을 동시에 누름 ──
  F: { name: "F", voicing: [
    { string: 6, fret: 1, finger: "1" },
    { string: 5, fret: 3, finger: "3" },
    { string: 4, fret: 3, finger: "4" },
    { string: 3, fret: 2, finger: "2" },
    { string: 2, fret: 1, finger: "1" },
    { string: 1, fret: 1, finger: "1" },
  ] },
  Bm: { name: "Bm", voicing: [
    { string: 6, fret: "x", finger: null },
    { string: 5, fret: 2, finger: "1" },
    { string: 4, fret: 4, finger: "3" },
    { string: 3, fret: 4, finger: "4" },
    { string: 2, fret: 3, finger: "2" },
    { string: 1, fret: 2, finger: "1" },
  ] },
};

// HAND_CONNECTIONS: 라이브러리 정적 값을 우선 사용, 없으면 폴백.
const HAND_CONNECTIONS =
  HandLandmarker.HAND_CONNECTIONS ||
  [
    [0, 1], [1, 2], [2, 3], [3, 4],        // 엄지
    [0, 5], [5, 6], [6, 7], [7, 8],        // 검지
    [5, 9], [9, 10], [10, 11], [11, 12],   // 중지
    [9, 13], [13, 14], [14, 15], [15, 16], // 약지
    [13, 17], [17, 18], [18, 19], [19, 20],// 새끼
    [0, 17],                               // 손바닥 아랫변
  ].map(([start, end]) => ({ start, end }));

// ── DOM 참조 ──
const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

const stage = $("stage");
const els = {
  startBtn: $("startBtn"),
  stopBtn: $("stopBtn"),
  cameraSelect: $("cameraSelect"),
  mirrorToggle: $("mirrorToggle"),
  skeletonToggle: $("skeletonToggle"),
  overlayMsg: $("overlayMsg"),
  hud: $("hud"),
  fps: $("fps"),
  handCount: $("handCount"),
  modelStatus: $("modelStatus"),
  camStatus: $("camStatus"),
  resStatus: $("resStatus"),
  errorBox: $("errorBox"),
  errorText: $("errorText"),
  errorClose: $("errorClose"),
  // Phase 2
  fretCount: $("fretCount"),
  calibMode: $("calibMode"),
  calibBtn: $("calibBtn"),
  calibClearBtn: $("calibClearBtn"),
  gridToggle: $("gridToggle"),
  calibStatus: $("calibStatus"),
  calibBanner: $("calibBanner"),
  // Phase 3
  fingerLabelToggle: $("fingerLabelToggle"),
  fingerReadout: $("fingerReadout"),
  // Phase 4
  chordSelect: $("chordSelect"),
  strictFingerToggle: $("strictFingerToggle"),
  capo: $("capo"),
  judgeBody: $("judgeBody"),
  // Phase 5
  smoothToggle: $("smoothToggle"),
  voiceToggle: $("voiceToggle"),
};

// ── 상태 ──
let handLandmarker = null; // 모델 인스턴스
let modelReady = false;
let stream = null;         // MediaStream
let running = false;       // 렌더 루프 동작 여부
let rafId = null;
let lastVideoTime = -1;    // 같은 프레임 재처리 방지
let latestResults = null;  // 최근 추론 결과(프레임 간 유지하여 부드럽게 표시)
let lastDetectTs = 0;      // detectForVideo 타임스탬프 단조 증가 보장용
let startToken = 0;        // startCamera 재진입(빠른 장치 전환) 방지용 세대 토큰

let MIRROR = els.mirrorToggle.checked;
// 새로고침/폼 복원 시 체크박스 상태가 정적 HTML과 달라질 수 있으므로,
// CSS 반전 클래스를 항상 체크박스(JS 단일 진실)에서 파생시킨다.
video.classList.toggle("mirrored", MIRROR);

// ── Phase 2/6: 지판 캘리브레이션 상태 ──
//  saved.points 의 (x,y)는 "원본 프레임 정규화 좌표(0~1)" — 랜드마크와 동일 좌표계라
//  toDisplay()로 변환하면 미러/해상도가 바뀌어도 항상 영상과 일치한다. (u,v)는 지판좌표.
const CALIB_KEY = "guitar-fret-calib-v2";
const calib = {
  active: false,   // 보정 클릭 수집 중인지
  K: 5,            // 기준 프렛
  mode: "corners", // "corners"(4점 빠름) | "multi"(프렛별 정밀)
  clickPts: [],    // 수집 중: {x,y(정규화), u,v(지판좌표)}
  targets: [],     // 클릭 순서별 기대 지판좌표 {u,v,n,side}
  saved: null,     // { mode, K, points: [{x,y,u,v}] }
};
let homography = null;          // { img2fret, fret2img, K } — Phase 3에서 재사용
let calibSuccessTimer = null;

// ── Phase 3: 손끝 → 줄/프렛 매핑 결과(Phase 4에서 재사용) ──
let detectedFingers = [];
let lastReadoutHtml = "";
// ── Phase 4: 코드 판정 ──
let lastJudgeHtml = "";

// FPS 측정
let fpsLast = performance.now();
let fpsFrames = 0;
let fpsValue = 0;

// =============================================================
// 유틸: 오버레이 메시지 / 에러 토스트
// =============================================================
function setOverlayMessage(html) {
  if (html == null) {
    els.overlayMsg.hidden = true;
    return;
  }
  els.overlayMsg.innerHTML = `<p>${html}</p>`;
  els.overlayMsg.hidden = false;
}

function showError(message) {
  els.errorText.textContent = message;
  els.errorBox.hidden = false;
}
els.errorClose.addEventListener("click", () => (els.errorBox.hidden = true));

// =============================================================
// 좌표 변환 (단일 진실 공급원)
//   정규화 랜드마크 → 캔버스 픽셀. 거울 모드면 x를 뒤집는다.
// =============================================================
function toDisplay(landmark) {
  const x = MIRROR ? (1 - landmark.x) : landmark.x;
  return { x: x * canvas.width, y: landmark.y * canvas.height };
}

// 화면 클릭(clientX/Y) → 캔버스 내부 픽셀. object-fit:cover 크롭/스케일을 역산.
function clientToCanvas(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const iw = canvas.width;
  const ih = canvas.height;
  if (!iw || !ih || !rect.width || !rect.height) return null;
  const scale = Math.max(rect.width / iw, rect.height / ih); // cover
  const dispW = iw * scale;
  const dispH = ih * scale;
  const offsetX = (rect.width - dispW) / 2;
  const offsetY = (rect.height - dispH) / 2;
  const cx = (clientX - rect.left - offsetX) / scale;
  const cy = (clientY - rect.top - offsetY) / scale;
  return {
    x: Math.min(iw, Math.max(0, cx)),
    y: Math.min(ih, Math.max(0, cy)),
  };
}

// =============================================================
// Phase 2: Homography (4점 DLT) — 외부 의존 없이 직접 구현
// =============================================================
// n×n 선형계 A·x=b 를 부분 피벗 가우스 소거로 푼다.
function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (!(Math.abs(M[piv][col]) > 1e-12)) return null; // 특이행렬(거의 일직선)/NaN 방어
    [M[col], M[piv]] = [M[piv], M[col]];
    const pivVal = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= pivVal;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

// 4점 대응(src→dst)으로 3×3 homography를 구한다. 반환: 길이 9 배열(h8=1).
function computeHomography(src, dst) {
  if (src.length !== 4 || dst.length !== 4) return null;
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const { x: X, y: Y } = dst[i];
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]);
    b.push(X);
    A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]);
    b.push(Y);
  }
  const h = solveLinearSystem(A, b);
  if (!h) return null;
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

// 4점 초과 대응을 최소제곱(정규방정식)으로 푼다 → 클릭 잡음 평균화로 정밀도↑.
// n=4면 정확해(computeHomography와 동일)를 준다.
function computeHomographyLS(src, dst) {
  const n = src.length;
  if (n < 4) return null;
  const rows = [];
  const bb = [];
  for (let i = 0; i < n; i++) {
    const { x, y } = src[i];
    const { x: X, y: Y } = dst[i];
    rows.push([x, y, 1, 0, 0, 0, -X * x, -X * y]); bb.push(X);
    rows.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]); bb.push(Y);
  }
  const AtA = Array.from({ length: 8 }, () => new Array(8).fill(0));
  const Atb = new Array(8).fill(0);
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    for (let i = 0; i < 8; i++) {
      Atb[i] += row[i] * bb[r];
      for (let j = 0; j < 8; j++) AtA[i][j] += row[i] * row[j];
    }
  }
  const h = solveLinearSystem(AtA, Atb);
  if (!h) return null;
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

function applyHomography(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  if (Math.abs(w) < 1e-12) return { x: NaN, y: NaN };
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

// 평균율: 너트~K프렛 구간에서 프렛 n의 정규화 위치(u). u(0)=0, u(K)=K.
function fretU(n, K) {
  const denom = 1 - Math.pow(2, -K / 12);
  if (denom === 0) return n; // K=0 방어
  return (K * (1 - Math.pow(2, -n / 12))) / denom;
}

// 저장된 보정(정규화 코너)에서 현재 표시 좌표계 기준 homography 쌍을 만든다.
//  fret 좌표 규약: u=프렛(0=너트..K), v=줄(0=6번줄..5=1번줄)
//  클릭 순서: [6줄·너트, 1줄·너트, 1줄·K, 6줄·K] → (0,0),(0,5),(K,5),(K,0)
function computeHomographies(saved) {
  if (!saved || !Array.isArray(saved.points) || saved.points.length < 4) return null;
  if (!Number.isFinite(saved.K)) return null;
  // 손상된 점이 toDisplay에서 throw하거나 NaN homography를 만들지 않도록 방어
  for (const p of saved.points) {
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.u) || !Number.isFinite(p.v)) return null;
  }
  const K = saved.K;
  const disp = saved.points.map(toDisplay); // {x,y,u,v} → toDisplay는 x,y만 사용(미러 반영)
  const fret = saved.points.map((p) => ({ x: p.u, y: p.v }));
  const exact = saved.points.length === 4;
  const img2fret = exact ? computeHomography(disp, fret) : computeHomographyLS(disp, fret);
  const fret2img = exact ? computeHomography(fret, disp) : computeHomographyLS(fret, disp);
  if (!img2fret || !fret2img) return null;
  return { img2fret, fret2img, K };
}

// =============================================================
// Phase 2: 보정 격자 / 진행 중 마커 그리기
// =============================================================
function drawFretGrid(H, K) {
  ctx.save();
  ctx.lineCap = "butt"; // 이전 프레임 drawHands가 남긴 "round" 누수 방지
  // 줄(가로 6선): 6번줄(v=0) ~ 1번줄(v=5)
  ctx.lineWidth = Math.max(1.5, canvas.width * 0.0025);
  ctx.strokeStyle = "rgba(76, 201, 240, 0.85)";
  for (let s = 0; s <= 5; s++) {
    const a = applyHomography(H, 0, s);
    const b = applyHomography(H, K, s);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  // 프렛(세로 K+1선): 평균율 위치, 너트는 굵은 금색
  for (let n = 0; n <= K; n++) {
    const u = fretU(n, K);
    const a = applyHomography(H, u, 0);
    const b = applyHomography(H, u, 5);
    if (n === 0) {
      ctx.lineWidth = Math.max(3, canvas.width * 0.006);
      ctx.strokeStyle = "#ffd166";
    } else {
      ctx.lineWidth = Math.max(1.5, canvas.width * 0.0032);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.82)";
    }
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    // 프렛 번호(프렛 공간 중앙, 6번줄 바깥쪽)
    if (n >= 1) {
      const uMid = (fretU(n - 1, K) + u) / 2;
      const lab = applyHomography(H, uMid, -0.45);
      drawLabel(String(n), lab.x, lab.y, "rgba(255,255,255,0.92)", 12, true);
    }
  }
  ctx.restore();
}

function drawCalibInProgress() {
  if (!calib.clickPts.length) return;
  const pts = calib.clickPts.map(toDisplay);
  ctx.save();
  ctx.strokeStyle = "#ffd166";
  ctx.fillStyle = "#ffd166";
  ctx.lineWidth = 2;
  if (pts.length > 1) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    if (calib.mode === "corners" && pts.length === 4) ctx.closePath();
    ctx.stroke();
  }
  pts.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fill();
    drawLabel(String(i + 1), p.x, p.y - 16, "#ffd166", 12, true);
  });
  ctx.restore();
}

// =============================================================
// Phase 3: 손끝 → 줄/프렛 매핑
// =============================================================
// 지판좌표 u(너트=0..K) → 프렛 번호. 프렛 공간 n(프렛선 n-1~n 사이)을 누르면 소리나는 프렛은 n.
function uToFret(u, K) {
  if (!(u >= 0)) return { fret: null, reason: "nut" }; // 너트 뒤 / NaN
  for (let n = 1; n <= K; n++) {
    if (u < fretU(n, K)) return { fret: n, reason: "on" };
  }
  return { fret: K + 1, reason: "beyond" }; // 보정 범위(프렛 K) 너머 → 지판 밖 취급
}

// v(0=6번줄..5=1번줄) → 줄번호(6..1). 지판 밖이면 null.
function vToString(v) {
  if (!Number.isFinite(v)) return null;
  const s = Math.round(v);
  if (v < -0.45 || v > 5.45 || s < 0 || s > 5) return null;
  return 6 - s;
}

// 모든 손의 운지 손끝을 지판좌표로 변환. onBoard=지판 위 유효 위치.
function mapFingertips(hands, results, H, t) {
  const out = [];
  if (!hands || !hands.length || !H) return out;
  const handedness = (results && (results.handednesses || results.handedness)) || [];
  hands.forEach((landmarks, hi) => {
    const cat = handedness[hi] && handedness[hi][0];
    const hand = cat ? cat.categoryName : null;
    for (const { idx, finger } of FRET_FINGERS) {
      const lm = landmarks[idx];
      if (!lm) continue;
      // Phase 5: One-Euro 스무딩으로 떨림 제거(토글 ON일 때)
      const sm = els.smoothToggle.checked ? getSmoothedTip(hi, idx, lm, t) : lm;
      const disp = toDisplay(sm);
      const f = applyHomography(H.img2fret, disp.x, disp.y);
      const string = vToString(f.y);
      const fr = uToFret(f.x, H.K);
      const onBoard =
        string != null && fr.fret != null && fr.fret >= 1 && fr.fret <= H.K;
      out.push({ finger, hand, u: f.x, v: f.y, string, fret: fr.fret, onBoard, disp });
    }
  });
  return out;
}

function drawFingerLabels(detected) {
  ctx.save();
  for (const d of detected) {
    if (!d.onBoard) continue;
    // 손끝 아래쪽에 "줄·프렛" 라벨(손가락 번호 위쪽 라벨과 겹치지 않게)
    drawLabel(
      `${d.finger}: ${d.string}번줄 ${d.fret}프렛`,
      d.disp.x,
      d.disp.y + 24,
      "#9dff9d",
      12,
      true
    );
  }
  ctx.restore();
}

// 우측 패널 readout 갱신(변경 시에만 DOM 업데이트 → 레이아웃 thrash 방지)
function updateFingerReadout(detected) {
  let html;
  if (!homography) {
    html = '<li class="dim">지판 보정 후 손끝을 지판에 올리면 표시됩니다</li>';
  } else {
    const onBoard = detected
      .filter((d) => d.onBoard)
      .sort((a, b) => a.string - b.string || a.fret - b.fret);
    if (!onBoard.length) {
      html = '<li class="dim">지판 위에 손끝이 없습니다</li>';
    } else {
      html = onBoard
        .map(
          (d) =>
            `<li><b>${FINGER_NAMES[d.finger] || d.finger}</b> → ${d.string}번줄 ${d.fret}프렛</li>`
        )
        .join("");
    }
  }
  if (html !== lastReadoutHtml) {
    els.fingerReadout.innerHTML = html;
    lastReadoutHtml = html;
  }
}

// =============================================================
// Phase 4: 코드 운지 판정
// =============================================================
function voicingToString(chord) {
  const by = {};
  for (const v of chord.voicing) by[v.string] = v.fret;
  return [6, 5, 4, 3, 2, 1].map((s) => (by[s] === "x" ? "x" : String(by[s]))).join("");
}

// detected(Phase 3) 와 목표 코드 비교. 각 음을 correct/wrong_finger/wrong_position/missing 으로 분류.
//  options.capo: 카포 프렛(>0이면 운지 음을 그만큼 위로 이동). 바레(같은 손가락+프렛 2줄↑)도 처리.
function evaluateVoicing(detected, chord, options = {}) {
  const strictFinger = !!options.strictFinger;
  const tol = options.fretTolerance || 0;
  const requireMute = !!options.requireMute;
  const capo = Math.max(0, options.capo || 0);

  // 카포: fret>0 음을 capo만큼 위로 이동
  const voicing = chord.voicing.map((v) =>
    typeof v.fret === "number" && v.fret > 0 ? { ...v, fret: v.fret + capo } : v
  );

  const onBoard = detected.filter((d) => d.onBoard && d.string != null && d.fret != null);
  const byFinger = {}; // 손가락 번호별 대표 검출(첫 onBoard)
  for (const d of onBoard) if (!(d.finger in byFinger)) byFinger[d.finger] = d;
  const fingerAt = (string, fret) =>
    onBoard.find((d) => d.string === string && Math.abs(d.fret - fret) <= tol);

  const required = voicing.filter((v) => typeof v.fret === "number" && v.fret > 0);

  // 같은 손가락+프렛에 2줄 이상 → 바레로 묶는다(검출은 손끝 1점뿐이므로 프렛 일치로 판정)
  const groups = new Map();
  for (const r of required) {
    const key = `${r.finger}@${r.fret}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }

  const notes = [];
  let correctCount = 0;
  const used = new Set();

  for (const grp of groups.values()) {
    const isBarre = grp.length >= 2 && grp[0].finger != null;
    if (isBarre) {
      const fret = grp[0].fret;
      const finger = grp[0].finger;
      const d = byFinger[finger];
      const barreOk = d && Math.abs(d.fret - fret) <= tol;
      if (barreOk) used.add(d);
      for (const r of grp) {
        let status;
        if (barreOk) {
          status = "correct";
          correctCount++;
        } else status = d ? "wrong_position" : "missing";
        notes.push({ string: r.string, fret: r.fret, finger: r.finger, status, detected: barreOk ? d : d || null, barre: true });
      }
    } else {
      for (const r of grp) {
        const pos = fingerAt(r.string, r.fret);
        let status, det = null;
        if (pos) {
          det = pos;
          if (!strictFinger || pos.finger === r.finger) {
            status = "correct";
            correctCount++;
          } else status = "wrong_finger";
          used.add(pos);
        } else {
          const dI = byFinger[r.finger];
          if (dI) {
            status = "wrong_position";
            det = dI;
            used.add(dI);
          } else status = "missing";
        }
        notes.push({ string: r.string, fret: r.fret, finger: r.finger, status, detected: det, barre: false });
      }
    }
  }

  // extra: 개방(0)/뮤트('x')여야 하는 줄을 누르고 있는 손가락
  const stateByString = {};
  for (const v of voicing) stateByString[v.string] = v.fret;
  const extras = [];
  for (const d of onBoard) {
    if (used.has(d)) continue;
    const st = stateByString[d.string];
    if (st === 0) extras.push({ finger: d.finger, string: d.string, fret: d.fret, targetState: "open", det: d });
    else if (st === "x" && requireMute)
      extras.push({ finger: d.finger, string: d.string, fret: d.fret, targetState: "mute", det: d });
    // st>0 인데 미사용: 잘못 짚힌 여분 손가락 — 1차에선 무시(중복 플래그 방지)
  }

  const total = required.length;
  const isCorrect = correctCount === total && extras.length === 0;
  return {
    score: total ? correctCount / total : 1,
    correctCount,
    total,
    isCorrect,
    notes,
    extras,
    corrections: buildCorrections(notes, extras),
  };
}

function buildCorrections(notes, extras) {
  const out = [];
  const seenBarre = new Set();
  for (const n of notes) {
    const name = FINGER_NAMES[n.finger] || `${n.finger}번`;
    if (n.barre) {
      const key = `${n.finger}@${n.fret}`;
      if (seenBarre.has(key)) continue;
      seenBarre.add(key);
      if (n.status === "missing" || n.status === "wrong_position") {
        out.push(`${name}로 ${n.fret}프렛 전체를 바레(여러 줄 동시에 누르기)하세요`);
      }
      continue;
    }
    if (n.status === "wrong_position" && n.detected) {
      out.push(`${name}를 ${n.detected.string}번줄 ${n.detected.fret}프렛 → ${n.string}번줄 ${n.fret}프렛으로 옮기세요`);
    } else if (n.status === "missing") {
      out.push(`${name}로 ${n.string}번줄 ${n.fret}프렛을 짚으세요`);
    } else if (n.status === "wrong_finger" && n.detected) {
      const cur = FINGER_NAMES[n.detected.finger] || n.detected.finger;
      out.push(`${n.string}번줄 ${n.fret}프렛은 ${name}로 짚으세요 (지금 ${cur})`);
    }
  }
  for (const e of extras) {
    const name = FINGER_NAMES[e.finger] || e.finger;
    out.push(`${e.string}번줄은 ${e.targetState === "open" ? "개방현" : "뮤트"}이니 ${name}를 떼세요`);
  }
  return out;
}

function setJudgeHtml(html) {
  if (html !== lastJudgeHtml) {
    els.judgeBody.innerHTML = html;
    lastJudgeHtml = html;
  }
}

const STATUS_ICON = { correct: "✅", wrong_position: "⚠️", wrong_finger: "⚠️", missing: "❌" };

function renderJudgePanel(chordKey, res) {
  if (!chordKey || !CHORDS[chordKey]) {
    setJudgeHtml('<p class="dim">목표 코드를 선택하세요</p>');
    return;
  }
  const chord = CHORDS[chordKey];
  const capo = Math.max(0, parseInt(els.capo.value, 10) || 0);
  const capoStr = capo > 0 ? ` · 카포 ${capo}` : "";
  const target = `<div class="judge-target">목표 <b>${chord.name}</b>${capoStr} · <code>${voicingToString(chord)}</code></div>`;

  if (!res) {
    setJudgeHtml(target + '<p class="dim">지판 보정 후 손끝을 올리면 판정됩니다</p>');
    return;
  }

  const okBadge = res.isCorrect
    ? confirmed
      ? ' <span class="ok">✅ 정확합니다! 🔒</span>'
      : ' <span class="ok">✅ 정확합니다!</span>'
    : "";
  const scoreHtml = `<div class="judge-score">정확도 ${res.correctCount}/${res.total}${okBadge}</div>`;

  const notesHtml =
    "<ul>" +
    res.notes
      .map((n) => {
        const name = FINGER_NAMES[n.finger] || n.finger;
        return `<li>${STATUS_ICON[n.status] || "•"} ${n.string}번줄 ${n.fret}프렛 <span class="dim">(${name})</span></li>`;
      })
      .join("") +
    "</ul>";

  let corrHtml = "";
  if (res.corrections.length) {
    corrHtml =
      '<div class="judge-corr"><h3>교정</h3><ul>' +
      res.corrections.map((c) => `<li>${c}</li>`).join("") +
      "</ul></div>";
  } else if (res.isCorrect) {
    corrHtml = '<div class="judge-corr ok">👍 완벽해요!</div>';
  }

  setJudgeHtml(target + scoreHtml + notesHtml + corrHtml);
}

// =============================================================
// Phase 5: 안정화(One-Euro 스무딩) + 교정 오버레이 + 음성/성공 효과
// =============================================================
class LowPass {
  constructor() { this.s = null; }
  filter(x, alpha) {
    this.s = this.s === null ? x : alpha * x + (1 - alpha) * this.s;
    return this.s;
  }
}
class OneEuro {
  constructor({ minCutoff = 1.7, beta = 0.4, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.tPrev = null;
    this.xLP = new LowPass();
    this.dxLP = new LowPass();
  }
  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, t) {
    if (this.tPrev === null) {
      this.tPrev = t;
      this.xPrev = x;
      this.xLP.filter(x, 1);
      return x;
    }
    let dt = t - this.tPrev;
    if (!(dt > 0)) dt = 1e-3;
    this.tPrev = t;
    const dx = (x - this.xPrev) / dt;
    this.xPrev = x;
    const edx = this.dxLP.filter(dx, this.alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xLP.filter(x, this.alpha(cutoff, dt));
  }
}
const ONEEURO_OPTS = { minCutoff: 1.7, beta: 0.4, dCutoff: 1.0 };
const tipFilters = new Map(); // "handIndex:landmarkIdx" → {x:OneEuro, y:OneEuro, seen}
let smoothFrameId = 0;
function getSmoothedTip(handIndex, idx, lm, t) {
  const key = handIndex + ":" + idx;
  let f = tipFilters.get(key);
  if (!f) {
    f = { x: new OneEuro(ONEEURO_OPTS), y: new OneEuro(ONEEURO_OPTS), seen: smoothFrameId };
    tipFilters.set(key, f);
  }
  f.seen = smoothFrameId;
  return { x: f.x.filter(lm.x, t), y: f.y.filter(lm.y, t) };
}

// 지판(줄,프렛) → 표시 픽셀(프렛 공간 중앙)
function pressPoint(string, fret, H) {
  const u = (fretU(fret - 1, H.K) + fretU(fret, H.K)) / 2;
  const v = 6 - string;
  return applyHomography(H.fret2img, u, v);
}

const COLOR_OK = "#36d399";
const COLOR_WARN = "#ffd166";
const COLOR_BAD = "#ff6b6b";

function drawRing(x, y, r, color) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.lineWidth = Math.max(2.5, canvas.width * 0.004);
  ctx.strokeStyle = color;
  ctx.stroke();
}
function drawDashedTarget(x, y, color, label) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const r = Math.max(10, canvas.width * 0.016);
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = Math.max(2, canvas.width * 0.0035);
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
  if (label) drawLabel(label, x, y, color, 13, true);
}
function drawArrow(x1, y1, x2, y2, color) {
  if (![x1, y1, x2, y2].every(Number.isFinite)) return;
  const head = Math.max(7, canvas.width * 0.011);
  const ang = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(2.5, canvas.width * 0.004);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
  ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawCorrectionOverlay(judge, H) {
  ctx.save();
  const tipStatus = new Map();
  for (const n of judge.notes) {
    const P = pressPoint(n.string, n.fret, H);
    if (n.status === "correct") {
      if (n.detected) tipStatus.set(n.detected, COLOR_OK);
    } else if (n.status === "wrong_finger") {
      if (n.detected) tipStatus.set(n.detected, COLOR_WARN); // 위치는 맞음 → 화살표 없이 경고
    } else if (n.status === "wrong_position") {
      if (n.detected) {
        tipStatus.set(n.detected, COLOR_WARN);
        drawArrow(n.detected.disp.x, n.detected.disp.y, P.x, P.y, COLOR_WARN);
      }
      drawDashedTarget(P.x, P.y, COLOR_WARN, FINGER_NAMES[n.finger]);
    } else if (n.status === "missing") {
      drawDashedTarget(P.x, P.y, COLOR_BAD, FINGER_NAMES[n.finger]);
    }
  }
  for (const e of judge.extras) if (e.det) tipStatus.set(e.det, COLOR_BAD);

  const r = Math.max(9, canvas.width * 0.012);
  for (const d of detectedFingers) {
    if (!d.onBoard) continue;
    drawRing(d.disp.x, d.disp.y, r, tipStatus.get(d) || "rgba(255,255,255,0.45)");
  }
  ctx.restore();
}

// ── 홀드 게이트 + 음성 코칭 + 성공 효과 ──
const HOLD_SEC = 0.7;
const VOICE_COOLDOWN = 3.0;
const VOICE_DEBOUNCE = 0.6;
let correctSince = null;
let confirmed = false;
let pendingMsg = null;
let pendingSince = 0;
let lastSpoken = "";
let lastSpokenTime = -999;
let audioCtx = null;

function updateHoldAndVoice(judge, t) {
  if (judge.isCorrect) {
    if (correctSince === null) correctSince = t;
    if (!confirmed && t - correctSince >= HOLD_SEC) {
      confirmed = true;
      onConfirmedSuccess(t);
    }
    pendingMsg = null;
  } else {
    correctSince = null;
    confirmed = false;
    if (els.voiceToggle.checked && judge.corrections.length) {
      const msg = judge.corrections[0];
      if (msg !== pendingMsg) {
        pendingMsg = msg;
        pendingSince = t;
      } else if (t - pendingSince >= VOICE_DEBOUNCE && msg !== lastSpoken && t - lastSpokenTime >= VOICE_COOLDOWN) {
        speak(msg);
        lastSpoken = msg;
        lastSpokenTime = t;
      }
    }
  }
}
function resetHold() {
  correctSince = null;
  confirmed = false;
  pendingMsg = null;
}
function onConfirmedSuccess(t) {
  stage.classList.add("flash-ok");
  setTimeout(() => stage.classList.remove("flash-ok"), 900);
  beep();
  if (els.voiceToggle.checked) {
    speak("정확합니다");
    lastSpoken = "정확합니다";
    lastSpokenTime = t;
  }
}
function speak(text) {
  try {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ko-KR";
    u.rate = 1.05;
    window.speechSynthesis.speak(u);
  } catch (e) { /* 무시 */ }
}
function beep() {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    if (!audioCtx) audioCtx = new AC();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = "sine";
    o.frequency.value = 880;
    const now = audioCtx.currentTime;
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);
    o.connect(g);
    g.connect(audioCtx.destination);
    o.start(now);
    o.stop(now + 0.26);
  } catch (e) { /* 무시 */ }
}

// =============================================================
// Phase 2: 보정 상태 머신 (시작/클릭/완료/취소/지우기/저장복원)
// =============================================================
// 보정 클릭 순서별 기대 지판좌표 생성
function buildCalibTargets(mode, K) {
  if (mode === "multi") {
    const t = [];
    for (let n = 0; n <= K; n++) {
      const u = fretU(n, K);
      t.push({ u, v: 0, n, side: 6 }); // 6번줄 끝
      t.push({ u, v: 5, n, side: 1 }); // 1번줄 끝
    }
    return t;
  }
  return [
    { u: 0, v: 0, n: 0, side: 6 },
    { u: 0, v: 5, n: 0, side: 1 },
    { u: K, v: 5, n: K, side: 1 },
    { u: K, v: 0, n: K, side: 6 },
  ];
}

function calibStepLabel(i) {
  const t = calib.targets[i];
  if (!t) return "";
  if (calib.mode === "corners") {
    const L = [
      "6번줄(가장 굵은 줄) × 너트(헤드 쪽 끝)",
      "1번줄(가장 얇은 줄) × 너트",
      `1번줄 × ${calib.K}번 프렛선`,
      `6번줄 × ${calib.K}번 프렛선`,
    ];
    return `${L[i]}이 만나는 점을 클릭`;
  }
  const fretName = t.n === 0 ? "너트(0프렛)" : `${t.n}번 프렛`;
  return `${fretName}선의 ${t.side}번줄 끝을 클릭`;
}

// 완료 토스트 타이머를 전환 시점마다 정리(재시작/취소/정지 시 stale 타이머가 배너를 숨기지 않게)
function clearCalibSuccessTimer() {
  if (calibSuccessTimer) {
    clearTimeout(calibSuccessTimer);
    calibSuccessTimer = null;
  }
}

function updateCalibBanner() {
  const n = calib.clickPts.length;
  const total = calib.targets.length;
  if (n >= total) return;
  els.calibBanner.classList.remove("success");
  els.calibBanner.textContent = `지판 보정 (${n + 1}/${total}) — ${calibStepLabel(n)}   ·   ESC: 취소`;
  els.calibBanner.hidden = false;
}

function startCalibration(mode) {
  if (!running) {
    showError("먼저 카메라를 시작한 뒤 보정하세요.");
    return;
  }
  if (calib.active) return cancelCalibration();
  clearCalibSuccessTimer();
  let K = parseInt(els.fretCount.value, 10);
  if (!Number.isFinite(K)) K = 5;
  K = Math.min(12, Math.max(3, K));
  els.fretCount.value = String(K);

  calib.active = true;
  calib.K = K;
  calib.mode = mode === "multi" ? "multi" : "corners";
  calib.clickPts = [];
  calib.targets = buildCalibTargets(calib.mode, K);
  els.fretCount.disabled = true; // 보정 중 K 변경 차단(스냅샷과 표시 불일치 방지)
  els.calibMode.disabled = true;
  stage.classList.add("calibrating");
  setOverlayMessage(null); // 큰 오버레이가 클릭을 가리지 않게
  els.calibBtn.textContent = "✕ 보정 취소";
  updateCalibBanner();
}

function onStageClick(e) {
  if (!calib.active) return;
  const c = clientToCanvas(e.clientX, e.clientY);
  if (!c) return;
  const t = calib.targets[calib.clickPts.length];
  if (!t) return;
  // 캔버스 내부 픽셀 → 원본 프레임 정규화(랜드마크와 동일 좌표계) + 기대 지판좌표(u,v)
  const xn = MIRROR ? 1 - c.x / canvas.width : c.x / canvas.width;
  const yn = c.y / canvas.height;
  calib.clickPts.push({
    x: Math.min(1, Math.max(0, xn)),
    y: Math.min(1, Math.max(0, yn)),
    u: t.u,
    v: t.v,
  });
  if (calib.clickPts.length >= calib.targets.length) finishCalibration();
  else updateCalibBanner();
}

function finishCalibration() {
  const candidate = { mode: calib.mode, K: calib.K, points: calib.clickPts.slice() };
  // 점들이 거의 일직선/중복이면 homography 특이 → 보정 거부하고 안내
  if (!computeHomographies(candidate)) {
    showError("보정 점들로 지판 격자를 만들 수 없습니다(거의 일직선/중복). 사각형 모서리에 가깝게 다시 찍어주세요.");
    cancelCalibration();
    return;
  }
  calib.saved = candidate;
  calib.active = false;
  calib.clickPts = [];
  els.fretCount.disabled = false;
  els.calibMode.disabled = false;
  stage.classList.remove("calibrating");
  els.calibBtn.textContent = "🎯 지판 보정 시작";
  saveCalib();
  updateCalibStatus();

  // 완료 토스트
  els.calibBanner.classList.add("success");
  els.calibBanner.textContent = "✅ 지판 보정 완료! 격자가 지판과 어긋나면 다시 보정하세요.";
  els.calibBanner.hidden = false;
  clearCalibSuccessTimer();
  calibSuccessTimer = setTimeout(() => {
    els.calibBanner.hidden = true;
    els.calibBanner.classList.remove("success");
    calibSuccessTimer = null;
  }, 2500);
}

function cancelCalibration() {
  clearCalibSuccessTimer();
  calib.active = false;
  calib.clickPts = [];
  els.fretCount.disabled = false;
  els.calibMode.disabled = false;
  stage.classList.remove("calibrating");
  els.calibBtn.textContent = "🎯 지판 보정 시작";
  els.calibBanner.hidden = true;
  els.calibBanner.classList.remove("success");
}

function clearCalibration() {
  calib.saved = null;
  homography = null;
  try { localStorage.removeItem(CALIB_KEY); } catch (e) { /* 무시 */ }
  updateCalibStatus();
}

function updateCalibStatus() {
  const ok = !!calib.saved;
  els.calibStatus.textContent = ok
    ? `됨 (${calib.saved.mode === "multi" ? "정밀" : "4점"}, K=${calib.saved.K})`
    : "안 됨";
  els.calibClearBtn.disabled = !ok;
}

function saveCalib() {
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(calib.saved)); } catch (e) { /* 무시 */ }
}

function loadCalib() {
  try {
    const s = localStorage.getItem(CALIB_KEY);
    if (!s) return;
    const o = JSON.parse(s);
    const k = Math.round(Number(o && o.K));
    const pointsOk =
      o &&
      Array.isArray(o.points) &&
      o.points.length >= 4 &&
      o.points.every((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.u) && Number.isFinite(p.v));
    // 신규 보정과 동일한 불변식(정수 3~12)을 요구하고, 실제로 homography가 만들어지는지까지 확인
    const candidate =
      pointsOk && k >= 3 && k <= 12
        ? { mode: o.mode === "multi" ? "multi" : "corners", K: k, points: o.points }
        : null;
    if (candidate && computeHomographies(candidate)) {
      calib.saved = candidate;
      els.fretCount.value = String(k);
    } else {
      try { localStorage.removeItem(CALIB_KEY); } catch (e) { /* 무시 */ }
    }
  } catch (e) {
    try { localStorage.removeItem(CALIB_KEY); } catch (e2) { /* 무시 */ }
  }
}

// =============================================================
// Phase 1: MediaPipe 모델 로드 (GPU 우선 → 실패 시 CPU 폴백)
// =============================================================
async function initHandLandmarker() {
  els.modelStatus.textContent = "로딩 중…";
  setOverlayMessage("AI 손 인식 모델을 불러오는 중입니다… (최초 1회, 수 초 소요)");

  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

  const baseConfig = {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, baseConfig);
  } catch (gpuErr) {
    console.warn("GPU delegate 실패 → CPU로 재시도:", gpuErr);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...baseConfig,
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
    });
  }

  modelReady = true;
  els.modelStatus.textContent = "준비됨";
  // 카메라가 아직 안 켜졌으면 시작 안내, 켜져 있으면 안내 숨김
  if (!running) {
    setOverlayMessage("아래 <b>카메라 시작</b>을 눌러주세요.");
  } else {
    setOverlayMessage(null);
  }
}

// =============================================================
// Phase 0: 카메라 시작 / 정지
// =============================================================
function buildConstraints(deviceId) {
  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
  };
  if (deviceId) videoConstraints.deviceId = { exact: deviceId };
  return { video: videoConstraints, audio: false };
}

async function startCamera(deviceId) {
  // 사전 점검: 보안 컨텍스트 / API 지원
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    showError("카메라는 보안 연결(https) 또는 localhost에서만 사용할 수 있습니다. file:// 로 직접 열지 말고 로컬 서버로 실행하세요. (README 참고)");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("이 브라우저는 카메라(getUserMedia)를 지원하지 않습니다. 최신 Chrome/Edge/Firefox 를 사용하세요.");
    return;
  }

  // 재진입(빠른 장치 전환 등) 방지용 세대 토큰. 이 호출보다 더 최신 호출이 시작되면 물러난다.
  const myToken = ++startToken;

  // 기존 스트림 정리(장치 전환 시)
  stopStreamTracks();

  setOverlayMessage("카메라 권한을 확인하는 중…");

  let newStream;
  try {
    newStream = await navigator.mediaDevices.getUserMedia(buildConstraints(deviceId));
  } catch (err) {
    if (myToken === startToken) {
      resetToStopped();
      handleGetUserMediaError(err);
    }
    return;
  }

  // 기다리는 동안 더 최신 startCamera가 시작됐다면, 방금 얻은 스트림을 정리하고 빠진다(누수 방지).
  if (myToken !== startToken) {
    newStream.getTracks().forEach((t) => t.stop());
    return;
  }

  stream = newStream;
  video.srcObject = stream;
  try {
    await video.play();
  } catch (playErr) {
    console.warn("video.play() 보류:", playErr);
  }
  if (myToken !== startToken) return;

  // 비디오 메타데이터(실제 해상도) 확보 후 캔버스 크기 동기화
  try {
    await waitForVideoReady();
  } catch (readyErr) {
    if (myToken === startToken) {
      resetToStopped();
      showError("카메라에서 영상 프레임을 받지 못했습니다. 다른 카메라를 선택하거나 장치를 확인한 뒤 다시 시도하세요.");
      els.camStatus.textContent = "오류";
      setOverlayMessage("아래 <b>카메라 시작</b>을 눌러 다시 시도하세요.");
    }
    return;
  }
  if (myToken !== startToken) return;

  syncCanvasSize();

  // 권한이 잡힌 뒤에야 장치 라벨이 보이므로 목록 갱신
  await populateCameraList();
  if (myToken !== startToken) return;

  els.camStatus.textContent = "동작 중";
  els.resStatus.textContent = `${video.videoWidth}×${video.videoHeight}`;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  els.hud.hidden = false;

  setOverlayMessage(modelReady ? null : "AI 손 인식 모델을 불러오는 중입니다…");

  // 렌더 루프 시작
  running = true;
  lastVideoTime = -1;
  if (rafId == null) rafId = requestAnimationFrame(renderLoop);
}

function stopStreamTracks() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

// 정지 상태로 완전히 되돌린다. 정상 종료/오류/장치전환 실패 등 모든 종료 경로가 이리로 수렴해
// 버튼·HUD·스트림·루프 상태가 항상 일관되게 유지된다. (오버레이 메시지는 호출부가 결정)
function resetToStopped() {
  if (calib.active) cancelCalibration();
  // 정지 시 남아있을 수 있는 완료 토스트/타이머도 정리
  clearCalibSuccessTimer();
  els.calibBanner.hidden = true;
  els.calibBanner.classList.remove("success");
  running = false;
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  stopStreamTracks();
  if (video.srcObject) video.srcObject = null;
  latestResults = null;
  lastVideoTime = -1;
  detectedFingers = [];
  lastReadoutHtml = "";
  els.fingerReadout.innerHTML = '<li class="dim">지판 보정 후 손끝을 지판에 올리면 표시됩니다</li>';
  homography = null;
  tipFilters.clear();
  resetHold();
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  lastJudgeHtml = "";
  renderJudgePanel(els.chordSelect.value, null);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  els.camStatus.textContent = "정지됨";
  els.resStatus.textContent = "--";
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.hud.hidden = true;
}

function stopCamera() {
  resetToStopped();
  setOverlayMessage(modelReady ? "아래 <b>카메라 시작</b>을 눌러주세요." : "AI 손 인식 모델을 불러오는 중입니다…");
}

function waitForVideoReady(timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 2 && video.videoWidth > 0) return resolve();

    let done = false;
    const cleanup = () => {
      done = true;
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("error", onError);
      clearTimeout(timer);
    };
    const onReady = () => {
      if (done) return;
      if (video.videoWidth > 0) {
        cleanup();
        resolve();
      }
    };
    const onError = () => {
      if (done) return;
      cleanup();
      reject(new Error("video element error"));
    };
    // 드물지만 프레임이 끝내 안 들어오는 장치/드라이버 문제 시 무한 대기 방지
    const timer = setTimeout(() => {
      if (done) return;
      cleanup();
      reject(new Error("video ready timeout"));
    }, timeoutMs);

    video.addEventListener("loadeddata", onReady);
    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("error", onError);
  });
}

function handleGetUserMediaError(err) {
  const name = err && err.name ? err.name : "";
  let msg;
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      msg = "카메라 권한이 거부되었습니다. 브라우저 주소창의 카메라 아이콘에서 권한을 허용한 뒤 다시 시도하세요.";
      break;
    case "NotFoundError":
    case "DevicesNotFoundError":
      msg = "사용 가능한 카메라를 찾지 못했습니다. 카메라 연결 상태를 확인하세요.";
      break;
    case "NotReadableError":
    case "TrackStartError":
      msg = "카메라를 다른 앱이 사용 중입니다. 줌/팀즈 등 다른 프로그램을 종료한 뒤 다시 시도하세요.";
      break;
    case "OverconstrainedError":
      msg = "요청한 카메라 설정을 만족하는 장치가 없습니다. 다른 카메라를 선택해보세요.";
      break;
    default:
      msg = `카메라를 시작하지 못했습니다: ${err && err.message ? err.message : name || "알 수 없는 오류"}`;
  }
  showError(msg);
  setOverlayMessage("아래 <b>카메라 시작</b>을 눌러 다시 시도하세요.");
  els.camStatus.textContent = "오류";
}

// =============================================================
// 캔버스 크기 동기화: 내부 해상도 = 실제 비디오 해상도
//   (CSS는 둘 다 무대를 가득 채우므로 표시 크기는 자동 일치)
// =============================================================
function syncCanvasSize() {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  }
}

// =============================================================
// 카메라 장치 목록
// =============================================================
async function populateCameraList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    const current = stream && stream.getVideoTracks()[0]
      ? stream.getVideoTracks()[0].getSettings().deviceId
      : "";

    els.cameraSelect.innerHTML = "";
    cams.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `카메라 ${i + 1}`;
      if (cam.deviceId === current) opt.selected = true;
      els.cameraSelect.appendChild(opt);
    });
    els.cameraSelect.disabled = cams.length <= 1;
  } catch (e) {
    console.warn("장치 목록 조회 실패:", e);
  }
}

// =============================================================
// 렌더 루프
// =============================================================
function renderLoop() {
  rafId = requestAnimationFrame(renderLoop);
  if (!running) return;

  // 비디오가 준비되지 않았으면 스킵
  if (video.readyState < 2 || video.videoWidth === 0) return;

  syncCanvasSize();

  // 같은 프레임은 재추론하지 않음(효율)
  if (modelReady && handLandmarker && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    // 타임스탬프는 단조 증가해야 함
    let ts = performance.now();
    if (ts <= lastDetectTs) ts = lastDetectTs + 1;
    lastDetectTs = ts;
    try {
      latestResults = handLandmarker.detectForVideo(video, ts);
    } catch (e) {
      console.warn("detectForVideo 오류:", e);
    }
  }

  // 그리기
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1) 지판 격자(보정됨 & 표시 ON) — 손 아래에 깔린다. homography는 Phase 3용으로 보관.
  if (calib.saved) {
    homography = computeHomographies(calib.saved);
    if (homography && els.gridToggle.checked) {
      drawFretGrid(homography.fret2img, homography.K);
    }
  } else {
    homography = null;
  }

  // 2) 손 골격
  const hands = latestResults && latestResults.landmarks ? latestResults.landmarks : [];
  if (els.skeletonToggle.checked) {
    drawHands(hands, latestResults);
  }

  // 3) Phase 3: 손끝 → 줄/프렛 매핑 (보정됨일 때만, Phase 5 스무딩 적용)
  const tSec = performance.now() / 1000;
  if (homography) {
    smoothFrameId++;
    detectedFingers = mapFingertips(hands, latestResults, homography, tSec);
    // 이번 프레임에 갱신되지 않은(사라진 손) 필터 정리
    for (const [k, f] of tipFilters) if (f.seen !== smoothFrameId) tipFilters.delete(k);
    if (els.fingerLabelToggle.checked) drawFingerLabels(detectedFingers);
  } else {
    detectedFingers = [];
    if (tipFilters.size) tipFilters.clear();
  }
  updateFingerReadout(detectedFingers);

  // 4) Phase 4/5: 코드 판정 + 교정 오버레이 + 안정화(홀드)/음성
  const chordKey = els.chordSelect.value;
  let judge = null;
  if (chordKey && CHORDS[chordKey] && homography) {
    judge = evaluateVoicing(detectedFingers, CHORDS[chordKey], {
      strictFinger: els.strictFingerToggle.checked,
      capo: Math.max(0, parseInt(els.capo.value, 10) || 0),
    });
  }
  if (judge) {
    drawCorrectionOverlay(judge, homography);
    updateHoldAndVoice(judge, tSec);
  } else {
    resetHold();
  }
  renderJudgePanel(chordKey, judge);

  // 5) 보정 진행 중 클릭 마커(맨 위)
  if (calib.active) drawCalibInProgress();

  updateHud(hands.length);
}

// =============================================================
// 손 그리기
// =============================================================
function drawHands(handsLandmarks, results) {
  ctx.save(); // ctx 상태(lineCap/strokeStyle 등)가 다음 프레임으로 누수되지 않게
  // 손마다 색 구분(왼손/오른손)
  const handedness = results && (results.handednesses || results.handedness) || [];

  handsLandmarks.forEach((landmarks, handIdx) => {
    const label = getHandLabel(handedness[handIdx]);
    const isLeft = label && label.categoryName === "Left";
    const boneColor = isLeft ? "#4cc9f0" : "#b388ff";

    // 1) 뼈대(연결선)
    ctx.lineWidth = Math.max(2, canvas.width * 0.004);
    ctx.strokeStyle = boneColor;
    ctx.lineCap = "round";
    for (const c of HAND_CONNECTIONS) {
      const a = toDisplay(landmarks[c.start]);
      const b = toDisplay(landmarks[c.end]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // 2) 관절 점
    const jointR = Math.max(2, canvas.width * 0.005);
    for (let i = 0; i < landmarks.length; i++) {
      if (FINGERTIPS[i]) continue; // 손끝은 아래에서 따로
      const p = toDisplay(landmarks[i]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, jointR, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
    }

    // 3) 손끝 강조 + 손가락 번호
    const tipR = Math.max(4, canvas.width * 0.009);
    for (const [idxStr, fingerNo] of Object.entries(FINGERTIPS)) {
      const i = Number(idxStr);
      const p = toDisplay(landmarks[i]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, tipR, 0, Math.PI * 2);
      ctx.fillStyle = "#ffd166";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      ctx.stroke();

      drawLabel(fingerNo, p.x, p.y - tipR - 4, "#ffd166", 11);
    }

    // 4) 손목 옆에 왼/오른손 라벨
    if (label) {
      const wrist = toDisplay(landmarks[0]);
      const pct = Math.round(label.score * 100);
      const text = `${label.categoryName === "Left" ? "왼손" : "오른손"} ${pct}%`;
      drawLabel(text, wrist.x, wrist.y + 18, boneColor, 13, true);
    }
  });
  ctx.restore();
}

function getHandLabel(categoryArray) {
  if (!categoryArray || !categoryArray.length) return null;
  // [{categoryName:'Left'|'Right', score, ...}]
  return categoryArray[0];
}

// 캔버스에 가독성 좋은 라벨(반투명 배경) 그리기
function drawLabel(text, x, y, color, fontSize, withBg) {
  ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  if (withBg) {
    const w = ctx.measureText(text).width + 12;
    const h = fontSize + 8;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    roundRect(x - w / 2, y - h / 2, w, h, 6);
    ctx.fill();
  }
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// =============================================================
// HUD (FPS / 손 개수)
// =============================================================
function updateHud(handCount) {
  fpsFrames++;
  const now = performance.now();
  const elapsed = now - fpsLast;
  if (elapsed >= 500) {
    fpsValue = Math.round((fpsFrames * 1000) / elapsed);
    fpsFrames = 0;
    fpsLast = now;
    els.fps.textContent = `FPS ${fpsValue}`;
  }
  els.handCount.textContent = `손 ${handCount}개`;
}

// =============================================================
// 이벤트 바인딩
// =============================================================
els.startBtn.addEventListener("click", () => {
  els.errorBox.hidden = true;
  const deviceId = els.cameraSelect.value || undefined;
  startCamera(deviceId);
});

els.stopBtn.addEventListener("click", stopCamera);

els.cameraSelect.addEventListener("change", () => {
  if (running) startCamera(els.cameraSelect.value || undefined);
});

els.mirrorToggle.addEventListener("change", () => {
  MIRROR = els.mirrorToggle.checked;
  video.classList.toggle("mirrored", MIRROR);
});

// Phase 2/6: 보정 컨트롤 (모드: 빠름 4점 / 정밀 프렛별)
els.calibBtn.addEventListener("click", () => startCalibration(els.calibMode.value));
els.calibClearBtn.addEventListener("click", clearCalibration);
stage.addEventListener("click", onStageClick);

// Phase 4/6: 코드 판정 컨트롤 (정지/일시정지 중에는 목표 코드만 표시)
els.chordSelect.addEventListener("change", () => renderJudgePanel(els.chordSelect.value, null));
els.strictFingerToggle.addEventListener("change", () => renderJudgePanel(els.chordSelect.value, null));
els.capo.addEventListener("change", () => renderJudgePanel(els.chordSelect.value, null));
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && calib.active) cancelCalibration();
});

// 창 크기 변화 시 캔버스 내부 해상도는 비디오 기준이라 그대로지만,
// 혹시 모를 비디오 해상도 변경(장치 전환 등) 대비해 동기화 한 번.
window.addEventListener("resize", syncCanvasSize);

// 뒤로가기/새로고침 폼 복원으로 체크박스 상태가 바뀌어도 미러 상태를 다시 맞춘다.
window.addEventListener("pageshow", () => {
  MIRROR = els.mirrorToggle.checked;
  video.classList.toggle("mirrored", MIRROR);
});

// 비디오 메타데이터가 늦게 도착해도 캔버스 동기화
video.addEventListener("loadedmetadata", syncCanvasSize);

// =============================================================
// 부팅: 모델을 미리 로드해 두고(카메라와 병렬), 실패해도 카메라는 동작
// =============================================================
// 저장된 지판 보정 복원
loadCalib();
updateCalibStatus();

initHandLandmarker().catch((err) => {
  console.error("모델 로드 실패:", err);
  els.modelStatus.textContent = "실패";
  showError("손 인식 모델을 불러오지 못했습니다. 인터넷 연결을 확인하세요. (카메라 영상은 계속 볼 수 있습니다.)");
  // 카메라가 이미 동작 중이면 오버레이로 화면을 덮지 않는다(영상은 계속 보여야 함).
  setOverlayMessage(running ? null : "아래 <b>카메라 시작</b>을 눌러주세요. (손 인식 없이 영상만 표시)");
});
