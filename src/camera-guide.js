export function evaluateCameraGuide(input, options = {}) {
  const checks = [];
  const videoWidth = input?.video?.videoWidth || 0;
  const videoHeight = input?.video?.videoHeight || 0;
  const canvasWidth = input?.canvasWidth || videoWidth || 1;
  const canvasHeight = input?.canvasHeight || videoHeight || 1;

  checks.push(checkVideoReady(videoWidth, videoHeight));
  checks.push(checkBrightness(input?.videoQuality));
  checks.push(checkContrast(input?.videoQuality));

  if (input?.homography) {
    checks.push(checkBoardSize(input.homography, canvasWidth, canvasHeight));
    checks.push(checkBoardCenter(input.homography, canvasWidth, canvasHeight));
    checks.push(checkBoardPerspective(input.homography));
  }

  checks.push(checkHandVisibility(input?.hands || [], input?.detectedFingers || []));
  checks.push(checkFps(input?.fps));

  const summary = summarizeCameraGuide(checks);
  return {
    ...summary,
    checks,
  };
}

export function summarizeCameraGuide(checks) {
  const badCount = checks.filter((c) => c.severity === "bad").length;
  const warnCount = checks.filter((c) => c.severity === "warn").length;
  const score = Math.max(0, Math.min(1, 1 - badCount * 0.28 - warnCount * 0.12));
  const level = badCount ? "poor" : warnCount ? "ok" : "good";
  const firstIssue = checks.find((c) => c.severity === "bad" || c.severity === "warn");
  const headline = firstIssue
    ? firstIssue.action
    : "카메라 배치가 안정적입니다.";
  return { score, level, headline };
}

export function sampleVideoQuality(video, options = {}) {
  if (!video || !video.videoWidth || !video.videoHeight || typeof document === "undefined") return null;
  const width = options.width || 160;
  const height = options.height || 90;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  let sum = 0;
  let sumSq = 0;
  let diff = 0;
  let prev = null;
  const count = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    const y = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    sum += y;
    sumSq += y * y;
    if (prev != null) diff += Math.abs(y - prev);
    prev = y;
  }
  const brightness = sum / count;
  const variance = Math.max(0, sumSq / count - brightness * brightness);
  return {
    brightness,
    contrast: Math.sqrt(variance),
    blurScore: Math.max(0, Math.min(1, diff / Math.max(1, count - 1) / 40)),
  };
}

function checkVideoReady(width, height) {
  if (!width || !height) {
    return {
      id: "video-ready",
      ok: false,
      severity: "bad",
      label: "카메라",
      message: "영상 프레임을 아직 받지 못했습니다.",
      action: "카메라를 시작하고 권한을 허용하세요.",
    };
  }
  return good("video-ready", "카메라", `${width}x${height} 영상이 들어오고 있습니다.`);
}

function checkBrightness(videoQuality) {
  if (!videoQuality) return warn("brightness", "밝기", "밝기 정보를 아직 계산하지 못했습니다.", "카메라가 켜진 뒤 잠시 기다리세요.");
  const b = videoQuality.brightness;
  if (b < 55) return warn("brightness", "밝기", `평균 밝기가 ${Math.round(b)}로 낮습니다.`, "조명을 켜거나 역광을 피하세요.");
  if (b > 220) return warn("brightness", "밝기", `평균 밝기가 ${Math.round(b)}로 높습니다.`, "강한 직사광이나 흰 배경 반사를 줄이세요.");
  return good("brightness", "밝기", "손과 지판을 구분할 밝기입니다.");
}

function checkContrast(videoQuality) {
  if (!videoQuality) return warn("contrast", "대비", "대비 정보를 아직 계산하지 못했습니다.", "카메라가 켜진 뒤 잠시 기다리세요.");
  const c = videoQuality.contrast;
  if (c < 18) return warn("contrast", "대비", `대비가 ${Math.round(c)}로 낮습니다.`, "지판과 손이 배경에서 구분되도록 배경을 바꾸세요.");
  return good("contrast", "대비", "손과 배경의 대비가 충분합니다.");
}

function checkBoardSize(homography, canvasWidth, canvasHeight) {
  const corners = getCorners(homography);
  const area = polygonArea(corners);
  const ratio = area / Math.max(1, canvasWidth * canvasHeight);
  if (ratio < 0.1) return bad("board-size", "지판 크기", `지판 영역이 화면의 ${Math.round(ratio * 100)}%입니다.`, "카메라를 더 가까이 두거나 기타를 화면 중앙으로 옮기세요.");
  if (ratio < 0.18) return warn("board-size", "지판 크기", `지판 영역이 화면의 ${Math.round(ratio * 100)}%입니다.`, "카메라를 조금 더 가까이 두세요.");
  if (ratio > 0.65) return warn("board-size", "지판 크기", `지판 영역이 화면의 ${Math.round(ratio * 100)}%입니다.`, "지판 전체가 보이도록 카메라를 조금 뒤로 빼세요.");
  return good("board-size", "지판 크기", `지판 영역이 화면의 ${Math.round(ratio * 100)}%입니다.`);
}

function checkBoardCenter(homography, canvasWidth, canvasHeight) {
  const corners = getCorners(homography);
  const center = corners.reduce((acc, p) => ({ x: acc.x + p.x / corners.length, y: acc.y + p.y / corners.length }), { x: 0, y: 0 });
  const dx = Math.abs(center.x - canvasWidth / 2) / canvasWidth;
  const dy = Math.abs(center.y - canvasHeight / 2) / canvasHeight;
  if (dx > 0.22 || dy > 0.22) return warn("board-center", "지판 중앙", "지판이 화면 중앙에서 벗어나 있습니다.", "기타 넥이 화면 중앙에 오도록 카메라나 기타를 옮기세요.");
  return good("board-center", "지판 중앙", "지판이 화면 중앙에 가깝습니다.");
}

function checkBoardPerspective(homography) {
  const [a, b, c, d] = getCorners(homography);
  const top = dist(a, b);
  const bottom = dist(d, c);
  const left = dist(a, d);
  const right = dist(b, c);
  const widthRatio = ratio(top, bottom);
  const heightRatio = ratio(left, right);
  if (widthRatio > 2.4 || heightRatio > 2.4) return warn("board-perspective", "정면 각도", "지판 원근 왜곡이 큽니다.", "카메라를 지판 정면에 가깝게 이동하세요.");
  return good("board-perspective", "정면 각도", "지판 각도가 판정 가능한 범위입니다.");
}

function checkHandVisibility(hands, detectedFingers) {
  if (!hands.length) return warn("hand-visibility", "손끝 보임", "손이 화면에 잡히지 않습니다.", "운지 손끝이 보이도록 손등이 가리지 않게 조정하세요.");
  const unclear = detectedFingers.filter((d) => d.contactState === "unknown" || d.contactState === "lifted").length;
  if (detectedFingers.length >= 4 && unclear >= 2) return warn("hand-visibility", "손끝 보임", "손끝 2개 이상이 안정적으로 보이지 않습니다.", "손등이 손끝을 가리지 않게 각도를 조정하세요.");
  return good("hand-visibility", "손끝 보임", "손끝이 추적되고 있습니다.");
}

function checkFps(fps) {
  if (Number.isFinite(fps) && fps < 15) return warn("fps", "FPS", `FPS가 ${Math.round(fps)}입니다.`, "다른 앱을 닫거나 해상도를 낮출 수 있는 카메라를 선택하세요.");
  return good("fps", "FPS", Number.isFinite(fps) ? `FPS ${Math.round(fps)}입니다.` : "FPS를 계산 중입니다.");
}

function getCorners(homography) {
  if (Array.isArray(homography?.corners)) return homography.corners;
  if (typeof homography?.fret2img === "function") return [homography.fret2img(0, 0), homography.fret2img(0, 5), homography.fret2img(homography.K, 5), homography.fret2img(homography.K, 0)];
  return [
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
  ];
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a.x * b.y - b.x * a.y;
  }
  return Math.abs(area) / 2;
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function ratio(a, b) {
  const min = Math.max(1e-6, Math.min(a, b));
  return Math.max(a, b) / min;
}

function good(id, label, message) {
  return { id, ok: true, severity: "good", label, message, action: message };
}

function warn(id, label, message, action) {
  return { id, ok: false, severity: "warn", label, message, action };
}

function bad(id, label, message, action) {
  return { id, ok: false, severity: "bad", label, message, action };
}
