export const CONTACT_STATES = {
  PRESSED: "pressed",
  HOVER: "hover",
  LIFTED: "lifted",
  UNKNOWN: "unknown",
};

const DEFAULTS = {
  pressedThreshold: 0.68,
  liftedThreshold: 0.35,
  stableVelocity: 0.08,
  movingVelocity: 0.3,
};

export function normalize01(value, min, max, invert = false) {
  if (!Number.isFinite(value) || !Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return invert ? 1 : 0;
  }
  const raw = Math.max(0, Math.min(1, (value - min) / (max - min)));
  return invert ? 1 - raw : raw;
}

export function computeFingerContactFeatures(input, options = {}) {
  const opts = { ...DEFAULTS, ...options };
  const mapped = input?.mapped || {};
  const reasonCodes = [];

  if (mapped.onBoard !== true) {
    return {
      cellConfidence: 0,
      curlConfidence: 0,
      stabilityConfidence: 0,
      depthConfidence: 0.5,
      velocityNormPerSec: Infinity,
      reasonCodes: ["off_board"],
    };
  }
  reasonCodes.push("on_board");

  const fretCenter = Number.isFinite(mapped.fret) ? mapped.fret - 0.5 : mapped.u;
  const uDistance = Math.abs((mapped.u ?? fretCenter) - fretCenter);
  const vCenter = Number.isFinite(mapped.string) ? 6 - mapped.string : mapped.v;
  const vDistance = Math.abs((mapped.v ?? vCenter) - vCenter);
  const cellConfidence = clamp01(0.55 * normalize01(uDistance, 0.5, 0, false) + 0.45 * normalize01(vDistance, 0.45, 0, false));
  if (cellConfidence >= 0.7) reasonCodes.push("near_cell_center");

  const posture = input?.posture || {};
  const pip = posture.pipAngle;
  const dip = posture.dipAngle;
  let curlConfidence = 0.5;
  if (Number.isFinite(pip) || Number.isFinite(dip)) {
    const pipScore = Number.isFinite(pip) ? normalize01(pip, 178, 145, false) : 0.5;
    const dipScore = Number.isFinite(dip) ? normalize01(dip, 176, 150, false) : 0.5;
    curlConfidence = clamp01((pipScore + dipScore) / 2);
  }
  if (posture.risk) curlConfidence *= 0.55;
  if (curlConfidence >= 0.65) reasonCodes.push("curled_finger");
  if (posture.risk) reasonCodes.push("flat_finger");

  const velocityNormPerSec = computeVelocity(input?.previousTip, input?.currentTip);
  const stabilityConfidence = Number.isFinite(velocityNormPerSec)
    ? normalize01(velocityNormPerSec, opts.movingVelocity, opts.stableVelocity, false)
    : 0.5;
  if (stabilityConfidence >= 0.7) reasonCodes.push("stable_tip");
  else reasonCodes.push("moving_tip");

  const depthConfidence = computeDepthConfidence(input);
  if (depthConfidence >= 0.65) reasonCodes.push("depth_contact_hint");

  return {
    cellConfidence,
    curlConfidence,
    stabilityConfidence,
    depthConfidence,
    velocityNormPerSec,
    reasonCodes,
  };
}

export function classifyFingerContact(input, options = {}) {
  const opts = { ...DEFAULTS, ...options };
  const mapped = input?.mapped || {};

  if (mapped.onBoard !== true) {
    return {
      state: CONTACT_STATES.LIFTED,
      confidence: 0.92,
      features: computeFingerContactFeatures(input, opts),
    };
  }

  if (!Number.isFinite(mapped.string) || !Number.isFinite(mapped.fret)) {
    return {
      state: CONTACT_STATES.UNKNOWN,
      confidence: 0.5,
      features: computeFingerContactFeatures(input, opts),
    };
  }

  const features = computeFingerContactFeatures(input, opts);
  const score =
    0.35 * features.cellConfidence +
    0.25 * features.curlConfidence +
    0.25 * features.stabilityConfidence +
    0.15 * features.depthConfidence;

  if (score >= opts.pressedThreshold) {
    return { state: CONTACT_STATES.PRESSED, confidence: clamp01(score), features };
  }
  if (score <= opts.liftedThreshold) {
    return { state: CONTACT_STATES.LIFTED, confidence: clamp01(1 - score), features };
  }
  return { state: CONTACT_STATES.HOVER, confidence: clamp01(score), features };
}

function computeVelocity(previousTip, currentTip) {
  if (!previousTip || !currentTip || !Number.isFinite(previousTip.t) || !Number.isFinite(currentTip.t)) {
    return Infinity;
  }
  const dt = currentTip.t - previousTip.t;
  if (!(dt > 0)) return Infinity;
  const dx = (currentTip.x ?? 0) - (previousTip.x ?? 0);
  const dy = (currentTip.y ?? 0) - (previousTip.y ?? 0);
  const dz = (currentTip.z ?? 0) - (previousTip.z ?? 0);
  return Math.hypot(dx, dy, dz) / dt;
}

function computeDepthConfidence(input) {
  const tip = input?.currentTip;
  const joints = input?.joints;
  const landmarks = input?.landmarks;
  if (!tip || !joints || !Array.isArray(landmarks)) return 0.5;
  const pip = landmarks[joints.pip];
  const dip = landmarks[joints.dip];
  if (!Number.isFinite(tip.z) || (!Number.isFinite(pip?.z) && !Number.isFinite(dip?.z))) return 0.5;
  const refZ = [pip?.z, dip?.z].filter(Number.isFinite).reduce((a, b, _, arr) => a + b / arr.length, 0);
  const delta = Math.abs(tip.z - refZ);
  return clamp01(0.45 + normalize01(delta, 0, 0.08) * 0.35);
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}
