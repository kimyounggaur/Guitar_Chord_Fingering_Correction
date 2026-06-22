import assert from "node:assert/strict";
import { classifyFingerContact, normalize01 } from "../src/contact-classifier.js";

const joints = { mcp: 5, pip: 6, dip: 7, tip: 8 };

function makeLandmarks({ tip = {}, pip = {}, dip = {}, mcp = {} } = {}) {
  const arr = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  arr[5] = { x: 0.4, y: 0.4, z: 0, ...mcp };
  arr[6] = { x: 0.42, y: 0.42, z: 0, ...pip };
  arr[7] = { x: 0.43, y: 0.45, z: 0, ...dip };
  arr[8] = { x: 0.44, y: 0.47, z: 0, ...tip };
  return arr;
}

assert.equal(normalize01(5, 0, 10), 0.5);
assert.equal(normalize01(0, 0, 10, true), 1);

{
  const res = classifyFingerContact({
    finger: "1",
    landmarks: makeLandmarks(),
    joints,
    mapped: { onBoard: false, string: null, fret: null, u: -1, v: -1 },
    currentTip: { x: 0.4, y: 0.4, z: 0, t: 1 },
  });
  assert.equal(res.state, "lifted");
  assert.ok(res.confidence >= 0.8);
}

{
  const res = classifyFingerContact({
    finger: "1",
    landmarks: makeLandmarks(),
    joints,
    mapped: { onBoard: true, string: 2, fret: 1, u: 1.5, v: 4.0 },
    previousTip: { x: 0.4, y: 0.4, z: 0, t: 1 },
    currentTip: { x: 0.405, y: 0.405, z: 0, t: 1.2 },
    posture: { pipAngle: 145, dipAngle: 150, risk: false },
  });
  assert.equal(res.state, "pressed");
  assert.ok(res.confidence >= 0.68);
}

{
  const res = classifyFingerContact({
    finger: "1",
    landmarks: makeLandmarks(),
    joints,
    mapped: { onBoard: true, string: 2, fret: 1, u: 1.5, v: 4.0 },
    previousTip: { x: 0.2, y: 0.2, z: 0, t: 1 },
    currentTip: { x: 0.42, y: 0.45, z: 0, t: 1.2 },
    posture: { pipAngle: 145, dipAngle: 150, risk: false },
  });
  assert.equal(res.state, "hover");
}

{
  const res = classifyFingerContact({
    finger: "1",
    landmarks: makeLandmarks(),
    joints,
    mapped: { onBoard: true, string: 2, fret: 1, u: 1.5, v: 4.0 },
    previousTip: { x: 0.4, y: 0.4, t: 1 },
    currentTip: { x: 0.405, y: 0.405, t: 1.2 },
    posture: { pipAngle: 178, dipAngle: 176, risk: true },
  });
  assert.notEqual(res.state, "pressed");
}
