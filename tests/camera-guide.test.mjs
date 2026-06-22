import assert from "node:assert/strict";
import { evaluateCameraGuide, summarizeCameraGuide } from "../src/camera-guide.js";

{
  const guide = evaluateCameraGuide({
    video: { videoWidth: 0, videoHeight: 0 },
    canvasWidth: 1280,
    canvasHeight: 720,
    homography: null,
    hands: [],
    detectedFingers: [],
    fps: 30,
    videoQuality: null,
  });
  assert.equal(guide.level, "poor");
  assert.ok(guide.checks.some((c) => c.id === "video-ready" && c.severity === "bad"));
}

{
  const guide = evaluateCameraGuide({
    video: { videoWidth: 1280, videoHeight: 720 },
    canvasWidth: 1280,
    canvasHeight: 720,
    homography: {
      fret2img: { a: 1 },
      K: 6,
      corners: [
        { x: 340, y: 230 },
        { x: 940, y: 240 },
        { x: 950, y: 500 },
        { x: 330, y: 490 },
      ],
    },
    hands: [[{}]],
    detectedFingers: [
      { contactState: "pressed" },
      { contactState: "pressed" },
      { contactState: "hover" },
    ],
    fps: 29,
    videoQuality: { brightness: 122, contrast: 41, blurScore: 0.62 },
  });
  assert.notEqual(guide.level, "poor");
  assert.ok(guide.checks.some((c) => c.id === "board-size"));
  assert.ok(summarizeCameraGuide(guide.checks).headline.length > 0);
}

{
  const guide = evaluateCameraGuide({
    video: { videoWidth: 1280, videoHeight: 720 },
    canvasWidth: 1280,
    canvasHeight: 720,
    homography: null,
    hands: [],
    detectedFingers: [],
    fps: 10,
    videoQuality: { brightness: 30, contrast: 10, blurScore: 0.1 },
  });
  assert.ok(guide.checks.some((c) => c.id === "brightness" && c.severity !== "good"));
  assert.ok(guide.checks.some((c) => c.id === "fps" && c.severity === "warn"));
}
