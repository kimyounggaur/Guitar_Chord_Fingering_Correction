import assert from "node:assert/strict";
import {
  buildTargetMap,
  detectBarreRequirements,
  scoreFingeringPrior,
} from "../src/fingering-prior.js";

const F_CHORD = {
  name: "F",
  voicing: [
    { string: 6, fret: 1, finger: "1" },
    { string: 5, fret: 3, finger: "3" },
    { string: 4, fret: 3, finger: "4" },
    { string: 3, fret: 2, finger: "2" },
    { string: 2, fret: 1, finger: "1" },
    { string: 1, fret: 1, finger: "1" },
  ],
};

{
  const target = buildTargetMap(F_CHORD, 1);
  assert.equal(target.get("6").fret, 2);
  assert.equal(target.get("5").fret, 4);
}

{
  const barres = detectBarreRequirements(F_CHORD);
  assert.equal(barres.length, 1);
  assert.equal(barres[0].finger, "1");
  assert.equal(barres[0].fret, 1);
}

{
  const prior = scoreFingeringPrior({
    chord: F_CHORD,
    detectedFingers: [
      { finger: "1", string: 6, fret: 1, contactState: "pressed" },
      { finger: "3", string: 5, fret: 3, contactState: "pressed" },
      { finger: "4", string: 4, fret: 3, contactState: "pressed" },
      { finger: "2", string: 3, fret: 2, contactState: "pressed" },
    ],
    useContactClassifier: true,
    strictFinger: true,
  });
  assert.ok(prior.score > 0.75);
  assert.equal(prior.violations.some((v) => v.type === "barre_required_but_missing"), false);
}

{
  const prior = scoreFingeringPrior({
    chord: F_CHORD,
    detectedFingers: [
      { finger: "3", string: 5, fret: 3, contactState: "pressed" },
      { finger: "4", string: 4, fret: 3, contactState: "pressed" },
    ],
    useContactClassifier: true,
    strictFinger: true,
  });
  assert.ok(prior.violations.some((v) => v.type === "barre_required_but_missing"));
}

{
  const prior = scoreFingeringPrior({
    chord: F_CHORD,
    detectedFingers: [
      { finger: "1", string: 5, fret: 4, contactState: "pressed" },
      { finger: "4", string: 2, fret: 1, contactState: "pressed" },
    ],
    useContactClassifier: true,
    strictFinger: true,
  });
  assert.ok(prior.violations.some((v) => v.type === "finger_crossing"));
}

{
  const prior = scoreFingeringPrior({
    chord: F_CHORD,
    detectedFingers: [
      { finger: "1", string: 6, fret: 1, contactState: "hover" },
      { finger: "2", string: 3, fret: 1, contactState: "pressed" },
      { finger: "4", string: 2, fret: 7, contactState: "pressed" },
    ],
    useContactClassifier: true,
    strictFinger: true,
  });
  assert.ok(prior.score < 0.9);
  assert.ok(prior.violations.some((v) => v.type === "stretch"));
}
