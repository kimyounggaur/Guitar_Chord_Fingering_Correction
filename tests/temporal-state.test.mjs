import assert from "node:assert/strict";
import { ContactSmootherRegistry, ContactStateSmoother } from "../src/temporal-state.js";

{
  const sm = new ContactStateSmoother({ minPressMs: 120, minLiftMs: 160 });

  let res = sm.update({ state: "pressed", confidence: 0.9 }, 1.0);
  assert.notEqual(res.state, "pressed");

  res = sm.update({ state: "pressed", confidence: 0.9 }, 1.13);
  assert.equal(res.state, "pressed");
  assert.equal(res.rawState, "pressed");
  assert.ok(res.stableMs >= 120);

  res = sm.update({ state: "lifted", confidence: 0.9 }, 1.2);
  assert.equal(res.state, "pressed");

  res = sm.update({ state: "lifted", confidence: 0.9 }, 1.38);
  assert.equal(res.state, "lifted");
}

{
  const registry = new ContactSmootherRegistry({ minPressMs: 0 });
  registry.markFrame(1);
  assert.equal(registry.update("0:8", { state: "pressed", confidence: 0.9 }, 1).state, "pressed");
  registry.markFrame(4);
  registry.sweepInactive(2);
  assert.equal(registry.size, 0);
}
