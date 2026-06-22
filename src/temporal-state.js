const DEFAULTS = {
  pressEnterConfidence: 0.7,
  pressExitConfidence: 0.45,
  hoverEnterConfidence: 0.4,
  minPressMs: 120,
  minLiftMs: 160,
  maxMissingMs: 220,
};

export class ContactStateSmoother {
  constructor(options = {}) {
    this.options = { ...DEFAULTS, ...options };
    this.reset();
  }

  reset() {
    this.state = "unknown";
    this.rawState = "unknown";
    this.rawSince = null;
    this.stateSince = null;
    this.confidence = 0;
  }

  update(raw, tSec) {
    const now = Number.isFinite(tSec) ? tSec : 0;
    const rawState = raw?.state || "unknown";
    const rawConfidence = Number.isFinite(raw?.confidence) ? raw.confidence : 0;

    if (rawState !== this.rawState) {
      this.rawState = rawState;
      this.rawSince = now;
    } else if (this.rawSince === null) {
      this.rawSince = now;
    }

    let nextState = this.state;
    if (this.state === "unknown") nextState = "lifted";

    const rawMs = Math.max(0, (now - this.rawSince) * 1000);

    if (
      rawState === "pressed" &&
      rawConfidence >= this.options.pressEnterConfidence &&
      rawMs >= this.options.minPressMs
    ) {
      nextState = "pressed";
    } else if (
      this.state === "pressed" &&
      rawState === "lifted" &&
      rawMs >= this.options.minLiftMs
    ) {
      nextState = "lifted";
    } else if (
      this.state === "pressed" &&
      rawConfidence < this.options.pressExitConfidence &&
      rawMs >= this.options.minLiftMs
    ) {
      nextState = rawState === "hover" ? "hover" : "lifted";
    } else if (
      this.state !== "pressed" &&
      rawState === "hover" &&
      rawConfidence >= this.options.hoverEnterConfidence
    ) {
      nextState = "hover";
    } else if (rawState === "lifted" && this.state !== "pressed") {
      nextState = "lifted";
    }

    const changed = nextState !== this.state;
    if (changed || this.stateSince === null) this.stateSince = now;
    this.state = nextState;
    this.confidence = rawConfidence;

    return {
      state: this.state,
      confidence: rawConfidence,
      stableMs: Math.round(rawMs),
      rawState,
      changed,
    };
  }
}

export class ContactSmootherRegistry {
  constructor(options = {}) {
    this.options = options;
    this.items = new Map();
    this.frameId = 0;
  }

  get size() {
    return this.items.size;
  }

  markFrame(frameId) {
    this.frameId = frameId;
  }

  update(key, raw, tSec) {
    let entry = this.items.get(key);
    if (!entry) {
      entry = { smoother: new ContactStateSmoother(this.options), seen: this.frameId };
      this.items.set(key, entry);
    }
    entry.seen = this.frameId;
    return entry.smoother.update(raw, tSec);
  }

  sweepInactive(maxAgeFrames = 2) {
    for (const [key, entry] of this.items) {
      if (this.frameId - entry.seen > maxAgeFrames) this.items.delete(key);
    }
  }

  clear() {
    this.items.clear();
  }
}
