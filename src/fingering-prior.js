const FINGER_NAME = { "1": "검지", "2": "중지", "3": "약지", "4": "새끼" };

export function buildTargetMap(chord, capo = 0) {
  const map = new Map();
  for (const v of chord?.voicing || []) {
    const fret = typeof v.fret === "number" && v.fret > 0 ? v.fret + capo : v.fret;
    map.set(String(v.string), { ...v, fret });
  }
  return map;
}

export function detectBarreRequirements(chord, capo = 0) {
  const groups = new Map();
  for (const v of chord?.voicing || []) {
    if (!(typeof v.fret === "number" && v.fret > 0 && v.finger)) continue;
    const fret = v.fret + capo;
    const key = `${v.finger}@${fret}`;
    if (!groups.has(key)) groups.set(key, { finger: v.finger, fret, strings: [] });
    groups.get(key).strings.push(v.string);
  }
  return [...groups.values()].filter((g) => g.strings.length >= 2);
}

export function scoreFingeringPrior(input, options = {}) {
  const chord = input?.chord;
  const capo = Math.max(0, input?.capo || 0);
  const useContactClassifier = !!input?.useContactClassifier;
  const strictFinger = !!input?.strictFinger;
  const targetMap = buildTargetMap(chord, capo);
  const pressed = (input?.detectedFingers || []).filter((d) => {
    if (!Number.isFinite(d.string) || !Number.isFinite(d.fret)) return false;
    return !useContactClassifier || d.contactState === "pressed";
  });
  const hover = (input?.detectedFingers || []).filter((d) => d.contactState === "hover");

  const violations = [];
  const suggestions = [];
  const fingerAdjustments = {};
  for (const d of pressed) {
    fingerAdjustments[d.finger] = { confidenceMultiplier: 1 };
  }

  checkBarreRequirements(chord, capo, pressed, violations);
  checkOpenMuteExtras(targetMap, pressed, violations);
  checkSingleFinger(pressed, violations);
  checkFingerCrossing(pressed, violations);
  checkStretch(pressed, violations);
  checkStrictFinger(targetMap, pressed, strictFinger, violations);
  checkHover(hover, violations);

  for (const violation of violations) {
    const text = describePriorViolation(violation);
    if (text) suggestions.push(text);
    if (violation.finger && fingerAdjustments[violation.finger]) {
      fingerAdjustments[violation.finger].confidenceMultiplier *= violation.severity === "bad" ? 0.65 : 0.8;
    }
  }

  const score = Math.max(0, Math.min(1, 1 - violations.reduce((sum, v) => sum + (v.severity === "bad" ? 0.2 : 0.1), 0)));
  return {
    score,
    violations,
    fingerAdjustments,
    suggestions: [...new Set(suggestions)].slice(0, 4),
  };
}

export function describePriorViolation(violation) {
  switch (violation.type) {
    case "open_extra":
      return `${violation.string}번줄은 개방현이므로 ${nameOf(violation.finger)}를 떼세요.`;
    case "single_finger_conflict":
      return `${nameOf(violation.finger)}가 서로 다른 프렛을 동시에 누르는 것으로 보입니다.`;
    case "barre_required_but_missing":
      return `${nameOf(violation.finger)}로 ${violation.fret}프렛 바레를 잡아야 합니다.`;
    case "finger_crossing":
      return `${nameOf(violation.lowerFinger)}는 낮은 프렛 쪽, ${nameOf(violation.higherFinger)}는 높은 프렛 쪽에 두는 것이 자연스럽습니다.`;
    case "stretch":
      return `프렛 간격이 ${violation.span}칸입니다. 손가락을 더 가까운 운지로 모으세요.`;
    case "strict_finger":
      return `${violation.string}번줄 ${violation.fret}프렛은 ${nameOf(violation.expected)}로 짚으세요.`;
    case "hover_many":
      return "눌림이 불확실한 손가락이 많습니다. 손끝을 더 안정적으로 눌러주세요.";
    default:
      return "";
  }
}

function checkBarreRequirements(chord, capo, pressed, violations) {
  const barres = detectBarreRequirements(chord, capo);
  for (const barre of barres) {
    const hasFingerAtFret = pressed.some((d) => d.finger === barre.finger && d.fret === barre.fret);
    if (!hasFingerAtFret) {
      violations.push({
        type: "barre_required_but_missing",
        severity: "warn",
        finger: barre.finger,
        fret: barre.fret,
        strings: barre.strings,
      });
    }
  }
}

function checkOpenMuteExtras(targetMap, pressed, violations) {
  for (const d of pressed) {
    const target = targetMap.get(String(d.string));
    if (target?.fret === 0) {
      violations.push({ type: "open_extra", severity: "warn", finger: d.finger, string: d.string });
    }
  }
}

function checkSingleFinger(pressed, violations) {
  const byFinger = groupBy(pressed, (d) => d.finger);
  for (const [finger, rows] of byFinger) {
    const frets = [...new Set(rows.map((d) => d.fret))];
    if (frets.length > 1) {
      violations.push({ type: "single_finger_conflict", severity: "bad", finger, frets });
    }
  }
}

function checkFingerCrossing(pressed, violations) {
  const reps = [...groupBy(pressed, (d) => d.finger)]
    .map(([finger, rows]) => ({ finger, fret: Math.min(...rows.map((d) => d.fret)) }))
    .filter((r) => ["1", "2", "3", "4"].includes(r.finger));
  for (const a of reps) {
    for (const b of reps) {
      if (Number(a.finger) < Number(b.finger) && a.fret > b.fret + 1) {
        violations.push({
          type: "finger_crossing",
          severity: "warn",
          lowerFinger: a.finger,
          higherFinger: b.finger,
        });
      }
    }
  }
}

function checkStretch(pressed, violations) {
  if (pressed.length < 2) return;
  const frets = pressed.map((d) => d.fret).filter(Number.isFinite);
  const span = Math.max(...frets) - Math.min(...frets);
  if (span > 4) violations.push({ type: "stretch", severity: "warn", span });
}

function checkStrictFinger(targetMap, pressed, strictFinger, violations) {
  if (!strictFinger) return;
  for (const d of pressed) {
    const target = targetMap.get(String(d.string));
    if (target && typeof target.fret === "number" && target.fret === d.fret && target.finger && target.finger !== d.finger) {
      violations.push({
        type: "strict_finger",
        severity: "warn",
        finger: d.finger,
        expected: target.finger,
        string: d.string,
        fret: d.fret,
      });
    }
  }
}

function checkHover(hover, violations) {
  if (hover.length >= 1) {
    violations.push({ type: "hover_uncertain", severity: "warn" });
  }
  if (hover.length >= 2) {
    violations.push({ type: "hover_many", severity: "warn" });
  }
}

function groupBy(items, fn) {
  const map = new Map();
  for (const item of items) {
    const key = fn(item);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(item);
  }
  return map;
}

function nameOf(finger) {
  return FINGER_NAME[finger] || finger;
}
