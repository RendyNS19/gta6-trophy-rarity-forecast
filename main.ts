// gta6-trophy-rarity-forecast-01 — entry: pooled logit/log-time trend forecaster.
//
// Model, per trophy:
//   * Work on the logit scale (rates clamped to 0.01..99.99 first, as the scorer clamps).
//   * Fit a weighted least-squares line of logit(rate) against x = ln(1 + hours since the
//     game's release), with recent reads weighted more (exponential half-life). Exophase
//     rates evolve roughly multiplicatively in time since release, so log-time is the
//     natural axis: equal steps in x are equal *relative* steps in age.
//   * Pool across trophies: each trophy's slope is shrunk toward the cross-trophy median
//     slope, so a trophy whose own reads are sparse or flat borrows the field's trend.
//   * The projection is anchored at the trophy's most recent observed rate:
//         median_logit = logit(last rate) + shrunk_slope * (x_target - x_last)
//     so a flat history yields exactly persistence, and a moving history extrapolates
//     its own (pooled) trend over the horizon.
//   * Guardrails: a trophy whose observed logit range is negligible keeps slope 0;
//     the projected median never falls more than a small dilution dip below the last
//     rate (earner counts are monotone; only new-player dilution can pull a rate down);
//     medians are capped (97% ordinary, tighter for the platinum, ~99% absolute).
//   * Spread: sigma = sqrt( (S * sqrt(horizon_days))^2 + (K * |projected move|)^2 ),
//     floored and capped — the random-walk term prices clock risk, the move term prices
//     extrapolation risk, so a bold projection honestly widens its own band.
// Deterministic: every input comes from the case file itself (no wall clock, no RNG).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- tuning constants -------------------------------------------------------
const RECENCY_HALF_LIFE_H = 18; // weight of a read halves every 18 h of age
const SHRINK_K = 6; // pseudo-count pulling each slope toward the pooled slope
const FLAT_RANGE = 0.05; // logit range below which a history counts as flat
const MAX_DILUTION_DIP = 0.3; // most the median may sit below the last rate, logit units
const CAP_ORDINARY = 97; // median ceiling for a normal trophy, percent
const CAP_PLATINUM = 50; // median ceiling for the platinum, percent
const CAP_PINNED = 99.9; // ceiling for a trophy pinned at ~100% in every read
const CAP_ABSOLUTE = 99; // ceiling for everything else, percent
const SPREAD_S = 0.55; // random-walk spread, logit units per sqrt(day)
const SPREAD_K = 0.4; // extra spread per unit of projected logit move
const SIGMA_FLOOR = 0.08;
const SIGMA_CAP = 1.4;
const DEFAULT_RATE_PCT = 1; // last-resort rate when a case carries no usable read

interface TrophyMeta {
  id: number;
  tier?: string;
}
interface ReadRow {
  at: string;
  rates: Array<number | null>;
}
interface CaseFile {
  levels?: number[];
  horizon_minutes: number;
  target: string;
  game?: { release_date?: string };
  trophies: Array<TrophyMeta>;
  history: Array<ReadRow>;
}

// --- small numeric helpers --------------------------------------------------
const clampRate = (pct: number): number => Math.min(99.99, Math.max(0.01, pct));
const logitOf = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const pctOf = (logit: number): number => 100 / (1 + Math.exp(-logit));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/** Standard-normal quantile: exact constants at the contest's five levels, Acklam's
 *  rational approximation anywhere else. */
function zFor(tau: number): number {
  const known: Array<[number, number]> = [
    [0.05, -1.6448536269514722],
    [0.25, -0.6744897501960817],
    [0.5, 0],
    [0.75, 0.6744897501960817],
    [0.95, 1.6448536269514722],
  ];
  for (const [t, z] of known) if (Math.abs(tau - t) < 1e-12) return z;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  let q: number;
  let r: number;
  if (tau < pLow) {
    q = Math.sqrt(-2 * Math.log(tau));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (tau <= 1 - pLow) {
    q = tau - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - tau));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((x, y) => x - y);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Weighted least-squares slope of y on x; null when x has no weighted variance. */
function wlsSlope(xs: number[], ys: number[], ws: number[]): number | null {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let j = 0; j < xs.length; j++) {
    sw += ws[j];
    sx += ws[j] * xs[j];
    sy += ws[j] * ys[j];
  }
  if (sw <= 0) return null;
  const mx = sx / sw;
  const my = sy / sw;
  let num = 0;
  let den = 0;
  for (let j = 0; j < xs.length; j++) {
    num += ws[j] * (xs[j] - mx) * (ys[j] - my);
    den += ws[j] * (xs[j] - mx) * (xs[j] - mx);
  }
  if (den < 1e-12) return null;
  return num / den;
}

// --- the forecast for one case ----------------------------------------------
function forecast(c: CaseFile): Array<{ id: number; q: number[] }> {
  const levels = c.levels && c.levels.length > 0 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const zs = levels.map(zFor);
  const n = c.trophies.length;
  const hist = c.history;
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;

  // Time axis: log hours since release. A date-only release_date means midnight UTC
  // at the start of release day, the same origin the house curve baseline uses.
  let releaseMs = Date.parse(c.game?.release_date ?? '');
  if (Number.isNaN(releaseMs) && c.game?.release_date) {
    releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  }
  if (Number.isNaN(releaseMs)) {
    releaseMs = hist.length > 0 ? Date.parse(hist[0].at) : Date.parse(c.target);
  }
  const xAt = (ms: number): number => Math.log(1 + Math.max(0, (ms - releaseMs) / 3600000));
  const readMs = hist.map((r) => Date.parse(r.at));
  const readX = readMs.map(xAt);
  const targetX = xAt(Date.parse(c.target));
  const lastMs = readMs.length > 0 ? readMs[readMs.length - 1] : releaseMs;
  const weights = readMs.map((ms) => Math.pow(0.5, Math.max(0, (lastMs - ms) / 3600000) / RECENCY_HALF_LIFE_H));

  // Most recent observed rate per trophy (nulls skipped), and where it was seen.
  const lastRate: Array<number | null> = new Array(n).fill(null);
  const lastX: number[] = new Array(n).fill(targetX);
  for (let i = 0; i < n; i++) {
    for (let j = hist.length - 1; j >= 0; j--) {
      const r = hist[j].rates[i];
      if (r !== null && r !== undefined && Number.isFinite(r)) {
        lastRate[i] = r;
        lastX[i] = readX[j];
        break;
      }
    }
  }
  // Fallback for a trophy never observed: the cross-trophy 10th percentile of the
  // observed last rates (rare trophies are the ones that go missing).
  const observed = lastRate.filter((r): r is number => r !== null).sort((a, b) => a - b);
  const fallbackRate =
    observed.length > 0 ? observed[Math.min(observed.length - 1, Math.floor(0.1 * observed.length))] : DEFAULT_RATE_PCT;

  // Per-trophy weighted fits on the logit/log-time plane.
  const ownSlope: Array<number | null> = new Array(n).fill(null);
  const ownWeight: number[] = new Array(n).fill(0);
  const logitRange: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let j = 0; j < hist.length; j++) {
      const r = hist[j].rates[i];
      if (r === null || r === undefined || !Number.isFinite(r)) continue;
      const y = logitOf(r);
      xs.push(readX[j]);
      ys.push(y);
      ws.push(weights[j]);
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
    if (xs.length > 0) logitRange[i] = hi - lo;
    if (xs.length >= 3) {
      ownSlope[i] = wlsSlope(xs, ys, ws);
      ownWeight[i] = xs.length;
    }
  }
  const pooledSlope = medianOf(ownSlope.filter((s): s is number => s !== null && Number.isFinite(s)));

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < n; i++) {
    const base = lastRate[i] ?? fallbackRate;
    const baseLogit = logitOf(base);

    // Shrink the trophy's own slope toward the pooled slope; a flat history or a
    // trophy with no fit of its own falls back to (a share of) the pooled trend.
    let slope: number;
    if (logitRange[i] < FLAT_RANGE) {
      slope = 0; // no significant movement of its own: persistence
    } else if (ownSlope[i] !== null && Number.isFinite(ownSlope[i])) {
      slope = (ownWeight[i] * (ownSlope[i] as number) + SHRINK_K * pooledSlope) / (ownWeight[i] + SHRINK_K);
    } else {
      slope = pooledSlope;
    }
    if (!Number.isFinite(slope)) slope = 0;

    let medianLogit = baseLogit + slope * Math.max(0, targetX - lastX[i]);
    // Earner counts only grow; a projected fall is dilution at most, and a small one.
    if (medianLogit < baseLogit - MAX_DILUTION_DIP) medianLogit = baseLogit - MAX_DILUTION_DIP;

    let medianPct = pctOf(medianLogit);
    const pinned = base >= 99.9 && logitRange[i] < FLAT_RANGE;
    const cap = pinned ? CAP_PINNED : c.trophies[i].tier === 'platinum' ? CAP_PLATINUM : Math.min(CAP_ORDINARY, CAP_ABSOLUTE);
    if (medianPct > cap) medianPct = cap;
    if (medianPct < 0.01) medianPct = 0.01;
    medianLogit = logitOf(medianPct);

    const move = Math.abs(medianLogit - baseLogit);
    let sigma = Math.sqrt(Math.pow(SPREAD_S * Math.sqrt(horizonDays), 2) + Math.pow(SPREAD_K * move, 2));
    sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));

    const q: number[] = [];
    let prev = 0;
    for (const z of zs) {
      let v = round4(pctOf(medianLogit + z * sigma));
      if (v < prev) v = prev;
      if (v > 100) v = 100;
      q.push(v);
      prev = v;
    }
    rows.push({ id: c.trophies[i].id, q });
  }
  return rows;
}

// --- driver -------------------------------------------------------------------
mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const parsed = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const trophies = forecast(parsed);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
