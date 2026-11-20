[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/RendyNS19/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/RendyNS19/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A deterministic forecaster for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for each of Grand Theft Auto VI's 55 PlayStation 5 trophies it predicts five quantiles
(5/25/50/75/95%) of the trophy's earn rate on Exophase at the case's target time,
scored by pinball loss on the logit scale.

## The model

`main.ts` (Node 22, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` and writes `<id>.out` files to `$MAKEFUN_OUTPUT_DIR`.

Per trophy:

- Rates are clamped to 0.01–99.99 and moved to the **logit scale**.
- A **weighted least-squares line** fits logit(rate) against
  `ln(1 + hours since release)`, with recent reads weighted more
  (18-hour half-life) — earn rates evolve multiplicatively in the game's age,
  so log-time is the natural axis.
- Slopes are **pooled across trophies**: each trophy's slope is shrunk toward the
  cross-trophy median slope, so sparse or flat trophies borrow the field's trend.
- The projection is **anchored at the most recent observed rate**:
  `median = logit(last rate) + shrunk_slope × (x_target − x_last)`. A flat history
  therefore reduces to persistence, while a moving history extrapolates its trend.
- Guardrails: a trophy with negligible observed logit movement keeps slope 0; the
  median never falls more than a small dilution dip below the last rate; medians
  are capped (97% ordinary trophies, tighter for the platinum).
- The spread prices both clock risk and extrapolation risk:
  `sigma = sqrt((0.55·√horizon_days)² + (0.4·|projected logit move|)²)`,
  floored at 0.08 and capped at 1.4 logit units; quantiles are
  `inv_logit(median_logit + z_τ · sigma)` rounded to 4 decimals.

Fully deterministic — no wall clock, no randomness; every input comes from the
case file itself.
