# ReliVE v2 differential-evidence policy

ReliVE v2 certifies a **calibrated dependence of model support on a frozen
visual evidence program**. It does not claim that an image intervention alone
proves a real-world claim true.

The historical strict label `STRICT_LOCAL_DEPENDENCE` remains diagnostic:
`ORIGINAL=SUPPORTED`, `KEEP_TARGET=SUPPORTED`, `DROP_TARGET=INSUFFICIENT`, and
every drop-control is `SUPPORTED`. It identifies a high-strength local subset,
but is not a necessary certificate condition: support can remain categorical
while its continuous score drops substantially.

For `K` controls, use conservative aggregation:

```
keep_control_worst = max(score(KEEP_MATCHED_CONTROL[k]))
drop_control_worst = min(score(DROP_MATCHED_CONTROL[k]))
g_keep = score(KEEP_TARGET) - keep_control_worst
g_drop = drop_control_worst - score(DROP_TARGET)
```

Automatic admission requires a calibration artifact bound to the policy and
passes its calibrated `tau_original`, `tau_keep_floor`, `tau_keep`, `tau_drop`,
`tau_evidence`, and `tau_operator` thresholds. The policy stores only symbols;
it never derives thresholds from pilot artifacts. Without calibration results
are `DIAGNOSTIC_ONLY`.

Automatic admission allows only `TIER_1_EXACT` controls (at least one) or
`TIER_2_MATCHED` controls (at least two). Every control must use the same
operator and frame/pixel modification extent, remain geometrically valid, and
avoid required evidence. Tier 2 additionally binds a policy/calibration
tolerance source and matches area curve, connectivity, foreground fraction,
salience, and occlusion. Tier 3, manual, random, unmatched, missing, or
silently downgraded controls are excluded.

Calibration, protocol development, and blind test remain full-video disjoint.
Blind truth cannot enter prompts, evidence selection, cache keys, or Adapt.
`POSTCONDITION_PERSISTENCE` additionally requires
`INTERACTION_END_LOCALIZED`, `POST_WINDOW_AVAILABLE`, and
`OPERATOR_SUPPORT_ABSENT` before it can enter automatic consideration.
Human-oracle provenance (`HUMAN_ORACLE`, `MANUAL_EVIDENCE`, `MANUAL_CONTROL`,
`TEST_GT`, `POST_HOC_EDIT`) taints the full chain and yields only
`ORACLE_DIAGNOSTIC_ONLY`; it cannot create automatic coverage or verification.

Adapt is bounded: each listed route at most once, one selected route per atomic
claim, R0 plus at most R1, and no R2. Nonunique routes abstain rather than
selecting the best candidate. Operator/control unavailability is `UNCERTAIN`.
