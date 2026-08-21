# Regime-change detection and recurring memory

## Detection signal

Use the residual from a prediction made before the new target was available. A rising residual can indicate graph change, noise change, marginal shift, or model misspecification; it is not proof that a causal mechanism changed.

## Included detector

`PageHinkley` accumulates deviations above a running mean and triggers after a configurable threshold and minimum warm-up period.

Key settings: tolerance `delta`, threshold, minimum instances, and whether the detector is reset after an alarm.

## Comparison detector

ADWIN from River is a useful alternative. Compare false alarms, detection delay, missed changes, and runtime under identical residual streams.

## Response options

1. Reset or inflate RLS covariance while preserving graph coefficients.
2. Temporarily decrease the forgetting factor.
3. Store the pre-change graph in bounded memory.
4. Compare the new graph to stored graph prototypes.
5. Restore a previous regime only when doing so improves past-only validation evidence.

## Evaluation

Known synthetic change points allow tolerance-window precision/recall, detection delay, false positives per 1,000 observations, and post-alarm forecasting recovery. Real datasets without graph-change labels should report alarm behavior descriptively, not causal-change accuracy.
