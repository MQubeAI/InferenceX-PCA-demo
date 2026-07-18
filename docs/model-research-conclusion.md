# Model research conclusion

## Final decision

- Select full-context TabFM for throughput (`metrics_tput_per_gpu`) research point estimates.
- Full-context grouped `config_id` evaluation: R2 **0.961979 +/- 0.008605** and MAE **338.540384**.
- Prefer conditional-scale split conformal as the single uncertainty research method; it is not an active production prediction service.
- Do not continue latency segmentation or residual modeling. Reject the median-TPOT two-stage approach. Do not implement VAE/CRVAE.
- No further expensive model runs are currently required.

## Evidence and calibration boundary

The throughput residual diagnostic finds strong conditional heteroskedasticity and non-Gaussian residuals, so a global Gaussian error bar is not appropriate. Conditional quantiles narrowly won at 50%, but conditional scale is selected for a consistent method across 50%, 80%, and 95% coverage.

The leakage-safe uncertainty evaluation deliberately reserves about half of each outer-training fold for TabFM context. Its point-model R2 is **0.913897**; this is not comparable to, and must not replace, the **0.961979** full-context result. Consequently its intervals are research-grade only and are not calibrated around the selected full-context point model.

At 95%, conditional scale achieved 95.34% empirical coverage, 2485.442 average width, and 4739.169 interval score, versus 8235.473 for global conformal. Its average width was 34.62% narrower than global conformal.

## Reporting policy

All subgroup rows remain in the artifact. Worst-subgroup undercoverage is ranked only among groups with at least 20 rows by default; the artifact records that threshold and how many smaller groups were excluded. This prevents one-row concurrency groups from becoming the headline failure.

## Latency decision

The median-TPOT two-stage candidates did not beat the global baseline consistently on aggregate and tail-focused error. Keep it as a weaker global research baseline only; do not pursue latency segmentation, a residual model, or another expensive latency run now.

## Artifact availability

- `full_context_throughput`: available
- `median_tpot_tail`: available
- `throughput_residuals`: available
- `throughput_uncertainty`: available
