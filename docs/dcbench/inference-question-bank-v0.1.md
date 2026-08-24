# Inference question bank v0.1

Generated from the frozen derived view and validation artifacts. Objective operations are not hidden chain-of-thought requirements.

## inference_v0_1_001 — direct_retrieval (easy)

**Question:** For the measured operating point with config_id 1, hardware h200, ISL 1024.0, OSL 1024.0, and concurrency 4, what throughput per GPU was reported?

**Gold answer:** 316.001301 tokens/s/GPU

**Why/evidence:** ['InferenceX July 20 aggregate'] — retrieve the exact aggregate row.

**Failure indicates:** retrieve, read metric failure.

## inference_v0_1_002 — direct_retrieval (easy)

**Question:** For the measured operating point with config_id 9, hardware mi355x, ISL 8192.0, OSL 1024.0, and concurrency 8, what throughput per GPU was reported?

**Gold answer:** 1756.174041 tokens/s/GPU

**Why/evidence:** ['InferenceX July 20 aggregate'] — retrieve the exact aggregate row.

**Failure indicates:** retrieve, read metric failure.

## inference_v0_1_003 — controlled_comparison (medium)

**Question:** Holding benchmark_type=agentic_traces, isl=nan, osl=nan, conc=1, model=dsv4, framework=sglang fixed, which of b300 and mi355x has higher measured throughput per GPU?

**Gold answer:** b300

**Why/evidence:** ['InferenceX July 20 aggregate'] — verify fixed variables → compare throughput.

**Failure indicates:** filter, compare failure.

## inference_v0_1_004 — quantitative_multirow_reasoning (medium)

**Question:** For the same controlled pair (b300 versus mi355x; benchmark_type=agentic_traces, isl=nan, osl=nan, conc=1, model=dsv4, framework=sglang), by what percent is the higher measured throughput per GPU greater than the lower?

**Gold answer:** 6.016%

**Why/evidence:** ['InferenceX July 20 aggregate'] — (higher/lower - 1) * 100.

**Failure indicates:** filter, calculate, compare failure.

## inference_v0_1_005 — configuration_tradeoff_reasoning (medium)

**Question:** Among these two measured candidates with benchmark_type=agentic_traces, isl=nan, osl=nan, conc=1, model=dsv4, framework=sglang, select the deployment maximizing throughput per GPU: b300 (1121.103) or mi355x (1057.489) tokens/s/GPU.

**Gold answer:** b300

**Why/evidence:** ['InferenceX July 20 aggregate'] — compare supplied measured candidates.

**Failure indicates:** retrieve, optimize failure.

## inference_v0_1_006 — planning_constrained_decision (hard)

**Question:** A demand of 10,000 output tokens/s must use one of two measured candidates with benchmark_type=agentic_traces, isl=nan, osl=nan, conc=1, model=dsv4, framework=sglang: b300 (1121.103 tokens/s/GPU) or mi355x (1057.489 tokens/s/GPU). Assuming throughput scales linearly only for this bookkeeping calculation, which requires fewer GPUs: ceil(demand / measured throughput per GPU)?

**Gold answer:** b300

**Why/evidence:** ['InferenceX July 20 aggregate'] — ceil(10000/a) → ceil(10000/b) → choose smaller.

**Failure indicates:** calculate, constrained_select failure.

## inference_v0_1_007 — research_result_interpretation (hard)

**Question:** In the controlled July Epoch-resolved cohort, did adding Epoch capabilities to hardware identity materially improve ordinary interpolation?

**Gold answer:** No; the difference was small and not material.

**Why/evidence:** ['controlled hardware validation'] — compare fixed-fold A and B → apply recorded interpretation.

**Failure indicates:** interpret_experiment failure.

## inference_v0_1_008 — research_result_interpretation (hard)

**Question:** What does near-equal capability-only and identity-only Random Forest accuracy support?

**Gold answer:** Physical capability descriptors preserve much of the predictive information; it does not prove reconstruction of GPU identity.

**Why/evidence:** ['controlled hardware validation'] — compare A/C → respect claim boundary.

**Failure indicates:** interpret_experiment, avoid_overclaim failure.

## inference_v0_1_009 — generalization_transfer_reasoning (hard)

**Question:** Which held-out SKU had the lowest capability-RF MAE?

**Gold answer:** mi325x

**Why/evidence:** ['held-out hardware transfer'] — compare held-out metrics.

**Failure indicates:** retrieve, rank failure.

## inference_v0_1_010 — generalization_transfer_reasoning (hard)

**Question:** Which held-out SKU had the highest capability-RF MdAPE?

**Gold answer:** mi355x

**Why/evidence:** ['held-out hardware transfer'] — compare held-out metrics.

**Failure indicates:** retrieve, rank failure.

## inference_v0_1_011 — support_uncertainty_reasoning (hard)

**Question:** Does the seven-SKU capability-distance versus MdAPE association establish a calibrated support threshold for future hardware?

**Gold answer:** No; it is descriptive across only seven SKUs and does not define a threshold.

**Why/evidence:** ['held-out hardware transfer'] — interpret descriptive Spearman evidence.

**Failure indicates:** interpret_statistics, avoid_overclaim failure.

## inference_v0_1_012 — planning_constrained_decision (hard)

**Question:** A planner wants a hardware-transfer case with an interpolation-like descriptor profile and the lower of the two such MdAPEs. Which SKU should it choose: GB200 or MI325X?

**Gold answer:** MI325X

**Why/evidence:** ['held-out hardware transfer'] — identify interpolation-like records → compare MdAPE.

**Failure indicates:** filter, compare, constrained_select failure.
