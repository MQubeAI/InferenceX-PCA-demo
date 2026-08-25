# Inference question bank v0.2 candidates

These 28 mechanically verified items are a separate candidate pool created because no target frontier-model API or approved interface was available for v0.1 baseline collection. v0.1 remains frozen. Candidate selection still requires review after real baseline results exist.

## inference_v0_2_candidate_001 — direct_retrieval (easy)

**Question:** For the measured operating point with config_id 8, hardware b200, ISL 1024.0, OSL 1024.0, and concurrency 4, what throughput per GPU was reported?

**Gold answer:** 521.576279 tokens/s/GPU

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** retrieve the measured aggregate row

## inference_v0_2_candidate_002 — direct_retrieval (easy)

**Question:** For the measured operating point with config_id 286, hardware b300, ISL 1024.0, OSL 1024.0, and concurrency 8192, what throughput per GPU was reported?

**Gold answer:** 14773.406592 tokens/s/GPU

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** retrieve the measured aggregate row

## inference_v0_2_candidate_003 — direct_retrieval (easy)

**Question:** For the measured operating point with config_id 205, hardware gb200, ISL 1024.0, OSL 1024.0, and concurrency 512, what throughput per GPU was reported?

**Gold answer:** 3791.287432 tokens/s/GPU

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** retrieve the measured aggregate row

## inference_v0_2_candidate_004 — direct_retrieval (easy)

**Question:** For the measured operating point with config_id 2, hardware mi355x, ISL 1024.0, OSL 1024.0, and concurrency 4, what throughput per GPU was reported?

**Gold answer:** 444.285194 tokens/s/GPU

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** retrieve the measured aggregate row

## inference_v0_2_candidate_005 — controlled_comparison (medium)

**Question:** The evidence gives two measured candidates with all workload and serving-configuration fields fixed. Which of b200 and b300 has higher throughput per GPU?

**Gold answer:** b300

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** verify every listed fixed field → compare measured throughput per GPU

## inference_v0_2_candidate_006 — controlled_comparison (medium)

**Question:** The evidence gives two measured candidates with all workload and serving-configuration fields fixed. Which of b300 and mi355x has higher throughput per GPU?

**Gold answer:** b300

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** verify every listed fixed field → compare measured throughput per GPU

## inference_v0_2_candidate_007 — controlled_comparison (medium)

**Question:** The evidence gives two measured candidates with all workload and serving-configuration fields fixed. Which of b300 and gb300 has higher throughput per GPU?

**Gold answer:** b300

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** verify every listed fixed field → compare measured throughput per GPU

## inference_v0_2_candidate_008 — controlled_comparison (medium)

**Question:** The evidence gives two measured candidates with all workload and serving-configuration fields fixed. Which of gb200 and gb300 has higher throughput per GPU?

**Gold answer:** gb300

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** verify every listed fixed field → compare measured throughput per GPU

## inference_v0_2_candidate_009 — quantitative_multirow_reasoning (medium)

**Question:** For the two controlled measured candidates in the evidence, By what percent is the higher measured throughput per GPU greater than the lower?

**Gold answer:** 4.080%

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** (higher / lower - 1) * 100

## inference_v0_2_candidate_010 — quantitative_multirow_reasoning (medium)

**Question:** For the two controlled measured candidates in the evidence, What is the absolute throughput-per-GPU difference between the two measured candidates?

**Gold answer:** 5.346 tokens/s/GPU

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** higher throughput minus lower throughput

## inference_v0_2_candidate_011 — quantitative_multirow_reasoning (medium)

**Question:** For the two controlled measured candidates in the evidence, What is the ratio of higher to lower measured throughput per GPU?

**Gold answer:** 1.0623

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** higher throughput divided by lower throughput

## inference_v0_2_candidate_012 — quantitative_multirow_reasoning (medium)

**Question:** Rank the three controlled measured candidates in the evidence from highest to lowest throughput per GPU, using ` > ` between hardware labels.

**Gold answer:** b200 > b300 > mi355x

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** verify fixed variables → sort measured throughput per GPU descending

## inference_v0_2_candidate_013 — configuration_tradeoff_reasoning (medium)

**Question:** With every listed field except concurrency fixed, which listed concurrency has the highest measured throughput per GPU?

**Gold answer:** 512

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** hold all non-concurrency fields fixed → select the largest measured throughput

## inference_v0_2_candidate_014 — quantitative_multirow_reasoning (hard)

**Question:** With every listed field except concurrency fixed, by what percent does measured throughput per GPU change from concurrency 64 to 256?

**Gold answer:** 166.250%

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** (throughput at higher concurrency / throughput at lower concurrency - 1) * 100

## inference_v0_2_candidate_015 — planning_constrained_decision (hard)

**Question:** A demand of 1,000 output tokens/s must use one listed measured candidate. For bookkeeping only, assume linear scaling and choose the hardware requiring the fewest GPUs: ceil(demand / measured throughput per GPU).

**Gold answer:** b200

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** calculate ceil(demand / measured throughput per GPU) for each candidate → choose the smallest count

## inference_v0_2_candidate_016 — planning_constrained_decision (hard)

**Question:** A demand of 1,000 output tokens/s must use one listed measured candidate. For bookkeeping only, assume linear scaling and choose the hardware requiring the fewest GPUs: ceil(demand / measured throughput per GPU).

**Gold answer:** b300

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** calculate ceil(demand / measured throughput per GPU) for each candidate → choose the smallest count

## inference_v0_2_candidate_017 — planning_constrained_decision (hard)

**Question:** A demand of 500 output tokens/s must use one listed measured candidate. For bookkeeping only, assume linear scaling and choose the hardware requiring the fewest GPUs: ceil(demand / measured throughput per GPU).

**Gold answer:** b200

**Evidence:** ['InferenceX July 20 aggregate']

**Operations:** calculate ceil(demand / measured throughput per GPU) for each candidate → choose the smallest count

## inference_v0_2_candidate_018 — research_result_interpretation (hard)

**Question:** Which controlled Random Forest representation has the highest pooled out-of-fold R²?

**Gold answer:** B_identity_plus_capabilities

**Evidence:** ['controlled hardware validation']

**Operations:** compare pooled out-of-fold R² across A/B/C/D

## inference_v0_2_candidate_019 — research_result_interpretation (hard)

**Question:** Which controlled Random Forest representation has the lowest pooled out-of-fold MAE?

**Gold answer:** B_identity_plus_capabilities

**Evidence:** ['controlled hardware validation']

**Operations:** compare pooled out-of-fold MAE across A/B/C/D

## inference_v0_2_candidate_020 — research_result_interpretation (hard)

**Question:** Did adding matched precision peak compute to the capability-only representation materially improve ordinary interpolation?

**Gold answer:** No; D was slightly worse than C on both R² and MAE.

**Evidence:** ['controlled hardware validation']

**Operations:** compare C and D under the fixed protocol

## inference_v0_2_candidate_021 — research_result_interpretation (hard)

**Question:** Does the A/B/C/D result support claiming that Epoch capabilities materially improve ordinary interpolation?

**Gold answer:** No; the ordinary-interpolation differences are small and not material.

**Evidence:** ['controlled hardware validation']

**Operations:** compare A/B/C/D → respect recorded claim boundary

## inference_v0_2_candidate_022 — generalization_transfer_reasoning (hard)

**Question:** Which held-out measured SKU has the lowest capability-Random-Forest MAE?

**Gold answer:** mi325x

**Evidence:** ['held-out hardware transfer']

**Operations:** rank held-out capability-RF MAE

## inference_v0_2_candidate_023 — generalization_transfer_reasoning (hard)

**Question:** Which held-out measured SKU has the highest capability-Random-Forest MdAPE?

**Gold answer:** mi355x

**Evidence:** ['held-out hardware transfer']

**Operations:** rank held-out capability-RF MdAPE

## inference_v0_2_candidate_024 — generalization_transfer_reasoning (hard)

**Question:** For held-out GB300, which has lower MAE: capability Random Forest or the nearest hardware-capability analogue baseline?

**Gold answer:** capability Random Forest

**Evidence:** ['held-out hardware transfer']

**Operations:** compare the two GB300 MAEs

## inference_v0_2_candidate_025 — generalization_transfer_reasoning (hard)

**Question:** Does the held-out-SKU study support calling this unreleased-hardware prediction?

**Gold answer:** No; it is observed held-out-SKU transfer, not unreleased-hardware prediction.

**Evidence:** ['held-out hardware transfer']

**Operations:** read the claim boundary → avoid unsupported extrapolation

## inference_v0_2_candidate_026 — support_uncertainty_reasoning (hard)

**Question:** Which held-out SKU has the largest nearest-training-hardware normalized capability distance?

**Gold answer:** mi355x

**Evidence:** ['held-out hardware transfer']

**Operations:** rank normalized nearest-hardware distances

## inference_v0_2_candidate_027 — support_uncertainty_reasoning (hard)

**Question:** Between MI300X and MI325X, which has the higher capability-Random-Forest MdAPE despite a larger nearest-hardware distance?

**Gold answer:** MI300X

**Evidence:** ['held-out hardware transfer']

**Operations:** filter MI300X and MI325X → compare distance and MdAPE

## inference_v0_2_candidate_028 — support_uncertainty_reasoning (hard)

**Question:** Does the seven-SKU distance-versus-MdAPE association define a calibrated support threshold for arbitrary future hardware?

**Gold answer:** No; it is descriptive across seven SKUs and does not define a calibrated threshold.

**Evidence:** ['held-out hardware transfer']

**Operations:** interpret seven-SKU descriptive association → avoid threshold overclaim
