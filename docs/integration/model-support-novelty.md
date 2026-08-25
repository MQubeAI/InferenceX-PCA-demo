# Future model support and novelty framework

An unseen model should eventually be labelled **SUPPORTED**, **WEAK SUPPORT**, or **UNSUPPORTED** from uncalibrated diagnostics, not invented thresholds. Candidate dimensions are total-parameter distance, active-parameter distance, dense/MoE mismatch, architecture-family mismatch, expert-count and active-expert novelty, layer and hidden-size distance, context-range violation, attention-architecture mismatch, and descriptor missingness pattern.

The current sources cannot calibrate this framework: all dimensions except sparse total parameters are missing as structured exact-version data. Missingness itself must not fingerprint a model identity or become a misleading feature.
