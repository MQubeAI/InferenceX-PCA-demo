# DC Bench inference/performance track specification

Categories: A direct retrieval; B controlled comparison; C quantitative
multi-row reasoning; D configuration tradeoff; E research-result
interpretation; F generalization/transfer; G support/uncertainty; H constrained
planning. Items must have unique, mechanically reproducible gold answers.

Answerability modes are `public_or_general`, `table_only`, `multirow_reasoning`,
`project_analysis`, `model_supported`, and `unsupported_current_system`.
Scored v0.1 uses table/multirow/project modes only. Evidence must preserve metric
units; fixed-variable comparisons must document their filter; unresolved maps,
missing physical fields, and arbitrary prediction requests are rejected.
