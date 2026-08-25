# ACP Bench design notes

This design review used the official [IBM ACPBench repository](https://github.com/IBM/ACPBench).
ACPBench separates atomic reasoning tasks, supplies bounded state/context, uses
machine-known answers, and supports independently scoreable formats. DC Bench
reuses those principles: each item names its evidence contract, provenance,
answer type, gold derivation, and context mode.

It does not copy ACPBench's action-state semantics. Inference work concerns
measured operating points, comparability, arithmetic, support, and bounded
decisions—not action applicability or plan reachability. Therefore DC Bench
gold answers come from frozen table queries and analysis JSON, and planning
items select among supplied measured candidates rather than inventing a future
system state.
