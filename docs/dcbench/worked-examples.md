# Worked examples

## 1. Hardware identity versus physical descriptors

InferenceX records the accelerator as a category; Epoch adds memory, bandwidth,
precision compute, and TDP descriptors. On the fixed 7,168-row cohort, RF A
(identity) had R² 0.8377 / MAE 784.23 and RF C (capabilities only) 0.8375 /
768.20. This preserves much predictive information without the name; it does
not prove the model reconstructs identity.

## 2. Entire SKU withheld: GB300

All GB300 rows were excluded. Its 531 rows / 254 configs were predicted from
other SKUs; RF C MAE was 1211.18, MdAPE 25.03%, within-20% 44.63%, and R²
0.818. B300 was the nearest capability analogue (distance 0.425). No GB300
throughput entered training. This is held-out measured-hardware transfer, not
unreleased-hardware prediction.

## 3. Support warning: MI300X

MI300X still beat its tested transfer baselines (MAE 560.99) but had MdAPE
38.24%, within-20% 26.88%, two out-of-range capability dimensions, and only 20
test configs. It illustrates why a future system needs support/abstention rules
rather than emitting every hardware what-if.
