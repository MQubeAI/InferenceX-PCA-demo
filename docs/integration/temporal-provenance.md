# Temporal provenance and as-of boundary

The InferenceX research snapshot is `db-dump/2026-07-20`. Its aggregate rows
have representative benchmark dates through 2026-07-17. Epoch’s local raw
snapshot manifest was generated on 2026-08-20, after the InferenceX cutoff.

Every derived hardware descriptor retains Epoch release date, last-modified
time, source snapshot, source file, and provenance. The integrated view computes
`epoch_temporal_descriptor_status`: accepted mapped rows must at least satisfy
`epoch_release_date <= benchmark representative date`; all 7,168 accepted-map
rows do so. H100/H200 rows are `unmapped_or_ambiguous`.

Release-date eligibility is necessary but not sufficient to prove that the
specific Epoch field was publicly available on that date. Because the only
available Epoch source snapshot is newer than the InferenceX snapshot, the
hardware representation and held-out-SKU results are explicitly post-hoc
descriptor studies. They are not as-of historical prediction claims and they
do not establish future-hardware prediction.

Future historical experiments must use a source snapshot archived at or before
their declared cutoff (or field-level publication provenance), fit mappings and
representations only from that allowed information, and record the cutoff,
source hashes, entity resolution state, and training partitions. Future
benchmark/model observations must likewise use their evaluation/release dates,
not their current database appearance alone.
