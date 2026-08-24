# Question quality audit v0.1

All 12 records were rebuilt from frozen sources and inspected. Each has a unique
gold, existing evidence path, July snapshot, deterministic derivation, correct
unit where numeric, and no H100/H200 physical-spec dependency. Controlled items
use declared fixed fields; project-analysis items cite validation JSON. Evidence
does not include the gold calculation, while evidence mode supplies enough rows
or artifact references to isolate reasoning from retrieval.

| IDs | Disposition | Main capability / likely failure |
|---|---|---|
| 001–002 | KEEP | exact retrieval; unit/row selection |
| 003–006 | KEEP | fair filtering, arithmetic, constrained decision |
| 007–008 | KEEP | project interpretation / avoid overclaim |
| 009–010 | KEEP | transfer metric ranking |
| 011–012 | KEEP | support restraint / support-aware choice |

Summary: KEEP 12, REVISE 0, DROP 0. Categories: direct 2; comparison 1;
quantitative 1; tradeoff 1; planning 2; project interpretation 2; transfer 2;
support 1. Difficulty: easy 2, medium 3, hard 7. Modes: table-only 2,
multi-row 4, project-analysis 6. Generic public knowledge is insufficient for
all but broad conceptual language; closed book remains meaningful as abstention
and prior-knowledge control. Remaining concern: v0.1 is intentionally small.
