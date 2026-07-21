# Release status and planned full release

## Materials available in the current repository

The revision-stage repository provides:

- the documented software environment,
- the procedural data-generation protocol,
- the official three-seed experiment configuration,
- formal training and evaluation commands,
- the sanitized 400-sequence seed manifest,
- a machine-readable protocol configuration.

These materials are intended to make the reported experimental design
auditable during peer review.

## Full release upon acceptance

Upon acceptance of the manuscript, the repository will be expanded with:

- the complete SOC-DS source code,
- baseline and loss-ablation model definitions,
- procedural generation and label-construction scripts,
- training, testing, statistical-analysis, and RAFT-validation scripts,
- the frozen corpus or durable data-access instructions,
- experiment configuration files and result tables,
- trained checkpoints where distribution size and hosting permit,
- checksums and a tagged release corresponding to the accepted article.

The external RAFT implementation and pretrained weights will be referenced
through their original distribution rather than redistributed without
permission.

## Versioning

The accepted-paper release will be tagged separately from the
revision-stage documentation so that the protocol reviewed by the journal
remains identifiable.
