# SOC-DS

Reproducibility materials for:

**SOC-DS: Distance-Stratified Space Density Estimation for Ground-Based
Optical Space-Object Population Assessment**

SOC-DS estimates three predefined camera-depth-stratified image-plane
density maps from two consecutive ground-based optical frames. The
reported study is a controlled procedural proof of concept. Its outputs
must not be interpreted as continuous three-dimensional physical density,
absolute range, orbit determination, object identification, or catalogue
maintenance.

## Current repository contents

| Resource | Description |
|---|---|
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Reference software, hardware, Blender, and RAFT dependencies |
| [`environment.yml`](environment.yml) | Reference Conda environment |
| [`requirements.txt`](requirements.txt) | Pip-oriented dependency specification |
| [`docs/DATA_GENERATION_PROTOCOL.md`](docs/DATA_GENERATION_PROTOCOL.md) | Camera, target motion, star field, labels, split, and corpus limitations |
| [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) | Official three-seed training, selection, and statistical protocol |
| [`configs/official_protocol.json`](configs/official_protocol.json) | Machine-readable formal experiment configuration |
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | Baseline, loss-ablation, testing, and RAFT-validation commands |
| [`manifests/dataset_seed_manifest.csv`](manifests/dataset_seed_manifest.csv) | Sanitized Blender and star-field seed manifest for 400 sequences |
| [`docs/RELEASE_POLICY.md`](docs/RELEASE_POLICY.md) | Current review-stage materials and planned full release |

## Protocol summary

- Procedural corpus: 400 sequences, 40,000 stored frames
- Nominal population scales: 50--500 in increments of 50
- Image resolution: 640 x 360
- Density/depth grid: 80 x 45
- Formal training seeds: 12345, 23456, 34567
- Training epochs: 30
- Optimizer: Adam, learning rate `1e-4`, weight decay `5e-4`
- Optical flow: frozen pretrained RAFT, four iterations, bidirectional mode
- Model selection:
  - stratified models: lowest validation stratum-wise MAE;
  - non-stratified controls: lowest validation total-count MAE.
- Main reporting: mean and standard deviation over independent seeds,
  paired sequence-level comparisons, stratified hierarchical-bootstrap
  confidence intervals, and paired Cohen's `d_z`.

## Important scope notes

- Camera-depth intervals are expressed in Blender world units, not
  kilometres.
- The corpus uses predefined depth strata and contains gaps around the
  decision thresholds.
- The generator contains controlled simulation-specific cues, including
  a scale-dependent motion prescription.
- The released frozen corpus defines the evaluated dataset.

## Quick start

Create the reference environment:

```bash
conda env create -f environment.yml
conda activate soc-ds
```

Review the machine-readable protocol:

```bash
python -m json.tool configs/official_protocol.json
```

Formal execution commands are documented in
[`docs/COMMANDS.md`](docs/COMMANDS.md).

## Release status

This revision-stage repository provides the complete experimental
protocol, dependency specification, seed manifest, and execution
commands. The full source code, frozen corpus access, analysis scripts,
and associated release artifacts will be deposited upon acceptance, as
described in [`docs/RELEASE_POLICY.md`](docs/RELEASE_POLICY.md).

## Citation

Citation metadata will be added when the article receives its final
bibliographic record.
