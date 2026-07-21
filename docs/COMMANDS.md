# Formal training and evaluation commands

The commands below document the official protocol. Replace paths enclosed
in angle brackets with local paths.

## Expected paths

```text
<DATASET_ROOT>/
  50/
    sample_0001/
    ...
  100/
  ...
  500/

<RAFT_CHECKPOINT>          pretrained raft-things checkpoint
<EXPERIMENT_ROOT>          output directory
```

## Run the complete experiment matrix

The orchestration script fixes the official seeds, split, checkpoint
criteria, RAFT mode, and loss-ablation coefficients.

### Retrained baselines

```bash
python pipeline_orchestrator.py baseline_train \
  --root_dir "<DATASET_ROOT>" \
  --experiment_root "<EXPERIMENT_ROOT>" \
  --raft_model "<RAFT_CHECKPOINT>"
```

```bash
python pipeline_orchestrator.py baseline_test \
  --root_dir "<DATASET_ROOT>" \
  --experiment_root "<EXPERIMENT_ROOT>" \
  --raft_model "<RAFT_CHECKPOINT>"
```

### Retrained loss ablations

```bash
python pipeline_orchestrator.py ablation_train \
  --root_dir "<DATASET_ROOT>" \
  --experiment_root "<EXPERIMENT_ROOT>" \
  --raft_model "<RAFT_CHECKPOINT>"
```

```bash
python pipeline_orchestrator.py ablation_test \
  --root_dir "<DATASET_ROOT>" \
  --experiment_root "<EXPERIMENT_ROOT>" \
  --raft_model "<RAFT_CHECKPOINT>"
```

## Explicit A0 example

### Training, seed 12345

```bash
python train_r2_variant.py \
  --root_dir "<DATASET_ROOT>" \
  --variant A0_full_socds \
  --run_label A0_full_socds \
  --output_dir "<EXPERIMENT_ROOT>/baseline_train/A0_full_socds/seed_12345" \
  --raft_model "<RAFT_CHECKPOINT>" \
  --flow_mode bidirectional \
  --raft_iters 4 \
  --train_start 1 \
  --train_end 30 \
  --val_start 31 \
  --val_end 35 \
  --frame_start 1 \
  --epochs 30 \
  --batch_size 1 \
  --workers 4 \
  --seed 12345 \
  --deterministic True \
  --checkpoint_metric stratum_mae \
  --early_stopping_patience 0 \
  --min_epochs 30 \
  --auto_resume True \
  --use_star_enhanced True \
  --fallback_to_raw True \
  --use_pretrained_frontend True
```

Repeat with seeds `23456` and `34567`, changing the output directory
accordingly.

### Testing the independently selected best checkpoint

```bash
python test_r2_variant.py \
  --root_dir "<DATASET_ROOT>" \
  --checkpoint "<EXPERIMENT_ROOT>/baseline_train/A0_full_socds/seed_12345/model_best.pth.tar" \
  --output_dir "<EXPERIMENT_ROOT>/baseline_test/A0_full_socds/seed_12345/best" \
  --raft_model "<RAFT_CHECKPOINT>" \
  --flow_mode bidirectional \
  --raft_iters 4 \
  --test_start 36 \
  --test_end 40 \
  --frame_start 1 \
  --workers 4 \
  --seed 12345 \
  --evaluation_label best \
  --evaluation_protocol official_independent_seed_best
```

## Checkpoint criterion for A5 and A6

The single projected-density and direct count-regression controls do not
produce distance-stratified outputs. Their training commands therefore
use:

```text
--checkpoint_metric total_mae
```

All other formal baselines and loss ablations use:

```text
--checkpoint_metric stratum_mae
```

## RAFT validation

```bash
python validate_raft_vs_simulator.py \
  --dataset_root "<DATASET_ROOT>" \
  --raft_model "<RAFT_CHECKPOINT>" \
  --output_dir "<EXPERIMENT_ROOT>/raft_validation" \
  --sample_start 36 \
  --sample_end 40 \
  --frame_start 1 \
  --use_star_enhanced True \
  --fallback_to_raw True \
  --raft_iters 4
```

The validation samples RAFT flow at each target's previous-frame
location and compares it with simulator-derived consecutive-frame pixel
displacement.
