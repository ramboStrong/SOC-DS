# Official experiment protocol

## Independent repetitions

All confirmatory baseline and loss-ablation experiments use three
independently trained runs with seeds:

```text
12345
23456
34567
```

The seed is applied to Python, NumPy, PyTorch CPU/CUDA, the DataLoader
generator, and worker-local random number generators. Deterministic
PyTorch algorithms are requested with warning-only fallback for
third-party operations without deterministic kernels.

## Input preprocessing

The main density network receives RGB images normalized with ImageNet
statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

RAFT receives the same star-enhanced RGB frames at 640 x 360 resolution,
converted from uint8 to float without ImageNet normalization.

## Optical flow

The formal protocol uses:

- a pretrained `raft-things` checkpoint,
- four RAFT update iterations,
- frozen RAFT parameters,
- separate forward flow from `I_(t-1)` to `I_t`,
- separate backward flow from `I_t` to `I_(t-1)`.

## Optimization

- Optimizer: Adam
- Learning rate: `1e-4`
- Weight decay: `5e-4`
- Epochs: 30
- Batch size: 1
- DataLoader workers: 4
- Learning-rate schedule: fixed
- Early stopping: disabled
- Frontend initialization: ImageNet pretrained where applicable

The full-model loss weights are:

```text
lambda_layer       = 1.0
lambda_total       = 0.2
lambda_consistency = 1.0
lambda_depth       = 0.1
```

The coefficients were chosen to balance the numerical magnitudes of the
individual loss components and were fixed before the formal multi-seed
experiments. For each loss-ablation variant, only the designated
coefficient is set to zero.

## Checkpoint selection

Checkpoint selection is performed independently for every trained seed:

- A0--A4 and B1--B4: lowest validation stratum-wise MAE;
- A5 and A6: lowest validation total-count MAE.

All 30 epochs are completed, and no test-set metric is used for model
selection.

## Dataset split

For each nominal population scale:

- training: sequences 1--30,
- validation: sequences 31--35,
- testing: sequences 36--40,
- first evaluated target frame: frame 1, paired with frame 0.

## Statistical reporting

Main metrics are summarized as the mean and sample standard deviation
over the three independently trained seeds.

Paired comparisons use the test sequence as the analysis unit.
Confidence intervals are obtained with a population-scale-stratified
hierarchical bootstrap. Paired Cohen's `d_z` is used as the standardized
effect size.

Differences are oriented so that positive values favour A0:

- error metrics: comparator minus A0;
- allocation diagonals: A0 minus comparator.

The machine-readable version of this protocol is available in
`configs/official_protocol.json`.
