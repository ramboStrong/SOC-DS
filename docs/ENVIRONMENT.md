# Environment and dependencies

## Reference software environment

The formal SOC-DS experiments were conducted with:

- Python 3.9.18
- PyTorch 2.0.1 with CUDA 11.8
- A torchvision release compatible with PyTorch 2.0.1
- Blender 5.2.0 LTS for procedural scene generation
- The official RAFT implementation with a pretrained `raft-things` checkpoint

`environment.yml` provides a reference Conda environment, while
`requirements.txt` provides an equivalent pip-oriented specification.
These files describe a compatible reconstruction of the reported
environment rather than a platform-independent bitwise lock file.

## Hardware used for the reported experiments

- CPU: Intel Core i9-13900KF
- System memory: 32 GB
- GPU: NVIDIA GeForce RTX 4090 with 24 GB memory
- Operating system used for the formal runs: Windows

The reported inference times are hardware- and implementation-specific.

## External components

### Blender

Install Blender 5.2.0 LTS separately. The generator uses OpenGL viewport
rendering under `SOLID` shading rather than an offline physically based
rendering pipeline.

### RAFT

The RAFT source tree and pretrained checkpoint are external dependencies.
Place the RAFT package so that the following imports resolve:

```python
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
```

The formal protocol uses the pretrained `raft-things` checkpoint, four
update iterations, frozen RAFT parameters, and separate forward and
backward evaluations.

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate soc-ds
```

Pip:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
