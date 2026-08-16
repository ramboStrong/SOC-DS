from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

@dataclass(frozen=True)
class PartitionSpec:
    partition_id: str
    num_strata: int
    thresholds: tuple[float, ...]
    threshold_source: str
    description: str

    @property
    def stratum_names(self) -> List[str]:
        return [f's{i}' for i in range(self.num_strata)]

    def assign_depth(self, depth: float) -> int:
        import numpy as np
        return int(np.searchsorted(np.asarray(self.thresholds), float(depth), side='right'))

def load_partition_config(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Partition configuration not found: {path}')
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if int(payload.get('schema_version', -1)) != 1:
        raise ValueError(f"Unsupported partition-config schema: {payload.get('schema_version')}")
    if not payload.get('partitions'):
        raise ValueError('Partition configuration contains no partitions.')
    return payload

def get_partition_spec(config: Dict, partition_id: str) -> PartitionSpec:
    for item in config['partitions']:
        if str(item['partition_id']) == str(partition_id):
            num_strata = int(item['num_strata'])
            thresholds = tuple((float(x) for x in item['thresholds']))
            if len(thresholds) != num_strata - 1:
                raise ValueError(f'{partition_id}: expected {num_strata - 1} thresholds, got {len(thresholds)}')
            if any((b <= a for a, b in zip(thresholds[:-1], thresholds[1:]))):
                raise ValueError(f'{partition_id}: thresholds must be strictly increasing: {thresholds}')
            return PartitionSpec(partition_id=str(item['partition_id']), num_strata=num_strata, thresholds=thresholds, threshold_source=str(item.get('threshold_source', '')), description=str(item.get('description', '')))
    available = [str(x['partition_id']) for x in config['partitions']]
    raise KeyError(f'Unknown partition_id={partition_id!r}. Available: {available}')

def list_partition_ids(config: Dict) -> List[str]:
    return [str(item['partition_id']) for item in config['partitions']]

def partition_label_dir(sample_dir: str | Path, partition_id: str) -> Path:
    return Path(sample_dir) / 'partition_sensitivity' / partition_id / 'layer_density_maps'

def parse_csv_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(',') if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]
