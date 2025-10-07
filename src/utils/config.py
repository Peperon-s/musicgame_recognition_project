from dataclasses import dataclass
from typing import Any, Tuple
import yaml, os

@dataclass(frozen=True)
class PathCfg:
    frame_dir: str
    dataset_dir: str
    models_dir: str
    logs_dir: str

@dataclass(frozen=True)
class PreprocessorCfg:
    target_size: Tuple[int, int]
    center_crop_size: Any
    to_grayscale: bool
    mean: Any
    std: Any

@dataclass(frozen=True)
class RuntimeCfg:
    prefer_gpu_module: bool = True

@dataclass(frozen=True)
class Config:
    paths: PathCfg
    preprocessor: PreprocessorCfg
    runtime: RuntimeCfg

def load_config(path= "project/config/default.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    p = y["paths"]; pp = y["preprocessor"]; rt = y.get("runtime", {})
    return Config(
        paths=PathCfg(**p),
        preprocessor=PreprocessorCfg(**pp),
        runtime=RuntimeCfg(**rt),
    )
        