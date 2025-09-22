"""
src/utils/config.py
--------------------
Carga y mapeo de configuración del proyecto desde settings.yaml
"""

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ProjectConfig:
    name: str
    seed: int


@dataclass
class PathsConfig:
    data_raw: str
    data_processed: str


@dataclass
class ModelConfig:
    sentiment_model: str
    use_cuda: bool


@dataclass
class APIConfig:
    host: str
    port: int


@dataclass
class Settings:
    project: ProjectConfig
    paths: PathsConfig
    model: ModelConfig
    api: APIConfig


def load_settings(path: str = "settings.yaml") -> Settings:
    """
    Lee settings.yaml y devuelve un objeto Settings
    con todas las secciones tipadas.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return Settings(
        project=ProjectConfig(**cfg["project"]),
        paths=PathsConfig(**cfg["paths"]),
        model=ModelConfig(**cfg["model"]),
        api=APIConfig(**cfg["api"]),
    )
