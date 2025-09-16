# tests/test_config.py
# -*- coding: utf-8 -*-
import os
from src.utils.config import load_settings

def test_load_settings_ok():
    cfg = load_settings()
    assert cfg.project.name == "recomendador_sentimientos_amazon"

    # Acepta 'random_seed' o 'seed' (según versión de Settings)
    seed = getattr(cfg, "random_seed", None)
    if seed is None:
        seed = getattr(cfg, "seed", None)
    if seed is None and hasattr(cfg, "project"):
        seed = getattr(cfg.project, "seed", None)
    assert isinstance(seed, int)

    assert isinstance(cfg.api.port, int)
    assert os.path.exists("settings.yaml")
    
