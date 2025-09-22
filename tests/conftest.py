# tests/conftest.py
"""
Este archivo asegura que los módulos dentro de 'src' se puedan importar
correctamente en los tests, tanto en ejecución local como en CI.
"""

import sys
import os
from pathlib import Path

# Ruta al directorio raíz del proyecto (donde está la carpeta 'src')
ROOT_DIR = Path(__file__).resolve().parent.parent / "recomendador_sentimientos_amazon"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print(f"[conftest] PYTHONPATH ajustado, ROOT_DIR añadido: {ROOT_DIR}")
