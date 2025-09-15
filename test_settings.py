from src.utils.config import load_settings

cfg = load_settings()
print("Nombre del proyecto:", cfg.project.name)
print("Seed:", cfg.project.seed)
print("Ruta raw:", cfg.paths.data_raw)
print("Ruta processed:", cfg.paths.data_processed)
print("Modelo:", cfg.model.sentiment_model)
print("Usar CUDA:", cfg.model.use_cuda)
print("API:", f"{cfg.api.host}:{cfg.api.port}")
