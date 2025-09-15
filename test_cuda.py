# Verificación básica de CUDA en PyTorch
# test_cuda.py
import torch
from transformers import pipeline

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Sentiment pipeline con aceleración si hay CUDA
device = 0 if torch.cuda.is_available() else -1
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
print(clf("This product is amazing, I loved it!"))
