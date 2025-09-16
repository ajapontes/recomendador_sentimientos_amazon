from pathlib import Path
import sys
import pandas as pd

IN_PATH = Path("./data/in/amazon_reviews_2015.snappy.parquet")

def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {in_path}")

    df = pd.read_parquet(in_path)
    if "product_category" not in df.columns:
        raise ValueError("El parquet no contiene la columna 'product_category'.")

    # Conteo de categorías (top 50)
    vc = df["product_category"].value_counts(dropna=False)
    print("=== Top categorías por cantidad ===")
    print(vc.head(50))

    # Lista completa (únicos) por si quieres revisarla toda
    print("\n=== Total categorías únicas:", df["product_category"].nunique(), "===\n")
    # Muestra algunas en orden alfabético
    print(sorted(df["product_category"].unique())[:50])

if __name__ == "__main__":
    main()
