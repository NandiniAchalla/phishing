from pathlib import Path
import pandas as pd

DATA_PATH = Path(r"E:\phishing-dataset")

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def build_dataset():
    print("Scanning dataset at:", DATA_PATH)

    rows = []

    html_files = list(DATA_PATH.rglob("*.html"))
    print("HTML files found:", len(html_files))

    for file in html_files:
        try:
            # folder like phishing_0001-0500 or not-phishing_0001-0500
            parent_folder = file.parents[1].name.lower()

            if "not-phishing" in parent_folder:
                label = 0
            elif "phishing" in parent_folder:
                label = 1
            else:
                continue  # skip unknown

            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            rows.append({
                "sample_id": file.parent.name,
                "label": label,
                "source_folder": parent_folder,
                "length": len(content)
            })

        except Exception:
            print("Skipped:", file)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH / "features.csv", index=False)

    print("Rows collected:", len(rows))
    print("Saved to:", OUTPUT_PATH / "features.csv")
    print("Dataset built successfully!")


if __name__ == "__main__":
    build_dataset()
