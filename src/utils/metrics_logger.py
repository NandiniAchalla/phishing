import json
from pathlib import Path


def save_metrics(results: dict, filename: str):
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / filename

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
