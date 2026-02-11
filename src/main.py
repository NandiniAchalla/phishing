import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data_loader.build_dataset import build_dataset

if __name__ == "__main__":
    build_dataset()
