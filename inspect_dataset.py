
from pathlib import Path

dataset_path = Path("dataset")

for split in ["train", "valid", "test"]:
    split_path = dataset_path / split

    print(f"\n--- {split.upper()} ---")

    for item in split_path.iterdir():
        print(item.name, "(directory)" if item.is_dir() else "(file)")

    images = list(split_path.rglob("*.jpg")) + \
             list(split_path.rglob("*.png"))

    labels = list(split_path.rglob("*.txt"))

    print("Images:", len(images))
    print("TXT annotations:", len(labels))
