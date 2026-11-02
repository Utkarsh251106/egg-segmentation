
from pathlib import Path
from collections import Counter

dataset_path = Path("dataset")

for split in ["train", "valid", "test"]:

    print(f"\n{'='*40}")
    print(f"{split.upper()}")
    print(f"{'='*40}")

    split_path = dataset_path / split

    for folder in split_path.iterdir():

        if folder.is_dir():
            files = [f for f in folder.rglob("*") if f.is_file()]

            extensions = Counter(
                f.suffix.lower() for f in files
            )

            print(f"\nFolder: {folder.name}")
            print("Total files:", len(files))
            print("Extensions:", extensions)

            print("Sample files:")
            for file in files[:5]:
                print(" ", file.name)

        elif folder.is_file():
            print("File:", folder.name)