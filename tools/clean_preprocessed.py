"""
Script để xóa preprocessed data và tạo lại với filtering mới
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
from config import arg

preprocessed_root = Path('data/preprocessed')
folder_name = f'{arg.dataset_code}_min_rating{arg.min_rating}-min_uc{arg.min_uc}-min_sc{arg.min_sc}'
dataset_folder = preprocessed_root.joinpath(folder_name)

print("=" * 80)
print("CLEANING OLD PREPROCESSED DATA")
print("=" * 80)
print(f"Dataset: {arg.dataset_code}")
print(f"Folder: {dataset_folder}")
print()

if dataset_folder.exists():
    print(f"Removing: {dataset_folder}")
    shutil.rmtree(dataset_folder)
    print("✓ Successfully removed old preprocessed data")
else:
    print("No preprocessed data found. Nothing to clean.")

print()
print("=" * 80)
print("Now run: python data_prepare.py")
print("Or with filtering: python data_prepare.py --use_text --use_image")
print("=" * 80)
