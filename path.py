from pathlib import Path
import os
import glob

TRAIN_NOISE_PATH_DIR = Path('./data/train')
TRAIN_CLEAN_PATH_DIR = Path('./data/clean')

noise_file = list(TRAIN_NOISE_PATH_DIR.rglob('*.flac'))
print(len(noise_file))