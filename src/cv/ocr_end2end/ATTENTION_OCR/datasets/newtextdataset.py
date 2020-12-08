from datasets import fsns
import os
import json

DEFAULT_DATASET_DIR = 'dataset_dir/'

def get_split(split_name, config_path, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = json.load(open(config_path,'r'))

  return fsns.get_split(split_name, dataset_dir, config)
