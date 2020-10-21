import json
import pickle
import shutil
from pathlib import Path


def copy_file(file_path, out_dir):
    out_file = Path(out_dir) / Path(file_path).name
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file_path, out_file)
    return out_file.as_posix()


def load_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(data, pkl_file):
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)
    return pkl_file


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file


def load_csv(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    return lines


def save_csv(lines, csv_file):
    with open(csv_file, "w") as f:
        f.write("\n".join(lines))
    return csv_file
