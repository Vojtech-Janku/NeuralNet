import os
import subprocess
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "downloaded_kaggle")
DATASET      = "zalando-research/fashionmnist"

TRAIN_RAW  = os.path.join(DOWNLOAD_DIR, "fashion-mnist_train.csv")
TEST_RAW   = os.path.join(DOWNLOAD_DIR, "fashion-mnist_test.csv")

def out(filename):
    return os.path.join(SCRIPT_DIR, filename)

SPLITS = [
    (TRAIN_RAW, out("fashion_mnist_train_vectors.csv"), out("fashion_mnist_train_labels.csv")),
    (TEST_RAW,  out("fashion_mnist_test_vectors.csv"),  out("fashion_mnist_test_labels.csv")),
]


def download():
    if os.path.exists(TRAIN_RAW) and os.path.exists(TEST_RAW):
        print("Raw files already present, skipping download.")
        return
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    subprocess.check_call([
        sys.executable, "-m", "kaggle",
        "datasets", "download", "-d", DATASET,
        "-p", DOWNLOAD_DIR, "--unzip",
    ])


def split(raw_path, vectors_path, labels_path):
    print(f"Splitting {os.path.basename(raw_path)} ...")
    with (
        open(raw_path) as src,
        open(vectors_path, "w") as vec_out,
        open(labels_path,  "w") as lab_out,
    ):
        next(src)  # skip header
        for line in src:
            label, _, pixels = line.partition(",")
            lab_out.write(label + "\n")
            vec_out.write(pixels)  # pixels already ends with \n


if __name__ == "__main__":
    download()
    for raw, vectors, labels in SPLITS:
        split(raw, vectors, labels)
    print("Done.")
