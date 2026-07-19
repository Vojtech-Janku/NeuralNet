import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

idx = int(sys.argv[1])

with open('data/downloaded_kaggle/fashion-mnist_test.csv') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    row = list(reader)[idx]

label_id = int(row[0])
pixels   = np.array(row[1:], dtype='uint8').reshape((28, 28))

plt.imshow(pixels, cmap='gray', interpolation='nearest')
plt.title(f"index={idx}  |  label={label_id}: {LABELS[label_id]}")
plt.axis('off')
plt.show()
