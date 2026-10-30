
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

dataset_path = Path("webcam_dataset")

classes = [
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign"
]

fig, axes = plt.subplots(5, 5, figsize=(12, 14))

for row, class_name in enumerate(classes):
    folder = dataset_path / class_name
    images = list(folder.glob("*.jpg"))

    selected = random.sample(images, min(5, len(images)))

    for col, image_path in enumerate(selected):
        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("webcam_samples.png", dpi=150)
plt.show()
