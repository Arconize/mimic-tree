import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, Evaluator

dataset = medmnist.TissueMNIST(split="train", download=True)

class_names = INFO["tissuemnist"]["label"]

image, label = dataset[0]

plt.figure(figsize=(4, 4))
plt.imshow(image, cmap="gray")
plt.title(f"Class: {class_names[str(label.item())]}\nLabel: {label.item()}")
plt.axis("off")

plt.savefig("tissuemnist_sample.png")

print("تصویر با موفقیت در فایل 'tissuemnist_sample.png' ذخیره شد.")
