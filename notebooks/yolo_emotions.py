from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import random

# Load trained classification model
model = YOLO('last.pt')

# test images
test_image_folders = r'D:\machine learning\emotions\test'

# get list of image files
image_folders = os.listdir(test_image_folders)

# plot settings
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
ax = ax.ravel()

for idx in range(16):
  img_folder = random.choice(image_folders)
  img_files = os.listdir(os.path.join(test_image_folders, img_folder))
  img_path = os.path.join(test_image_folders, img_folder, random.choice(img_files))
  image = cv2.imread(img_path)

  # perform classification
  result = model(img_path)[0]

  print(model.names)
  print(result)
  print(result.probs)

  # get predicted class name
  class_id = int (result.probs.top1)
  class_name = model.names[class_id]
  confidence = result.probs.top1conf.item()

  # convert BGR to RGB for plotting
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Plot the image with class name and confidence
  ax[idx].imshow(image_rgb)
  ax[idx].set_title(f'Actual: {img_folder}\nPredicted: {class_name} ({confidence:.2f})', fontsize=12)
  ax[idx].axis('off')

plt.tight_layout()
plt.show()