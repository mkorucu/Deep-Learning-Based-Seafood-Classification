from IPython import display
display.clear_output()

import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

import subprocess
try:
    subprocess.run(["wget", "https://app.roboflow.com/ds/4zNvzaDrdj?key=o2h2h76X50", "-O", "file.zip"], check=True)
    print("Download successful!")
except subprocess.CalledProcessError as e:
    print(f"Error downloading file: {e}")

model = YOLO('yolov8n.pt')
training_results = model.train(data= "file.zip",epochs=20,save_period=5,patience=20)

import os
folder_path = "Roboflow_labeled_data/test/images"
# Get all file names in the folder
file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

print("File Names in the Folder:")
for file_name in file_paths:
    model.predict(source=file_name, conf=0.1, save=True)