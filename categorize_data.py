import os
from pathlib import Path
import shutil

## Load all images
dataset_path = Path(r"C:\Users\ASUS\.cache\kagglehub\datasets\norbertelter\pcb-defect-dataset\versions\2\pcb-defect-dataset\train\images")
all_images = list(dataset_path.glob('*.jpg'))
print("Total images: ", len(all_images))

## Classify Defect types each image
origanized_path = Path("organized_dataset")

# Create directories for each defect
defect_types = ["missing_hole", "open_circuit", "short", "spur", "spurious_copper", "mouse_bite", "unknown"]
for defect_type in defect_types:
    (origanized_path / defect_type).mkdir(parents=True, exist_ok=True)
# Copy images to categorized directories
for img in all_images:
    defect_type = None
    for type in defect_types:
        if type in img.stem:
            defect_type = type
    defect_type = "unknown" if defect_type == None else defect_type
    dest = origanized_path / defect_type / img.name
    shutil.copy(img, dest)
    