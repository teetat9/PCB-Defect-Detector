# PCB Defect Detector

Detecting defects in printed circuit boards using computer vision.

Dataset source:
https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset

## Progress
- [x] Dec 29: Dataset loaded, visualized 6 defect types
- [ ] Next: Build simple classification model

## Defect Types
- Missing hole
- Open circuit  
- Short circuit
- Spur
- Spurious copper
- Mouse bite

### Option 1: Using Conda (Recommended)

```bash
conda create -n pcb_detector python=3.10 -y
conda activate pcb_detector
pip install -r requirements.txt
```

### Option 2: Using Python venv

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Categorize the dataset

Run the script below to organize PCB images into folders based on defect type:

```bash
python categorize_data.py
```

After running the script, the dataset will be structured for training and evaluation.