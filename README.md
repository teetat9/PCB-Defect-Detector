# PCB Defect Detector

Detecting defects in printed circuit boards using computer vision.
Data from https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset

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

## Setup
Using conda:
conda create -n pcb_detector python=3.10
conda activate pcb_detector
pip install -r requirements.txt

Using venv:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt