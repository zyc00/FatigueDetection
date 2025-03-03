# FatigueDetection

## Install
```
pip install -e . &&
pip install matplotlib dlib opencv-python
```

## Usage
To test the algorithm and visualize results
```
python test.py
```

To call the fatigue detection function
```
from fatigue import fatigue_detection
fatigue_detection(images)
```