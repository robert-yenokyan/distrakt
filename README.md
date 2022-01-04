# Distrakt Python (=>3.8) package. 
## Setup Requirements

__*Requires pip version 20.0.2*__
  
  * *To downgrade pip version: `pip install pip==20.0.2`.*

# Installation
1. Clone the directory.
2. Activate the environment in which you want to install `distrakt`.
3. `cd` to the newly cloned directory `/object_tracking`.
4. Run `pip install -e .`


# Usage
```python
from distrakt import Distrakt

tracker = Distrakt("img.jpeg")

distance = tracker.get_distance()
print(distance)
```