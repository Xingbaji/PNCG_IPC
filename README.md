This repository hosts the official implementation of the paper 
"Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity"。

Requirements：
```python3 -m pip install -U taichi meshtaichi_patcher```

Run：
```
cd PNCG_IPC/demo
python cubic_demos
```

### Project Structure
The repository is organized as follows:

algorithm/: Contains core algorithms, infrastructure, and functions critical to our method.
The workflow starts with base_deformer, progresses through collision_detection_v2, and finally integrates within pncg_base_ipc.

demo/: A collection of demonstrations showcasing the capabilities and performance of our method.

util/model_loading: Contains code for loading models necessary for our demonstrations and tests.

math_utils/: Hosts essential utility functions for elastic deformations, primarily within elastic_util.py and matrix_util.py.

model/: Stores model files utilized in the demonstrations and testing phases.

Please use ubuntu system, the performance of our code in windows is worse than ubuntu.

#### TODO:
Further documentation detailing specific implementation details will be provided separately to offer clear insight into the methodologies employed.


