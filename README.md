This repository hosts the official implementation of the paper [Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity](https://xingbaji.github.io/PNCG_project_page/).

### Run demo
Requirements：
```pip install -r requirements.txt```

Run：
```
cd PNCG_IPC/demo
python cubic_demos
```

### Project Structure
The repository is organized as follows:

- algorithm/: Contains core algorithms, infrastructure, and functions critical to our method.
The workflow starts with base_deformer, progresses through collision_detection_v2, and finally integrates within pncg_base_ipc.

- demo/: A collection of demonstrations showcasing the capabilities and performance of our method.

- util/model_loading: Contains code for loading models necessary for our demonstrations and tests.

- math_utils/: Hosts essential utility functions for elastic deformations, primarily within elastic_util.py and matrix_util.py.

- model/: Stores model files utilized in the demonstrations and testing phases.

### Note
- The newest version of Numpy may cause Core Dumped, I don't figure out why it happens. It seems to due to the version difference of meshtaichi package and Numpy. Older version such as numpy1.26, is fine.
- When using MeshTaichi to load a model, the xxxx.face file is necessary, while the face information is not required. So some .face files contain just one face.
Or You can just delete following code in \_\_init\_\_.py in meshtaichi_patcher
```
        ans["face"] = read_tetgen(f'{base_name}.face')[0].reshape(-1, 3)
```
- Please run in linux system, the performance of our code in windows is worse than linux.
- Feel free to contact me at shenxing03@corp.netease.com / shenxingsx@zju.edu.cn or create a Github issue if you have questions regarding setting up the repository, running examples or adding new examples.
- A introduction of our paper can be found at others/Paper介绍.pdf.

### Citation
Please consider citing our paper if your find our research or this codebase helpful:

    @inproceedings{10.1145/3641519.3657490,
    author = {Shen, Xing and Cai, Runyuan and Bi, Mengxiao and Lv, Tangjie},
    title = {Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity},
    year = {2024},
    isbn = {9798400705250},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3641519.3657490},
    doi = {10.1145/3641519.3657490},
    booktitle = {ACM SIGGRAPH 2024 Conference Papers},
    articleno = {96},
    numpages = {11},
    keywords = {GPU, Nonlinear conjugate gradient method, Physics-based simulation},
    location = {Denver, CO, USA},
    series = {SIGGRAPH '24}
    }
