
# **DiffusionNet**: Accelerating the solution of Time-Dependent partial differential equations using deep learning

Citation
```
@article{asem2020diffusionnet,
  title={DiffusionNet: Accelerating the solution of Time-Dependent partial differential equations using deep learning},
  author={Asem, Mahmoud},
  journal={arXiv preprint arXiv:2011.10015},
  year={2020}
}
```

### Install requirements
`
pip install -r requirements.txt
`



# <mark> **I. Transient Heat transfer Section**  </mark>

### Contents

| Script| Description |
| ------------- | ------------- |
| `solver.py`  |  Alternate direction implicit scheme solver for transient heat transfer  |
| `DiffusionNet.py`  | Content Cell  |
| `generator.py`  |  data generation utility functions for Transient heat transfer  |
| `visualize.py`  | Heat map comparison utility function |
|`speedup_analysis.py` |speedup analysis section utility functions |


| Notebook| Description |
| ------------- | ------------- |
| `Reproduce models.ipynb`  |  Reproduce trained models  |
| `Heatmaps figure.ipynb`  | Reproduce and visualize heat maps comparisons |
| `Speedup figures.ipynb`  |  Reproduce speedups plots for Iterations / Gridsize ( GPU+CPU specific )  |
|`Loss plots figures` |Reproduce loss plots|
____
### 1) Reproduce trained models

To reproduce trained models, under **Reproduce models** notebook

![image](https://i.imgur.com/3QsCoMW.png)

1) Choose the appropriate parameters for `grid size` ,`step`,number of `batches` to be trained on.
2)  Tick the **train** checkbox to start training, reproduced models will be saved under `ReproducedModels` and `ReproducedLogs` folders.
___

### 2) Reproduce loss plots figures

![image](https://i.imgur.com/XyC8gEx.png)

Under `Loss plots figures.ipynb` notebook choose the desired log from drop down menu and check `plot`
___
### 3) Reproduce Heat maps comparisons figure


![image](https://i.imgur.com/yAJV3wj.png)

Under visualization notebook, **Choose the following parameters,** 

| Parameter  | Description |
| ------------- | ------------- |
| `Step`  |  step size  |
| `Content Cell`  | Content Cell  |
|`Grid size` | Grid size NxN|
|`bc1` | Bottom boundary condition|
|`bc2` | Left boundary condition|
|`bc3` | Top boundary condition|
|`bc4` | Right boundary condition|
|`ic`  | Initial condition|
|`t00` | Initial input step to model|

<br>

1) Tick **analyze** to view Timing of numerical solution and deep learning solution, and Error metrics.
2) Ticks **plot** to view the heat maps of Numerical solution , Deep learning solution and the absolute difference between them, respectively. 
3) Tick **save** to save the resultant heat maps (optional).
4) <mark>Make sure the loaded model Step[10,100] and gridsizes[12,24,48,96] matches that of selected heat maps parameters above <mark>

<br>

![image](https://i.imgur.com/wImxA0A.png)


___


### 4) Reproduce Speedup figures


![image](https://i.imgur.com/CLCqAaX.png)

Under Speedup Figures notebook, **Choose the following parameters,** 

| Parameter  | Description |
| ------------- | ------------- |
| `Speedup analysis`  |  Iteration analysis / Gridsize analysis  |
| `P`  | Deep learning prediction step  |
|`Grid size` | Grid size NxN|


<br>

Tick **Start** to start analysis and then plot figures as in **below**


<br>

![image](https://i.imgur.com/gNfOtI7.png)


___


# <mark>  **II. Inviscid burgers Section** </mark>

### Contents

| Script| Description |
| ------------- | ------------- |
| `utils.py`  |  Utility functions   |
| `DiffusionNet.py`  | Content Cell  |
| `generator.py`  |  data generation utility functions for Transient heat transfer  |



| Notebook| Description |
| ------------- | ------------- |
| `Reproduce models.ipynb`  |  Reproduce trained models  |
| `Reproduce Figures.ipynb`  | Reproduce Data representation,Error histogram and Sample plot figure |
| `Reproduce Test data prediction.ipynb`  |  Generate the prediction of DiffusionNet for the given test data  |



