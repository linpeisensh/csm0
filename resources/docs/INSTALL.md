## Installation

The code for the Canonical Surface Mapping is built using pytorch and pytorch3d.

#### Packages

The following are the packages required to run the code

- python>=3.8
- pip
- trimesh
- scipy
- imageio
- opencv
- tensorboard
- pytorch
- torchvision
- cudatoolkit=10.2
- fvcore
- pytorch3d
- matplotlib
- rtree=0.9.4
- scikit-image

#### Conda
Download the latest conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

To create a conda environment with all the required packages using conda run the following command

```
conda env create -f resources/env.yml
```
> The above command creates a conda environment with the name **acsm**

To update an existing environment with name 'acsm' use
```
conda env update -f resources/env.yml
```

To activate the environment run the following command

```
conda activate acsm
```