# ImAge_workflow
This repository provides Python-based code to replicate the imaging-based chromatin and epigenetic age (ImAge) with more flexibility.
The ImAge is introduced in the following publications:
- Alvarez-Kuglen, M., Ninomiya, K., Qin, H., Rodriguez, D., Fiengo, L., Farhy, C., Hsu, W.-M., Kirk, B., Havas, A., Feng, G.-S., et al. (2024). ImAge quantitates aging and rejuvenation. Nat. Aging. https://doi.org/10.1038/s43587-024-00685-1.
(Full article access is also available at https://www.scintillon.org/terskikh-publications)

## Overall workflow
The main workflow to calculate ImAge readouts consists of four steps: 1. illumination correction, 2. Nuclei segmentation, 3. image feature extraction, and 4. ImAge axis construction.
![](/repo_assets/workflow.png)

## Prerequisites
### Folder structure
In the original script, we assume the folder structure below:
![](/repo_assets/folder.png)

### Installation
Following installation and Python scripts are verified to work on the windows subsystem of linux (WSL), Ubuntu-18.04 running on windows 11 on a desktop PC consisting of Intel Xeon CPU (Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz) with 64GB of RAM and 1 NVIDIA Quadro P4000 with 8GB RAM.

We used Poetry for package management and version control. The user should set up a Python 3.10 environment with [Poetry](https://github.com/python-poetry/poetry) installed. 
We recommend using an Anaconda (or Miniconda) to make an environment (see [official installation guide](https://docs.anaconda.com/free/anaconda/install/) for Anaconda installation). Set up the virtual environment using conda (assuming Anaconda is already installed):
```
# setup Python environment with Python 3.10
conda create -n ImAge_workflow python=3.10

# once the installation is done, load the installed environment
conda activate ImAge_workflow
```

Install poetry on the virtual environment using pip (if portry is already installed in the base environment, then skip this step):
```
pip install poetry
```

Install dependency packages using poetry
```
# navigate to the location where this repository is cloned
cd /PATH_TO_CLONED_REPOSITORY/ImAge_workflow
poetry install
```

Our code includes GPU implementation in the segmentation step. Cuda and CudaToolkit installation is taken care of by poetry. For details on installation, please refer https://www.tensorflow.org/install/pip
Run the following commands to enable cuda on the conda environment:
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'NVIDIA_PACKAGE_DIR=$(dirname $(python -c "import nvidia;print(nvidia.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'for dir in $NVIDIA_PACKAGE_DIR/*; do if [ -d "$dir/lib" ]; then export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"; fi done' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```
## Usage
Change working directory to ImAge_workflow and execute Python script in the appropriate order. By default Each Python script is aligned in the appropreate order in main.py. Files are named with a prefix starting with o*, which stands for the order of the process. (e.g., o3_extract_features.py has to be executed after o2_segmentation.py). Each script creates results files (e.g., segmentation, image features, ImAge readouts).

```shell
python workflow/main.py
```

Visualize results and export raw ImAge readouts for additional analysis on external software
```shell
python workflow/visualization.py
```
This visualization will replicate Fig.6 b-e, Extended Data Fig. 8 b-d and Extended Data Fig. 9 a in the publication [ImAge quantitates aging and rejuvenation. Nature Aging](https://www.nature.com/articles/s43587-024-00685-1))


### Sample data availability
Sample data is available at https://osf.io/mkc9u/
You can download and extract files and copy them as instructed in the Folder structure section. This will replicate the results for Figure 1 in our publication, [ImAge quantitates aging and rejuvenation. Nature Aging]([http://dx.doi.org/10.21203/rs.3.rs-3479973/v1](https://www.nature.com/articles/s43587-024-00685-1)).

## Related publication
- Ninomiya, K., and Terskikh, A.V. (2024). Imaging the epigenetic landscape in single cells to study aging trajectories. Nat Aging. https://doi.org/10.1038/s43587-024-00689-x.
- Farhy, C., Hariharan, S., Ylanko, J., Orozco, L., Zeng, F.-Y., Pass, I., Ugarte, F., Forsberg, E.C., Huang, C.-T., Andrews, D.W., et al. (2019). Improving drug discovery using image-based multiparametric analysis of the epigenetic landscape. Elife 8. 10.7554/eLife.49683.

