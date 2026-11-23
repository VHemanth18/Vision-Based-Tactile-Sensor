# Installation

The project has three independent parts with different dependencies. Install only what you need.

| Part | Folder | Requirements file | Needs |
|---|---|---|---|
| Learning pipeline (training, evaluation, prediction, experiments) | `tactile_force/`, `experiments/`, `main.py` | `requirements.txt` | Python ≥ 3.9, PyTorch ≥ 2.1, GPU recommended |
| Dataset generation in simulation (Phase 1) | `data_generation/` | `requirements-sim.txt` | TACTO, PyBullet, pybulletX, Hydra |
| Real-world recording (Phase 2) | `real_rig/` | `requirements-rig.txt` | DIGIT sensor, ESP32 + HX711 + load cell on USB |

The experiments in this repository were run with Python 3.11, PyTorch 2.11 (CUDA 12.8) on an NVIDIA RTX 5070
(Linux). The code also runs on Windows and on Google Colab.

## 1. Learning pipeline

**Linux / macOS**

```bash
git clone https://github.com/VHemanth18/Vision-Based-Tactile-Sensor.git
cd Vision-Based-Tactile-Sensor
python -m venv .venv
source .venv/bin/activate
# PyTorch with the command for your CUDA version from https://pytorch.org/get-started/locally/, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest -q          # 9 passed
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/VHemanth18/Vision-Based-Tactile-Sensor.git
cd Vision-Based-Tactile-Sensor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
python -m pytest -q
```

On Windows add `train.num_workers=0` (or `--num-workers 0`) if the DataLoader worker processes fail.

**Conda** works the same way: `conda create -n vbts python=3.11 && conda activate vbts`, then the `pip` lines.

Always run the commands from the repository root (the folder that contains `main.py`). Do not run
`python tactile_force/train.py` directly: the package uses relative imports.

## 2. Dataset generation (Phase 1)

TACTO is installed from source; it brings the DIGIT sensor model and the renderer (pyrender/OpenGL).

```bash
pip install -r requirements-sim.txt
pip install git+https://github.com/facebookresearch/tacto.git
```

The generator also needs two files from the TACTO repository (`examples/conf/`): the Hydra config `digit.yaml`
and the background image `bg_digit_240_320.jpg`, and one URDF per indenter. See
[dataset_preparation.md](dataset_preparation.md) and [`data_generation/conf/README.md`](../data_generation/conf/README.md).

Headless Linux servers need an off-screen OpenGL backend, e.g. `export PYOPENGL_PLATFORM=egl` (or `osmesa`).

## 3. Real-world recording (Phase 2)

```bash
pip install -r requirements-rig.txt
```

The ESP32 must run a sketch that reads the HX711 and prints the weight over USB serial (lines `Weight: <value> g`
or the fast format `W:<ms>,<grams>`, which `record_both_sensors.py` parses). Linux users need access to the serial
port (`sudo usermod -aG dialout $USER`, then log in again) and to the DIGIT USB video device.

## 4. Google Colab

Upload the repository and the dataset as zip files to Google Drive, then:

```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/work && cd /content/work && unzip -q -o /content/drive/MyDrive/tactile/Vision-Based-Tactile-Sensor.zip && unzip -q -o /content/drive/MyDrive/tactile/TACTO_DATASETS.zip -d data
%cd /content/work/Vision-Based-Tactile-Sensor
!pip install -q pandas opencv-python-headless pyyaml pyarrow
!python run_split_experiment.py --data-root /content/work/data/TACTO_DATASETS --num-workers 2 --out /content/drive/MyDrive/tactile/outputs/split_test-hexagonal_val-hash
```

Unzip the dataset to the local disk of the runtime: reading hundreds of thousands of PNGs directly from Drive is
very slow.

## 5. Check the installation without the dataset

```bash
python tools/make_synthetic_dataset.py --out synthetic_dataset --shapes ball_array_indenter heart_indenter ring_indenter star_indenter triangle_indenter asterisk_indenter hash_indenter hexagonal_indenter
python run_split_experiment.py --data-root synthetic_dataset --out outputs/synthetic_split --epochs 2 --cpu-small \
    --set preprocess.mask_top_px=22 "preprocess.crop=[0,0.5,0,1]"
```

This trains both models for two epochs on fake frames and writes the full output tree under `outputs/synthetic_split/`.
