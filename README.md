# StructRecon

This directory contains the code and configs used by the StructRecon paper.

## Double-blind note
- Author and institution metadata are removed from the paper template.
- The code repository does not require personal credentials or private endpoints.

## Requirements
- Python 3.8
- CUDA-capable GPU (recommended)
- Conda (recommended)

## Setup
From `paper-code`:
```bash
conda env create -f environment.yml
conda activate structrecon
```

## Smoke tests
From `paper-code`:
```bash
python -m compileall .
python run.py --help
python preoptimize_from_tsdf.py --help
python visualizer.py --help
```

## Data layout
The dataset loader is selected by `dataset` in the config.

ICPARK:
```text
<input_folder>/
  color/
    *.jpg or *.png
  depth/
    *.png
  pose/ or poses/
    *.txt
```

## Configuration
Base config: `configs/structrecon.yaml`  
Scene override example: `configs/Parking/parking.yaml`

At minimum, verify:
- `data.input_folder`
- `data.output`
- camera intrinsics and image size (`cam`)
- `mapping.bound` (and `mapping.marching_cubes_bound` if used)

Run-time flags can override config values:
- `--input_folder`
- `--output`

## Run StructRecon
From `paper-code`:
```bash
python run.py configs/structrecon.yaml
```

Scene-specific run:
```bash
python run.py configs/Parking/parking.yaml
```

With overrides:
```bash
python run.py configs/structrecon.yaml --input_folder <path> --output <path> --multi_gpu
```

## Optional: build OSM prior TSDF
Edit constants in `src/Osm2Tsdf.py` (for example `OSM_PATH`, `OUTPUT_DIR`, `VOXEL`, `TRUNC`) and run:
```bash
python src/Osm2Tsdf.py
```

Then update:
- `model.prior_tsdf_path`
- `model.prior_tsdf_origin_xyz`
- `mapping.bound`

## Optional: preoptimize geometry planes from TSDF
```bash
python preoptimize_from_tsdf.py --config configs/structrecon.yaml
```

Optional arguments:
```bash
python preoptimize_from_tsdf.py --config configs/structrecon.yaml --iterations 3000 --output output/custom/
```

If you pre-segment planes into blocks:
```bash
python src/preprocess_planes.py configs/Parking/parking.yaml --visualize
```

## Visualization
```bash
python visualizer.py configs/structrecon.yaml --output <output_folder>
```

Optional flags:
```bash
python visualizer.py configs/structrecon.yaml --output <output_folder> --save_rendering --top_view
```

## Evaluation
ATE:
```bash
python src/tools/eval_ate.py configs/structrecon.yaml --output <output_folder>
```

Reconstruction metrics:
```bash
python src/tools/eval_recon.py --rec_mesh <mesh.ply> --gt_mesh <gt_mesh.ply> -3d
```

Cull mesh:
```bash
python src/tools/cull_mesh.py configs/structrecon.yaml --input_mesh <mesh.ply>
```

## Outputs
Typical outputs under `data.output`:
- `ckpts/` checkpoints
- `mesh/` reconstructed meshes
- `tracking_vis/` and `mapping_vis/` visualization frames
