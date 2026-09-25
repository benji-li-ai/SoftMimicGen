# SoftMimicGen

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver.svg)](https://isaac-sim.github.io/IsaacLab/v2.2.1/index.html)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)

A data generation pipeline for scalable robot learning in deformable object manipulation.

[🌐 Website](https://softmimicgen.github.io/) | [📄 Paper](https://arxiv.org/abs/2603.25725)

## Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/NVlabs/SoftMimicGen.git
cd SoftMimicGen

# Run the install script (creates conda env, installs Isaac Sim, Isaac Lab, SoftMimicGen, and downloads datasets)
bash softmimicgen.sh

# Activate the environment
conda activate softmimicgen
```

### Verify Installation

```bash
# List available environments
python scripts/environments/list_envs.py

# Run a zero-action agent
python scripts/environments/zero_agent.py --task Isaac-Jenga-Franka-IK-Rel-v0 --num_envs 4
```

## Data Generation Pipeline

The SoftMimicGen pipeline consists of three steps: **record** (teleoperate), **annotate** (segment subtasks), and **generate** (create new demonstrations). Pre-annotated datasets are provided for all tasks.

> **Note:** Humanoid tasks require the `--enable_pinocchio` flag for all commands below.

For the package-backed YAM cell task, bounded iteration commands, and the fresh-recording
contract, see [Native YAM cell recording and generation](EXTRAPOLATION_PLAN.md).

### Generate from Pre-Annotated Datasets

Annotated datasets are downloaded automatically during installation to `datasets/annotated_dataset/`. New demonstrations can be generated as follows:

```bash
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cuda \
    --num_envs 1 \
    --generation_num_trials 100 \
    --input_file ./datasets/annotated_dataset/<annotated_dataset>.hdf5 \
    --output_file ./datasets/generated_dataset/<generated_dataset>.hdf5 \
    --enable_cameras
```

### Re-Annotate Datasets

The provided datasets can be re-annotated, or custom recorded demonstrations can be annotated with subtask boundaries:

```bash
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cuda \
    --task <MIMIC_TASK_NAME> \
    --input_file ./datasets/annotated_dataset/<annotated_dataset>.hdf5 \
    --output_file ./datasets/annotated_dataset/<re_annotated_dataset>.hdf5 \
    --enable_cameras
```


### Available Tasks

| Task | Task ID | Mimic ID | Robot |
|---|---|---|---|
| Teddy Pick and Place | `Isaac-Teddy-GR1T2-Abs-v0` | `Isaac-Teddy-GR1T2-Abs-Mimic-v0` | GR1T2 |
| Towel | `Isaac-Towel-GR1T2-Abs-v0` | `Isaac-Towel-GR1T2-Abs-Mimic-v0` | GR1T2 |
| Rope Manipulation | `Isaac-Rope-Franka-IK-Rel-v0` | `Isaac-Rope-Franka-IK-Rel-Mimic-v0` | Franka |
| Jenga | `Isaac-Jenga-Franka-IK-Rel-v0` | `Isaac-Jenga-Franka-IK-Rel-Mimic-v0` | Franka |
| Towel Fold | `Isaac-Fold-Towel-Franka-IK-Rel-v0` | `Isaac-Fold-Towel-Franka-IK-Rel-Mimic-v0` | Franka |
| Cube Stack | `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0` | `Isaac-Stack-Soft-Cube-Franka-IK-Rel-Mimic-v0` | Franka |
| Tissue Lift | `Isaac-Tissue-PSM-IK-Rel-v0` | `Isaac-Tissue-PSM-IK-Rel-Mimic-v0` | Surgical PSM |
| Threading | `Isaac-Thread-PSM-IK-Rel-v0` | `Isaac-Thread-PSM-IK-Rel-Mimic-v0` | Surgical PSM |
| Towel Fold | `Isaac-Fold-Towel-Yam-Joint-v0` | `Isaac-Fold-Towel-Yam-Joint-Mimic-v0` | YAM |
| Cell Towel Fold | `Isaac-Fold-Cell-Towel-Yam-Joint-v0` | `Isaac-Fold-Cell-Towel-Yam-Joint-Mimic-v0` | YAM |
| Bag Loading | `Isaac-Bag-Yam-Joint-v0` | `Isaac-Bag-Yam-Joint-Mimic-v0` | YAM |

## Contribution Guidelines

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must
include a Developer Certificate of Origin sign-off.

## License

This project is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for details. Third-party components retain their own
licenses as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you find this work useful, please cite the paper using the following bibtex:

```
@article{moghani2026softmimicgen,
  title={SoftMimicGen: A Data Generation System for Scalable Robot Learning in Deformable Object Manipulation},
  author={Moghani, Masoud and Azizian, Mahdi and Garg, Animesh and Zhu, Yuke and Huver, Sean and Mandlekar, Ajay},
  journal={arXiv preprint arXiv:2603.25725},
  year={2026}
}
```
