<p align="center">
  <img src="docs/teaser.png" width="100%">
</p>

<p align="left">
  <a href="https://behavior.stanford.edu/index.html">
    <img
      src="https://img.shields.io/badge/BEHAVIOR--1K-Website-red?logo=googleplay&logoColor=white"
      alt="BEHAVIOR-1K Website"
    />
  </a>
  <a href="https://behavior.stanford.edu/challenge/leaderboard.html#privileged-information-track">
    <img
      src="https://img.shields.io/badge/BEHAVIOR--1K-Leaderboard-5865F2?logo=googleplay&logoColor=white"
      alt="BEHAVIOR-1K Leaderboard"
    />
  </a>
  <a href="https://huggingface.co/sunshk/comet_submission">
    <img
        src="https://img.shields.io/badge/Comet--Submission-HuggingFace-green?logo=huggingface&logoColor=brightyellow"
        alt="Comet Submission"
    />
  </a>
  <a href="docs/report.pdf">
    <img
      src="https://img.shields.io/badge/Comet--Submission-Paper-red?logo=arxiv&logoColor=red"
      alt="Implementation Report"
    />
  </a>
</p>

# Openpi Comet

> [!TIP]
> OpenPi Comet is the submission of Team Comet for the [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/index.html). This repository provides a unified framework for pre-training, post-training, data generation and evaluation of π0.5 (Pi05) models on BEHAVIOR-1K.

Our [[submission]](https://behavior.stanford.edu/challenge/leaderboard.html#privileged-information-track) achieved a Q-score of 0.2514, securing 2nd place overall and finishing just behind the winning team by a narrow margin—highlighting both the strong competitiveness of our approach and the effectiveness of our end-to-end VLA training strategy. 

<p align="center">
  <img src="docs/leaderboard.png" width="80%">
</p>


This codebase contains:
1. Distributed OpenPi training infrastructure
2. Various pre-training setup, including hierarchical instructions (global, subtask, skill) and multimodal observations (RGB, depth, point cloud, segmentation, bounding boxes, human pointing)
3. Post-training via Rejection Sampling Fine-Tuning (RFT) with automated dataset construction
4. Data generation scripts such as teleoperation and simulation rollouts using existing policy
5. Model zoo of pretrained VLA checkpoints trained on 1M+ robot interactions

Please check our [[Report]](./docs/report.pdf) for more details.

## Updates

- [Dec 6, 2025] Released the full submission codebase and pre-trained weights.
- [TODO] Upload our RFT dataset.


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Repo Clone

```bash
git clone https://github.com/mli0603/comet-2025-b1k-challenge.git
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```
This finetuning instruction is adapted from the original [openpi repo](https://github.com/Physical-Intelligence/openpi).

## Installation

Openpi use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
cd baselines/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

source .venv/bin/activate

# Install behavior for server deploy 
cd $PATH_TO_BEHAVIOR_1K
uv pip install -e bddl
uv pip install -e OmniGibson[eval]
```

## Model Zoo 🤗

We provide a suite of base VLA model checkpoints trained on 1M+ robot trajectories, ideal for BEHAVIOR-1K fine-tuning.

|   Task ID | Task Name                               | HF URL                                                                                                                                                                                                                                                   |
|----------:|:----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|         0 | turning_on_radio                        | [comet_submission/turning_on_radio](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-12-18-33-57_pi05_b1k-turning_on_radio_cs32_bs64_lr2.5e-6_step15k_re_jax/14999)                                                                      |
|         1 | picking_up_trash                        | [comet_submission/picking_up_trash](https://huggingface.co/sunshk/comet_submission/tree/main/17-36-51_pi05_b1k-pt12_cs32_bs32_lr2.5e-5_step100k_gpu80_jax/75000)                                                                                         |
|         2 | putting_away_Halloween_decorations      | [comet_submission/putting_away_Halloween_decorations](https://huggingface.co/sunshk/comet_submission/tree/main/11-31-51_pi05_b1k-putting_away_Halloween_decorations_cs32_bs32_lr2.5e-6_step15k_sft_pt50_merge_25k_jax/14999)                             |
|         3 | cleaning_up_plates_and_food             | [comet_submission/cleaning_up_plates_and_food](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                    |
|         4 | can_meat                                | [comet_submission/can_meat](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                       |
|         5 | setting_mousetraps                      | [comet_submission/setting_mousetraps](https://huggingface.co/sunshk/comet_submission/tree/main/03-32-55_pi05_b1k-pt50_merge_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/20000)                                                                                 |
|         6 | hiding_Easter_eggs                      | [comet_submission/hiding_Easter_eggs](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-49-17_pi05_b1k-hiding_Easter_eggs_official_rollsamplelower_lr2.5e-6_step20k_sft_resume_jax/19999)                                           |
|         7 | picking_up_toys                         | [comet_submission/picking_up_toys](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                |
|         8 | rearranging_kitchen_furniture           | [comet_submission/rearranging_kitchen_furniture](https://huggingface.co/sunshk/comet_submission/tree/main/03-32-55_pi05_b1k-pt50_merge_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/45000)                                                                      |
|         9 | putting_up_Christmas_decorations_inside | [comet_submission/putting_up_Christmas_decorations_inside](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                        |
|        10 | set_up_a_coffee_station_in_your_kitchen | [comet_submission/set_up_a_coffee_station_in_your_kitchen](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-49-17_pi05_b1k-set_up_a_coffee_station_in_your_kitchen_official_rollsamplelower_lr2.5e-6_step20k_sft_resume_jax/15000) |
|        11 | putting_dishes_away_after_cleaning      | [comet_submission/putting_dishes_away_after_cleaning](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-41-28_pi05_b1k-putting_dishes_away_after_cleaning_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/19999)          |
|        12 | preparing_lunch_box                     | [comet_submission/preparing_lunch_box](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-41-27_pi05_b1k-preparing_lunch_box_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/19999)                                        |
|        13 | loading_the_car                         | [comet_submission/loading_the_car](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                |
|        14 | carrying_in_groceries                   | [comet_submission/carrying_in_groceries](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/35000)                                                                          |
|        15 | bringing_in_wood                        | [comet_submission/bringing_in_wood](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                               |
|        16 | moving_boxes_to_storage                 | [comet_submission/moving_boxes_to_storage](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-12-19-00-46_pi05_b1k-pt10_merge1112_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/40000)                                                             |
|        17 | bringing_water                          | [comet_submission/bringing_water](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-12-15-45-53_pi05_b1k-pt10_re-pt12-49k-m_cs32_bs64_lr2.5e-6_step50k_gpu160_jax/45000)                                                                  |
|        18 | tidying_bedroom                         | [comet_submission/tidying_bedroom](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                |
|        19 | outfit_a_basic_toolbox                  | [comet_submission/outfit_a_basic_toolbox](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-49-18_pi05_b1k-outfit_a_basic_toolbox_official_rollsamplelower_lr2.5e-6_step20k_sft_resume_jax/15000)                                   |
|        20 | sorting_vegetables                      | [comet_submission/sorting_vegetables](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                             |
|        21 | collecting_childrens_toys               | [comet_submission/collecting_childrens_toys](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                      |
|        22 | putting_shoes_on_rack                   | [comet_submission/putting_shoes_on_rack](https://huggingface.co/sunshk/comet_submission/tree/main/17-36-51_pi05_b1k-pt12_cs32_bs32_lr2.5e-5_step100k_gpu80_jax/85000)                                                                                    |
|        23 | boxing_books_up_for_storage             | [comet_submission/boxing_books_up_for_storage](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                    |
|        24 | storing_food                            | [comet_submission/storing_food](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                   |
|        25 | clearing_food_from_table_into_fridge    | [comet_submission/clearing_food_from_table_into_fridge](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                           |
|        26 | assembling_gift_baskets                 | [comet_submission/assembling_gift_baskets](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                        |
|        27 | sorting_household_items                 | [comet_submission/sorting_household_items](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-12-22-39-15_pi05_b1k-pt50_merge1112_hq_re_cs32_bs64_lr2.5e-6_step50k_gpu400_jax/15000)                                                       |
|        28 | getting_organized_for_work              | [comet_submission/getting_organized_for_work](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                     |
|        29 | clean_up_your_desk                      | [comet_submission/clean_up_your_desk](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                             |
|        30 | setting_the_fire                        | [comet_submission/setting_the_fire](https://huggingface.co/sunshk/comet_submission/tree/main/06-20-28_pi05_b1k-pt12_merge_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/40000)                                                                                   |
|        31 | clean_boxing_gloves                     | [comet_submission/clean_boxing_gloves](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                            |
|        32 | wash_a_baseball_cap                     | [comet_submission/wash_a_baseball_cap](https://huggingface.co/sunshk/comet_submission/tree/main/06-20-28_pi05_b1k-pt12_merge_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/40000)                                                                                |
|        33 | wash_dog_toys                           | [comet_submission/wash_dog_toys](https://huggingface.co/sunshk/comet_submission/tree/main/11-31-51_pi05_b1k-wash_dog_toys_cs32_bs32_lr2.5e-6_step15k_sft_pt50_merge_25k_jax/14999)                                                                       |
|        34 | hanging_pictures                        | [comet_submission/hanging_pictures](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-15-03-13-35_pi05_b1k-hanging_pictures_roll1114_cs32_bs64_lr2.5e-6_step15k_re-pt50-49k_jax/14999)                                                    |
|        35 | attach_a_camera_to_a_tripod             | [comet_submission/attach_a_camera_to_a_tripod](https://huggingface.co/sunshk/comet_submission/tree/main/06-20-28_pi05_b1k-pt12_merge_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/25000)                                                                        |
|        36 | clean_a_patio                           | [comet_submission/clean_a_patio](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                  |
|        37 | clean_a_trumpet                         | [comet_submission/clean_a_trumpet](https://huggingface.co/sunshk/comet_submission/tree/main/20-36-46_pi05_b1k-clean_a_trumpet_cs32_bs32_lr2.5e-5_step30k_jax/25000)                                                                                      |
|        38 | spraying_for_bugs                       | [comet_submission/spraying_for_bugs](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-41-30_pi05_b1k-spraying_for_bugs_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/15000)                                            |
|        39 | spraying_fruit_trees                    | [comet_submission/spraying_fruit_trees](https://huggingface.co/sunshk/comet_submission/tree/main/06-46-38_pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/49999)                                                                                     |
|        40 | make_microwave_popcorn                  | [comet_submission/make_microwave_popcorn](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-10-12-57-44_pi05_b1k-pt7_cs32_bs64_lr2.5e-5_step50k_gpu80_jax/49999)                                                                          |
|        41 | cook_cabbage                            | [comet_submission/cook_cabbage](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                   |
|        42 | chop_an_onion                           | [comet_submission/chop_an_onion](https://huggingface.co/sunshk/comet_submission/tree/main/03-32-55_pi05_b1k-pt50_merge_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/45000)                                                                                      |
|        43 | slicing_vegetables                      | [comet_submission/slicing_vegetables](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-16-15-41-27_pi05_b1k-slicing_vegetables_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/19999)                                          |
|        44 | chopping_wood                           | [comet_submission/chopping_wood](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                  |
|        45 | cook_hot_dogs                           | [comet_submission/cook_hot_dogs](https://huggingface.co/sunshk/comet_submission/tree/main/2025-11-12-16-22-00_pi05_b1k-pt10_merge1112_re-pt12-49k-m_cs32_bs64_lr2.5e-6_step50k_gpu160_jax/40000)                                                         |
|        46 | cook_bacon                              | [comet_submission/cook_bacon](https://huggingface.co/sunshk/comet_submission/tree/main/06-46-38_pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/49999)                                                                                               |
|        47 | freeze_pies                             | [comet_submission/freeze_pies](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/35000)                                                                                    |
|        48 | canning_food                            | [comet_submission/canning_food](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                   |
|        49 | make_pizza                              | [comet_submission/make_pizza](https://huggingface.co/sunshk/comet_submission/tree/main/10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000)                                                                                     |



### Finetune OpenPi

Each time we launch the training, we need to compute the normalization statistics for the training data in advance: 

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_b1k-turning_on_radio
```

This will create `norm_stats.json` under `assets/pi0_b1k/behavior-1k/2025-challenge-demos`, which will be used to normalize the training data.

After this, update the configs in `src/openpi/training/config.py` to be the task name you want (or None to include all tasks), for example, you can update the configs as follows for the `turning_on_radio` task:

```python
TrainConfig(
    name="pi05_b1k-turning_on_radio",
    exp_name="openpi",
    project_name="B1K",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
    data=LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        base_config=DataConfig(
            prompt_from_task=True,
            episodes_index=list(range(200)),
            behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
            tasks=["turning_on_radio"],
            fine_grained_level=0,  # 0: global instruction, 1: subtask instruction, 2: skill instruction
            train_task_type="regular",  # regular | cumulate | mixture
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "The Model Path you want to finetune from, e.g., gs://openpi-assets/checkpoints/pi05_base/params\
        or the checkpoint from our model zoo"
    ),
    num_train_steps=30_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        peak_lr=2.5e-5,
        decay_steps=30_000,
    ),
    freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
    ema_decay=None,
    checkpoint_base_dir=".",
    num_workers=8,
    batch_size=8 * 32,
),
```

Then run the following command to fintune OpenPi:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi05_b1k-turning_on_radio \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --weight_loader.params_path="The Model Path you want to finetune from, e.g., gs://openpi-assets/checkpoints/pi05_base/params \
    or the checkpoint from our model zoo" # also be configurable in the config
```

### Pre-train OpenPi

To support distributed training, we update `src/openpi/training/data_loader.py` for data sharding, and the `src/openpi/training/checkpoints_dist.py` and `scripts/train_dist.py` for distributed checkpointing management and training. To launch the pretrain, run the following command:

```bash
# set dist training envs
export MASTER_ADDR=${SERVICE_PREFIX}-0.${SUBDOMAIN}
export WORLD_SIZE=${LEPTON_JOB_TOTAL_WORKERS}
export WORLD_RANK=${LEPTON_JOB_WORKER_INDEX}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12350

config_name=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40
exp_name=pi05_b1k-pt50_pretrain

python scripts/compute_norm_stats.py --config-name ${config_name}

python scripts/train_dist.py ${config_name} --exp_name=${exp_name} --overwrite
```

### Post-train OpenPi using Rejection Sampling fine-tuning (RFT)

To perform RFT, you need to first deploy the finetuned checkpoint, and then rollout the episodes in the BEHAVIOR-1K Simulator. We also observe that the `pose perturbator` helps improve the robustness of the RFT Algorithm. 

1. Copy the `openpi_comet/data_generation/rollout/learning` to `BEHAVIOR-1K/OmniGibson/omnigibson/learning`.
```bash
cp -r data_generation/rollout/learning/* BEHAVIOR-1K/OmniGibson/omnigibson/learning/
```
NOTE: be careful to the latest commit of the BEHAVIOR-1K repo.

2. Run the RFT rollout in parallel:

```bash
python OmniGibson/omnigibson/learning/eval.py policy=websocket \
    save_rollout=true \
    perturb_pose=true \
    eval_on_train_instances=false \
    task.name=turning_on_radio \
    log_path=./outputs/rft \
    use_parallel_evaluator=false \
    parallel_evaluator_start_idx=0 \
    parallel_evaluator_end_idx=10 \
    model.port=8000 \
    env_wrapper._target_=omnigibson.learning.wrappers.RolloutRGBWrapper
```
where `parallel_evaluator_start_idx` and `parallel_evaluator_end_idx` are the start and end index of the parallel rollout, we can distribute the rollout to multiple GPUs by splitting the total number of instances into multiple parts.

3. Build the RFT dataset:
After the rollout, you can build the RFT dataset by running the following command:

```bash
python data_generation/rollout/create_rft_dataset.py \
    --rollout_dir $PATH_TO_ROLLOUT_DATASET \
    --rft_dir $PATH_TO_RFT_DATASET
```

Then, we can perform RFT training on the RFT dataset. Please refer to the [RFT training config](src/openpi/training/config.py) for more details.

### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:

    ```
    source .venv/bin/activate
    uv run scripts/serve_b1k.py --task_name=$TASK_NAME policy:checkpoint --policy.config=pi0_b1k --policy.dir=$PATH_TO_CKPT
    ```
    This opens a connection listening on 0.0.0.0:8000. Please check the `scripts/serve_b1k.py` for more details.


2. Run the evaluation on BEHAVIOR:

    Assume you have behavior env installed (check https://github.com/StanfordVL/BEHAVIOR-1K for more details), run the following command within the BEHAVIOR-1K directory:
    ```
    conda activate behavior 
    python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio log_path=$LOG_PATH
    ```


## FAQs

If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. We appreciate your feedback and contributions!

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{comet2025behavior1k,
  title={Comet Submission for BEHAVIOR-1K Challenge},
  author={Comet Team},
  url={https://github.com/mli0603/comet-2025-b1k-challenge},
  year={2025}
}
```

<!-- ```bibtex
@article{comet2025behavior1k,
  title={Comet Submission for BEHAVIOR-1K Challenge},
  author={Comet Team},
  journal={arXiv preprint arXiv:2512.06000},
  url={https://github.com/mli0603/comet-2025-b1k-challenge},
  year={2025}
}
``` -->
