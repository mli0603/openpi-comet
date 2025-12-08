"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.b1k_policy as b1k_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str = None

    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # episodes index to use for training
    episodes_index: list[int] | None = None

    # tasks to use for training
    tasks: list[str] | None = None

    # tasks to use for training
    modalities: list[str] = dataclasses.field(default_factory=lambda: ["rgb"])

    # tolerance decoding
    tolerance_s: float = 1e-4

    # fine-grained level of orchestrators to use for training
    fine_grained_level: int = (0,)  # 0, 1, 2

    # type of task to use for training
    train_task_type: str = ("regular",)  # regular | cumulate | mixture

    # whether to return seg instance
    return_seg_instance: bool = False

    # type of rgb to use for training
    train_rgb_type: str = "regular"  # regular | box | point

    # skill list to use for training
    skill_list: list[str] = dataclasses.field(default_factory=lambda: ["all"])


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        meta_input_transforms = []
        meta_output_transforms = []

        if self.model_delta_action_mask:
            delta_action_mask = _transforms.make_bool_mask(*self.model_delta_action_mask)
            meta_input_transforms.append(_transforms.DeltaActions(delta_action_mask))
            meta_output_transforms.append(_transforms.AbsoluteActions(delta_action_mask))

        if self.rearrange_action_indices is not None:
            meta_input_transforms.append(
                _transforms.ArrangeStateActions(indices=self.rearrange_action_indices)
            )
            meta_output_transforms.append(
                _transforms.RearrangeStateActions(indices=self.rearrange_action_indices)
            )

        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {}
                    if model_config.fast_model_tokenizer_kwargs is None
                    else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        ),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Meta image keys to use for training
    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=GroupFactory
    )
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim, model_type=model_config.model_type
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KRGBDDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    depth_as_pcd: bool = False

    pcd_downsample: int = 9

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/egocentric_depth": "observation.images.depth.head",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    meta_image_keys=self.meta_image_keys,
                    depth_as_pcd=self.depth_as_pcd,
                    pcd_downsample=self.pcd_downsample,
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # Learning rate schedule to use for training.
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule
    )

    # Optimizer to use for training.
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)

    # EMA decay to use for training.
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./outputs/assets/train"

    # Special assets base directory for B1K data.
    b1k_assets_dir: str | None = None

    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # How often (in steps) to log validation metrics.
    val_log_interval: int = 100
    # Validation batch size (optional, defaults to batch_size if not set)
    val_batch_size: int | None = None
    # Number of validation batches to average for validation loss
    val_num_batches: int = 10
    # Optionally, repo_id for validation set (if different from train)
    val_repo_id: str | None = None
    val_episodes_index: list[int] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        if self.b1k_assets_dir and pathlib.Path(self.b1k_assets_dir).exists():
            logging.info(f"Using B1K assets directory: {self.b1k_assets_dir}")
            return pathlib.Path(self.b1k_assets_dir).resolve()
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")

        return (pathlib.Path(self.checkpoint_base_dir) / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def eps_index_fn(*indexs):
    eps_index = []
    for item in indexs:
        if isinstance(item, (list, tuple)):
            eps_index.extend(list(range(item[0], item[1])))
        else:
            eps_index.extend(list(range(item)))
    return eps_index


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # 0. Base Model Configs
    TrainConfig(
        name="pi05_b1k-base",
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
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
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
    TrainConfig(
        name="pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k",
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
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
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
    # 1. pretrain configs
    TrainConfig(
        name="pi05_b1k-pt7_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=[
                    "bringing_water",
                    "cook_hot_dogs",
                    "make_microwave_popcorn",
                    "picking_up_trash",
                    "putting_shoes_on_rack",
                    "turning_on_radio",
                    "wash_a_baseball_cap",
                ],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt10_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=[
                    "bringing_water",
                    "cook_hot_dogs",
                    "make_microwave_popcorn",
                    "picking_up_trash",
                    "putting_shoes_on_rack",
                    "setting_the_fire",
                    "turning_on_radio",
                    "wash_a_baseball_cap",
                    "hanging_pictures",
                    "moving_boxes_to_storage",
                ],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt12_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=[
                    "turning_on_radio",
                    "picking_up_trash",
                    "hiding_Easter_eggs",
                    "wash_a_baseball_cap",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "make_microwave_popcorn",
                    "bringing_water",
                    "tidying_bedroom",
                    "putting_shoes_on_rack",
                    "setting_the_fire",
                    "cook_hot_dogs",
                ],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=[
                    "turning_on_radio",
                    "picking_up_trash",
                    "hiding_Easter_eggs",
                    "wash_a_baseball_cap",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "make_microwave_popcorn",
                    "bringing_water",
                    "tidying_bedroom",
                    "putting_shoes_on_rack",
                    "setting_the_fire",
                    "cook_hot_dogs",
                    "freeze_pies",
                    "putting_away_Halloween_decorations",
                    "cleaning_up_plates_and_food",
                    "can_meat",
                    "setting_mousetraps",
                    "picking_up_toys",
                    "rearranging_kitchen_furniture",
                    "putting_up_Christmas_decorations_inside",
                    "set_up_a_coffee_station_in_your_kitchen",
                    "putting_dishes_away_after_cleaning",
                    "preparing_lunch_box",
                    "loading_the_car",
                    "carrying_in_groceries",
                    "bringing_in_wood",
                    "moving_boxes_to_storage",
                    "outfit_a_basic_toolbox",
                    "sorting_vegetables",
                    "collecting_childrens_toys",
                    "boxing_books_up_for_storage",
                    "storing_food",
                    "clearing_food_from_table_into_fridge",
                    "assembling_gift_baskets",
                    "sorting_household_items",
                    "getting_organized_for_work",
                    "clean_up_your_desk",
                    "clean_boxing_gloves",
                    "wash_dog_toys",
                    "clean_a_patio",
                    "clean_a_trumpet",
                    "spraying_for_bugs",
                    "spraying_fruit_trees",
                    "cook_cabbage",
                    "chop_an_onion",
                    "slicing_vegetables",
                    "chopping_wood",
                    "cook_bacon",
                    "canning_food",
                    "make_pizza",
                ],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    # 2. SFT Configs
    TrainConfig(
        name="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=["turning_on_radio"],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_your_pretrained_checkpoint"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    # ... add more SFT configs here ...
    # 3. RFT Configs
    TrainConfig(
        name="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_rft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos-rft",
                tasks=["turning_on_radio"],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_your_pretrained_checkpoint"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_rft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos-rft",
                tasks=[
                    "turning_on_radio",
                    "picking_up_trash",
                    "hiding_Easter_eggs",
                    "wash_a_baseball_cap",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "make_microwave_popcorn",
                    "bringing_water",
                    "tidying_bedroom",
                    "putting_shoes_on_rack",
                    "setting_the_fire",
                    "cook_hot_dogs",
                    "freeze_pies",
                    "putting_away_Halloween_decorations",
                    "cleaning_up_plates_and_food",
                    "can_meat",
                    "setting_mousetraps",
                    "picking_up_toys",
                    "rearranging_kitchen_furniture",
                    "putting_up_Christmas_decorations_inside",
                    "set_up_a_coffee_station_in_your_kitchen",
                    "putting_dishes_away_after_cleaning",
                    "preparing_lunch_box",
                    "loading_the_car",
                    "carrying_in_groceries",
                    "bringing_in_wood",
                    "moving_boxes_to_storage",
                    "outfit_a_basic_toolbox",
                    "sorting_vegetables",
                    "collecting_childrens_toys",
                    "boxing_books_up_for_storage",
                    "storing_food",
                    "clearing_food_from_table_into_fridge",
                    "assembling_gift_baskets",
                    "sorting_household_items",
                    "getting_organized_for_work",
                    "clean_up_your_desk",
                    "clean_boxing_gloves",
                    "wash_dog_toys",
                    "clean_a_patio",
                    "clean_a_trumpet",
                    "spraying_for_bugs",
                    "spraying_fruit_trees",
                    "cook_cabbage",
                    "chop_an_onion",
                    "slicing_vegetables",
                    "chopping_wood",
                    "cook_bacon",
                    "canning_food",
                    "make_pizza",
                ],
                fine_grained_level=0,  # 0, 1, 2
                train_task_type="regular",  # regular | cumulate | mixture
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_your_pretrained_checkpoint"),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        # closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        # closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        # raise ValueError(f"Config '{config_name}' not found.{closest_str}")
        return _CONFIGS_DICT[""]

    return _CONFIGS_DICT[config_name]
