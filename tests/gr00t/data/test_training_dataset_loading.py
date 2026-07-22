# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
import numpy as np
import pandas as pd


def _modality_configs() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=["cam"]),
        "mask": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
        "action": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["instruction"]),
    }


def _make_loader() -> LeRobotEpisodeLoader:
    loader = object.__new__(LeRobotEpisodeLoader)
    loader.episodes_metadata = [{"episode_index": 7, "length": 5}]
    loader.modality_configs = _modality_configs()
    loader.overlap_episode_io = False
    loader._load_parquet_data = MagicMock(
        return_value=pd.DataFrame(
            {
                "state.arm": [np.array([i, i + 1]) for i in range(5)],
                "action.arm": [np.array([i + 2]) for i in range(5)],
                "language.instruction": [f"step {i}" for i in range(5)],
            }
        )
    )

    def load_video(_episode_id, indices):
        return {"cam": np.stack([np.full((2, 2, 3), index) for index in indices])}

    def load_mask(_episode_id, indices):
        return {"cam": np.stack([np.full((2, 2), index) for index in indices])}

    loader._load_video_data = MagicMock(side_effect=load_video)
    loader._load_mask_data = MagicMock(side_effect=load_mask)
    return loader


def test_training_loader_decodes_only_required_media_indices():
    loader = _make_loader()

    episode = loader.load_episode_data(0, np.array([1, 3]))

    np.testing.assert_array_equal(episode.video_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(episode.mask_indices, [1, 3])
    np.testing.assert_array_equal(loader._load_video_data.call_args.args[1], [0, 1, 2, 3])
    np.testing.assert_array_equal(loader._load_mask_data.call_args.args[1], [1, 3])

    step = extract_step_data(
        episode,
        step_index=3,
        modality_configs=loader.modality_configs,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    assert step.images["cam"].shape == (2, 2, 2, 3)
    assert step.masks["cam"].shape == (1, 2, 2)
    np.testing.assert_array_equal(step.states["arm"], [[3, 4]])
    np.testing.assert_array_equal(step.actions["arm"], [[5]])
    assert step.text == "step 3"


def test_episode_getitem_keeps_dataframe_contract():
    loader = _make_loader()

    episode = loader[0]

    assert isinstance(episode, pd.DataFrame)
    assert len(episode) == 5
    assert "video.cam" in episode.columns
    assert "mask.cam" in episode.columns


def test_shard_loader_preloads_episodes_with_bounded_workers():
    dataset = object.__new__(ShardedSingleStepDataset)
    dataset.shard_load_workers = 2
    dataset.allow_padding = False
    dataset.sharded_episodes = [
        [
            (0, np.array([0, 1])),
            (1, np.array([2])),
            (2, np.array([3])),
        ]
    ]
    dataset.episode_loader = MagicMock()
    dataset.episode_loader.load_episode_data.side_effect = lambda idx, *_args: f"episode-{idx}"
    dataset.get_datapoint = MagicMock(side_effect=lambda episode, step: (episode, step))

    datapoints = dataset.get_shard(0)

    assert datapoints == [
        ("episode-0", 0),
        ("episode-0", 1),
        ("episode-1", 2),
        ("episode-2", 3),
    ]
    assert dataset.episode_loader.load_episode_data.call_args_list == [
        call(0, dataset.sharded_episodes[0][0][1], False),
        call(1, dataset.sharded_episodes[0][1][1], False),
        call(2, dataset.sharded_episodes[0][2][1], False),
    ]
