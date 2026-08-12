# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        self.lr_scheduler_total_steps = kwargs.pop("lr_scheduler_total_steps", None)
        self.exact_data_resume = kwargs.pop("exact_data_resume", False)
        self.resume_training_contract = kwargs.pop("resume_training_contract", {})
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Build a scheduler against the final recovery horizon.

        ``TrainingArguments.max_steps`` may be a temporary stop point in a
        hosted-notebook stage. A fixed total keeps LR values identical across
        1000 -> 2000 -> 3000 resumes and a single uninterrupted 3000-step run.
        """
        scheduler_steps = self.lr_scheduler_total_steps or num_training_steps
        if scheduler_steps < num_training_steps:
            raise ValueError(
                "lr_scheduler_total_steps cannot be shorter than the current training run: "
                f"{scheduler_steps} < {num_training_steps}"
            )
        return super().create_scheduler(scheduler_steps, optimizer=optimizer)

    def _validate_resumed_scheduler_horizon(self, checkpoint: str) -> None:
        config_path = Path(checkpoint) / "experiment_cfg" / "conf.yaml"
        if not config_path.is_file():
            if self.lr_scheduler_total_steps is not None:
                raise ValueError(
                    f"Checkpoint {checkpoint} has no experiment_cfg/conf.yaml; cannot verify "
                    "the fixed LR-scheduler horizon for an exact resume"
                )
            return
        from omegaconf import OmegaConf

        previous = OmegaConf.load(config_path)
        previous_total = OmegaConf.select(
            previous, "training.lr_scheduler_total_steps", default=None
        )
        if previous_total != self.lr_scheduler_total_steps:
            raise ValueError(
                "Refusing an LR-inconsistent resume: checkpoint "
                f"lr_scheduler_total_steps={previous_total!r}, current run="
                f"{self.lr_scheduler_total_steps!r}. Use the same final schedule horizon "
                "for every stage."
            )
        mismatches = {}
        for dotted_key, current_value in self.resume_training_contract.items():
            previous_value = OmegaConf.select(previous, dotted_key, default=None)
            if dotted_key == "training.exact_data_resume" and previous_value is None:
                previous_value = False
            if hasattr(previous_value, "value"):
                previous_value = previous_value.value
            if previous_value != current_value:
                mismatches[dotted_key] = (previous_value, current_value)
        if mismatches:
            raise ValueError(
                f"Refusing a resume with a different optimization/training contract: {mismatches}"
            )

    def _adapter_only_checkpoints_enabled(self) -> bool:
        return os.environ.get("GR00T_ACTION_DIT_LORA_ADAPTER_CHECKPOINTS", "0") == "1"

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """Save compact LoRA state for numbered checkpoints only.

        A normal/final ``save_model`` still delegates to Transformers so the
        merged root export is a standalone checkpoint. This opt-in path is
        deliberately limited to single-process training: silently emitting a
        partial state from a sharded model would make resume irreproducible.
        """
        output_path = Path(output_dir or self.args.output_dir)
        is_numbered_checkpoint = output_path.name.startswith("checkpoint-")
        if not (self._adapter_only_checkpoints_enabled() and is_numbered_checkpoint):
            return super()._save(output_dir=str(output_path), state_dict=state_dict)

        if int(getattr(self.args, "world_size", 1)) != 1:
            raise RuntimeError(
                "Adapter-only Action-DiT LoRA checkpoints currently require world_size=1"
            )

        from safetensors.torch import save_file
        from transformers.trainer import TRAINING_ARGS_NAME

        from gr00t.model.action_dit_lora import ADAPTER_METADATA_NAME, adapter_checkpoint_metadata

        root = self.model.module if hasattr(self.model, "module") else self.model
        metadata = adapter_checkpoint_metadata(root)
        trainable_names = set(metadata["trainable_parameter_names"])
        adapter_state = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in root.named_parameters()
            if name in trainable_names
        }
        missing = sorted(trainable_names.difference(adapter_state))
        if missing:
            raise RuntimeError(f"Trainable tensors missing from adapter checkpoint: {missing}")

        output_path.mkdir(parents=True, exist_ok=True)
        # Remove only known model-weight files from this exact checkpoint. A
        # failed earlier full save must not leave shards that look resumable.
        for pattern in (
            "model*.safetensors",
            "model.safetensors.index.json",
            "pytorch_model*.bin",
            "pytorch_model.bin.index.json",
        ):
            for stale_path in output_path.glob(pattern):
                stale_path.unlink()
        save_file(
            adapter_state,
            str(output_path / "model.safetensors"),
            metadata={"format": "pt"},
        )
        (output_path / ADAPTER_METADATA_NAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if hasattr(root, "config"):
            root.config.save_pretrained(output_path)
        torch.save(self.args, output_path / TRAINING_ARGS_NAME)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # Exact replay is scientifically cleaner but can be expensive for video
        # datasets. Fast mode uses a deterministic stage-specific reseed.
        self.args.ignore_data_skip = not self.exact_data_resume
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0 and not self.exact_data_resume:
            # ``new_seed`` MUST be the same on every rank: ``ShardedMixtureDataset``
            # builds its shard schedule from this seed and partitions disjointly
            # by index, so a per-rank delta here would cause sample duplication
            # / loss across ranks. Both inputs are rank-symmetric (the dataset's
            # own seed was set rank-symmetrically at __init__, and global_step
            # is read from TrainerState which is broadcast via rendezvous).
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. This stage is reproducible, but its data order "
                "is not bitwise-equivalent to an uninterrupted run."
            )
        elif curr_global_step > 0:
            logging.warning(
                "Exact data resume enabled: replaying/skipping the deterministic iterable "
                "stream to global step %s; startup may be slow",
                curr_global_step,
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Pre-load TrainerState before super().train() so get_train_dataloader
        can read self.state.global_step (stateful samplers rely on this).
        ``resume_from_checkpoint=True`` with no checkpoint raises rather than
        silently starting fresh.
        """
        if resume_from_checkpoint is True:
            latest_checkpoint = get_last_checkpoint(self.args.output_dir)
            if latest_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )
        elif resume_from_checkpoint in (False, None):
            latest_checkpoint = None
        else:
            latest_checkpoint = resume_from_checkpoint  # caller passed an explicit path

        if latest_checkpoint is not None:
            latest_checkpoint = str(latest_checkpoint)
            self._validate_resumed_scheduler_horizon(latest_checkpoint)
            from gr00t.model.action_dit_lora import validate_adapter_checkpoint

            validate_adapter_checkpoint(self.model, latest_checkpoint)
            logging.info(f"Resuming from checkpoint {latest_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(latest_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=latest_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

                # Log a sample of ground-truth vs predicted action tokens from
                # the first batch element so users can verify the model is
                # learning the right behaviors.
                shifted_labels = inputs["labels"][:1, 1:].cpu()
                shifted_preds = preds[:1, :-1]
                mask_0 = shifted_labels[0] != -100
                gt_tokens = shifted_labels[0][mask_0][:20]
                if self.action_offset is not None:
                    gt_tokens = gt_tokens - self.action_offset
                gt_sample = gt_tokens.tolist()
                pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
                logging.info(
                    "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                    "  GT:   %s\n  Pred: %s",
                    self.state.global_step,
                    gt_sample,
                    pred_sample,
                )

        return (loss, outputs) if return_outputs else loss
