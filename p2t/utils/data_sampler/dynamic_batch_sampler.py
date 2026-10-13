# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic Batch Sampler for DAPO-style dynamic sampling."""

import math
from typing import Iterator, List

from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler[List[int]]):
    """
    A batch sampler that allows dynamic adjustment of batch size during training.
    
    This is useful for DAPO's dynamic sampling strategy where we may need different
    batch sizes depending on how many valid samples we need to collect after filtering.
    
    Args:
        sampler: Base sampler to draw samples from
        base_batch_size: Default batch size
        drop_last: Whether to drop the last incomplete batch
    
    Example:
        >>> from torch.utils.data import RandomSampler
        >>> sampler = RandomSampler(dataset)
        >>> batch_sampler = DynamicBatchSampler(sampler, base_batch_size=32)
        >>> dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        >>> 
        >>> # During training, adjust batch size dynamically
        >>> remaining_samples = 10
        >>> batch_sampler.set_next_batch_size(remaining_samples * 2)  # Request 20 samples
    """
    
    def __init__(
        self,
        sampler: Sampler[int],
        base_batch_size: int,
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.base_batch_size = base_batch_size
        self.drop_last = drop_last
        
        # Dynamic batch size control
        self._next_batch_size = base_batch_size
        self._current_batch_size = base_batch_size
        
        # State tracking for checkpoint/resume
        self._num_batches_yielded = 0
        self._total_samples_yielded = 0
        
        # Indices already drawn from the main sampler in the current epoch
        self._sampled_indices_in_epoch = set()

        # Supplementary pool: indices returned by DAPO to be sampled again
        self.supp_idx_list = []
        
        
    def set_next_batch_size(self, size: int):
        """
        Set the batch size for the next batch.
        Will be clamped to at least 1.
        """
        self._next_batch_size = max(1, size)
        print(f"[DynamicBatchSampler] Set next batch size to {self._next_batch_size}")
        
    def reset_batch_size(self):
        """Reset to base batch size."""
        self._next_batch_size = self.base_batch_size
        print(f"[DynamicBatchSampler] Reset next batch size to {self.base_batch_size}")
    
    def reset_for_next_epoch(self):
        self._sampled_indices_in_epoch.clear()
        self._num_batches_yielded = 0
        self._total_samples_yielded = 0
        self.supp_idx_list = []
    
    def get_remaining_samples_count(self) -> int:
        """
        Return the number of remaining sampleable items in the current epoch.

        Includes two parts:
        1. Items not yet drawn from the main sampler
        2. Items in supp_idx_list (the supplement pool)

        Returns:
            int: number of remaining un-sampled items
        """
        remaining_from_sampler = len(self.sampler) - len(self._sampled_indices_in_epoch)
        
        remaining_from_supp = len(self.supp_idx_list) if hasattr(self, 'supp_idx_list') else 0
        
        return remaining_from_sampler + remaining_from_supp
    

    def __iter__(self) -> Iterator[List[int]]:
        # Each batch is filled from the supplementary pool first (FIFO), then from the main
        # sampler, so returned prompts are reused in the next step. Supplementary indices
        # are already in _sampled_indices_in_epoch. The batch size is re-read after every
        # yield, since the driver may change it between batches.
        batch = []
        self._current_batch_size = self._next_batch_size
        main_iter = iter(self.sampler)
        main_exhausted = False

        while True:
            # ---- fill a batch: supplement pool first, then main sampler ----
            while len(batch) < self._current_batch_size:
                if self.supp_idx_list:
                    batch.append(self.supp_idx_list.pop(0))
                elif not main_exhausted:
                    try:
                        idx = next(main_iter)
                    except StopIteration:
                        main_exhausted = True
                        continue
                    # skip indices already sampled this epoch (e.g. after resume)
                    if idx in self._sampled_indices_in_epoch:
                        continue
                    batch.append(idx)
                else:
                    break   # both sources exhausted

            # ---- yield once a full batch is assembled ----
            if len(batch) == self._current_batch_size:
                self._sampled_indices_in_epoch.update(batch)
                print(f'====> reach batch_size {self._current_batch_size}.')

                yield batch
                self._num_batches_yielded += 1
                self._total_samples_yielded += len(batch)

                batch = []
                self._current_batch_size = self._next_batch_size
                continue

            # both sources exhausted
            break

        # Handle remaining (incomplete) samples
        if len(batch) > 0 and not self.drop_last:
            self._sampled_indices_in_epoch.update(batch)

            yield batch
            self._num_batches_yielded += 1
            self._total_samples_yielded += len(batch)

        self.reset_for_next_epoch()

        
    def __len__(self) -> int:
        # Use base_batch_size for length calculation (approximate)
        if self.drop_last:
            return len(self.sampler) // self.base_batch_size
        else:
            return math.ceil(len(self.sampler) / self.base_batch_size)
    
    def state_dict(self):
        """Save state for checkpoint, including the sampled indices and the supplementary pool."""
        return {
            'num_batches_yielded': self._num_batches_yielded,
            'total_samples_yielded': self._total_samples_yielded,
            'next_batch_size': self._next_batch_size,
            'current_batch_size': self._current_batch_size,
            'base_batch_size': self.base_batch_size,
            'sampled_indices_in_epoch': [int(i) for i in self._sampled_indices_in_epoch],
            'supp_idx_list': [int(i) for i in self.supp_idx_list],
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self._num_batches_yielded = state_dict.get('num_batches_yielded', 0)
        self._total_samples_yielded = state_dict.get('total_samples_yielded', 0)
        self._next_batch_size = state_dict.get('next_batch_size', self.base_batch_size)
        self._current_batch_size = state_dict.get('current_batch_size', self.base_batch_size)
        self.base_batch_size = state_dict.get('base_batch_size', self.base_batch_size)
        
        sampled_indices = state_dict.get('sampled_indices_in_epoch', [])
        self._sampled_indices_in_epoch = set(sampled_indices)

        self.supp_idx_list = list(state_dict.get('supp_idx_list', []))

        print(f"[DynamicBatchSampler] Loaded state: yielded {self._num_batches_yielded} batches, "
              f"{self._total_samples_yielded} samples, {len(self._sampled_indices_in_epoch)} indices already sampled, "
              f"{len(self.supp_idx_list)} indices in supp pool")

    def add_to_supp_idx_pool(self, idx_list):
        """
        Add indices to the supplement pool, allowing them to be resampled.

        Args:
            idx_list: list of indices to add
        """
        self.supp_idx_list += idx_list
        # The indices stay in _sampled_indices_in_epoch, so they are not counted twice.
        print(f"[DynamicBatchSampler] Added {len(idx_list)} indices to supp_idx_list. "
              f"Total remaining: {self.get_remaining_samples_count()} (supp: {len(self.supp_idx_list)})")
        
    



