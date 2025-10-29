# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
from abc import abstractmethod
from accelerate import Accelerator
from accelerate.logging import get_logger

from openlrm.runners.abstract import Runner


logger = get_logger(__name__)


class Inferrer(Runner):

    EXP_TYPE: str = None

    def __init__(self, freeze_gaussian=True):  # Add freeze_gaussian parameter
        super().__init__()

        torch._dynamo.config.disable = True
        self.accelerator = Accelerator()

        self.model : torch.nn.Module = None
        self.freeze_gaussian = freeze_gaussian
        
        # Check for FMGaussianSplatting checkpoint
        self.fmgaussian_trained = self._check_fmgaussian_checkpoint()

    def _check_fmgaussian_checkpoint(self):
        """Check if FMGaussianSplatting model has been trained"""
        # Check if checkpoint exists in default location
        checkpoint_path = "./exps/endonerf/pulling/point_cloud"
        if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
            return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def device(self):
        return self.accelerator.device

    @abstractmethod
    def _build_model(self, cfg):
        pass

    @abstractmethod
    def infer_single(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self):
        pass

    def run(self):
        
        # Run inference
        logger.info("Running inference...")
        self.infer()
