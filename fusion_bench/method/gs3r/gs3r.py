import random
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import os
import torch
from omegaconf import DictConfig
import lightning as L
from torch import Tensor, nn
from transformers import CLIPVisionModel

from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool import Seq2SeqLMPool
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils import timeit_context, seed_everything_by_time
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub, state_dict_add
from fusion_bench.utils.type import StateDictType

from .utils import (
    gs3r_utils,
    check_parameterNamesMatch,
    check_state_dicts_equal,
    state_dict_to_vector,
    vector_to_state_dict,
)


class GS3RForCLIP(BaseAlgorithm, LightningFabricMixin): # For Acc & BWT Calculation
    def __init__(
        self,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        remove_keys: Optional[List[str]] = None,
        k = -1,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        iter_num: int = 400,
        opt_only_last: bool = True,
        init_recon: bool = False,
        **kwargs,
    ):
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.remove_keys = remove_keys if remove_keys is not None else []
        self.k = k
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.iter_num = iter_num
        self.opt_only_last = opt_only_last
        self.init_recon = init_recon
        super().__init__(**kwargs)

    def run(self, modelpool):
        if self.seed is not None:
            L.seed_everything(self.seed)
        else:
            seed_everything_by_time(self.fabric)

        # Load the pre-trained model and the fine-tuned models
        pretrained_model = modelpool.load_pretrained_model()

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)
        
        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)

        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")

        finetuned_models = [modelpool.load_model(model_name) for model_name in model_names] 

        ptm_check = pretrained_model.state_dict()
        ft_checks = [model.state_dict() for model in finetuned_models]
        check_parameterNamesMatch(ft_checks + [ptm_check])

        with timeit_context("Flattening out Checkpoints"):
            task_vectors = [state_dict_sub(check, ptm_check) for check in ft_checks]

        merged_vec = task_vectors[0]

        if self.evaluate_on_every_step:
            self.taskpool._is_setup = False
            self.taskpool._test_datasets = DictConfig(
                {model_names[0]: self._test_datasets[model_names[0]]}
            )
            pretrained_model_copy = deepcopy(pretrained_model)
            merged_vec_copy = deepcopy(merged_vec)

            pretrained_model_copy.load_state_dict(
                state_dict_add(merged_vec_copy, pretrained_model_copy.state_dict())
            )
            report = self.taskpool.evaluate(pretrained_model_copy)
            save_to_json(report, Path(self.log_dir) / "report_0.json")

        if self.save_on_every_step:
            self.save_merged_model(pretrained_model_copy, 0)
            del(pretrained_model_copy)
            del(merged_vec_copy)

        # Storage for save rotation_matrices
        singular_values = {}

        
        # Iterative Merge and Evaluation
        for model_idx, (target_vec, model_name) in enumerate(zip(task_vectors[1:], model_names[1:])):
            model_idx += 1
            
            # Merge ##
            if self.k == -1 :
                print('Full rank')
                if self.iter_num == 0:
                    print('No Iteration')
                    merged_vec, singular_values = gs3r_utils.gs3r_merge([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values) 
                else:
                    merged_vec, singular_values = gs3r_utils.gs3r_merge_opt([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values, ntask=len(task_vectors), iter_num=self.iter_num, opt_only_last=self.opt_only_last, init_recon=self.init_recon) 
                
            else:
                print(f'Trunctaed - rank {self.k}')
                merged_vec, singular_values = gs3r_utils.gs3r_merge_trunc([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values, k_s=self.k)

            if self.evaluate_on_every_step:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in model_names[: model_idx + 1]}
                )
                pretrained_model_copy = deepcopy(pretrained_model)
                merged_vec_copy = deepcopy(merged_vec)

                pretrained_model_copy.load_state_dict(
                    state_dict_add(merged_vec_copy, pretrained_model_copy.state_dict())
                )
                report = self.taskpool.evaluate(pretrained_model_copy)
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

            if self.save_on_every_step:
                self.save_merged_model(pretrained_model_copy, model_idx)
                del(pretrained_model_copy)
                del(merged_vec_copy)

        
        # Return Final version
        pretrained_model.load_state_dict(
            state_dict_add(merged_vec, pretrained_model.state_dict())
        )

        return pretrained_model

    
    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        merged_model.save_pretrained(
            Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        )



class GS3RForFlanT5(
    BaseAlgorithm,
    LightningFabricMixin,
):

    def __init__(
        self,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        remove_keys: Optional[List[str]] = None,
        k = -1,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        iter_num: int = 400,
        opt_only_last: bool = True,
        init_recon: bool = False,
        **kwargs,
    ):
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.remove_keys = remove_keys if remove_keys is not None else []
        self.k = k
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.iter_num = iter_num
        self.opt_only_last = opt_only_last
        self.init_recon = init_recon
        super().__init__(**kwargs)


    def run(self, modelpool):
        if self.seed is not None:
            L.seed_everything(self.seed)
        else:
            seed_everything_by_time(self.fabric)

        pretrained_model = modelpool.load_model("_pretrained_")

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        finetuned_models = [modelpool.load_model(model_name) for model_name in model_names]

        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")

        self.taskpool = deepcopy(self._program.taskpool) # fusion_bench.compat.taskpool.flan_t5_glue_text_generation.FlanT5GLUETextGenerationTaskPool
        model_names = [x.split('-')[-1] for x in model_names]

        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names_trimmed.json")

        self._test_datasets = deepcopy(model_names)

        ptm_check = pretrained_model.state_dict()
        ft_checks = [model.state_dict() for model in finetuned_models]
        check_parameterNamesMatch(ft_checks + [ptm_check])

        with timeit_context("Flattening out Checkpoints"):
            task_vectors = [state_dict_sub(check, ptm_check) for check in ft_checks]

        merged_vec = task_vectors[0]

        if self.evaluate_on_every_step:
            self.taskpool._all_task_names = self._test_datasets[:1]
            pretrained_model_copy = deepcopy(pretrained_model)
            merged_vec_copy = deepcopy(merged_vec)

            pretrained_model_copy.load_state_dict(
                state_dict_add(merged_vec_copy, pretrained_model_copy.state_dict())
            )
            report = self.taskpool.evaluate(pretrained_model_copy)
            save_to_json(report, Path(self.log_dir) / "report_0.json")

        if self.save_on_every_step:
            self.save_merged_model(pretrained_model_copy, 0)
            del(pretrained_model_copy)
            del(merged_vec_copy)

        # Storage for save rotation_matrices
        singular_values = {}

        
        # Iterative Merge and Evaluation
        for model_idx, (target_vec, model_name) in enumerate(zip(task_vectors[1:], model_names[1:])):
            model_idx += 1
            
            # Merge ##
            if self.k == -1 :
                print('Full rank')
                if self.iter_num == 0:
                    print('No Iteration')
                    merged_vec, singular_values = gs3r_utils.gs3r_merge([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values) 
                else:
                    merged_vec, singular_values = gs3r_utils.gs3r_merge_opt_flan([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values, ntask=len(task_vectors), iter_num=self.iter_num, opt_only_last=self.opt_only_last, init_recon=self.init_recon) 
            else:
                print(f'Trunctaed - rank {self.k}')
                if self.iter_num == 0:
                    print('No Iteration')   
                    merged_vec, singular_values = gs3r_utils.gs3r_merge_trunc([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values, k_s=self.k)
                else:
                    print('Trunc + Opt')
                    merged_vec, singular_values = gs3r_utils.gs3r_merge_opt_flan_trunc([merged_vec, target_vec], self.remove_keys, self.fabric.device, idx=model_idx, singular_values=singular_values, ntask=len(task_vectors), iter_num=self.iter_num, opt_only_last=self.opt_only_last, init_recon=self.init_recon, k_s=self.k) # Nothing but actual sym


            if self.evaluate_on_every_step:
                self.taskpool._all_task_names = self._test_datasets[:model_idx + 1]
                pretrained_model_copy = deepcopy(pretrained_model)
                merged_vec_copy = deepcopy(merged_vec)

                pretrained_model_copy.load_state_dict(
                    state_dict_add(merged_vec_copy, pretrained_model_copy.state_dict())
                )
                report = self.taskpool.evaluate(pretrained_model_copy)
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

              

            if self.save_on_every_step:
                self.save_merged_model(pretrained_model_copy, model_idx)
                del(pretrained_model_copy)
                del(merged_vec_copy)

        
        # Return Final version
        pretrained_model.load_state_dict(
            state_dict_add(merged_vec, pretrained_model.state_dict())
        )

        return pretrained_model

    def save_merged_model(self, merged_model, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        merged_model.save_pretrained(
            Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        )

