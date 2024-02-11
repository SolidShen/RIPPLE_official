import os
import copy
from dataclasses import dataclass
from loguru import logger
import torch 
import yaml 
import time
import json
import gc
import random
import numpy as np
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu
from src.utils.string_utils import (
    SuffixManager,
    load_conversation_template,
)

from src.utils.attack_manager import (
    get_embedding_matrix,
    get_embeddings,
)

from src.utils.opt_utils import (
    get_filtered_cands,
    get_asr,
    get_logits,
    get_loss,
    get_bleu_score,
    get_jaccard_score,
    generate,
)
from src.eval.predictors import (
   GPTFuzzPredictor,
   JailbrokenPredictor,
   TDCPredictor,
   EnsemblePredictor, 
)

RTA_TOKEN = 306




# MODEL_DIR = {
#     "gptfuzz": "hubert233/GPTFuzz",
#     "jailbroken": "/data/guangyu_shen/tdc_llm/ripple/data/cls/evaluator",
#     "tdc": "TDC2023/Llama-2-13b-chat-cls-test-phase"
# }

        
@dataclass
class PromptMetrics:
    
    prompt_ids: torch.Tensor # the generated prompt tokens
    prompt_str: str # the generated prompt string
    
        
    # ==> measure whether the attack is successful
    asr: float  # measure the overlap between the generated response and the target response
    loss: float # measure the loss between the generated response and the target response
    success_word_bleu_score: float # measure the BLEU score between the generated response and the target response
    success_char_bleu_score: float # measure the BLEU score between the generated response and the target response
    success_bleu_score: float
    success_jaccard_score: float 
    
    # ==> measure whether the attack is stealthy
    overlap_word_bleu_score: float # measure the BLEU score between the generated prompts and the initial target
    overlap_char_bleu_score: float # measure the BLEU score between the generated prompts and the initial target 
    overlap_jaccard_score: float # measure the jaccard distance between the generated prompts and the initial target 
    
    generation: str
     
    def __init__(
        self,
        prompt_ids,
        prompt_str,
        asr, 
        loss,
        success_word_bleu_score,
        success_char_bleu_score,
        success_jaccard_score,
        overlap_word_bleu_score,
        overlap_char_bleu_score,
        overlap_jaccard_score,
        generation,
        ):
        self.prompt_ids = prompt_ids
        self.prompt_str = prompt_str
        self.asr = asr
        self.loss = loss
        self.success_word_bleu_score = success_word_bleu_score
        self.success_char_bleu_score = success_char_bleu_score
        self.success_jaccard_score = success_jaccard_score
        self.overlap_word_bleu_score = overlap_word_bleu_score
        self.overlap_char_bleu_score = overlap_char_bleu_score
        self.overlap_jaccard_score = overlap_jaccard_score
        self.generation = generation 

        
##################################################
# Ripple Class: 
# generate multiple prompts for a given behavior and target
###################################################
class Ripple:
     
    def __init__(
        self,
        model_name,
        tokenizer,
        model,
        config,
        logger,
        ):

        self.model_name = model_name
        self.tokenizer = tokenizer 
        self.model = model 
        self.config = config
        self.logger = logger  

        if self.config.judge_model == "gptfuzz":
            self.judge_model = GPTFuzzPredictor(self.config.judge_model_filepath[self.config.judge_model])
        elif self.config.judge_model == "tdc":
            self.judge_model = TDCPredictor(self.config.judge_model_filepath[self.config.judge_model])
        elif self.config.judge_model == "jailbroken":
            self.judge_model = JailbrokenPredictor(self.config.judge_model_filepath[self.config.judge_model])
        elif self.config.judge_model == "ensemble":
            self.judge_model = EnsemblePredictor(MODEL_DIR=self.config.judge_model_filepath)
        else:
            raise NotImplementedError(f"judge model not implemented")                
        
        # TODO: setup CONV_TEMPLATE here for different LLMs 
        self.CONV_TEMPLATE, self.COMPATIBLE_TOKEN = load_conversation_template(model_name)
        self.EOS_STR = "</s>" 
        

    def generate_prompts(
        self, 
        behavior, 
        target, 
        result_filepath,
    ):
        
         
        self.logger.info("Generating prompts ...") 
        self.logger.info("="*50)
        self.logger.info(f"Behavior: {behavior}")
        self.logger.info("="*50)
        self.logger.info(f"Target: {target}")
        self.logger.info("="*50)
        
        self.__record = {
            "prompt": [],
            "generation": [],
            "overhead": [],
            "success_jaccard_score": [],
            "overlap_jaccard_score": [],
        } 
        
        self.CONTINUE = True 
        self.NOT_ALLOWED_TOKENS = None 
        
        start_time = time.time()
                
        while self.CONTINUE:
            
            init_prompt_ids, init_prompt_str = self.__init_prompt(target)
            
            if self.config.init_strategy == "!":
                if "llama2" in self.model_name or "alpaca" in self.model_name:
                    instruction = f"{behavior}. "
                elif "vicuna" in self.model_name or "baichuan" in self.model_name or "falcon" in self.model_name:
                    instruction = f"{behavior}."
                else:
                    raise NotImplementedError(f"model name {self.model_name} not implemented")
            else: 
                
                if "llama2" in self.model_name or "alpaca" in self.model_name  or "falcon" in self.model_name:
                    instruction = " "
                elif "vicuna" in self.model_name or "baichuan" in self.model_name:
                    instruction = ""
                else:
                    raise NotImplementedError(f"model name {self.model_name} not implemented")
             
            suffix_manager = SuffixManager(
                tokenizer=self.tokenizer,
                conv_template=self.CONV_TEMPLATE,
                instruction=instruction,
                target=target,
                adv_string=init_prompt_str,
            )
            
            compatible = self.__check_compatibility(suffix_manager, init_prompt_str) 
            if compatible == False:
                self.logger.info(f"Current target is uncompatible! skip ...")
                return None           
            
            REDUCE_OVERLAPPING = False
            
            best_prompt_metrics = None 
            
            for step in range(self.config.num_steps):

                prompt_str = init_prompt_str if step == 0 else prompt_metrics.prompt_str

                
                input_ids = suffix_manager.get_input_ids(adv_string=prompt_str).to(self.model.device)


                # TODO here gonly get grad cands, put syn_cands inside greedy search
                cands = self.__get_cands(
                    input_ids=input_ids,
                    suffix_manager=suffix_manager
                )
                
                
                if self.config.search_method == "greedy": 
                    
                    prompt_metrics = self.__greedy_search(
                        input_ids=input_ids,
                        target=target,
                        suffix_manager=suffix_manager,
                        cands=cands,
                        step=step,
                        init_prompt_ids=init_prompt_ids,
                        reduce_overlapping=REDUCE_OVERLAPPING,
                    )

                elif self.config.search_method == "gcg":
                    prompt_metrics = self.__gcg(
                        input_ids=input_ids,
                        target=target,
                        suffix_manager=suffix_manager,
                        cands=cands,
                        step=step,
                    )
                
                else:
                    raise NotImplementedError(f"search method {self.config.search_method} not implemented") 
                
                prompt_metrics = self.__evaluate(
                    suffix_manager, 
                    prompt_metrics, 
                    target)
                
                self.logger.info("="*50)
                self.logger.info(f"--> Prompt: {prompt_metrics.prompt_str}")
                self.logger.info("="*50)
                self.logger.info(f"--> Target: {target}")
                self.logger.info("="*50)
                self.logger.info(f"--> Generation: {prompt_metrics.generation}")
                self.logger.info("="*50)
                self.logger.info(f"--> Success Jaccard Score: {prompt_metrics.success_jaccard_score:.4f}")
                self.logger.info("="*50)
                self.logger.info(f"--> Overlap Jaccard Score: {prompt_metrics.overlap_jaccard_score:.4f}")
                self.logger.info("="*50)
                
               
                 
                # start reduce overlapping if prompt can achieve jailbreaking asr  
                REDUCE_OVERLAPPING = self.__check_reduce_overlapping(behavior, prompt_metrics) 
                 
                SUCCESS, ASR_SUCCESS = self.__check_success_criteria(behavior, prompt_metrics) 

                best_prompt_metrics = self.__update_best_prompt(behavior, best_prompt_metrics, prompt_metrics) 

                
                if ASR_SUCCESS:
                    self.__record["prompt"].append(prompt_metrics.prompt_str)
                    self.__record["generation"].append(prompt_metrics.generation) 
                    overhead = time.time() - start_time
                    self.__record["overhead"].append(overhead)
                    self.__record["success_jaccard_score"].append(prompt_metrics.success_jaccard_score)
                    self.__record["overlap_jaccard_score"].append(prompt_metrics.overlap_jaccard_score)
                                         
                if SUCCESS:
                    self.logger.info("Jailbreaking success!")
                    self.logger.info("="*50)

                    if prompt_metrics.prompt_str not in self.__record["prompt"]:
                        self.__record["prompt"].append(prompt_metrics.prompt_str)
                        self.__record["generation"].append(prompt_metrics.generation) 
                        overhead = time.time() - start_time
                        self.__record["overhead"].append(overhead)
                        self.__record["success_jaccard_score"].append(prompt_metrics.success_jaccard_score)
                        self.__record["overlap_jaccard_score"].append(prompt_metrics.overlap_jaccard_score)
                         
                        
                    break 
                 
                del cands
                gc.collect()
            
                # early stop 
                if self.config.early_stop and step == 0 and prompt_metrics.loss > 1:
                    break     
                
             
            
            #* add best prompt to record if not success 
            if not self.CONTINUE and best_prompt_metrics is not None:
                self.logger.debug(f"Add best prompt to record")
                if best_prompt_metrics.prompt_str not in self.__record["prompt"]:
                    self.__record["prompt"].append(best_prompt_metrics.prompt_str)
                    self.__record["generation"].append(best_prompt_metrics.generation)
                    overhead = time.time() - start_time
                    self.__record["overhead"].append(overhead)
                    self.__record["success_jaccard_score"].append(best_prompt_metrics.success_jaccard_score)
                    self.__record["overlap_jaccard_score"].append(best_prompt_metrics.overlap_jaccard_score)
            
            self.CONTINUE = False 
             
        if len(self.__record["generation"]) > 0:
            self.logger.info(f"Dump results...")
            self.__dump_results(behavior, target, result_filepath)
        
        return self.__record  
        
         
    def __update_best_prompt(self, behavior, best_prompt, metrics):
        
        pred = self.judge_model.predict([behavior, metrics.generation])
        JUDGE_SUCCESS = True if pred == 1 else False
        
        JACCARD_SUCCESS = True if metrics.success_jaccard_score >= self.config.jailbreaking_threshold else False
         
         
        if JUDGE_SUCCESS and JACCARD_SUCCESS:
        # if metrics.success_jaccard_score >= self.config.jailbreaking_threshold:
            if best_prompt is None:
                best_prompt = copy.deepcopy(metrics)
            else:
                if metrics.overlap_jaccard_score <= best_prompt.overlap_jaccard_score:
                    best_prompt = copy.deepcopy(metrics)
        
        return best_prompt
                    
    
    def __init_prompt(self, target):
        if self.config.init_strategy == "!":
            init_prompt_str = "!" + " !" * (self.config.prompt_len - 1)
        elif self.config.init_strategy == "toxic_target":
            raw_init_prompt_str = f'Repeat: "{target}'
            # remove <\s> if it is the last token
            raw_init_prompt_str = raw_init_prompt_str[:-4] if raw_init_prompt_str.endswith(self.EOS_STR) else raw_init_prompt_str
            raw_init_prompt_ids = self.tokenizer(
                raw_init_prompt_str, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            raw_init_prompt_len = raw_init_prompt_ids.shape[-1]
            init_prompt_str = (
                raw_init_prompt_str + " !" * (self.config.prompt_len - 1 - raw_init_prompt_len)
            )
        elif self.config.init_strategy == "repeat&complete":
            truncate_len = self.config.init_target_len if len(target.split()) > self.config.init_target_len else len(target.split())
            target_prefix = " ".join(target.split()[:truncate_len])
            raw_init_prompt_str = f"repeat and complete: \"{target_prefix}"
            # raw_init_prompt_str = f"Repeat and complete: \"{target_prefix}"
            raw_init_prompt_str = raw_init_prompt_str[:-4] if raw_init_prompt_str.endswith(self.EOS_STR) else raw_init_prompt_str
            raw_init_prompt_ids = self.tokenizer(
                raw_init_prompt_str, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            # logger.debug(raw_init_prompt_ids)
            # input()
            raw_init_prompt_len = raw_init_prompt_ids.shape[-1]
            if "falcon" in self.model_name or "baichuan" in self.model_name:
                #* " !"  decoded as two ids in falcon tokenizers
                init_prompt_str = (
                    raw_init_prompt_str + " a" + " a" * (self.config.prompt_len - 1 - raw_init_prompt_len)
                )
                
            else:
                init_prompt_str = (
                    raw_init_prompt_str + " !" + " !" * (self.config.prompt_len - 1 - raw_init_prompt_len)
                )

        else:
            raise ValueError("Invalid init strategy")
        
        init_prompt_ids = self.tokenizer(init_prompt_str, return_tensors="pt", add_special_tokens=True)["input_ids"]
        logger.info(f"Initial prompt: {init_prompt_str}, string word len: {len(init_prompt_str.split())}, prompt ids shape: {init_prompt_ids.shape}")    
        logger.debug(f"init prompt ids: {init_prompt_ids}")
         
        return init_prompt_ids, init_prompt_str

   
    def __check_compatibility(self, suffix_manager, prompt):
        prompt_input_ids = suffix_manager.get_input_ids(prompt)[: suffix_manager._assistant_role_slice.stop]
        return False if prompt_input_ids[-1] != self.COMPATIBLE_TOKEN else True 

     
    @torch.no_grad()
    def __select_best_prompt(
        self, 
        input_ids, 
        target, 
        original_prompt_ids,
        curr_prompt_ids,
        suffix_manager,
    ):

        
        prompt_str_list = get_filtered_cands(
                self.tokenizer,
                original_prompt_ids,
                prompt_len=self.config.prompt_len,
                suffix_manager=suffix_manager,
                filter_cand=True,
                curr_control=curr_prompt_ids,
            ) 
        self.logger.debug(f"number of prompts after filtering: {len(prompt_str_list)}")
        
        
        logits, ids = get_logits(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            control_slice=suffix_manager._control_slice,     
            test_controls=prompt_str_list,
            return_ids=True,
            batch_size=self.config.batch_size,
        )

        
        loss_list, no_calib_loss_list = get_loss(
            logits,
            ids,
            suffix_manager._target_slice,
            rta_coef=self.config.rta_coef,
            head_ce_coef=self.config.head_ce_coef,
            tail_ce_coef=self.config.tail_ce_coef,
            loss_type=self.config.loss_type,
        )
        
        asr_list = get_asr(
            logits,
            ids,
            suffix_manager._target_slice,
        )
        
        word_bleu_score_list = get_bleu_score(
            prompt_str_list,
            target,
            level="word",
        )
        
        char_bleu_score_list = get_bleu_score(
            prompt_str_list,
            target,
            level="char",
        )
        
        jaccard_score_list = get_jaccard_score(
            prompt_str_list,
            target,
            self.tokenizer,
        )
        assert len(prompt_str_list) == len(loss_list) == len(asr_list) == len(word_bleu_score_list) == len(char_bleu_score_list) == len(jaccard_score_list), f"prompt_str_list: {len(prompt_str_list)}, loss_list: {len(loss_list)}, asr_list: {len(asr_list)}, word_bleu_score_list: {len(word_bleu_score_list)}, char_bleu_score_list: {len(char_bleu_score_list)}, jaccard_score_list: {len(jaccard_score_list)}" 
        for sample_id in range(len(prompt_str_list)):
            self.logger.debug(f"candidate prompt id: {sample_id}    |   loss: {loss_list[sample_id]:.4f}    |   asr: {asr_list[sample_id]:.4f}  |   word_bleu_score: {word_bleu_score_list[sample_id]:.4f}  |   char_bleu_score: {char_bleu_score_list[sample_id]:.4f}  |   jaccard_score: {jaccard_score_list[sample_id]:.4f}")
            # self.logger.debug(f"candidate prompt id: {sample_id}    |   loss: {no_calib_loss_list[sample_id]:.4f}    |   asr: {asr_list[sample_id]:.4f}  |   word_bleu_score: {word_bleu_score_list[sample_id]:.4f}  |   char_bleu_score: {char_bleu_score_list[sample_id]:.4f}  |   jaccard_score: {jaccard_score_list[sample_id]:.4f}")
        
        #* if there are multiple samples has 1.00 ASR pick the one more diverse   
        high_asr_num = (asr_list >= 0.999).nonzero(as_tuple=True)[0].shape[0] 
        if high_asr_num > 0:
            idx = torch.where(asr_list >= 0.999)[0]
            tmp_best_prompt_id = torch.topk(
                torch.tensor([jaccard_score_list[i] for i in idx]),
                1,
                largest=False,
            ).indices 
            best_prompt_id = idx[tmp_best_prompt_id]
        else:
            best_prompt_id = torch.topk(
                loss_list, 1, largest=False
            ).indices
        
        
        best_prompt = prompt_str_list[best_prompt_id]
        
        return PromptMetrics(
            prompt_ids=self.tokenizer(best_prompt, return_tensors="pt", add_special_tokens=False).input_ids,
            prompt_str=best_prompt,
            asr=asr_list[best_prompt_id].item(),
            loss=loss_list[best_prompt_id].item(),
            # loss=no_calib_loss_list[best_prompt_id].item(),
            success_word_bleu_score=None,
            success_char_bleu_score=None,
            success_jaccard_score=None,
            overlap_word_bleu_score=word_bleu_score_list[best_prompt_id],
            overlap_char_bleu_score=char_bleu_score_list[best_prompt_id],
            overlap_jaccard_score=jaccard_score_list[best_prompt_id],
            generation=None,
            )  
        
    
    def __evaluate(
        self,
        suffix_manager,
        prompt_metrics,
        target,
    ):
        input_ids = suffix_manager.get_input_ids(adv_string=prompt_metrics.prompt_str).to(self.model.device)
        
        output_ids = generate(
            self.model,
            self.tokenizer,
            input_ids,
            suffix_manager._assistant_role_slice, 
        )  
        
        gen_str = self.tokenizer.decode(output_ids).strip()
        success_char_bleu_score = get_bleu_score(
            [gen_str],
            target,
            level="char",
        )[0]
        success_word_bleu_score = get_bleu_score(
            [gen_str],
            target,
            level="word",
        )[0]
        success_jaccard_score = get_jaccard_score([target], gen_str, self.tokenizer)[0]
        
        # update prompt_metrics
        prompt_metrics.success_char_bleu_score = success_char_bleu_score
        prompt_metrics.success_word_bleu_score = success_word_bleu_score
        prompt_metrics.success_jaccard_score = success_jaccard_score
        prompt_metrics.generation = gen_str
         
        return prompt_metrics 
         
    @torch.no_grad()
    def __gcg(
        self,
        input_ids,
        target,
        suffix_manager,
        cands,
        step,
    ):

        control_toks = input_ids[suffix_manager._control_slice]
        target_toks = input_ids[suffix_manager._target_slice]
        self.logger.debug(f"--> prompt ids: {control_toks} with shape: {control_toks.shape}")
        self.logger.debug(f"--> target ids: {target_toks} with shape: {target_toks.shape}")
        
        control_toks = control_toks.to(cands.device)
        
        original_control_toks = control_toks.repeat(self.config.batch_size, 1)
        # random select batch_size positions 
        new_token_pos = torch.randint(
            0,
            len(control_toks),
            (self.config.batch_size, 1),
            device=control_toks.device,
        ).squeeze(1).type(torch.int64)
        
        new_token_val = torch.gather(
            cands[new_token_pos], 1,
            torch.randint(0, self.config.topk, (self.config.batch_size, 1),
            device=cands.device)
        ) 
        original_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        
        best_prompt = self.__select_best_prompt(
                input_ids=input_ids,
                target=target,
                original_prompt_ids=original_control_toks,
                curr_prompt_ids=control_toks,
                suffix_manager=suffix_manager,
        )
        self.logger.debug(f"best prompt: {best_prompt.prompt_str}")
        self.logger.debug(f"best prompt ids: {best_prompt.prompt_ids}")
        self.logger.info(
                f"step: {step}  |   loss: {best_prompt.loss:.4f}    |   asr: {best_prompt.asr:.4f}"
            ) 
            
        
        return best_prompt 
         
     
    #* current only support beam_size = 1, hence greedy
    @torch.no_grad()
    def __greedy_search(
        self,
        input_ids,
        target,
        suffix_manager,
        cands,
        step,
        init_prompt_ids,
        reduce_overlapping,
    ):
        control_toks = input_ids[suffix_manager._control_slice]
        self.logger.debug(f"control toks: {control_toks}")
        self.logger.debug(f"control str: {self.tokenizer.decode(control_toks, skip_special_tokens=True)}")
        self.logger.debug(f"input toks: {input_ids}")
        self.logger.debug(f"input str: {self.tokenizer.decode(input_ids, skip_special_tokens=True)}")
        
        target_toks = input_ids[suffix_manager._target_slice]
        force_mutate_ratio = self.config.force_mutate_ratio if reduce_overlapping else 0.0    
        force_mutate_pos = [x for x in range(len(control_toks)) if (control_toks[x] in target_toks) or (control_toks[x] in init_prompt_ids[:, :10].to(control_toks.device))]
        self.logger.debug(f"--> overlapping pos: {force_mutate_pos}")
        force_mutate_pos = [pos for pos in force_mutate_pos if random.random() < force_mutate_ratio]
        self.logger.debug(f"--> prompt ids: {control_toks} with shape: {control_toks.shape}")
        self.logger.debug(f"--> target ids: {target_toks} with shape: {target_toks.shape}")
        self.logger.debug(f"--> force mutate pos: {force_mutate_pos}")
        

        
        control_toks = control_toks.to(cands.device)
        
        if self.config.selective_position != cands.shape[0]:
            if reduce_overlapping:
                not_force_mutate_pos = [pos for pos in range(len(control_toks)) if pos not in force_mutate_pos]
                if len(force_mutate_pos) > self.config.selective_position:
                    position_list = force_mutate_pos
                else:
                    position_list = np.random.choice(
                        not_force_mutate_pos, self.config.selective_position - len(force_mutate_pos), replace=False
                    )
                    position_list = np.concatenate([position_list, force_mutate_pos])
            else:    
                position_list = np.random.choice(
                    range(len(control_toks)), self.config.selective_position, replace=False
                )
            if step == 0:
                # add 0, 1, 2, 3, 4, 5 in position_list if not in 
                position_list = np.concatenate([position_list, np.array([0, 1, 2, 3, 4, 5])])
                # remove duplicate
                position_list = np.unique(position_list)
            position_list = np.sort(position_list)
            self.logger.debug(f"position list: {position_list} with shape: {len(position_list)}") 
        else:
            position_list = range(len(control_toks))
        
        
        beam_pool = [control_toks]

        
        
        
        for pos in position_list: 
                
            top_indices_pos = cands[pos, :]
            
            original_control_toks = []
            
            for sample_id in range(len(beam_pool)):
                if (step == 0 and pos in [0, 1, 2, 3, 4]) or (pos in force_mutate_pos):
                    redundant_toks = (top_indices_pos == control_toks[pos].to(top_indices_pos.device)).nonzero(as_tuple=True)[0]
                    if pos in force_mutate_pos:
                        self.logger.debug(redundant_toks)
                        self.logger.debug(top_indices_pos)
                        self.logger.debug(control_toks[pos])
                    if len(redundant_toks) > 0:
                        for redundant_tok in redundant_toks:
                            top_indices_pos[redundant_tok] = top_indices_pos[-1]
                        
                         
                    tmp_control_toks = beam_pool[sample_id].repeat(self.config.topk, 1)
                else:
                    tmp_control_toks = beam_pool[sample_id].repeat(self.config.topk + 1, 1)
                
                tmp_control_toks[:self.config.topk, pos] = top_indices_pos
                original_control_toks.append(tmp_control_toks)

            original_control_toks = torch.cat(original_control_toks, dim=0)
            

            
 
            best_prompt = self.__select_best_prompt(
                input_ids=input_ids,
                target=target,
                original_prompt_ids=original_control_toks,
                curr_prompt_ids=beam_pool[0],
                suffix_manager=suffix_manager,
            )
            
            self.logger.debug(f"best prompt: {best_prompt.prompt_str}")
            self.logger.debug(f"best prompt ids: {best_prompt.prompt_ids}, shape: {best_prompt.prompt_ids.shape}")
            
             
            beam_pool = [best_prompt.prompt_ids] 
            
            is_force_muate = True if pos in force_mutate_pos else False
            tag = "force_mutate" if is_force_muate else "optional_mutate"
             
            self.logger.info(
                f"step: {step}  |   pos: {pos}[{tag}]  |   loss: {best_prompt.loss:.4f}    |   asr: {best_prompt.asr:.4f}    |   overlap_jaccard_score: {best_prompt.overlap_jaccard_score:.4f}"
            )
            
        
        return best_prompt 
            
         
    def __get_cands(
        self,
        input_ids,
        suffix_manager,
    ):
        grad_cands = self.__estimate_gradient(
                input_ids=input_ids,
                suffix_manager=suffix_manager,
            )
        syn_cands = self.__estimate_synonym(
                input_ids=input_ids,
                suffix_manager=suffix_manager,
            )
        
        assert grad_cands.shape == syn_cands.shape, \
            f"grad_cands shape: {grad_cands.shape}, syn_cands shape: {syn_cands.shape}"
        
        if self.NOT_ALLOWED_TOKENS is not None:
            for cand_pos in range(grad_cands.shape[0]):
                grad_cands[cand_pos, self.NOT_ALLOWED_TOKENS[cand_pos]] = -np.infty
                syn_cands[cand_pos, self.NOT_ALLOWED_TOKENS[cand_pos]] = -np.infty 
                
        _, grad_cands = grad_cands.topk(self.config.topk, dim=1)
        _, syn_cands = syn_cands.topk(self.config.topk + 1, dim=1)
        syn_cands = syn_cands[:, 1:]
        
        if self.config.candidate_type == "grad":
            return grad_cands
        elif self.config.candidate_type == "syn":
            return syn_cands
        elif self.config.candidate_type == "mix":
            cand_num = grad_cands.shape[1] // 2
            mix_cands = torch.cat([grad_cands[:, :cand_num], syn_cands[:, :cand_num]], dim=1)
            return mix_cands
        else:
            raise ValueError("Invalid candidate type")  
    
    
    def __estimate_synonym(
        self,
        input_ids,
        suffix_manager,
    ):
        embed_weights = get_embedding_matrix(self.model)
        input_slice = suffix_manager._control_slice
        prompt = input_ids[input_slice]

        #! put it in cpu to avoid OOM
        syn_cands = list(map(lambda x: torch.nn.functional.cosine_similarity(embed_weights, embed_weights[prompt[x]], dim=-1).detach().cpu(), range(prompt.shape[0])))
        # check decoding result, remove if too similar
        syn_cands = torch.stack(syn_cands, dim=0)
        self.logger.debug(syn_cands.shape)
        syn_cands = syn_cands.to(input_ids.device)
        # sort_syn_cands_idx = torch.stack(sort_syn_cands_idx, dim=0)

        sort_syn_cands, sort_syn_cands_idx = syn_cands.topk(self.config.topk, dim=1)

        for idx in range(sort_syn_cands_idx.shape[0]):
            decode_og_tok = self.tokenizer.decode(prompt[idx])
            decode_og_char_len = len(decode_og_tok)
            if decode_og_char_len > 2:    
                for cand_idx in range(sort_syn_cands_idx.shape[1]):
                    decode_syn = self.tokenizer.decode(sort_syn_cands_idx[idx, cand_idx])
                    char_bleu_score = sentence_bleu([decode_syn.lower().strip()], decode_og_tok.lower().strip())
                    if char_bleu_score > 0.5:
                        
                        syn_cands[idx, sort_syn_cands_idx[idx, cand_idx]] = 0.0
        
        
        if self.NOT_ALLOWED_TOKENS is not None:
            for cand_pos in range(syn_cands.shape[0]):
                syn_cands[cand_pos, self.NOT_ALLOWED_TOKENS[cand_pos]] = -np.infty 
                
         
        return syn_cands 
        
         
 
    def __estimate_gradient(
        self,
        input_ids,
        suffix_manager,
    ):
        input_slice = suffix_manager._control_slice
        target_slice = suffix_manager._target_slice
        loss_slice = suffix_manager._loss_slice
        
        embed_weights = get_embedding_matrix(self.model)
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            device=self.model.device,
            dtype=embed_weights.dtype,
        )

        one_hot.scatter_(
            1,
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype),
        )
        
        one_hot.requires_grad_()    
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
        embeds = get_embeddings(self.model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:, : input_slice.start, :],
                input_embeds,
                embeds[:, input_slice.stop :, :],
            ],
            dim=1,
        )
        
        logits = self.model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        
        pred_logits = logits[0, loss_slice, :]
        pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
        
        # debugging log 
        #######################################
        '''
        for i in range(pred_probs.shape[0]):
            pred_prob = pred_probs[i]
            k = 10 if i == 0 or i == (pred_probs.shape[0] - 1) else 2
            prob_values, indices = pred_prob.topk(k)
            logit_values = pred_logits[i, indices]
            for j in range(len(indices)):
                self.logger.debug(
                    f"top {j} prediction for {i}-th token: {indices[j]} with prob {prob_values[j]:.4f} with logit: {logit_values[j]:.4f} token: {self.tokenizer.decode(indices[j])}[{indices[j]}]"
                )
            
            gt_token = targets[i]
            self.logger.debug(
                f"***** gt token: {gt_token} gt token logit: {pred_logits[i, gt_token]:.4f} *****"
            )
            self.logger.debug("\n")
        self.logger.debug("*" * 25)
        '''
        #######################################

        if self.config.loss_type == "CE":
            loss = torch.nn.CrossEntropyLoss(reduction="none")(logits[0, loss_slice, :], targets)
            rta_loss = torch.softmax(logits[0, loss_slice], dim=-1)[:, RTA_TOKEN]
            rta_loss_coef = torch.zeros_like(rta_loss)
            rta_loss_coef[0] = self.config.rta_coef
            rta_loss = (rta_loss * rta_loss_coef).sum(dim=-1)
            
            ce_loss_coef = torch.ones_like(loss)
            ce_loss_coef[0] = self.config.head_ce_coef
            ce_loss_coef[-1] = self.config.tail_ce_coef
            
            ce_loss = (loss * ce_loss_coef).mean(dim=-1)
            loss = ce_loss + rta_loss
            
        elif self.config.loss_type == "CW":
            raise NotImplementedError("CW loss is not implemented yet!") 
        else:
            raise ValueError("Invalid loss type")
        
        loss.backward()
        
        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)
        return -grad 
         
            
     
    def __check_success_criteria(self, behavior, metrics):
        
        pred = self.judge_model.predict([behavior, metrics.generation])
        JUDGE_SUCCESS = True if pred == 1 else False
         
        JACCARD_SUCCESS = True if metrics.success_jaccard_score >= self.config.jailbreaking_threshold else False
        
        ASR_SUCCESS = True if JUDGE_SUCCESS and JACCARD_SUCCESS else False
        
        
        SUCCESS = True \
            if ASR_SUCCESS \
            and metrics.overlap_jaccard_score <= self.config.overlapping_threshold \
            else False
            
        return SUCCESS, ASR_SUCCESS
        
    
    def __check_reduce_overlapping(self, behavior, metrics):

        pred = self.judge_model.predict([behavior, metrics.generation])
        self.logger.info(f"--> judge model prediction: {pred}")
        JUDGE_SUCCESS = True if pred == 1 else False
        JACCARD_SUCCESS = True if metrics.success_jaccard_score >= self.config.jailbreaking_threshold else False
        ASR_SUCCESS = True if JUDGE_SUCCESS and JACCARD_SUCCESS else False

        return ASR_SUCCESS 

    def __dump_results(self, behavior, target, result_filepath):
        
        log = dict()
        log["behavior"] = behavior
        log["target"] = target
        log["prompt"] = self.__record["prompt"]
        log["generation"] = self.__record["generation"]
        log["overhead"] = self.__record["overhead"]
        log["success_jaccard_score"] = self.__record["success_jaccard_score"]
        log["overlap_jaccard_score"] = self.__record["overlap_jaccard_score"]
        self.logger.info(log)
        with open(result_filepath, "w") as f:
            json.dump(log, f, indent=4)
            
