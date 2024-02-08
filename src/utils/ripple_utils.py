import torch
import fcntl
import json
import random
import numpy as np
import yaml
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import time
import re
import errno
import os


PERIOD_TOKEN_ID = 29889

@dataclass
class Config:
    temp: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float 
    max_query: int
    max_tokens: int
    
    search_method: str
    seed: int
    topk: int
    batch_size: int
    num_steps: int
    selective_position: int
    prompt_len: int
    init_strategy: str
    rta_coef: float
    head_ce_coef: float
    tail_ce_coef: float
    loss_type: str
    max_prompt_num: int
    force_mutate_ratio: float 
    jailbreaking_threshold: float
    overlapping_threshold: float
    candidate_type: str
    init_target_len: int
    judge_model: str
    judge_model_filepath: dict
    early_stop: bool
    model_filepath: str
    parsing_method: str
    
        
    def __init__(self, config_filepath):
        config = yaml.safe_load(open(config_filepath, "r"))
        self.temp = config["temp"]
        self.top_p = config["top_p"]
        self.frequency_penalty = config["frequency_penalty"]
        self.presence_penalty = config["presence_penalty"]
        self.max_query = config["max_query"]
        self.max_tokens = config["max_tokens"]
        self.seed = config["seed"]
        self.model_filepath = config["model_filepath"]
        self.judge_model_filepath = config["judge_model_filepath"] 
        self.search_method = config["search_method"]
        self.topk = config["topk"]
        self.batch_size = config["batch_size"]
        self.num_steps = config["num_steps"]
        self.selective_position = config["selective_position"]
        self.prompt_len = config["prompt_len"]
        self.init_strategy = config["init_strategy"]
        self.rta_coef = config["rta_coef"]
        self.head_ce_coef = config["head_ce_coef"]
        self.tail_ce_coef = config["tail_ce_coef"]
        self.loss_type = config["loss_type"]
        self.max_prompt_num = config["max_prompt_num"]
        self.force_mutate_ratio = config["force_mutate_ratio"] 
        self.jailbreaking_threshold = config["jailbreaking_threshold"]
        self.overlapping_threshold = config["overlapping_threshold"]
        self.candidate_type = config["candidate_type"] 
        self.init_target_len = config["init_target_len"]
        self.judge_model = config["judge_model"]
        self.early_stop = config["early_stop"]
        self.parsing_method = config["parsing_method"]
        
def load_everything(
    model_name,
    model_filepath,
):
    # ! do not add cuda() when using multi-gpu
    if "llama2" in model_name:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_filepath,
                torch_dtype=torch.float16,
                device_map="balanced",       
                # device_map="auto",       
            )
            .eval()
        )
        
        use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(
            model_filepath,
            padding_side="left",
            use_fast=use_fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
    
    elif "vicuna" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
                model_filepath,
                padding_side="left")
        model = (AutoModelForCausalLM.from_pretrained(
            model_filepath,
            torch_dtype=torch.float16,
            device_map="balanced",       
            )
            .eval()     
        ) 
        tokenizer.pad_token = tokenizer.eos_token
    
    elif "falcon" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_filepath,
            padding_side="left")
        model = (AutoModelForCausalLM.from_pretrained(
            model_filepath,
            torch_dtype=torch.float16,
            device_map="balanced",       
            )
            .eval()     
        ) 
        tokenizer.pad_token = tokenizer.eos_token
    
    elif "baichuan" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_filepath, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_filepath, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(model_filepath)
        model.generation_config.max_new_tokens = 256
        model = model.eval()
    elif "alpaca" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_filepath)
        model = AutoModelForCausalLM.from_pretrained(model_filepath, device_map="auto", torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.eos_token
        model = model.eval()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
     

    return model, tokenizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_all_targets(
    parsing_method,
    sample_filepath,
    tokenizer,
    prompt_len,
    logger,
):
    parsed_targets = {}
    samples = json.load(open(sample_filepath, "r"))
    for behavior in samples:
        parsed_targets[behavior] = parsing_targets(
            parsing_method=parsing_method,
            sample_filepath=sample_filepath,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            behavior=behavior,
            logger=logger,
        )
    
    json.dump(parsed_targets, open(f"{sample_filepath}.parsed", "w"), indent=4) 
    return parsed_targets


def parsing_targets(
    parsing_method,
    samples,
    tokenizer,
    prompt_len,
    logger,
):
    pattern = r'\b.*?(?<!\d)\.\s*'
    # samples = json.load(open(sample_filepath, "r"))[behavior]
    if parsing_method == "auto":
        BUFFER_TOKEN_NUM = 5
        targets = []
        for sample in samples:

            sample_toks = tokenizer(sample, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            if sample_toks.shape[-1] > (prompt_len - BUFFER_TOKEN_NUM):
                matches = re.finditer(pattern, sample)
                max_period_pos = 0
                for match in matches:
                    sub_sample = sample[:match.end()-1]
                    sub_sample_tok_len = tokenizer(sub_sample, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]
                    if sub_sample_tok_len < (prompt_len - BUFFER_TOKEN_NUM):
                        max_period_pos = match.end()-1
                
                sub_sample = sample[:max_period_pos] + "</s>"
            else:
                sub_sample = sample + "</s>"
            
            sub_sample = sub_sample.lstrip("\n")
            sub_sample = sub_sample.lstrip("0123456789. ") 
            sub_sample = sub_sample.rstrip("\n") 
             
            if len(sub_sample.split()) > 10:
                targets.append(sub_sample) 

    else:
        raise NotImplementedError(f"parsing_method: {parsing_method} is not implemented yet.")
    
    for target_id, target in enumerate(targets):
        logger.info(f"target: {target_id}/{len(targets)}, target: {target}")
    
     
    return targets 
