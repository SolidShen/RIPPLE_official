import torch 
import gc 
from nltk.translate.bleu_score import sentence_bleu
RTA_TOKEN = 306




def get_filtered_cands(
    tokenizer, 
    control_cand,
    prompt_len, 
    suffix_manager,
    filter_cand=True, 
    curr_control=None,
):
    curr_control = curr_control.to(control_cand.device)
    
    cands = []
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            tmp_prompt_ids = suffix_manager.get_input_ids(adv_string=decoded_str)[suffix_manager._control_slice]
            if tmp_prompt_ids.shape[-1] == prompt_len:
                #! new added 
                if (tmp_prompt_ids == control_cand[i].to(tmp_prompt_ids.device)).all():
                    cands.append(decoded_str)
        else:
            cands.append(decoded_str)
    
     
    if len(cands) == 0:
        print(f"Warning: no valid candidates, return curr_control")
        
        #* reset suffix manager
        tmp_curr_control = suffix_manager.get_input_ids(adv_string=tokenizer.decode(curr_control.squeeze(), skip_special_tokens=True))[suffix_manager._control_slice]
        
        return [tokenizer.decode(curr_control.squeeze(), skip_special_tokens=True)]
    
     
    #* reset suffix manager
    suffix_manager.get_input_ids(adv_string=cands[0])
    return cands

def get_asr(
    logits, 
    ids, 
    target_slice,
):
    asr_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    asr_logits = logits[:, asr_slice, :]
    asr_ids = ids[:, target_slice]
    asr_preds = asr_logits.argmax(-1)
    return (asr_ids == asr_preds).float().mean(dim=-1)


# 
def get_logits(
    *,
    model,
    tokenizer,
    input_ids,
    control_slice,
    test_controls=None,
    return_ids=False,
    batch_size=512,
):
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=model.device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len)
        )
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}"
        )

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError(
            (
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), "
                f"got {test_ids.shape}"
            )
        )

    locs = (
        torch.arange(control_slice.start, control_slice.stop)
        .repeat(test_ids.shape[0], 1)
        .to(model.device)
    )
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids,
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids
        gc.collect()
        return (
            forward(
                model=model,
                input_ids=ids,
                attention_mask=attn_mask,
                batch_size=batch_size,
            ),
            ids,
        )
    else:
        del locs, test_ids
        logits = forward(
            model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size
        )
        del ids
        gc.collect()
        return logits


def get_loss(
    logits, 
    ids, 
    target_slice, 
    loss_type, 
    rta_coef, 
    head_ce_coef, 
    tail_ce_coef,
):
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    
    if loss_type == "CE":
        crit = torch.nn.CrossEntropyLoss(reduction="none")
        loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
        rta_loss = torch.softmax(logits[:, loss_slice, :], dim=-1)[:, :, RTA_TOKEN]
        rta_loss_coef = torch.zeros_like(rta_loss)
        rta_loss_coef[0] = rta_coef
        ce_loss_coef = torch.ones_like(loss)
        ce_loss_coef[:, 0] = head_ce_coef
        ce_loss_coef[:, -1] = tail_ce_coef
        rta_loss = (rta_loss * rta_loss_coef).mean(dim=-1) 
        ce_loss = (loss * ce_loss_coef).mean(dim=-1)
        
         
        no_calib_loss = loss.mean(dim=-1)
        
        return ce_loss + rta_loss, no_calib_loss + rta_loss
    elif loss_type == "CW":
        pred_logits = logits[:, loss_slice, :]
        targets = ids[:, target_slice]
        max_logits = pred_logits.max(dim=-1).values 
        target_logits = torch.gather(pred_logits, -1, targets.unsqueeze(-1)).squeeze(-1)
        
        ce_loss_coef = torch.ones_like(target_logits)
        ce_loss_coef[:, 0] = head_ce_coef
        ce_loss_coef[:, -1] = tail_ce_coef
        
        eps = 1e-1
        cw_loss = (max_logits + eps - target_logits) * ce_loss_coef 
        loss = cw_loss.sum(dim=-1)
        return loss
    else:
        raise ValueError(f"loss_type must be CE or CW, got {loss_type}")
            

def get_bleu_score(
    prompt_str_list,
    target,
    level,
):
    if level == "char":
        return list(map(lambda x: sentence_bleu([x], target), prompt_str_list))
    elif level == "word":
        return list(map(lambda x: sentence_bleu([x.split()], target.split()), prompt_str_list))
    else:
        raise ValueError(f"level must be word or char, got {level}")
    
    
def get_jaccard_score(
    prompt_str_list,
    target,
    tokenizer,
):
    def jaccard_sim_score(sen_a, sen_b, tokenizer):
        tokens_a = tokenizer.tokenize(sen_a.lower())
        tokens_b = tokenizer.tokenize(sen_b.lower())
        
        jaccard_score = 0
        jaccard_score = [1 if tok in tokens_b else 0 for tok in tokens_a]
        return sum(jaccard_score) / len(tokens_a)

    return list(map(lambda x: jaccard_sim_score(x, target, tokenizer), prompt_str_list))

def get_candidates():
    raise NotImplementedError

# 
def get_logits(
    *,
    model,
    tokenizer,
    input_ids,
    control_slice,
    test_controls=None,
    return_ids=False,
    batch_size=512,
):
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=model.device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len)
        )
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}"
        )

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError(
            (
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), "
                f"got {test_ids.shape}"
            )
        )

    locs = (
        torch.arange(control_slice.start, control_slice.stop)
        .repeat(test_ids.shape[0], 1)
        .to(model.device)
    )
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids,
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids
        gc.collect()
        return (
            forward(
                model=model,
                input_ids=ids,
                attention_mask=attn_mask,
                batch_size=batch_size,
            ),
            ids,
        )
    else:
        del locs, test_ids
        logits = forward(
            model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size
        )
        del ids
        gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i : i + batch_size]
        else:
            batch_attention_mask = None

        logits.append(
            model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        )

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)

def generate(
    model,
    tokenizer,
    input_ids,
    assistant_role_slice,
    max_new_tokens=256,
    do_sample=False,
):
    
    input_ids = input_ids[: assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn_masks,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    
    return output_ids[assistant_role_slice.stop :]

    