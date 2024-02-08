import torch
import fastchat 

'''
Based on the paper "Universal and Transferable Adversarial Attacks on Aligned Language Models"
Source: https://github.com/llm-attacks/llm-attacks
Original author: Andy Zou
'''

def parse_model_name(template_name):
    if "llama2" in template_name or "llama-2" in template_name:
        template_name = "llama-2"
    elif "vicuna" in template_name:
        template_name = "vicuna_v1.1"
    elif "falcon" in template_name:
        template_name = "falcon"
    elif "baichuan2" in template_name:
        template_name = "baichuan-chat"
    elif "alpaca" in template_name:
        template_name = "alpaca"
    else:
        raise ValueError(f"Unknown model name: {template_name}")
    return template_name

def load_conversation_template(template_name):

    conv_template = fastchat.model.get_conversation_template(parse_model_name(template_name))
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    print(conv_template)

    if conv_template.name == 'llama-2':
        compatible_token = 29871
        conv_template.roles = ('[INST]', ' [/INST] ')
    elif conv_template.name == "vicuna_v1.1":
        compatible_token = 29901
        conv_template.roles = ('USER', 'ASSISTANT')
    elif conv_template.name == "falcon":
        compatible_token = 37
        conv_template.roles = ('User', 'Assistant')
    elif conv_template.name == "baichuan-chat":
        compatible_token = 196
        # conv_template.roles = (" <reserved_102> ", ' <reserved_103> ')
        conv_template.roles=("<reserved_106>", "<reserved_107>")
        conv_template.sep_style = fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    elif conv_template.name == "alpaca":
        compatible_token = 29901
        # compatible_token = 13291
        conv_template.roles = ("### Instruction", "### Response") 
    else:
        raise NotImplementedError(f"haven't implemented compatible token for {conv_template.name}")
     
    return conv_template, compatible_token


class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):
        
        
        if adv_string is not None:
            self.adv_string = adv_string
        
        if self.conv_template.name == 'llama-2' or self.conv_template.name == "alpaca": 
            self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        elif self.conv_template.name == 'vicuna_v1.1' or self.conv_template.name == 'falcon' or self.conv_template.name == "baichuan-chat":
            self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction}{self.adv_string}")
        else:
            raise NotImplementedError(f"haven't implemented compatible token for {self.conv_template.name}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()
            
        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids
        

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            
            
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            
            self._control_slice = slice(self._goal_slice.stop, len(toks))
            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

        else:
            python_tokenizer = False or "vicuna" in self.conv_template.name or "falcon" in self.conv_template.name or "alpaca" in self.conv_template.name
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True
            
            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                if "alpaca" in self.conv_template.name:
                    self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-2))
                elif "baichuan" in self.conv_template.name:
                    self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
                else: 
                    self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))
                
                separator = ' ' if self.instruction else ''
                if self.instruction:
                    if self.conv_template.name == 'llama-2' or self.conv_template.name == 'falcon' or self.conv_template.name == "baichuan-chat":
                        separator = ''
                    elif self.conv_template.name == 'vicuna_v1.1' or self.conv_template.name == "alpaca":
                        separator = ' '
                else:
                    separator = ''    
                    
                
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                
                
                if "alpaca" in self.conv_template.name:
                    self._control_slice = slice(self._goal_slice.stop, len(toks)-2) 
                elif "baichuan" in self.conv_template.name:
                    self._control_slice = slice(self._goal_slice.stop, len(toks))
                else:
                    self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                
                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
                
                
                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
                
            else:
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt
    
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

