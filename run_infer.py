import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
# from accelerate import PartialState

refined_model = "GameLLM-Pretrain"

from huggingface_hub import login
login(token = "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") 
tokenizer = AutoTokenizer.from_pretrained(refined_model)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(refined_model, device_map="auto")

input_text = "Below is a game description, your task is to generate the EFG file for this game desciption.\nGame Description: A and B are negotiating the division of 10,000 pounds in cash, following this rule: first, A proposes a distribution plan. If B accepts, the negotiation ends. If B rejects, then B proposes a new plan. In this scenario, if A accepts B's plan, the negotiation ends. If A rejects, A must propose a new plan, and B has no right to reject and must accept. Additionally, due to negotiation costs and interest losses, with each additional round of negotiation, the amount of cash each party receives is reduced by a factor of 0.95.\nEFG:\n"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_beams=1)
print(tokenizer.decode(outputs[0]))
