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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# from accelerate import PartialState
from huggingface_hub import login
login(token = "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") # Add token here

# def formatting_prompts_func(data):
#     # print("==================", len(data["Game_Description"]))
#     # print(data)
#     output_texts = []

#     # for i in range(len(example['question'])):
#     #     text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
#     #     output_texts.append(text)
#     for i in range(len(data["Game_Description"])):
#         text = f"""Below is a game description, your task is to generate the EFG file for this game desciption.
#         ### Game Description: 
#         {data["Game_Description"][i]}
#          ### EFG: 
#         {data["EFG"][i]}\n\n
#         """
#         # print("====================")
#         # print(text)
#         # print("====================")
#         output_texts.append(text)
#     # print(text)
#     return output_texts

# instruction_template = "### Game Description:"
# response_template = " ### EFG:"


# device_string = PartialState().process_index

# local_rank = os.getenv("LOCAL_RANK")
# device_string = "cuda:" + str(local_rank)

# Model from Hugging Face hub
base_model = "meta-llama/Meta-Llama-3-8B"

# New instruction dataset
# guanaco_dataset = "zczlsde/gametheory"

# Fine-tuned model
# new_model = "llama-2-7b-chat-guanaco"

# dataset = load_dataset(guanaco_dataset, split="train")
dataset = load_dataset("json", data_files="data.json", split="train")
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
# print(device_string)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map="auto"
)
# model.config.use_cache = False
# model.config.pretraining_tp = 1
print("=======================")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("=======================")
peft_params = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results_1",
    num_train_epochs=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="adafactor",
    save_steps=50,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    # dataset_text_field="text",
    # formatting_func=formatting_prompts_func,
    # data_collator=collator,
    max_seq_length=None,
    tokenizer=tokenizer,
    dataset_batch_size = 1,
    args=training_params,
    packing=False,
)

trainer.train()

# Save Model
refined_model = "GameLLM-Pretrain_test"
trainer.model.save_pretrained(refined_model)
tokenizer.save_pretrained(refined_model)
