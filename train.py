from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import create_reference_model, PPOConfig, PPOTrainer
from iTrainingLogger import iSummaryWriter
from peft import get_peft_model, LoraConfig
import nashpy as nash
import numpy as np
import torch
import pandas as pd
import markdown
import datetime
import time

EXPERIMENT_NAME = 'LLaMa3_Game_Payoff_Table_Generation'
EXPERIMENT_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M")
LOG_PATH = f'./logs/{EXPERIMENT_NAME}_{EXPERIMENT_TIME}'
writer = iSummaryWriter(log_path=LOG_PATH, log_name='LLaMa3_Game')
writer = iSummaryWriter(log_path='./logs', log_name='LLaMa3_Game')
# Load the peft model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model_ref = create_reference_model(model)
model = get_peft_model(model, lora_config)


# PPO parameters
config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=8,
    forward_batch_size=4, # could be increased
)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)
# device = ppo_trainer.accelerator.device

messages = [
    {"role": "system", "content": """You are generating payoff table given the game description. The format of payoff table is like the following:
| Player 1 \\ Player 2 | Action 3  | Action 4   |
|----------------------|-----------|------------|
| Action 1             | (5, 5)    | (7, 8)     |
| Action 2             | (8, 7)    | (6, 6)     |
    """},
    {"role": "user", "content": """
        Number of Players: 2

        Game Description:
        The game may describe the situation of, say, player I as a restaurant owner, who can provide food of good quality (T) or bad quality (B), and a potential customer, player II, who may decide to eat there (l) or not (r). The customer prefers l to r only if the quality is good. 

        Could you generate a payoff table for this game (Use only the single letter in the table to represent the actions)?
    """},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

question_tensors = []
for _ in range(config.batch_size):
    question_tensors.append(input_ids)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

true_game = np.array([[2, 0], [3, 1]],
                     [[8, 8], [2, 2]])

def find_nash_equilibria(payoff_table_1, payoff_table_2):
    """
    Find Nash equilibria for two games.
    """
    game_1 = nash.Game(payoff_table_1[0], payoff_table_1[1])
    game_2 = nash.Game(payoff_table_2[0], payoff_table_2[1])
    
    equilibria_1 = list(game_1.support_enumeration())
    equilibria_2 = list(game_2.support_enumeration())
    
    return equilibria_1, equilibria_2

def compare_nash_equilibria(matrix_1, matrix_2):
    """
    Compare Nash equilibria of two payoff matrices.
    """

    # Find Nash equilibria
    equilibria_1, equilibria_2 = find_nash_equilibria(matrix_1, matrix_2)
    total_equilibria = len(equilibria_1) + len(equilibria_2)
    # Find intersection of equilibria sets
    common_equilibria = set(equilibria_1) & set(equilibria_2)

    return common_equilibria, len(common_equilibria), total_equilibria

# Train the model
for epoch in range(100):
    # logs, timing = dict(), dict()
    # t0 = time.time()
    logs = dict()
    rewards = []
    response_tensors = []
    for _ in range(config.batch_size):
        outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        t = time.time()
        response = outputs[0][input_ids.shape[-1]:] ## TODO: check if this is correct
        response_tensors.append(response)
        # Reward calculation
        # Get the table first
        string = tokenizer.decode(response, skip_special_tokens=True)
        start = string.index("|")  # find the first occurrence of '|' which starts the table
        end = string.index("\n\n", start)  # find the end of the table by double newline
        table_string = string[start:end].strip()
        html_text = markdown.markdown(table_string, extensions=['tables'])
        tables = pd.read_html(html_text)
        table = tables[0]
        # Sort the table
        df = table.reindex(sorted(table.columns), axis=1)
        df.set_index(df.columns[0], inplace=True)
        df_sorted = df.sort_index()
        df_sorted = df_sorted.applymap(lambda x: tuple(map(int, x.strip("()").split(","))))
        
        eq, n_eq, total_eq = compare_nash_equilibria(true_game, df_sorted.to_numpy())

        reward = (2*n_eq)/total_eq
        rewards.append(torch.tensor(reward).to(model.device))
    
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    logs.update(stats)
    
    logs['env/reward_mean'] = (sum(rewards) / len(rewards)).cpu()
    # logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    # logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")
    
    # print('Random Sample 5 text(s) of model output:')
    # for i in range(5):                                                        
        # print(f'{i+1}. {random.choice(texts)}')
    if epoch != 0 and epoch % 100 == 0:
        save_path = f'{LOG_PATH}/EPOCH{epoch}'
        model.save_pretrained(save_path, push_to_hub=False)
        tokenizer.save_pretrained(save_path, push_to_hub=False)

    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)

    writer.add_scalar("objective/entropy", stats["objective/entropy"], epoch)
    writer.add_scalar("objective/kl", stats["objective/kl"], epoch)
    # writer.add_scalar("objective/logprobs", stats["objective/logprobs"], epoch)
    # writer.add_scalar("ppo/mean_non_score_reward", stats["ppo/mean_non_score_reward"], epoch)
    writer.record()
        
        
    





