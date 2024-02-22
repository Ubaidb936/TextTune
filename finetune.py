
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftConfig, PeftModel
from multiprocessing import cpu_count
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import transformers
from promptFuncs.py import generate_prompt

BASE_MODEL_ID = ""
DATASET_NAME = "gbharti/finance-alpaca"
D_MAP = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
CHARACTER = "stock_investor"





#Load the dataset, It should be a huggingface dataset........
data = load_dataset(DATASET_NAME, split='train')



#Load the tokenizer....
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, add_eos_token=True, trust_remote_code=True)


#Generate the prompt from a datafield.....
prompt = [generate_prompt(data_point) for data_point in data]
data = data.add_column("prompt", prompt)


#Tokenize the prompt
data = data.map(lambda sample: tokenizer(sample["prompt"]),num_proc=cpu_count(), batched=True)

#remove columns 
# data = data.remove_columns(['Context', 'Response'])


#split the data into train and test
data = data.shuffle(seed=1234)
data = data.train_test_split(test_size=0.1)
train_data = data["train"]
test_data = data["test"]




# QLora..Preparing to load the Model in 4 bit...
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,  #to quantize the model to 4-bits when you load it
    bnb_4bit_use_double_quant = True, #uses a nested quantization scheme to quantize the already quantized weights
    bnb_4bit_quant_type = "nf4", #uses a special 4-bit data type for weights initialized from a normal distribution (NF4) recommended from the paper....
    bnb_4bit_compute_dtype=torch.bfloat16 #While 4-bit bitsandbytes stores weights in 4-bits, the computation still happens in 16 or 32-bit
)

#Loading the Model
model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=D_MAP,
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=True,
)


#Finding the target layers to apply lora to....
import bitsandbytes as bnb
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)



#TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"] Mistral 7B 
#Finding the target modules to apply lora to.......
TARGET_MODULES = find_all_linear_names(model)


#Preparing Lora Model
lora_config = LoraConfig(
    r=8,                                   # Number of quantization levels
    lora_alpha=32,                         # Hyperparameter for LoRA
    target_modules=TARGET_MODULES,         # Modules to apply LoRA to
    lora_dropout=0.05,                     # Dropout probability
    bias="none",                           # Type of bias
    task_type="CAUSAL_LM"                  # Task type (in this case, Causal Language Modeling)
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


# Training
NUM_WARMUP_STEPS = 30   
MAX_STEPS = 200
EVAL_FREQ = MAX_STEPS/2
SAVE_FREQ = MAX_STEPS/2
LOG_FREQ = MAX_STEPS/2
LR_SCHEDULER_TYPE="cosine"   
WEIGHT_DECAY=0.01
PROMPT_COLUMN_NAME = "prompt" 


training_args = transformers.TrainingArguments (
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, # Effective batch size becomes 4
        dataloader_drop_last=True,
        gradient_checkpointing=True,  # Only some activations are saved to calculate gradients during the backward pass, saves memory but slows training
        fp16=True,   # mixed precision
        bf16=False,
        warmup_steps = NUM_WARMUP_STEPS,
        max_steps = MAX_STEPS,
        learning_rate = 2e-4,
        logging_steps = 1,
        output_dir = character,
        optim = "paged_adamw_8bit",
        evaluation_strategy = "steps", # or epoch
        save_strategy = "steps",
        eval_steps = EVAL_FREQ,
        save_steps = SAVE_FREQ,
        logging_steps = LOG_FREQ,
        logging_dir = f"{character}/logs", 
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        weight_decay=WEIGHT_DECAY,     # avoids Overfitting
        # push_to_hub=True,
        include_tokens_per_second=True,
)



tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field=PROMPT_COLUMN_NAME,
    peft_config=lora_config,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Training starts........................")
model.config.use_cache = False
trainer.train()

