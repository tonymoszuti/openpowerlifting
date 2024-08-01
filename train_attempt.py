import torch
import transformers
from accelerate import PartialState
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, logging, set_seed
from peft import LoraConfig
import argparse
import os
import multiprocessing
from trl import SFTTrainer
import numpy as np
import psycopg2
from datasets import Dataset

os.environ['HF_TOKEN'] = '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'


# Define and configure argument parser to handle command-line parameters.
parser = argparse.ArgumentParser()

# Configuration settings and parameters
parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--db_host", type=str, default="postgres-db")
parser.add_argument("--db_port", type=str, default="5432")
parser.add_argument("--db_name", type=str, default="openpowerlifting")
parser.add_argument("--db_user", type=str, default="admin")
parser.add_argument("--db_password", type=str, default="admin")
parser.add_argument("--db_query", type=str, default="SELECT * FROM powerlifting_results_final")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--micro_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--attention_dropout", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="finetune_llama3")
parser.add_argument("--num_proc", type=int, default=2)
parser.add_argument("--push_to_hub", type=bool, default=True)
parser.add_argument("--resume_from_checkpoint", type=str, default="finetune_llama3/checkpoint-1000")

args = parser.parse_args(args=[])

### Defining key functions

# Function to print number of trainable parameters in the model
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Function to fetch data from PostgreSQL and prepare it for training
def fetch_data_from_db(args):
    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password
    )
    cur = conn.cursor()
    cur.execute(args.db_query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Convert rows to a dataset compatible format
    data = [{"content": " ".join(map(str, row))} for row in rows]
    return Dataset.from_list(data)

def main(args):
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Load model
    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )
    print_trainable_parameters(model)

    # Fetch data from PostgreSQL
    data = fetch_data_from_db(args)

    # Setup the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            fp16=args.fp16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        peft_config=peft_config,
        dataset_text_field="content",
    )

    # Launch
    print("Training...")
    trainer.train()
    print("Saving the last checkpoint of the model")

    output_model_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(output_model_dir, exist_ok=True)
    trainer.save_model(output_model_dir)
    print("Training Done! 💥")

# Set the random seed for reproducibility
set_seed(args.seed)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Set logging level to reduce output
logging.set_verbosity_error()

# Run the main function
main(args)
