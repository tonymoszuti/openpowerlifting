import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, logging, set_seed
from peft import LoraConfig
import argparse
import os
import multiprocessing
from trl import SFTTrainer
import numpy as np

os.environ['HF_TOKEN'] = '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'


# Define and configure argument parser to handle command-line parameters.
parser = argparse.ArgumentParser()

# Configuration settings and parameters
parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-smol")
parser.add_argument("--subset", type=str, default="data/sql")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--dataset_text_field", type=str, default="content")

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
parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
parser.add_argument("--num_proc", type=int, default=2)#was none since i have 2 gpus why not
parser.add_argument("--push_to_hub", type=bool, default=True)
parser.add_argument("--resume_from_checkpoint", type=str, default="finetune_starcoder2/checkpoint-1000")

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

def main(args):
    # quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # LoRA configuration
    # Configuring Low-Rank Adaptation (LoRA), to inject trainable low-rank adapters into pre-trained models without modifying the original weights.
    peft_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj",],
        task_type="CAUSAL_LM",
    )
    # These target modules chosen will have adaptors on them ^^ (main modules in transformer architecture)
    # Task type is causal language modelling as we don't have a paired dataset

 # load model and dataset
    #hugging face authentication
    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )
    print_trainable_parameters(model)

# It loads a dataset with specified parameters, including handling of multiple processes for efficient data loading.
    data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        token=token,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
    )

# setup the trainer - specifically using supervised fine tuning trainer from transformers library
# SFTTrainer internally handles tokenisation and padding
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
            resume_from_checkpoint=args.resume_from_checkpoint,  # Specify checkpoint to resume from
        ),
        peft_config=peft_config,
        dataset_text_field=args.dataset_text_field,
    )

# launch
    print("Training...")

# This method performs the training loop, where the model is trained on the dataset over a number of steps defined by the max_steps parameter.
# The trainer handles the forward and backward passes, gradient accumulation, and any other training-related tasks.

    trainer.train()
    print("Saving the last checkpoint of the model")

# Saves the trained model's weights and configuration to the specified directory.
# This method creates a directory named final_checkpoint within the output_dir and stores the model files there.
# These files can later be used to reload the model without needing to retrain it.

    output_model_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(output_model_dir, exist_ok=True)
    trainer.save_model(output_model_dir)

    print("Training Done! 💥")

# Sets the random seed for reproducibility - ensure that the random number generation in the training process is deterministic, which helps in obtaining consistent results across different runs.
# This affects operations such as shuffling of the dataset, initialization of model parameters, and any other randomness in the training process.
set_seed(args.seed)

# create output directory, check it exists before training
os.makedirs(args.output_dir, exist_ok=True)

# Sets the logging level of the Hugging Face Transformers library to only show error messages.
# This reduces the amount of logging output during the training process
logging.set_verbosity_error()

# Run function to configure model, load dataset, set up trainer and training loop
main(args)