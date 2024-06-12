import torch
from accelerate import PartialState
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AdamW
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import numpy as np
from tqdm.auto import tqdm 

# Load the model without bitsandbytes and quantization
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")

# Load SQL part of Stack Smol dataset
stack_smol_dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/sql")

# Print information about the loaded dataset
print(stack_smol_dataset)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")

# Define the function to tokenize the dataset
def tokenize_function(examples):
    inputs = examples['content']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    
    
    # Create labels
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    return model_inputs

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
tokenized_dataset = stack_smol_dataset.map(tokenize_function, batched=True, remove_columns=stack_smol_dataset["train"].column_names)

# Split dataset into training and validation sets if not already split
if "train" not in tokenized_dataset or "validation" not in tokenized_dataset:
    split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
    tokenized_dataset = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

# Print tokenized dataset to check
print(tokenized_dataset)

# create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, 
                                          references=labels)}

peft_config = LoraConfig(
    task_type="SEQ2SEQ", # sequence classification
    r=4, # intrinsic rank of trainable weight matrix
    lora_alpha=32, # this is like a learning rate
    lora_dropout=0.01, # probablity of dropout
    target_modules = ["c_proj", "c_attn", "q_attn"]
)
 # we apply lora to query layer only

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Hyperparameters
lr = 1e-3  # Learning rate
batch_size = 4  # Batch size
num_epochs = 10  # Number of epochs

# Define the model name or identifier
model_name = "starcoder2-3b"

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_name + "-peft-lora-text-sql",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=[]
)

# Check if MPS (Metal Performance Shaders) is available
#if torch.backends.mps.is_available():
    #device = torch.device("mps")
    #print("Using MPS device")
#else:
    #device = torch.device("cpu")
    #print("MPS device not found. Using CPU")
#model.to(device)
#print("Device:", device)

# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("CUDA device not found. Using CPU")
model.to(device)
print("Device:", device)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # adjust scheduler parameters accordingly

# Move model to device
model.to(device)

# Set model in train mode
model.train()

# creater trainer object
trainer = Trainer(
    model=model, # our peft model
    args=training_args, # hyperparameters
    train_dataset=tokenized_dataset["train"], # training data
    eval_dataset=tokenized_dataset["validation"], # validation data
    tokenizer=tokenizer, # define tokenizer
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics, # evaluates model using compute_metrics() function from before

)

# train model
trainer.train()


# Get the path to the latest checkpoint
#latest_checkpoint = "path/to/latest/checkpoint"  # Update this with the actual path

# Resume training from the latest checkpoint
#trainer.train(resume_from_checkpoint=latest_checkpoint)