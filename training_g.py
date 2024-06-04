import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
# Load the CSV file containing the example SQL queries and natural language questions
df = pd.read_csv("./sql_training_data.csv")  # Ensure the correct path

# Extract the queries and labels from the DataFrame
queries = df["question"].tolist()
labels = df["sql_query"].tolist()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("defog/llama-3-sqlcoder-8b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Adding padding token

model = AutoModelForCausalLM.from_pretrained("defog/llama-3-sqlcoder-8b")
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings in case new tokens were added


max_len = 128 ### adjust as desired, always powers of 2 (max possible is 512)
              ### I lowered it to run faster

# Tokenize queries and labels
tokenized_queries = tokenizer(queries, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")
tokenized_labels = tokenizer(labels, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")

# Define a custom dataset class
class SQLDataset(Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels

    def __len__(self):
        return len(self.queries["input_ids"])

    def __getitem__(self, idx):
        query_input_ids = self.queries["input_ids"][idx]
        query_attention_mask = self.queries["attention_mask"][idx]
        label_input_ids = self.labels["input_ids"][idx]

        # Shift labels to the right
        # Shift labels to the right manually
        labels = torch.zeros_like(label_input_ids)
        labels[1:] = label_input_ids[:-1]
        labels[0] = tokenizer.pad_token_id
        
        return {
            "input_ids": query_input_ids, 
            "attention_mask": query_attention_mask,
            "labels": labels
        }

# Create a DataLoader for training
dataset = SQLDataset(tokenized_queries, tokenized_labels)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()  # Set the model to training mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
 
# Save the fine-tuned model
model.save_pretrained("fine_tuned_sqlcoder3b")
