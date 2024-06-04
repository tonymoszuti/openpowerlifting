from psycopg2 import connect, DatabaseError
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from torch.utils.data import Dataset, DataLoader
import torch



def connect_to_postgres():
    try:
        # Establishing the connection
        connection = connect(
            dbname="openpowerlifting",
            user="admin",
            password="admin",
            #host="postgres-db",
            host="localhost",
            port="5432"
        )
        
        print("Connection to PostgreSQL DB successful")

        return connection

    except (Exception, DatabaseError) as error:
        print(f"Error: {error}")
        return None


if __name__ == "__main__":
    # Connect to the database
    conn = connect_to_postgres()

    if conn:
        try:
            # Define the SQL query
            query = "SELECT * FROM powerlifting_results_final"

            # Load data into a pandas DataFrame
            df = pd.read_sql(query, conn)

        except Exception as e:
            print(f"Error executing query: {e}")

        finally:
            # Close the database connection
            conn.close()

# Load the CSV file containing the example SQL queries and natural language questions
df = pd.read_csv("sql_training_data.csv")

# Extract the queries and labels from the DataFrame
queries = df["question"].tolist()
labels = df["sql_query"].tolist()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("defog/llama-3-sqlcoder-8b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Adding padding token

model = AutoModelForCausalLM.from_pretrained("defog/llama-3-sqlcoder-8b")
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings in case new tokens were added

# Tokenize queries and labels
tokenized_queries = tokenizer(queries, truncation=True, padding=True, max_length=512, return_tensors="pt")
tokenized_labels = tokenizer(labels, truncation=True, padding=True, max_length=512, return_tensors="pt")

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
        labels = torch.cat([torch.tensor([tokenizer.pad_token_id]), label_input_ids[:-1]])
        
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
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_sqlcoder3b")