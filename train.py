import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("data.csv") # Using your specified file name: data.csv

# 1. CREATE THE 'text' COLUMN: Combine the Headline and Article Body
# This is crucial because your tokenizer expects a single text column.
# We use a special token [SEP] (Separator) to explicitly separate the two parts.
df['text'] = df['Headline'] + " [SEP] " + df['Article Body (Snippet)']

# 2. CONVERT LABELS: real=1, fake=0
# Note: The column name in your CSV is 'Label' (with a capital L). 
# We'll also convert the text labels to uppercase for safety before mapping.
df["label"] = df["Label"].str.upper().map({"REAL": 1, "FAKE": 0}) 

# 3. DROP UNNECESSARY COLUMNS now that 'text' and 'label' are ready
# We also drop '__index_level_0__' which is often created by Dataset.from_pandas()
df = df.drop(columns=['ID', 'Label', 'Headline', 'Article Body (Snippet)'])

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# -----------------------
# Load tokenizer & model
# -----------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    # This correctly targets the 'text' column we created
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

# Required for Trainer: Remove the raw text column and the extra index column
train_ds = train_ds.remove_columns(["text", "__index_level_0__"])
test_ds = test_ds.remove_columns(["text", "__index_level_0__"])

train_ds.set_format("torch")
test_ds.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# -----------------------
# Training settings
# -----------------------
training_args = TrainingArguments(
    output_dir="model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

# Start training
trainer.train()

# Save final model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Training complete! Model saved in folder: model/")