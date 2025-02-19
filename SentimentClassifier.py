!pip install --upgrade pip
!pip install torch torchvision torchaudio
!pip install transformers accelerate bitsandbytes
!pip install git+https://github.com/huggingface/peft.git

import os  # Provides functions to interact with the operating system
import torch  # PyTorch for deep learning and tensor computations
import pandas as pd  # Pandas for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Splitting dataset into train and validation sets
from torch.utils.data import Dataset, DataLoader  # Creating custom datasets and data loaders
from transformers import (
    AutoTokenizer,  # Tokenizer for processing text data
    AutoModelForCausalLM,  # Pre-trained causal language model
    TrainingArguments,  # Configuration settings for training
    Trainer,  # Trainer class for handling model training and evaluation
    BitsAndBytesConfig  # Configuration for model quantization (reducing memory usage)
)
from peft import LoraConfig, get_peft_model  # LoRA (Low-Rank Adaptation) for efficient fine-tuning
import random  # Standard library for generating random numbers
import numpy as np  # Numerical computations and handling arrays

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # Ensures same results on multi-GPU
np.random.seed(SEED)
random.seed(SEED)

# Ensure deterministic behavior in CuDNN (may slow down training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load training and test datasets
df_train = pd.read_csv("/train.csv")
df_test = pd.read_csv("/test.csv")

# Split training data into train and validation sets
train_df, val_df = train_test_split(
    df_train,
    test_size=0.2,
    random_state=42,
    stratify=df_train["label"]
)

def make_prompt(sentence):
    """
    Generates an instruction-based prompt for the sentiment classification task.
    """
    return (
        "Analyze the sentiment of the following sentence in an Indian language. Respond with exactly one word: Positive or Negative\n\n"
        "Sentence:\n"
        f"{sentence}\n\n"
        "Sentiment:\n"
    )

class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset class for sentiment classification.
    """
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
       
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        prompt = make_prompt(row["sentence"])
        label = row["label"]  # "Positive" or "Negative"

        # Combine prompt and label
        full_text = prompt + label

        # Tokenize input text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        
        # Labels are the same as input_ids for causal language modeling
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Load tokenizer and model
model_name = "/transformers/8b-instruct/2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not set

# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # Use 8-bit quantization to reduce memory usage
)

# Load pre-trained model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,  # Apply 8-bit quantization
    device_map="auto"  # Automatically distribute layers across available GPUs
)

# Define LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("LoRA parameters added. Trainable parameters prepared.")

# Prepare training and validation datasets
train_dataset = SentimentDataset(train_df, tokenizer)
val_dataset = SentimentDataset(val_df, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="lora_output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=100,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model at the end of each epoch
    fp16=True,  # Enable mixed precision training
    report_to="none",  # Prevent logging to external platforms
    load_best_model_at_end=True,  # Load best model checkpoint after training
    metric_for_best_model="eval_loss",  # Use validation loss to track best model
    greater_is_better=False  # Lower validation loss is better
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Set model to evaluation mode
model.eval()

predictions = []

# Generate predictions on test dataset
for i, row in df_test.iterrows():
    prompt_text = make_prompt(row["sentence"])
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.01,  # Reduce randomness in output
            do_sample=True
        )

    full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
   
    # Extract predicted sentiment label
    splitted = full_decoded.split("Sentiment:")
    if len(splitted) > 1:
        predicted_label = splitted[-1].strip()
    else:
        predicted_label = "Negative"  # Default fallback label

    # Assign final label based on heuristic
    if "Positive" in predicted_label:
        final_label = "Positive"
    elif "Negative" in predicted_label:
        final_label = "Negative"
    else:
        final_label = "Negative"  # Fallback for unexpected outputs
   
    predictions.append({"ID": row["ID"], "label": final_label})

# Save predictions to CSV file
submission_df = pd.DataFrame(predictions)
submission_df.to_csv("output.csv", index=False)
print("output.csv")
