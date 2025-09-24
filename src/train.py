import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer , GPT2LMHeadModel , Trainer , TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

# read & prepare processed data 
df = pd.read_csv(r"Chatbot for mental with RAG\data\processed_Mental_Health_FAQ.csv")
text = df["text"]
dataset = Dataset.from_pandas(df[["text"]])

# define function to load model and tokenizer
def load_model_and_tokrnizer(model_name , tokenizer_name):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # configuration 
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id


    return tokenizer , model 


# define Tokenization Function to get input_ids & attention mask 
def tokenize_function(tokenizer , examples):
    
    return tokenizer(examples["text"],padding="max_length",  
                             truncation=True,max_length=128  )

# split Data to train&test
def split_train_test(tokenized_dataset):
    # split Dataset to train ,test
    train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test["train"]
    test_dataset = train_test["test"]

    return train_dataset , test_dataset

def train_model(tokenizer,model ,train_dataset , test_dataset ):
    # GPT is CLM (Casual Langauge Model) : predict next token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)    
    #  training args

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6,  
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,
        weight_decay=0.1,
        logging_dir="./logs",
        logging_steps=10,
        logging_first_step=True,
        logging_strategy="steps",
        report_to="none",
        save_steps=300,
        overwrite_output_dir=True,
        eval_strategy="epoch", 
        disable_tqdm=False
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,  
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.train()
    trainer.save_model(r"Models\mental_health_gpt2")
    tokenizer.save_pretrained(r"Models\mental_health_gpt2")
    print(f"[INFO] model trained and saved ")


# load tokenizer & model
tokenizer , model = load_model_and_tokrnizer(model_name="gpt2" , tokenizer_name="gpt2")
# applay tokenizer on preprocessed data 
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# split data
train_dataset , test_dataset = split_train_test(tokenized_dataset)
# apply model
train_model(tokenizer,model ,train_dataset , test_dataset )