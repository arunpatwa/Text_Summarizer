from transformers import pipeline, set_seed

import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch

nltk.download("punkt")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "google-t5/t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# load the dataset over Here...
dataset_meta_review = load_dataset("zqz979/meta-review")

# create the pipeline over here and import metrics libraries ...
pipe = pipeline('summarization', model = model_ckpt )
rouge_metric = load_metric('rouge')

# Text PreProcessing is done over here...
from datasets import Dataset, DatasetDict
test_dataset = pd.DataFrame(dataset_meta_review['test'])
# Dropping nan values rows from the dataframe
test_dataset = test_dataset.dropna()
# Setting the optimized datafame in dataset created...
dataset_meta_review['test'] = Dataset.from_pandas(test_dataset)

# Saving ram from crashing...
test_dataset = None

train_dataset = pd.DataFrame(dataset_meta_review['train'])
train_dataset = train_dataset.dropna()

# lowercasing all the data over here...
train_dataset['Input'] = train_dataset['Input'].str.lower()

import re

# Removing any html tags if any ...
train_dataset['Input'] = train_dataset['Input'].str.replace('<.*?>', '', regex=True)

# Removnig any Url from the dataframe ...
train_dataset['Input'] = train_dataset['Input'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)

import string
exclude = string.punctuation

# Removing Puntuation from the Input column of the train data ...
train_dataset['Input'] =  train_dataset['Input'].str.replace('[{}]'.format(string.punctuation), '')


# now for the output column as well...
train_dataset['Output'] = train_dataset['Output'].str.lower()


# Doing above preprocessing with Output column of train as well...
train_dataset['Output'] = train_dataset['Output'].str.replace('<.*?>', '', regex=True)
train_dataset['Output'] = train_dataset['Output'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)

train_dataset['Output'] =  train_dataset['Output'].str.replace('[{}]'.format(string.punctuation), '')

dataset_meta_review['train'] = Dataset.from_pandas(train_dataset)
train_dataset = None

validation_dataset = pd.DataFrame(dataset_meta_review['validation'])
validation_dataset = validation_dataset.dropna()
dataset_meta_review['validation'] = Dataset.from_pandas(validation_dataset)
validation_dataset = None



def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]



def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                               batch_size=4, device=device,
                               column_text="article",
                               column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):

        inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                        padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device),
                         length_penalty=0.8, num_beams=8, max_length=128)
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

        # Finally, we decode the generated texts,
        # replace the  token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
               for s in summaries]

        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]


        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE scores.
    score = metric.compute()
    return score


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['Input'] , max_length = 1024, truncation = True )
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['Output'], max_length = 128, truncation = True )
        
    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }
    
dataset_meta_pt = dataset_meta_review.map(convert_examples_to_features, batched = True)

from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)



trainer_args = TrainingArguments(
    output_dir='pegasus-meta-review', num_train_epochs=1, warmup_steps=1000,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=50, save_steps=1e6,
    gradient_accumulation_steps=16,learning_rate = 0.7
) 


trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_meta_pt["train"], 
                  eval_dataset=dataset_meta_pt["validation"])

