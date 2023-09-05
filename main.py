# %%
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline

# %%
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased").cuda()

# %%
dataset = load_dataset("NgThVinh/dsc_model")
dataset.with_format("torch")
dataset

# %%
# dataset.push_to_hub('NgThVinh/dsc_model')

# %%
dataset['train'][:5]

# %%
dataset['train'].features

# %%
dataset['train'][0]

# %%
# max_length = 0
# for sen in dataset['train']['document']:
#     length = len(tokenizer.tokenize(sen))
#     max_length = max(length, max_length)
# max_length

# %%
def create_input_sentence(document, claim):
    return f"Given claim-document pair where claim: \"{claim}\", document: \"{document}\". Classify the claim to which class it belongs. If the claim contains information about the document, its label will be SUPPORTED, otherwise, its label will be REFUTED. In case the information of the claim cannot be verified based on the given document, its label will be NEI"

# %%
print(create_input_sentence(dataset['train'][100]['document'], dataset['train'][100]['claim']))

# %%
def preprocess_function(examples):
    inputs = tokenizer.encode_plus(
        create_input_sentence(examples["claim"], examples["document"]),
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    label = tokenizer.encode_plus(
        examples["label"],
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )

    examples["input_ids"] = inputs['input_ids'][0]
    examples["attention_mask"] = inputs['attention_mask'][0]

    examples['labels'] = label['input_ids'][0]
    
    return examples

# %%
print(preprocess_function(dataset['train'][100]))

# %%
train_dataset = dataset["train"].map(preprocess_function, remove_columns=dataset["train"].column_names)
test_dataset = dataset["test"].map(preprocess_function, remove_columns=dataset["test"].column_names)

# %%
# from transformers import DefaultDataCollator

# data_collator = DefaultDataCollator()

# %%
training_args = TrainingArguments(
    output_dir="dsc_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    # data_collator=data_collator,
)

trainer.train()
