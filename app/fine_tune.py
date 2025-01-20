# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from datasets import load_dataset

# def preprocess_squad(batch, tokenizer):
#     inputs = [f"Context: {context}\nQuestion: {question}" for context, question in zip(batch["context"], batch["question"])]
#     targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in batch["answers"]]
#     tokenized_inputs = tokenizer(inputs, truncation=True, padding=True, max_length=512)
#     tokenized_targets = tokenizer(targets, truncation=True, padding=True, max_length=128)
#     tokenized_inputs["labels"] = tokenized_targets["input_ids"]
#     return tokenized_inputs

# def fine_tune_squad():
#     model_name = "distilgpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Add padding token if missing
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.resize_token_embeddings(len(tokenizer))  # Resize embeddings for the new token

#     # Load SQuAD dataset
#     dataset = load_dataset("squad")

#     # Tokenize dataset
#     tokenized_dataset = dataset.map(lambda x: preprocess_squad(x, tokenizer), batched=True)

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir="./distilgpt2-squad",
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=4,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         logging_dir="./logs",
#     )

#     # Define Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset["train"],
#         eval_dataset=tokenized_dataset["validation"],
#         tokenizer=tokenizer,
#     )

#     # Train model
#     trainer.train()
#     model.save_pretrained("./distilgpt2-squad")
#     tokenizer.save_pretrained("./distilgpt2-squad")

# if __name__ == "__main__":
#     fine_tune_squad()




# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from datasets import load_dataset

# def preprocess_squad(batch, tokenizer):
#     inputs = [f"Context: {context}\nQuestion: {question}" for context, question in zip(batch["context"], batch["question"])]
#     targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in batch["answers"]]

#     # Tokenize inputs
#     tokenized_inputs = tokenizer(
#         inputs,
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#     )

#     # Tokenize targets
#     tokenized_targets = tokenizer(
#         targets,
#         truncation=True,
#         padding="max_length",
#         max_length=128,
#     )

#     # Add labels and replace padding tokens with -100
#     tokenized_inputs["labels"] = [
#         [(label if label != tokenizer.pad_token_id else -100) for label in labels]
#         for labels in tokenized_targets["input_ids"]
#     ]

#     return tokenized_inputs

# def fine_tune_squad():
#     model_name = "distilgpt2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Add padding token if missing
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.resize_token_embeddings(len(tokenizer))

#     # Load and preprocess SQuAD dataset
#     dataset = load_dataset("squad")
#     tokenized_dataset = dataset.map(lambda x: preprocess_squad(x, tokenizer), batched=True)

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir="./distilgpt2-squad",
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=4,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         logging_dir="./logs",
#         remove_unused_columns=False,
#     )

#     # Define Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset["train"],
#         eval_dataset=tokenized_dataset["validation"],
#         tokenizer=tokenizer,
#     )

#     # Train the model
#     trainer.train()
#     model.save_pretrained("./distilgpt2-squad")
#     tokenizer.save_pretrained("./distilgpt2-squad")

# if __name__ == "__main__":
#     fine_tune_squad()




from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

def preprocess_squad(batch, tokenizer):
    inputs = [f"Context: {context}\nQuestion: {question}" for context, question in zip(batch["context"], batch["question"])]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in batch["answers"]]

    # Tokenize inputs and targets
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(
            targets,
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    # Combine inputs and labels
    tokenized["labels"] = tokenized_targets["input_ids"]

    return tokenized

def fine_tune_squad():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Load and preprocess SQuAD dataset
    dataset = load_dataset("squad")
    tokenized_dataset = dataset.map(
        lambda batch: preprocess_squad(batch, tokenizer),
        batched=True,
        remove_columns=["id", "title", "context", "question", "answers"],
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./distilgpt2-squad",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        remove_unused_columns=False,
        report_to="none",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()
    model.save_pretrained("./distilgpt2-squad")
    tokenizer.save_pretrained("./distilgpt2-squad")

if __name__ == "__main__":
    fine_tune_squad()
