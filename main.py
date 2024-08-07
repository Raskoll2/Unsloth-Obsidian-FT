import dropbox
import os
from typing import Dict, Union
import datasets
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, HfApi
from trl import SFTTrainer

def download_all_md_files(dbx: dropbox.Dropbox, folder_path: str, data_path: str) -> None:
    if os.path.exists(os.path.join(data_path, "processed_dataset")):
        print("Dataset already exists. Skipping download.")
        return

    entries: Dict[str, Union[dropbox.files.FileMetadata, dropbox.files.FolderMetadata]] = list_folder(dbx, folder_path)
    for filename, entry in entries.items():
        if isinstance(entry, dropbox.files.FileMetadata) and filename.endswith(".md"):
            download_file(dbx, folder_path, filename, data_path)

def list_folder(dbx: dropbox.Dropbox, folder_path: str) -> Dict[str, Union[dropbox.files.FileMetadata, dropbox.files.FolderMetadata]]:
    path: str = "/" + folder_path.strip("/")
    try:
        res = dbx.files_list_folder(path)
        return {entry.name: entry for entry in res.entries}
    except dropbox.exceptions.ApiError as err:
        print("Folder listing failed:", err)
        return {}

def download_file(dbx: dropbox.Dropbox, folder_path: str, filename: str, data_path: str) -> None:
    file_path: str = os.path.join(folder_path, filename)
    local_path: str = os.path.join(data_path, filename)
    try:
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        metadata, res = dbx.files_download(file_path)
        
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if local_size == metadata.size:
                print(f"Skipping {filename}, already downloaded.")
                return

        if metadata.size < 100:
            print(f"Skipping {filename}, file size is too small.")
            return

        with open(local_path, "wb") as f:
            f.write(res.content)
        print(f"Downloaded: {filename}")

    except dropbox.exceptions.HttpError as err:
        print(f"Error downloading {filename}: {err}")

def process_data(data_path: str) -> datasets.Dataset:
    processed_dataset_path = os.path.join(data_path, "processed_dataset")

    if os.path.exists(processed_dataset_path):
        print("Loading existing dataset...")
        dataset = datasets.load_from_disk(processed_dataset_path)
    else:
        data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".md") and os.path.getsize(os.path.join(data_path, f)) >= 500]

        if data_files:
            dataset = datasets.load_dataset('text', data_files=data_files)
            dataset.save_to_disk(processed_dataset_path)
            print("Dataset processed and saved.")
        else:
            print("No files to process.")
            return None

    return dataset

def unsloth(data: datasets.Dataset, huggingface_token: str, model_name: str) -> any:
    # Set up the environment for Unsloth model fine-tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_seq_length = 512
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="openai-community/gpt2",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=load_in_4bit,
        use_auth_token=huggingface_token
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Data Preparation
    prompt = "{}"
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = [prompt.format(text) + EOS_TOKEN for text in examples["text"]]
        return {"text": texts}

    dataset = data.map(formatting_prompts_func, batched=True)

    # Fine-tuning with SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=0.0005,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()

    # Save and upload the model
    model.save_pretrained("outputs")
    tokenizer.save_pretrained("outputs")

    # Publish the model to Hugging Face
    api = HfApi()
    api.upload_folder(
        path_or_fileobj="outputs",
        path_in_repo=".",
        repo_id=model_name,
        token=huggingface_token
    )
    
    print("Model successfully published to Hugging Face.")

    # Print model stats
    print("Training statistics:")
    print(trainer_stats)

    return trainer_stats

if __name__ == "__main__":
    dropbox_token = input("Dropbox token (optional): ")
    folder_path = "/Vault/Obsidian Vault"
    data_path = "./Data"
    huggingface_token = input("Huggingface Token: ")
    model_name = input("Hugging Face Model Name: ")

    # Download Data
    dbx = dropbox.Dropbox(dropbox_token)
    download_all_md_files(dbx, folder_path, data_path)
    print("Successfully downloaded data")

    # Process data
    data = process_data(data_path)
    if data:
        # Train the model
        processed_data = unsloth(data, huggingface_token, model_name)
        print("Data processing and fine-tuning completed")
    else:
        print("No data to train on.")
