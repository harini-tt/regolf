import modal

# Image that installs all dependencies using uv.
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch",
    "transformers", 
    "wandb", 
    "numpy",
    "datasets",
    "accelerate"
)

app = modal.App("gpt-training", image=image)

@app.function(gpu="H200:8", secrets=[modal.Secret.from_name("wandb-secret")])
def train_gpt():
    # Import all dependencies inside the function to avoid conflicts
    import torch
    import torch.nn as nn
    import wandb
    import numpy as np
    import accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    
    # Initialize wandb
    import os
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.init(project="gpt-oss-20b-training", name="dummy-loss-training")
    
    # Load gpt-oss-20b model and tokenizer
    model_name = "openai/gpt-oss-20b"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load GSM8K math dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    # Use a subset for faster training
    dataset = dataset.select(range(min(1000, len(dataset))))
    
    # Preprocess data - format as instruction-following
    def format_math_problem(example):
        question = example["question"]
        answer = example["answer"]
        # Format as instruction-response for training
        text = f"Question: {question}\n\nAnswer: {answer}<|endoftext|>"
        return {"text": text}
    
    dataset = dataset.map(format_math_problem)
    
    # Create data collator for batching
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encoding = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        return encoding
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*3)
    num_epochs = 3
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log every 50 batches
            if num_batches % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": current_lr,
                    "step": epoch * len(dataloader) + num_batches
                })
        
        avg_loss = total_loss / num_batches
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "learning_rate": 1e-4
        })
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    wandb.finish()
    return f"Training completed! Final loss: {avg_loss:.4f}"

@app.local_entrypoint()
def main():
    result = train_gpt.remote()  # runs in the cloud
    print(result)
