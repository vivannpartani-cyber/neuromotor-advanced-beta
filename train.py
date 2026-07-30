import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb

from model import DeepResidualMapper
from nsd_dataset import NSDDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Neuromotor Decoder on NSD")
    parser.add_argument("--fmri_path", type=str, required=True, help="Path to NSD fMRI HDF5")
    parser.add_argument("--clip_path", type=str, required=True, help="Path to CLIP embeddings HDF5")
    parser.add_argument("--subject", type=str, default="subj01", help="NSD subject ID")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save model weights")
    parser.add_argument("--batch_size", type=int, default=128, help="Global batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--input_dim", type=int, required=True, help="Number of voxels for this subject")
    parser.add_argument("--wandb_project", type=str, default="neuromotor-nsd", help="WandB project name")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Initialize HuggingFace Accelerate
    # Automatically handles multi-GPU (DDP), mixed precision (fp16), and gradient accumulation
    accelerator = Accelerator(log_with="wandb")
    
    # Initialize wandb tracker
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args)
    )

    # 2. Setup Dataset and DataLoader
    accelerator.print(f"Loading dataset for {args.subject}...")
    train_dataset = NSDDataset(args.fmri_path, args.clip_path, subject=args.subject, split="train")
    # In a real cluster, increase num_workers. Using 0 here for stability if HDF5 isn't configured for it.
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Setup Model, Optimizer, and Loss
    model = DeepResidualMapper(input_dim=args.input_dim, hidden_dim=4096, output_dim=1024, num_blocks=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss() # Or CosineEmbeddingLoss depending on CLIP representation
    
    # 4. Prepare everything with Accelerate
    # This automatically moves models to the correct GPUs and wraps dataloaders for distributed sampling
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Training Loop
    accelerator.print("Starting distributed training...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # Only show progress bar on the main process
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")
        
        for fmri, clip_emb in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                # Forward pass
                preds = model(fmri)
                
                # Compute Loss
                loss = criterion(preds, clip_emb)
                
                # Backward pass (Accelerate handles scaling for mixed precision)
                accelerator.backward(loss)
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Log to WandB
                accelerator.log({"train_loss": loss.item()}, step=global_step)
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        progress_bar.close()
        avg_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        # 6. Save Checkpoint (Only main process saves to prevent file corruption)
        if accelerator.is_main_process and (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            # Unwrap model before saving to remove DDP wrappers
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), save_path)
            accelerator.print(f"Saved checkpoint to {save_path}")

    accelerator.end_training()
    accelerator.print("Training complete! 🎉")

if __name__ == "__main__":
    main()
