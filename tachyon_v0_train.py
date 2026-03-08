import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Tokenizer
from tachyon_v0_model import TachyonV0
import os
import signal
import sys

# --- CONFIGURATION ---
BATCH_SIZE = 1
BLOCK_SIZE = 1024
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "tachyon_v0.pt"
DATASET_PATH = "dataset.txt"

class SimpleStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        if not os.path.exists(DATASET_PATH):
            print(f"CRITICAL: {DATASET_PATH} not found. Please provide data.")
            return
        
        buf = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer.encode(line)
                buf.extend(tokens)
                while len(buf) > self.block_size:
                    yield torch.tensor(buf[:self.block_size]), torch.tensor(buf[1:self.block_size+1])
                    buf = buf[self.block_size:]

def train():
    print(f"--- TachyonV0 Training Initialization ---")
    print(f"Device: {DEVICE} | Model: 4096d/64L")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model = TachyonV0(n_embd=4096, n_layer=64, block_size=BLOCK_SIZE).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def save_and_exit(sig, frame):
        print("\n[SIGNAL RECEIVED] CTRL+C detected. Saving current progress to tachyon_v0.pt...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Save complete. Exiting gracefully.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_and_exit)

    print(f"Auto-save interval: 100,000 steps.")
    print("--- STARTING TRAINING ---")
    model.train()
    step = 0
    
    while True:
        dataset = SimpleStreamingDataset(tokenizer, BLOCK_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 1 == 0:
                print(f"Step {step:8d} | Loss: {loss.item():.4f}")
            
            # 10万ステップごとに保存
            if step > 0 and step % 100000 == 0:
                print(f"\n[AUTO-SAVE] Reached {step} steps. Saving to {MODEL_PATH}...")
                torch.save(model.state_dict(), MODEL_PATH)
            
            step += 1

if __name__ == "__main__":
    train()
