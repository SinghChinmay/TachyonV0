import torch
from transformers import GPT2Tokenizer
from tachyon_v0_model import TachyonV0
import os

def chat():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- TachyonV0 Chat Mode (Active on {device}) ---")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TachyonV0(n_embd=4096, n_layer=64, block_size=1024).to(device)
    
    model_path = "tachyon_v0.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Weights loaded from {model_path}")
    else:
        print("Warning: No weights found. Model is currently in its primal state.")

    model.eval()

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        input_ids = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0).to(device)
        generated = input_ids
        
        print("TachyonV0: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(100): # 最大100トークン生成
                idx_cond = generated[:, -1024:]
                logits, _ = model(idx_cond)
                
                # サンプリング (Top-K = 40, Temp = 0.7)
                logits = logits[:, -1, :] / 0.7
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                
                token_str = tokenizer.decode(next_token[0])
                print(token_str, end="", flush=True)
                if "<|endoftext|>" in token_str: break
        print()

if __name__ == "__main__":
    chat()
