#!/usr/bin/env python3
"""
Calcola la quantizzazione GGUF ottimale per un modello MoE dato:
- numero di layer
- hidden size
- numero di esperti
- lunghezza del contesto (token)

Suppone un modello simile a Mixtral 8x7B:
  - intermediate size = hidden_size * 3.5
  - vocab size = 32000
  - dimensione modello FP16 derivata dai parametri

Restituisce la quantizzazione con la massima qualità che entra in 24 GB VRAM
(inclusi modello, KV cache e attivazioni).
"""

import sys
import math

def calculate_params(layers, hidden_size, num_experts, vocab_size=32000):
    intermediate_size = int(hidden_size * 3.5)
    # Attention parameters per layer: Q,K,V,O weights + biases + layer norms
    attn = 4 * (hidden_size ** 2) + 8 * hidden_size  # 4*H^2 + 4*H (biases) + 2*H (layer norm)
    # FFN per expert (assuming SwiGLU? but we use simple 2*hidden*intermediate)
    # Using formula derived: 7*H^2 + 4.5*H per expert
    ffn_per_expert = 7 * (hidden_size ** 2) + 4.5 * hidden_size
    ffn = num_experts * ffn_per_expert
    layer_params = attn + ffn
    total_params = vocab_size * hidden_size + layers * layer_params
    return total_params

def main():
    if len(sys.argv) != 5:
        print("Usage: ./calc_optimal_quant.py <layers> <hidden_size> <num_experts> <context_len>")
        sys.exit(1)

    try:
        layers = int(sys.argv[1])
        hidden_size = int(sys.argv[2])
        num_experts = int(sys.argv[3])
        context_len = int(sys.argv[4])
    except ValueError:
        print("All arguments must be integers")
        sys.exit(1)

    total_params = calculate_params(layers, hidden_size, num_experts)
    # Model size in FP16 (2 bytes per parameter)
    BYTES_PER_PARAM_FP16 = 2
    GB = 1024 ** 3
    model_size_fp16_gb = (total_params * BYTES_PER_PARAM_FP16) / GB

    # KV cache and activations formulas (in GB)
    # KV cache: 2 (key+value) * layers * seq_len * hidden_size * 2 bytes (fp16) / GB
    # Actually from derivation: KV_cache_GB = (4 * layers * seq_len * hidden_size) / GB
    kv_cache_gb = (4 * layers * context_len * hidden_size) / GB
    # Activations: layers * seq_len * hidden_size * 2 bytes / GB
    act_gb = (2 * layers * context_len * hidden_size) / GB

    # Quantization options: (name, bits_per_weight, quality)
    quants = [
        ("IQ3_XXS", 2.06, "low"),
        ("IQ4_XS", 3.0, "medium"),
        ("Q4_K_M", 4.0, "good"),
        ("Q5_K_M", 5.0, "high"),
    ]

    best = None
    best_idx = -1
    for idx, (name, bits, qual) in enumerate(quants):
        model_size_gb = model_size_fp16_gb * (bits / 16.0)
        total_vram = model_size_gb + kv_cache_gb + act_gb
        if total_vram <= 24.0:  # VRAM limit
            if best is None or idx > best_idx:
                best = (name, bits, qual, model_size_gb, kv_cache_gb, act_gb, total_vram)
                best_idx = idx

    print(f"Model parameters: {total_params:,}")
    print(f"FP16 model size: {model_size_fp16_gb:.2f} GB")
    print(f"Context length: {context_len} tokens")
    print(f"KV cache size: {kv_cache_gb:.2f} GB")
    print(f"Activations size: {act_gb:.2f} GB")
    print("-" * 50)
    if best:
        name, bits, qual, msize, ksize, asize, total = best
        print(f"Best quantization: {name} ({bits} bits/weight, {qual} quality)")
        print(f"  Model size: {msize:.2f} GB")
        print(f"  KV cache: {ksize:.2f} GB")
        print(f"  Activations: {asize:.2f} GB")
        print(f"  Total VRAM: {total:.2f} GB (<= 24 GB)")
    else:
        print("No quantization fits in 24 GB VRAM.")
        print("Consider reducing context length or using CPU offload.")
        # Show the smallest option
        name, bits, qual, *_ = quants[0]
        msize = model_size_fp16_gb * (quants[0][1] / 16.0)
        total = msize + kv_cache_gb + act_gb
        print(f"Smallest option (IQ3_XXS) would need {total:.2f} GB")

if __name__ == "__main__":
    main()