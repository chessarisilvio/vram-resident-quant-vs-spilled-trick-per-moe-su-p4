# Confronto delle quantizzazioni GGUF per MoE

Questo documento riassume le caratteristiche principali delle quantizzazioni GGUF più rilevanti per i modelli Mixture‑of‑Experts (MoE) su una GPU NVIDIA P40 (24 GB VRAM). I dati provengono dall'analisi presente in `DIMENSIONAMENTO_MOE.md`.

| Quantizzazione | Bits per peso | Dimensione modello (GB) | Velocità inferenza* | Qualità percepita** | Contesto massimo consigliato (token) |
|----------------|---------------|--------------------------|---------------------|---------------------|--------------------------------------|
| **IQ3_XXS**   | ~2.06         | 11.2                     | ★★★★★ (molto veloce) | ★★☆☆☆ (qualità ridotta) | 16 k (VRAM totale 23.2 GB) |
| **IQ4_XS**    | ~3.0          | 16.3                     | ★★★★☆               | ★★★☆☆ (qualità media) | 8 k (VRAM totale 22.3 GB) |
| **Q4_K_M**    | ~4.0          | 21.8                     | ★★★☆☆               | ★★★★☆ (qualità buona) | 4 k (VRAM totale 24.8 GB) – *non entra* |
| **Q5_K_M**    | ~5.0          | 27.2                     | ★★☆☆☆               | ★★★★★ (qualità alta) | 4 k (VRAM totale 30.2 GB) – *non entra* |

**Note:**
- *Velocità inferenza* è valutata rispetto alla stessa architettura di modello; più basso è il numero di bit, più veloce è l'esecuzione, ma la precisione diminuisce.
- **Qualità percepita** è una stima basata su benchmark pubblici (perdita di accuratezza rispetto al modello FP16). IQ3_XXS può mostrare degradazione significativa su compiti di ragionamento complesso, mentre Q5_K_M mantiene quasi tutta la precisione originale.
- Il *contesto massimo consigliato* indica la lunghezza di sequenza (token) che può essere gestita entro il limite di 24 GB VRAM, includendo modello, KV‑cache e attivazioni. I valori sono derivati dalle formule di `DIMENSIONAMENTO_MOE.md`.
- Per carichi di lavoro che richiedono contesti lunghi (≥8 k token) si consiglia **IQ4_XS**; per scenari con contesto breve (≤4 k) e necessità di alta qualità, **Q4_K_M** è l'opzione più equilibrata, sebbene richieda più VRAM di quella disponibile.
- Quando la VRAM è limitata, il *trick* consiste nell'usare una quantizzazione più aggressiva (IQ3_XXS) e, se necessario, ridurre la lunghezza del contesto o sfruttare lo **spill‑to‑CPU** per la KV‑cache.

---

*Le stelle indicano la valutazione relativa all'interno di questo set di quantizzazioni.*

**Riferimenti**
- `DIMENSIONAMENTO_MOE.md` – calcoli di dimensione VRAM per Mixtral 8×7B.
- Documentazione ufficiale GGUF di llama.cpp per le specifiche di quantizzazione.
