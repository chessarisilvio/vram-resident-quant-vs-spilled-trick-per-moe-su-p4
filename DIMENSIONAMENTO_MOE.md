# Dimensionamento VRAM per modelli MoE su P40 (24GB)

## Premessa
Questo documento presenta un'analisi teorica dello spazio VRAM richiesto per eseguire modelli Mixture-of-Experts (MoE) con diverse quantizzazioni GGUF su una GPU NVIDIA P40 con 24GB di VRAM.
Consideriamo il modello Mixtral 8x7B come riferimento (46.7 miliardi di parametri totali).

## Ipotesi di calcolo
- **Parametri totali**: 46.7B (Mixtral 8x7B)
- **Numero di layer**: 32
- **Hidden size**: 4096
- **Batch size**: 1 (inferenza singola)
- **Precisione KV cache e attivazioni**: float16 (2 bytes)
- **Lunghezze di contesto testate**: 4096 (4k), 8192 (8k), 16384 (16k) token
- **Quantizzazioni GGUF considerate**:
  - IQ3_XXS: ~2.0625 bit per peso
  - IQ4_XS: ~3.0 bit per peso
  - Q4_K_M: ~4.0 bit per peso
  - Q5_K_M: ~5.0 bit per peso

## Formule utilizzate
1. **Dimensione pesi (VRAM)**:
   \[
   \text{Model Size (GB)} = \frac{\text{Total Parameters} \times \text{Bits per Weight}}{8 \times 1024^3}
   \]

2. **Dimensione KV cache**:
   \[
   \text{KV Cache (GB)} = \frac{2 \times \text{num\_layers} \times \text{batch\_size} \times \text{seq\_len} \times \text{hidden\_size} \times 2}{1024^3}
   \]
   (fattore 2 per key e value, 2 bytes per float16)

3. **Dimensione buffer attivazioni**:
   \[
   \text{Activation (GB)} = \frac{\text{num\_layers} \times \text{batch\_size} \times \text{seq\_len} \times \text{hidden\_size} \times 2}{1024^3}
   \]

4. **Totale VRAM richiesta**: Somma delle tre componenti sopra.

## Risultati

| Quantizzazione | Context (token) | Model Size (GB) | KV Cache (GB) | Activation (GB) | **Total (GB)** | Fits in 24GB? |
|----------------|-----------------|-----------------|---------------|-----------------|----------------|---------------|
| IQ3_XXS        | 4096            | 11.21           | 2.00          | 1.00            | **14.21**      | ✅ Yes        |
| IQ3_XXS        | 8192            | 11.21           | 4.00          | 2.00            | **17.21**      | ✅ Yes        |
| IQ3_XXS        | 16384           | 11.21           | 8.00          | 4.00            | **23.21**      | ✅ Yes        |
| IQ4_XS         | 4096            | 16.31           | 2.00          | 1.00            | **19.31**      | ✅ Yes        |
| IQ4_XS         | 8192            | 16.31           | 4.00          | 2.00            | **22.31**      | ✅ Yes        |
| IQ4_XS         | 16384           | 16.31           | 8.00          | 4.00            | **28.31**      | ❌ No         |
| Q4_K_M         | 4096            | 21.75           | 2.00          | 1.00            | **24.75**      | ❌ No         |
| Q4_K_M         | 8192            | 21.75           | 4.00          | 2.00            | **27.75**      | ❌ No         |
| Q4_K_M         | 16384           | 21.75           | 8.00          | 4.00            | **33.75**      | ❌ No         |
| Q5_K_M         | 4096            | 27.18           | 2.00          | 1.00            | **30.18**      | ❌ No         |
| Q5_K_M         | 8192            | 27.18           | 4.00          | 2.00            | **33.18**      | ❌ No         |
| Q5_K_M         | 16384           | 27.18           | 8.00          | 4.00            | **39.18**      | ❌ No         |

## Conclusioni
- **IQ3_XXS** rimane entro i 24GB anche con contesto 16k (23.21 GB), lasciando ~0.79 GB di margine per overhead di sistema e frammentazione.
- **IQ4_XS** entra con contesto fino a 8k (22.31 GB), ma supera il limite a 16k.
- **Q4_K_M** e superiori non entrano nemmeno con contesto 4k a causa della dimensione dei pesi già prossima al limite.
- Per sfruttare al meglio la P40 (24GB) con modelli MoE come Mixtral 8x7B, si consiglia l'uso di quantizzazioni aggressive come IQ3_XXS o IQ4_XS, a seconda della lunghezza di contesto richiesta.
- Il trick consiste nel preferire una quantizzazione più piccola ma 100% residente in VRAM piuttosto than una quantizzazione migliore che causa spill su RAM, penalizzando drasticamente la latenza.

## Note
- I calcoli sono teorici e non includono overhead aggiuntivi come memoria per il grafo di computazione, buffer di sistema o frammentazione.
- In pratica, lasciare un margine di sicurezza del 5-10% è consigliabile.
- Le dimensioni dei file GGUF possono variare leggermente a causa di overhead di metadati e allineamento.
EOF