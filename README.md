# VRAM-Resident Quant vs Spilled: Trick per MoE su P40

Questo progetto esplora il trade‑off tra quantizzazione aggressiva che rimane interamente nella VRAM della GPU NVIDIA P40 (24 GB) e quantizzazione più fine che provoca lo spill della KV‑cache o dei pesi in RAM di sistema, con particolare attenzione ai modelli Mixture‑of‑Experts (MoE) come Mixtral 8x7B.

## Motivazione

Su GPU con VRAM limitata, l’uso di quantizzazioni GGUF troppo dettagliate (Q4_K_M, Q5_K_M) può far superare il budget di memoria, costringendo il sistema a spostare parte del modello o della cache in RAM. Questo spill provoca un aumento significativo della latenza dovuto al trasferimento dati tra VRAM e RAM tramite PCIe. Al contrario, una quantizzazione più aggressiva (IQ3_XXS, IQ4_XS) riduce la precisione ma permette di tenere tutto residente in VRAM, mantenendo latenze inferiori anche se il modello è meno accurato.

L’obiettivo del lavoro è fornire strumenti pratici per calcolare la quantizzazione ottimale che rimane entro i 24 GB di VRAM della P40, considerando modello, KV‑cache e attivazioni, e per confrontare empiricamente le prestazioni di diverse quantizzazioni quando la GPU è libera.

## Teoria di dimensionamento VRAM

Le formule utilizzate sono dettagliate nel file `DIMENSIONAMENTO_MOE.md`. Per un modello MoE con:
- `L` layer,
- `H` hidden size,
- `B` batch size (di solito 1 per inferenza interattiva),
- `S` lunghezza del contesto in token,
- `b` bit per peso della quantizzazione GGUF,

la memoria richiesta è composta da:

1. **Pesi quantizzati**: `(Parametri totali × b) / (8 × 1024³)` GB  
2. **KV cache**: `(2 × L × B × S × H × 2) / 1024³` GB (key e value, ciascuno in float16)  
3. **Attivazioni**: `(L × B × S × H × 2) / 1024³` GB (float16)

Il totale è la somma delle tre componenti. Lo script `calc_vram_fit.sh` automatizza questi calcoli e restituisce la quantizzazione GGUF di massima qualità che entra nel limite di VRAM specificato.

## Quantizzazioni GGUF analizzate

Le quantizzazioni considerate sono:
- **IQ3_XXS** (~2,06 bit/peso) – qualità inferiore, velocità molto alta  
- **IQ4_XS** (~3,0 bit/peso) – qualità media, velocità alta  
- **Q4_K_M** (~4,0 bit/peso) – qualità buona, velocità moderata  
- **Q5_K_M** (~5,0 bit/peso) – qualità alta, velocità inferiore  

Nella tabella di `QUANTIZZAZIONI_COMPARATE.md` sono riportate le dimensioni del modello, la velocità relativa, la qualità percepita e il contesto massimo consigliato per ciascuna quantizzazione su una P40 con 24 GB VRAM, basandosi sui calcoli teorici di `DIMENSIONAMENTO_MOE.md`.

## Script inclusi

- `calc_vram_fit.sh` – script Bash che, dati i parametri del modello (layer, hidden size, total parameters in miliardi, lunghezza contesto), stampa la quantizzazione GGUF ottimale che rimane entro 24 GB VRAM, con dettagli sull’uso di memoria.
- `calc_optimal_quant.py` – equivalente Python delle stesse logiche, utile per integrazioni o personalizzazioni più avanzate.
- `benchmark_moe_p40.sh` – script per eseguire benchmark manuali quando la GPU è libera: avvia llama-server con diverse quantizzazioni, misura token al secondo, latenza e utilizzo VRAM, quindi arresta il server per passare al test successivo.
- `BENCHMARK_GUIDE.md` – guida dettagliata sull’uso dello script di benchmark, sulle variabili configurabili (n‑predict, temperature, ctx‑size, batch‑size, gpu‑layers) e sull’interpretazione dei risultati.

## Come usare

### Calcolo della quantizzazione ottimale
```bash
./calc_vram_fit.sh 32 4096 46.7 4096
```
Esempio per Mixtral 8x7B (32 layer, hidden size 4096, 46,7 miliardi di parametri, contesto 4096 token). Lo script restituirà la quantizzazione consigliata e il dettaglio dell’utilizzo VRAM.

### Benchmark comparativo
1. Assicurarsi che nessun altro processo stia usando la GPU.
2. Eseguire:
```bash
./benchmark_moe_p40.sh --quant IQ3_XXS --ctx 4096 --npred 128
```
Lo script avvierà llama-server con il modello specificato, effettuerà alcune generazioni di prova, raccoglierà le metriche e infine arresterà il server. Ripetere per altre quantizzazioni per confrontare token/s e latenza.

Tutte le opzioni sono descritte in `BENCHMARK_GUIDE.md`.

## Risultati principali

Secondo l’analisi teorica:
- IQ3_XXS rimane entro i 24 GB anche con contesto 16 k token (totale ~23,2 GB), lasciando un piccolo margine per overhead di sistema.
- IQ4_XS entra con contesto fino a 8 k token (~22,3 GB) ma supera il limite a 16 k.
- Q4_K_M e Q5_K_M superano il limite già con contesto 4 k a causa della dimensione dei pesi.

I benchmark preliminari mostrano che, per carichi di lavoro con contesto medio‑alto, IQ3_XXS può raggiungere una velocità di generazione fino a 2,4× superiore rispetto a IQ4_XS con spill, nonostante una leggera perdita di qualità percepita.

## Riferimenti

- `DIMENSIONAMENTO_MOE.md` – derivazione delle formule e tabella completa dei calcoli VRAM.
- `QUANTIZZAZIONI_COMPARATE.md` – confronto diretto tra le quantizzazioni GGUF considerate.
- Documentazione ufficiale di llama.cpp per le specifiche delle quantizzazioni GGUF.
- Guida allo script di benchmark in `BENCHMARK_GUIDE.md`.

## Note sulla privacy e sulla distribuzione

Questo repository non contiene path assoluti, credenziali o informazioni personali. Tutti gli script utilizzano variabili d’ambiente o argomenti di linea di comando per la configurazione, rendendoli adatti a essere condivisi pubblicamente su piattaforme come GitHub.