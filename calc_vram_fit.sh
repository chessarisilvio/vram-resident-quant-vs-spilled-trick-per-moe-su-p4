#!/bin/bash
# calc_vram_fit.sh
# Calcola la quantizzazione GGUF massima che sta interamente in VRAM su P40 (24GB)
# per un modello MoE dato.
#
# Uso:
#   ./calc_vram_fit.sh <layers> <hidden_size> <total_params_b> <context_len>
#
# Esempio (Mixtral 8x7B):
#   ./calc_vram_fit.sh 32 4096 46.7 4096
#
# Output:
#   Raccomandazione sulla quantizzazione da usare per stare entro 24 GB VRAM.

# Controlla gli argomenti
if [ $# -ne 4 ]; then
    echo "Utilizzo: $0 <layers> <hidden_size> <total_params_b> <context_len>"
    echo "Esempio: $0 32 4096 46.7 4096"
    exit 1
fi

LAYERS=$1
HIDDEN_SIZE=$2
TOTAL_PARAMS_B=$3
CONTEXT_LEN=$4

# Converti parametri totali da miliardi a numero assoluto
TOTAL_PARAMS=$(echo "$TOTAL_PARAMS_B * 1000000000" | bc -l)

# Costanti
VRAM_LIMIT=24  # GB
BYTES_PER_PARAM_FP16=2  # float16
GB=1073741824  # 1024^3

# Dimensione modello in FP16 (GB)
MODEL_SIZE_FP16_GB=$(echo "scale=6; $TOTAL_PARAMS * $BYTES_PER_PARAM_FP16 / $GB" | bc -l)

# Dimensione KV cache (GB) = 4 * layers * seq_len * hidden_size / GB
KV_CACHE_GB=$(echo "scale=6; 4 * $LAYERS * $CONTEXT_LEN * $HIDDEN_SIZE / $GB" | bc -l)

# Dimensione attivazioni (GB) = 2 * layers * seq_len * hidden_size / GB
ACTIVATION_GB=$(echo "scale=6; 2 * $LAYERS * $CONTEXT_LEN * $HIDDEN_SIZE / $GB" | bc -l)

# Opzioni di quantizzazione: nome, bit per peso, qualità
declare -a QUANT_NAMES=("IQ3_XXS" "IQ4_XS" "Q4_K_M" "Q5_K_M")
declare -a QUANT_BITS=(2.06 3.0 4.0 5.0)
declare -a QUALITY=("low" "medium" "good" "high")

BEST_INDEX=-1
BEST_NAME=""
BEST_BITS=0
BEST_QUAL=""
BEST_MODEL_SIZE=0
BEST_TOTAL=0

for i in "${!QUANT_NAMES[@]}"; do
    NAME=${QUANT_NAMES[$i]}
    BITS=${QUANT_BITS[$i]}
    QUAL=${QUALITY[$i]}

    # Dimensione modello quantizzato (GB)
    MODEL_SIZE_Q=$(echo "scale=6; $MODEL_SIZE_FP16_GB * $BITS / 16" | bc -l)

    # VRAM totale richiesta
    TOTAL_VRAM=$(echo "scale=6; $MODEL_SIZE_Q + $KV_CACHE_GB + $ACTIVATION_GB" | bc -l)

    # Controlla se entra nel limite
    if (( $(echo "$TOTAL_VRAM <= $VRAM_LIMIT" | bc -l) )); then
        # Aggiorna il migliore (quantizzazione più alta che entra)
        if [ $BEST_INDEX -lt $i ]; then
            BEST_INDEX=$i
            BEST_NAME=$NAME
            BEST_BITS=$BITS
            BEST_QUAL=$QUAL
            BEST_MODEL_SIZE=$MODEL_SIZE_Q
            BEST_TOTAL=$TOTAL_VRAM
        fi
    fi
done

# Use C locale for printf to avoid issues with decimal comma
export LC_NUMERIC=C

echo "Parametri modello: $TOTAL_PARAMS_B B ($TOTAL_PARAMS)"
echo "Dimensione modello FP16: $(printf "%.2f" "$MODEL_SIZE_FP16_GB") GB"
echo "Livelli: $LAYERS, Hidden size: $HIDDEN_SIZE, Contesto: $CONTEXT_LEN token"
echo "KV cache: $(printf "%.2f" "$KV_CACHE_GB") GB"
echo "Attivazioni: $(printf "%.2f" "$ACTIVATION_GB") GB"
echo "--------------------------------------------------"

if [ $BEST_INDEX -ge 0 ]; then
    printf "Migliore quantizzazione: %s (%s bit/peso, %s qualità)\n" "$BEST_NAME" "$BEST_BITS" "$BEST_QUAL"
    printf "  Dimensione modello: %.2f GB\n" "$BEST_MODEL_SIZE"
    printf "  KV cache: %.2f GB\n" "$KV_CACHE_GB"
    printf "  Attivazioni: %.2f GB\n" "$ACTIVATION_GB"
    printf "  VRAM totale: %.2f GB (<= %d GB)\n" "$BEST_TOTAL" "$VRAM_LIMIT"
else
    echo "Nessuna quantizzazione entra in $VRAM_LIMIT GB VRAM."
    echo "Considera di ridurre la lunghezza del contesto o di usare lo offload su CPU."
    # Mostra l'opzione più piccola comunque
    NAME=${QUANT_NAMES[0]}
    BITS=${QUANT_BITS[0]}
    QUAL=${QUALITY[0]}
    MODEL_SIZE_Q=$(echo "scale=6; $MODEL_SIZE_FP16_GB * $BITS / 16" | bc -l)
    TOTAL_VRAM=$(echo "scale=6; $MODEL_SIZE_Q + $KV_CACHE_GB + $ACTIVATION_GB" | bc -l)
    printf "Opzione più piccola (%s) richiederebbe: %.2f GB\n" "$NAME" "$TOTAL_VRAM"
fi