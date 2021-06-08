# Global variables
DATA="mnist"
N_RUNS=1

# Architecture variables
N_VISIBLE=784
N_LAYERS=3
N_VARIABLES=$((4*N_LAYERS))
BATCH_SIZE=128
EPOCHS="1 1 1"

# Optimization variables
MH="hs"
N_AGENTS=1
N_ITER=1

# Iterates through all possible seeds
for SEED in $(seq 1 $N_RUNS); do
    # Optimizes an architecture
    python dbn_optimization.py ${DATA} ${MH} -n_visible ${N_VISIBLE} -n_layers ${N_LAYERS} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -seed ${SEED} --use_gpu

    # Evaluates an architecture
    python dbn_evaluation.py ${MH}_${N_AGENTS}ag_${N_VARIABLES}var_${N_ITER}it.pkl ${DATA} -n_visible ${N_VISIBLE} -n_layers ${N_LAYERS} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -seed ${SEED} --use_gpu

    # Stores files in output folder
    mv ${MH}_${N_AGENTS}ag_${N_VARIABLES}var_${N_ITER}it.pkl outputs/${MH}_${N_AGENTS}ag_${N_VARIABLES}var_${N_ITER}it_${SEED}.pkl
    mv ${MH}_${N_AGENTS}ag_${N_VARIABLES}var_${N_ITER}it.pkl.txt outputs/${MH}_${N_AGENTS}ag_${N_VARIABLES}var_${N_ITER}it_${SEED}.txt
done