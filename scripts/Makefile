# --- Configuration ---
N_SCENARIOS=500
N_AGENTS=6
N_TASKS=12
RADIUS=250

PYTHON=python3

.PHONY: all clean gen run plot run-cap1 run-cap2 run-cap3 run-cap4

all: clean gen run plot
	@echo ">>> Pipeline Completed Successfully!"

# --- 1. Clean ---
clean:
	@echo ">>> Cleaning old data..."
	rm -rf ../test/scenarios/ 
	rm -rf ../test/results/

# --- 2. Generate Scenarios ---
gen:
	@echo ">>> Generating $(N_SCENARIOS) scenarios..."
	@$(PYTHON) generate_scenarios.py --agents $(N_AGENTS) --tasks $(N_TASKS) --count $(N_SCENARIOS) > /dev/null 2>&1

# --- 3. Run Experiments (Batch) ---
run: run-cap1 run-cap2 run-cap3 run-cap4
	@echo ">>> All Simulations Finished."

run-cap1:
	@echo ">>> [Experiment A] Running Capacity = 1..."
	@echo "Run Original PI (Capacity = 1)..."
	@$(PYTHON) run_batch.py --algo original --network row    --max-tasks-per-agent 1 --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "Original PI (Capacity = 1) Done!"
	@echo "Run Dynamic PI (Capacity = 1)..."
	@$(PYTHON) run_batch.py --algo dynamic  --network radius --max-tasks-per-agent 1 --agents $(N_AGENTS) --tasks $(N_TASKS) --radius $(RADIUS) > /dev/null 2>&1
	@echo "Dynamic PI (Capacity = 1) Done!"

run-cap2:
	@echo ">>> [Experiment B] Running Capacity = 2..."
	@echo "Run Original PI (Capacity = 2)..."
	@$(PYTHON) run_batch.py --algo original --network row    --max-tasks-per-agent 2 --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "Original PI (Capacity = 2) Done!"
	@echo "Run Dynamic PI (Capacity = 2)..."
	@$(PYTHON) run_batch.py --algo dynamic  --network radius --max-tasks-per-agent 2 --agents $(N_AGENTS) --tasks $(N_TASKS) --radius $(RADIUS) > /dev/null 2>&1
	@echo "Dynamic PI (Capacity = 2) Done!"

run-cap3:
	@echo ">>> [Experiment C] Running Capacity = 3..."
	@echo "Run Original PI (Capacity = 3)..."
	@$(PYTHON) run_batch.py --algo original --network row    --max-tasks-per-agent 3 --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "Original PI (Capacity = 3) Done!"
	@echo "Run Dynamic PI (Capacity = 3)..."
	@$(PYTHON) run_batch.py --algo dynamic  --network radius --max-tasks-per-agent 3 --agents $(N_AGENTS) --tasks $(N_TASKS) --radius $(RADIUS) > /dev/null 2>&1
	@echo "Dynamic PI (Capacity = 3) Done!"

run-cap4:
	@echo ">>> [Experiment D] Running Capacity = 4..."
	@echo "Run Original PI (Capacity = 4)..."
	@$(PYTHON) run_batch.py --algo original --network row    --max-tasks-per-agent 4 --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "Original PI (Capacity = 4) Done!"
	@echo "Run Dynamic PI (Capacity = 4)..."
	@$(PYTHON) run_batch.py --algo dynamic  --network radius --max-tasks-per-agent 4 --agents $(N_AGENTS) --tasks $(N_TASKS) --radius $(RADIUS) > /dev/null 2>&1
	@echo "Dynamic PI (Capacity = 4) Done!"

# --- 4. Plot Results ---
plot:
	@echo ">>> Plotting results..."
	@$(PYTHON) plot_results.py
