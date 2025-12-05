N_SCENARIOS=20
N_AGENTS=6
N_TASKS=12
RADIUS=50

.PHONY: all clean gen run plot run-original run-v1 run-v2

all: clean gen run plot
	@echo "All done!"

# --- Clean ---
clean:
	rm -rf ../test/scenarios/
	rm -rf ../test/results/

# --- Generate scenarios ---
gen: clean
	./generate_scenarios.py --agents $(N_AGENTS) --tasks $(N_TASKS) --count $(N_SCENARIOS) > /dev/null 2>&1
	@echo "Generate Scenarios Finished!"

# --- Run all algorithms (in parallel if using -j) ---
run: run-original run-v1 run-v2 run-v3
	@echo "All runs finished!"

run-original:
	./run_batch.py --algo original --network row --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "Original PI Finished!"

run-v1:
	./run_batch.py --algo v1 --network radius --radius $(RADIUS) --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "PI v1 Finished!"

run-v2:
	./run_batch.py --algo v2 --network radius --radius $(RADIUS) --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "PI v2 Finished!"

run-v3:
	./run_batch.py --algo v3 --network radius --radius $(RADIUS) --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "PI v3 Finished!"

# --- Plot ---
plot:
	./plot_results.py
