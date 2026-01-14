N_SCENARIOS=500
N_AGENTS=12
N_TASKS=24
N_MAX_TASKS_PER_AGENT=1
RADIUS=250

.PHONY: all clean gen run plot run-original run-dynamic

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
run: run-original run-dynamic
	@echo "All runs finished!"

run-original:
	./run_batch.py --algo original --network row --agents $(N_AGENTS) --tasks $(N_TASKS) > /dev/null 2>&1
	@echo "Original PI Finished!"

run-dynamic:
	./run_batch.py --algo dynamic --network radius --radius $(RADIUS) --agents $(N_AGENTS) --tasks $(N_TASKS) --max-tasks-per-agent $(N_MAX_TASKS_PER_AGENT) > /dev/null 2>&1
	@echo "PI Dynamic Finished!"

# --- Plot ---
plot:
	./plot_results.py
