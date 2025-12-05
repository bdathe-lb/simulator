#!/usr/bin/env bash

# Check if the first argument (algo) is provided.
if [ -z "$1" ]; then
    echo "Error: Please specify the algo of runs." >&2
    exit 1
fi

ALGO=$1

if ! [[ "$ALGO" =~ ^(original|v1|v2)$ ]]; then
    echo "Error: Invalid ALGO. Must be one of: original, V1, V2" >&2
    exit 1
fi

# --- Argument Handing ---
# Check if the first argument (run count) is provided.
if [ -z "$2" ]; then
    echo "Error: Please specify the number of runs." >&2
    exit 1
fi

# Assign the first argument to TOTAL_RUNS.
TOTAL_RUNS=$2

# Basic check to ensure the input is a positive number.
if ! [[ "$TOTAL_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: The number of runs must be a positive integer." >&2
    exit 1
fi

# --- Configuration ---
# Current project directory
[[ "$ALGO" == "original" ]] && ARGS="--network radius --radius 350" || ARGS="--network row"
# Main command to execute
MAIN_COMMAND="python3 /home/bdathe/projects/simulator/main.py --algo $ALGO $ARGS --quiet"
# Path to the Python script
VALIDATOR_SCRIPT="/home/bdathe/projects/simulator/scripts/validate_allocation.py"

# --- Define the function to run a single test (Crucial for Parallel) ---
# This function encapsulates the logic that runs in parallel.
run_single_test() {
    # Execute the main command, pipe output to the validator, and suppress stderr of the main command.
    ${MAIN_COMMAND} 2>/dev/null | ${VALIDATOR_SCRIPT}
    
    # Output the exit status code of the last command in the pipeline (the validator)
    # This output is collected by 'parallel' for later counting.
    echo $?
}

# --- Export variables and the function so 'parallel' can access them ---
export -f run_single_test
export MAIN_COMMAND
export VALIDATOR_SCRIPT

echo "--- Starting Allocation Success Rate Test (Parallel) ---"
echo "Total Runs: ${TOTAL_RUNS}"
echo "Using $(nproc) parallel processes (approx.)." # Use nproc to show the estimated number of cores used by -j0
echo "--------------------------------------------------------"

# --- CORE PARALLEL EXECUTION ---
# 1. seq 1 $TOTAL_RUNS: Generates a sequence of numbers from 1 up to TOTAL_RUNS.
# 2. | parallel: Pipes the sequence into 'parallel'.
# 3. -j0: Tells parallel to use all available CPU cores.
# 4. run_single_test: Executes the defined function TOTAL_RUNS times.
# RESULTS collects the exit codes (0 or 1) printed by run_single_test.
RESULTS=$(seq 1 $TOTAL_RUNS | parallel -j0 run_single_test)

# --- Results Calculation and Reporting ---
# Count the number of successful runs (exit code 0).
SUCCESS_COUNT=$(echo "$RESULTS" | grep -c '^0$')
# Calculate the number of failed runs.
FAILURE_COUNT=$((TOTAL_RUNS - SUCCESS_COUNT))
# Check if the 'bc' command (for floating-point arithmetic) is available.
# 2>&1 > /dev/null redirects both stdout and stderr to /dev/null, silencing the check.
if command -v bc 2>&1 > /dev/null; then
    # Use 'bc' for accurate floating-point percentage calculation.
    SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc)
else
    # Fallback to Bash's integer arithmetic (less accurate).
    SUCCESS_RATE=$((SUCCESS_COUNT * 100 / $TOTAL_RUNS))
    echo "Warning: 'bc' not found. Success rate calculated using integer arithmetic."
fi

echo
echo "--------------------------------------------"
echo "Tests Completed: ${TOTAL_RUNS}"
echo "Successful Allocations: ${SUCCESS_COUNT}"
echo "Failed Allocations: ${FAILURE_COUNT}"
echo "Success Rate: ${SUCCESS_RATE}%"
echo "--------------------------------------------"
