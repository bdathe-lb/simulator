#!/usr/bin/env bash

# --- Argument Handling ---
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

# Check if the second argument (run count) is provided.
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

# --- Variables Initialization ---
SUCCESS_COUNT=0
FAILURE_COUNT=0
CURRENT_RUN=0

echo "--- Starting Allocation Success Rate Test ---"
echo "Total Runs: ${TOTAL_RUNS}"
echo "Algorithm : ${ALGO}"
echo "--------------------------------------------"

# --- Loop Execution and Validation ---
while [ $CURRENT_RUN -lt $TOTAL_RUNS ]; do
    # Increment the current run counter.
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Print the current test progress.
    printf "Running test %d/%d ...\r" $CURRENT_RUN $TOTAL_RUNS

    # [FIX 4] Added 'python3' before the validator script to ensure it runs 
    # even if the file doesn't have +x permission.
    ${MAIN_COMMAND} 2>/dev/null | python3 ${VALIDATOR_SCRIPT}
    
    # Capture the exit code of the validator script.
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi

done

# Print an empty line to clear the progress output.
echo ""

# --- Results Calculation and Reporting ---
if command -v bc 2>&1 > /dev/null; then
    SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc)
else
    SUCCESS_RATE=$((SUCCESS_COUNT * 100 / $TOTAL_RUNS))
    echo "Warning: 'bc' not found. Success rate calculated using integer arithmetic."
fi

echo "--------------------------------------------"
echo "Tests Completed: ${TOTAL_RUNS}"
echo "Successful Allocations: ${SUCCESS_COUNT}"
echo "Failed Allocations: ${FAILURE_COUNT}"
echo "Success Rate: ${SUCCESS_RATE}%"
echo "--------------------------------------------"
