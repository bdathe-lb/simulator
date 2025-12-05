#!/usr/bin/env python3

import re
import sys
import argparse
from itertools import chain

def extract_allocation_result(data):
    """
    Extract each Agent and its list of completed tasks from the simulation result text.

    Args:
        data (str): Complete string data containing the simulation results.

    Returns:
        dict: Keys are Agent numbers (int), values are lists of task IDs (list of int).
    """
    # define regular expression
    pattern = r'Agent\s*(\d+)\s*\(.+\)\s*completed:\s*\[(.*)\]'

    # use 're.findall' to return a tuple list: [('0', '1, 2, 0'), ('1', '5, 4, 3'), ...]
    matches = re.findall(pattern, data)

    # store final result
    results = {}

    for agent_id_str, tasks_str in matches:
        # extract Agent ID
        agent_id = int(agent_id_str)

        # process task list string
        if tasks_str.strip():
            tasks_list = [int(t.strip()) for t in tasks_str.split(',')]
        else:
            tasks_list = []
        
        results[agent_id] = tasks_list

    return results

def check_all_intersections_empty(lists):
    """
    Check if the intersection of the lists is empty.

    Args:
        lists: the lists to be checked.
    """
    total_elements = sum(len(lst) for lst in lists)
    unique_elements = len(set(chain.from_iterable(lists)))
    return total_elements == unique_elements


def parse_args():
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run multi-agent multi-task allocation simulation.")

    # --- Action Flags (Boolean) ---
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        dest="verbose_action",
        help="Output the final result."
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    # read all data from standard input
    input_data = sys.stdin.read()

    if not input_data:
        print('Error: No data received from standard input.', file=sys.stderr)
        sys.exit(1)

    # extract results
    allocation_results = extract_allocation_result(input_data)

    # vertify if the allocation was successful
    all_task_lists = list(allocation_results.values())
    success = check_all_intersections_empty(all_task_lists)

    if args.verbose_action:
        if success:
            print('Success!')
        else:
            print('Fail!')
            sys.exit(1)
    else:
        if not success:
            sys.exit(1)
