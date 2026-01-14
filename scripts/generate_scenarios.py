#!/usr/bin/env python3

import sys
import os
import argparse

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import main

# Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test", "scenarios")

def ensure_dirs():
    """Creates necessary output directories if they don't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def generate(args):
    """Generates N random scenario files."""
    ensure_dirs()

    print(f"Generating {args.count} scenarios...")
    
    # Reuse main.py logic to setup and save
    args.save = True
    args.load = False
    args.algorithm = 'dynamic' # Dummy algo, doesn't affect position generation
    
    for i in range(args.count):
        filename = os.path.join(OUTPUT_DIR, f"scenario_{i}.txt")
        args.file = filename
        
        # This function generates random positions and saves them to args.file
        main.setup_scenario(args)
        
    print(f"Done. {args.count} scenarios saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random testing scenarios.")
    parser.add_argument("-n", "--count", type=int, default=20, help="Number of scenarios to generate")
    parser.add_argument("-a", "--agents", type=int, default=6)
    parser.add_argument("-t", "--tasks", type=int, default=12)
    parser.add_argument("--network", type=str, default='radius') 
    # Dummy args required by main.setup_scenario signatures
    parser.add_argument("--radius", type=float, default=350.0) 
    parser.add_argument("--max-tasks-per-agent", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--algo", dest="algorithm", type=str, default="v2") 
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--file", type=str, default="")

    args = parser.parse_args()
    generate(args)
