# run_pipeline.py
import os
import time
import subprocess
import argparse

def run_command(command, description):
    """Run a command and display output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    if process.returncode != 0:
        print(f"Error running: {description}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    return True

def main():
    parser = argparse.ArgumentParser(description="GNN Cancer Mutation Analysis Pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation studies")
    parser.add_argument("--skip-attention", action="store_true", help="Skip attention analysis")
    args = parser.parse_args()
    
    # Ensure necessary directories exist
    required_dirs = [
        'data/raw', 'data/processed', 'data/graphs', 'data/results',
        'models', 'models/checkpoints'
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    if not args.skip_preprocessing:
        run_command("python preprocess_data.py", "Data Preprocessing")

    run_command("python build_graph.py", "Graph Construction")

    if not args.skip_training:
        run_command("python train.py", "Model Training")

    if not args.skip_evaluation:
        run_command("python evaluate_models.py", "Model Evaluation & Comparison")

    if not args.skip_ablation:
        run_command("python ablation_studies.py", "Ablation Studies")

    if not args.skip_attention:
        run_command("python analyze_attention.py", "Attention Mechanism Analysis")

    print("\n🎉 All stages completed successfully!")

if __name__ == "__main__":
    main()