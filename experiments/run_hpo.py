from engine.searchers import RandomSearcher
from engine.tuners import HPOTuner
from experiments.settings import CONFIG_SPACE, NUM_EVALUATIONS
from experiments.objectives import cnn_objective

def main():
    print("=== Starting K-MNIST Hyperparameter Optimization ===")
    
    # 1. Initialize the Search Strategy
    searcher = RandomSearcher(config_space=CONFIG_SPACE)
    
    # 2. Initialize the Orchestrator (We are tuning the CNN here)
    tuner = HPOTuner(searcher=searcher, objective_func=cnn_objective)
    
    # 3. Run the Experiment
    print(f"Running {NUM_EVALUATIONS} trials using Random Search...\n")
    tuner.run(number_of_trials=NUM_EVALUATIONS)
    
    # 4. Display the ultimate champion configuration
    print("\n" + "="*50)
    print("🏆 HPO COMPLETE 🏆")
    print(f"Best Error (Loss): {tuner.incumbent_error:.4f}")
    print(f"Best Configuration: {tuner.incumbent}")
    print("="*50)

if __name__ == "__main__":
    main()