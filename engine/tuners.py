import time
import copy

class HPOTuner:
    """The orchestrator that runs the search loop and records results."""
    
    def __init__(self, searcher, objective_func):
        self.searcher = searcher
        self.objective = objective_func
        
        # Bookkeeping
        self.incumbent = None
        self.incumbent_error = float('inf')
        self.trajectory = []
        self.records = []

    def run(self, number_of_trials: int):
        for i in range(number_of_trials):
            start_time = time.time()
            
            # 1. Ask the searcher for the next best guess
            config = self.searcher.sample_configuration()
            
            # 2. Execute the objective (train the model)
            print(f"Trial {i+1}: Testing {config}")
            error = self.objective(config)
            
            # 3. Feed the result back to the searcher
            self.searcher.update(config, error)
            
            runtime = time.time() - start_time
            self._bookkeeping(config, error, runtime)
            print(f"Result: Error {error:.4f} | Runtime {runtime:.2f}s")

    def _bookkeeping(self, config, error, runtime):
        self.records.append({"config": config, "error": error, "runtime": runtime})
        
        # Keep track of the 'incumbent' (the best model found so far)
        if error < self.incumbent_error:
            self.incumbent = copy.deepcopy(config)
            self.incumbent_error = error
            
        self.trajectory.append(self.incumbent_error)


class SuccessiveHalvingTuner:
    """Multi-Fidelity bandit-based optimization (Algorithm 1)."""
    
    def __init__(self, searcher, objective_func, min_budget: int = 1, max_budget: int = 9, eta: int = 3):
        self.searcher = searcher
        self.objective = objective_func
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        
        # Calculate K (number of rounds) and initial population N
        # using: K = log_eta(max_budget / min_budget)
        self.K = math.floor(math.log(self.max_budget / self.min_budget, self.eta))
        self.N = self.eta ** self.K  # Initial configurations S_0
        
        self.incumbent = None
        self.incumbent_error = float('inf')
        self.records = []

    def run(self):
        print(f"Initializing SHA: K={self.K} rounds, starting with {self.N} configurations.")
        
        # 1. Initialize S_0 (Sample initial configurations)
        configs = [self.searcher.sample_configuration() for _ in range(self.N)]
        current_budget = self.min_budget
        
        # 2. The Halving Loop
        for k in range(self.K + 1):
            print(f"\n--- SHA Round {k} | Budget: {current_budget} epochs | Surviving Configs: {len(configs)} ---")
            
            round_results = []
            
            # Evaluate all surviving configurations at the current budget
            for i, config in enumerate(configs):
                start_time = time.time()
                
                # Execute pull c(lambda, R_k)
                error = self.objective(config, current_budget)
                round_results.append((error, config))
                
                runtime = time.time() - start_time
                self.records.append({"round": k, "budget": current_budget, "error": error, "runtime": runtime})
                print(f"  Config {i+1}/{len(configs)} -> Error: {error:.4f}")
                
                # Global bookkeeping
                if error < self.incumbent_error:
                    self.incumbent_error = error
                    self.incumbent = copy.deepcopy(config)
            
            # Sort configurations by observed loss (lowest is best)
            round_results.sort(key=lambda x: x[0])
            
            # If we just finished the final round at max_budget, stop!
            if k == self.K:
                break
                
            # Retain top 1/eta fraction for the next round
            num_survivors = max(1, len(configs) // self.eta)
            configs = [result[1] for result in round_results[:num_survivors]]
            
            # Increase budget for the next round: R_{k+1} = R_k * eta
            current_budget = min(current_budget * self.eta, self.max_budget)
    
    def plot_successive_halving(self):
        """Generates the Multi-Fidelity scatter plot from the D2L textbook."""
        if not self.records:
            print("No records found. Run the tuner first!")
            return

        # Extract all the budgets (X-axis) and errors (Y-axis) we saved during the run
        budgets = [record['budget'] for record in self.records]
        errors = [record['error'] for record in self.records]

        # Create a beautiful, professional academic plot
        plt.figure(figsize=(10, 6))
        plt.scatter(budgets, errors, alpha=0.7, color='royalblue', edgecolors='black', s=50)
        
        # Formatting to match the textbook style
        unique_rungs = sorted(list(set(budgets)))
        plt.xticks(unique_rungs, [str(r) for r in unique_rungs])
        
        plt.title(f"Successive Halving Pruning\n(Started with {self.N} configs, $\eta$={self.eta})", fontsize=14)
        plt.xlabel("Budget (Epochs)", fontsize=12)
        plt.ylabel("Validation Error", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save it for your thesis document!
        plt.tight_layout()
        plt.savefig('results/successive_halving_plot.png', dpi=300)
        plt.show()