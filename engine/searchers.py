import random
import math
import itertools
from abc import ABC, abstractmethod


class HPOSearcher(ABC):
    """HPO Searcher basic structure."""
    
    @abstractmethod
    def sample_configuration(self) -> dict:
        """Returns a new candidate configuration."""
        raise NotImplementedError
    
    @abstractmethod
    def update(self, config: dict, score: float):
        """Updates the internal state with the result of a trial."""
        pass


class RandomSearcher(HPOSearcher):
    """Implementation of Random Search."""
    
    def __init__(self, config_space: dict):
        self.config_space = config_space

    def sample_configuration(self) -> dict:
        """Samples uniformly from the provided ranges."""
        return {
            name: self._sample_domain(domain)
            for name, domain in self.config_space.items()
        }

    def _sample_domain(self, domain):
        """Helper to sample from different types of search spaces."""
        if domain['type'] == 'continuous':
            low, high = domain['range']
            return random.uniform(low, high)
        elif domain['type'] == 'discrete':
            return random.choice(domain['values'])
        return None

    def update(self, config: dict, score: float):
        """Random Search doesn't need to update state, but we satisfy the contract."""
        pass


class GridSearcher(HPOSearcher):
    """
    Evaluates combinations in a hyperparameter grid. 
    Automatically discretizes continuous and log spaces.
    """
    
    def __init__(self, config_space: dict, resolution: int = 3):
        self.config_space = config_space
        self.resolution = resolution  # How many points to slice a continuous range into
        self.grid = self._build_grid()
        self.current_index = 0

    def _build_grid(self) -> list:
        keys = []
        value_lists = []
        
        for name, domain in self.config_space.items():
            keys.append(name)
            
            if domain['type'] == 'discrete':
                # Pass discrete lists exactly as they are
                value_lists.append(domain['values'])
                
            elif domain['type'] == 'continuous':
                low, high = domain['range']
                # e.g., range(0.1, 0.7) with resolution=3 -> [0.1, 0.4, 0.7]
                points = np.linspace(low, high, self.resolution).tolist()
                value_lists.append(points)
                
            elif domain['type'] == 'log':
                low, high = domain['range']
                # e.g., range(1e-4, 1e-1) with resolution=3 -> [0.0001, 0.00316, 0.1]
                points = np.geomspace(low, high, self.resolution).tolist()
                value_lists.append(points)
                
            else:
                raise ValueError(f"Unknown domain type: {domain['type']}")
                
        # Generate the Cartesian product of all lists
        combinations = list(itertools.product(*value_lists))
        
        return [dict(zip(keys, combo)) for combo in combinations]

    def sample_configuration(self) -> dict:
        if self.current_index >= len(self.grid):
            raise StopIteration("Grid Search has evaluated all possible combinations!")
            
        config = self.grid[self.current_index]
        self.current_index += 1
        return config

    def update(self, config: dict, score: float):
        pass


class GeneticSearcher(HPOSearcher):
    """
    An evolutionary algorithm that breeds hyperparameters.
    Uses Elitism, Uniform Crossover, and Point Mutation.
    """
    
    def __init__(self, config_space: dict, pop_size: int = 10, mutation_rate: float = 0.2):
        self.config_space = config_space
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        
        # Generational Memory
        self.generation_number = 1
        self.current_results = []  # Stores (config, score) for the current generation
        
        # We start by generating a random "Adam and Eve" population
        self.untested_queue = [self._random_config() for _ in range(self.pop_size)]

    def _random_config(self) -> dict:
        """Helper to generate a completely random configuration."""
        config = {}
        for name, domain in self.config_space.items():
            if domain['type'] == 'continuous':
                config[name] = random.uniform(*domain['range'])
            elif domain['type'] == 'log':
                low, high = domain['range']
                config[name] = math.exp(random.uniform(math.log(low), math.log(high)))
            elif domain['type'] == 'discrete':
                config[name] = random.choice(domain['values'])
        return config

    def sample_configuration(self) -> dict:
        """Yields the next config. If the queue is empty, it evolves a new generation!"""
        if not self.untested_queue:
            self._evolve_generation()
            
        # Pop the next configuration waiting in line
        return self.untested_queue.pop(0)

    def update(self, config: dict, score: float):
        """Saves the fitness score of the config to the current generation pool."""
        self.current_results.append((config, score))

    def _evolve_generation(self):
        """The biological core: Selection, Crossover, and Mutation."""
        print(f"\n🧬 EVOLVING GENERATION {self.generation_number} -> {self.generation_number + 1} 🧬")
        
        # 1. Evaluate Fitness (Sort by lowest error)
        self.current_results.sort(key=lambda x: x[1])
        
        next_generation = []
        
        # 2. Elitism: The absolute best 2 configs survive untouched!
        next_generation.append(self.current_results[0][0])
        next_generation.append(self.current_results[1][0])
        
        # 3. Selection: Keep the top 50% as the "breeding pool"
        num_parents = max(2, self.pop_size // 2)
        breeding_pool = [result[0] for result in self.current_results[:num_parents]]
        
        # 4. Crossover & Mutation to fill the rest of the population
        while len(next_generation) < self.pop_size:
            # Pick two random parents from the elite pool
            parent1, parent2 = random.sample(breeding_pool, 2)
            
            # Uniform Crossover: 50/50 chance to inherit a gene from parent 1 or 2
            child = {}
            for key in self.config_space.keys():
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
                
            # Point Mutation: Chance to randomly mutate a gene
            for key, domain in self.config_space.items():
                if random.random() < self.mutation_rate:
                    # Generate a single random gene to replace the inherited one
                    dummy_config = self._random_config()
                    child[key] = dummy_config[key]
                    
            next_generation.append(child)
            
        # Reset the queues for the new generation
        self.untested_queue = next_generation
        self.current_results = []
        self.generation_number += 1