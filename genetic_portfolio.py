import pandas as pd
import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

class GeneticPortfolioOptimizer:
    def __init__(self, returns_file: str, pop_size: int = 200, mutation_rate: float = 0.05):
        self.returns_data = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        self.tickers = list(self.returns_data.columns)
        self.n_assets = len(self.tickers)

        self.POP_SIZE = pop_size
        self.MUTATION_RATE = mutation_rate
        self.risk_free_rate = 0.02 / 12  

        self.momentum_scores = self.compute_momentum_scores()
        self.capm_scores = self.compute_capm_scores()

        print(f"Optimizer initialized with {self.n_assets} assets")

    def random_individual(self) -> np.ndarray:
        individual = np.random.choice([0, 1], size=self.n_assets)
        if individual.sum() == 0:
            individual[random.randint(0, self.n_assets - 1)] = 1
        return individual

    def compute_momentum_scores(self) -> pd.Series:
        return self.returns_data.mean()

    def compute_capm_scores(self) -> pd.Series:
        market_returns = self.returns_data.mean(axis=1)
        expected_market_return = market_returns.mean()
        betas = {}
        for ticker in self.tickers:
            df = pd.DataFrame({"asset": self.returns_data[ticker], "market": market_returns}).dropna()
            if len(df) < 10:
                betas[ticker] = 0  
                continue
            beta, _ = np.polyfit(df["market"], df["asset"], 1)
            betas[ticker] = beta
        rf = self.risk_free_rate
        capm_returns = {
            ticker: rf + betas[ticker] * (expected_market_return - rf)
            for ticker in self.tickers
        }
        actual_returns = self.returns_data.mean()
        return actual_returns - pd.Series(capm_returns)

    def fitness(self, individual: np.ndarray) -> float:
        if individual.sum() == 0:
            return -1000
        weights = individual / individual.sum()
        portfolio_returns = (self.returns_data * weights).sum(axis=1).dropna()
        if len(portfolio_returns) == 0:
            return -1000
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        if std_return == 0:
            return -1000
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        momentum_score = np.dot(self.momentum_scores.fillna(0), weights)
        capm_score = np.dot(self.capm_scores.fillna(0), weights)
        return sharpe_ratio + 0.3 * momentum_score + 0.3 * capm_score

    def select_parent(self, population: List[np.ndarray]) -> np.ndarray:
        ind1, ind2 = random.sample(population, 2)
        return ind1 if self.fitness(ind1) > self.fitness(ind2) else ind2

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        point = random.randint(1, self.n_assets - 2)
        child = np.concatenate([parent1[:point], parent2[point:]])
        if child.sum() == 0:
            child[random.randint(0, self.n_assets - 1)] = 1
        return child

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for i in range(self.n_assets):
            if random.random() < self.MUTATION_RATE:
                mutated[i] = 1 - mutated[i]  # bit flip
        if mutated.sum() == 0:
            mutated[random.randint(0, self.n_assets - 1)] = 1
        return mutated

    def optimize(self, generations=50, population_size=100, mutation_rate=0.1, elite_size=2, verbose=True):
        population = [self.random_individual() for _ in range(population_size)]

        # init_diversity = np.mean([
        #     np.sum(np.abs(ind1 - ind2)) for i, ind1 in enumerate(population)
        #     for j, ind2 in enumerate(population) if i < j
        # ])
        # print(f"[DEBUG] Initial population diversity: {init_diversity:.2f}")

        best_scores_history = []

        for generation in range(generations):
            population = sorted(population, key=self.fitness, reverse=True)
            best_individual = population[0]
            best_score = self.fitness(best_individual)
            best_scores_history.append(best_score)

            # === DEBUG: Statistiques sur le portefeuille courant ===
            weights = best_individual / best_individual.sum()
            portfolio_returns = (self.returns_data * weights).sum(axis=1).dropna()
            cumulative_value = (1 + portfolio_returns).cumprod() * 1000 
            final_value = cumulative_value.iloc[-1] if len(cumulative_value) > 0 else 0

            fitness_values = [self.fitness(ind) for ind in population]
            fitness_std = np.std(fitness_values)
            diversity = np.mean([np.sum(np.abs(ind - best_individual)) for ind in population])
            num_assets = best_individual.sum()
            max_weight = weights.max()
            nonzero_assets = np.count_nonzero(weights > 0.01)

            # print(f"\n[DEBUG] Generation {generation}")
            # print(f" - Best score: {best_score:.4f}")
            # print(f" - Fitness std: {fitness_std:.6f}")
            # print(f" - Diversity wrt best: {diversity:.2f}")
            # print(f" - Assets in portfolio: {int(num_assets)}")
            # print(f" - Max weight: {max_weight:.3f}")
            # print(f" - Assets >1%: {nonzero_assets}")
            # print(f" - Final value: {final_value:.2f}€")

            if verbose:
                print(f"Generation {generation}: Score = {best_score:.4f} | Final value = {final_value:.2f}€")

            next_generation = population[:elite_size]

            while len(next_generation) < population_size:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)
                child = self.crossover(parent1, parent2)
                if np.random.rand() < mutation_rate:
                    child = self.mutate(child)
                next_generation.append(child)

            population = next_generation

        best_individual = max(population, key=self.fitness)
        best_weights = best_individual / best_individual.sum()

        print(f"\n[INFO] Optimization completed.")
        print(f"Best score achieved: {self.fitness(best_individual):.4f}")
        return best_weights,best_score,best_scores_history


    def print_portfolio_summary(self, individual: np.ndarray, top_n: int = 10):
        weights = individual / individual.sum()
        portfolio_df = pd.DataFrame({
            'Ticker': self.tickers,
            'Weight': weights
        }).sort_values('Weight', ascending=False)

        print(f"\nTop {top_n} allocations:")
        print(portfolio_df.head(top_n).to_string(index=False, float_format='%.3f'))

        portfolio_returns = (self.returns_data * weights).sum(axis=1).dropna()
        annual_return = portfolio_returns.mean() * 12
        annual_volatility = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = (annual_return - 0.02) / annual_volatility

        print(f"\nPortfolio stats:")
        print(f"Yearly return: {annual_return:.2%}")
        print(f"Yearly volatility: {annual_volatility:.2%}")
        print(f"Sharpe ratio: {sharpe_ratio:.3f}")

    def plot_optimization_history(self, scores_history: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(scores_history)
        plt.title('Fitness score evolution (Sharpe ratio)')
        plt.xlabel('Generation')
        plt.ylabel('Fitness score')
        plt.grid(True, alpha=0.3)
        plt.show()

    def backtest_portfolio(self, individual: np.ndarray) -> pd.DataFrame:
        weights = individual / individual.sum()
        portfolio_returns = (self.returns_data * weights).sum(axis=1).dropna()
        cumulative_value = (1 + portfolio_returns).cumprod() * 1000

        backtest_df = pd.DataFrame({
            'Date': portfolio_returns.index,
            'Monthly_Return': portfolio_returns.values,
            'Portfolio_Value': cumulative_value.values
        })

        return backtest_df

def main():
    optimizer = GeneticPortfolioOptimizer(
        returns_file="monthly_returns.csv",
        pop_size=200,
        mutation_rate=0.05
    )

    best_portfolio, best_score, history = optimizer.optimize(
        generations=300,
        verbose=True
    )

    print("\n" + "="*50)
    print("OPTIMAL PORTFOLIO")
    print("="*50)
    optimizer.print_portfolio_summary(best_portfolio, top_n=15)

    backtest_results = optimizer.backtest_portfolio(best_portfolio)
    print(f"\nPerformance:")
    print(f"Initial value: 1000€")
    print(f"Final value: {backtest_results['Portfolio_Value'].iloc[-1]:.2f}€")
    print(f"Total return: {(backtest_results['Portfolio_Value'].iloc[-1]/1000 - 1):.2%}")

    try:
        optimizer.plot_optimization_history(history)
    except ImportError:
        print("No graph")

if __name__ == "__main__":
    main()
