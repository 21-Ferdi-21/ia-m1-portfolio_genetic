# genetic_portfolio.py
import pandas as pd
import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

class GeneticPortfolioOptimizer:
    def __init__(self, returns_file: str, pop_size: int = 100, mutation_rate: float = 0.05):
        """
        Optimiseur de portfolio par algorithme génétique
        
        Args:
            returns_file: Chemin vers le fichier CSV des rendements
            pop_size: Taille de la population
            mutation_rate: Taux de mutation
        """
        self.returns_data = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        self.tickers = list(self.returns_data.columns)
        self.n_assets = len(self.tickers)

        
        # Paramètres de l'algorithme génétique
        self.POP_SIZE = pop_size
        self.MUTATION_RATE = mutation_rate
        
        # Métriques pour l'évaluation
        self.risk_free_rate = 0.02 / 12  # Taux sans risque mensuel (2% annuel)
        self.momentum_scores = self.compute_momentum_scores()
        self.capm_scores = self.compute_capm_scores()

        
        print(f"Portfolio optimizer initialisé avec {self.n_assets} actifs")
        print(f"Période: {self.returns_data.index[0]} à {self.returns_data.index[-1]}")
    
    def random_individual(self) -> np.ndarray:
        """
        Crée un individu aléatoire représentant une allocation de portfolio
        
        Returns:
            Array de poids sommant à 1.0, représentant l'allocation
        """
        # Générer des poids aléatoires
        weights = np.random.random(self.n_assets)
        # Normaliser pour que la somme = 1
        return weights / weights.sum()
    
    def compute_momentum_scores(self) -> pd.Series:
        """
        Calcule le score de momentum : moyenne des rendements des 6 derniers mois
        """
        momentum = self.returns_data[-6:].mean()
        return momentum

    def compute_capm_scores(self) -> pd.Series:
        """
        Calcule la différence entre le rendement réel et le rendement attendu (CAPM)
        """
        market_returns = self.returns_data.mean(axis=1)
        expected_market_return = market_returns.mean()
        
        betas = {}
        alphas = {}

        for ticker in self.tickers:
            asset_returns = self.returns_data[ticker]
            df = pd.DataFrame({'asset': asset_returns, 'market': market_returns}).dropna()
            if len(df) < 10:
                betas[ticker] = np.nan
                continue
            beta, alpha = np.polyfit(df['market'], df['asset'], 1)
            betas[ticker] = beta
            alphas[ticker] = alpha

        rf = self.risk_free_rate
        capm_returns = {ticker: rf + betas[ticker] * (expected_market_return - rf) if not np.isnan(betas[ticker]) else np.nan for ticker in self.tickers}
        actual_returns = self.returns_data.mean()
        
        capm_scores = actual_returns - pd.Series(capm_returns)
        return capm_scores

    def fitness(self, individual: np.ndarray) -> float:
        portfolio_returns = (self.returns_data * individual).sum(axis=1).dropna()
        if len(portfolio_returns) == 0:
            return -1000
        
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        if std_return == 0:
            return -1000
        
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return

        # Ajout des scores de momentum et CAPM
        momentum_score = np.dot(self.momentum_scores.fillna(0), individual)
        capm_score = np.dot(self.capm_scores.fillna(0), individual)

        # Pénalité pour concentration excessive
        concentration_penalty = -10 * max(0, max(individual) - 0.3)

        # Score final pondéré
        return sharpe_ratio + 0.3 * momentum_score + 0.3 * capm_score + concentration_penalty

    
    def select_parent(self, population: List[np.ndarray]) -> np.ndarray:
        """
        Sélection par tournoi : retourne le meilleur entre deux individus choisis au hasard
        """
        ind1, ind2 = random.sample(population, 2)
        return ind1 if self.fitness(ind1) > self.fitness(ind2) else ind2
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Croisement uniforme : combine les gènes des deux parents
        """
        # Croisement uniforme
        child = np.where(np.random.random(self.n_assets) > 0.5, parent1, parent2)
        
        # Normaliser pour maintenir la somme = 1
        return child / child.sum()
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutation : chaque gène a une chance d'être modifié
        """
        mutated = individual.copy()
        
        for i in range(self.n_assets):
            if random.random() < self.MUTATION_RATE:
                # Mutation gaussienne
                noise = np.random.normal(0, 0.1)
                mutated[i] = max(0, mutated[i] + noise)  # Garder les poids positifs
        
        # Renormaliser
        return mutated / mutated.sum()
    
    def optimize(self, generations: int = 300, verbose: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """
        Lance l'optimisation génétique
        
        Args:
            generations: Nombre de générations
            verbose: Afficher les progrès
            
        Returns:
            Tuple (meilleur_individu, meilleur_score, historique_scores)
        """
        # Initialiser la population
        population = [self.random_individual() for _ in range(self.POP_SIZE)]
        
        # Historique des scores
        best_scores_history = []
        
        print(f"Démarrage de l'optimisation génétique ({generations} générations)...")
        
        for generation in range(generations):
            # Évaluation et tri de la population
            population = sorted(population, key=self.fitness, reverse=True)
            
            # Meilleur individu de cette génération
            best_individual = population[0]
            best_score = self.fitness(best_individual)
            best_scores_history.append(best_score)
            
            # Affichage des progrès
            if verbose and (generation % 100 == 0 or generation < 10):
                print(f"Génération {generation}: Score = {best_score:.4f}")
                if generation % 200 == 0 and generation > 0:
                    self.print_portfolio_summary(best_individual)
            
            # Élitisme : garder les 10% meilleurs
            elite_size = self.POP_SIZE // 10
            new_population = population[:elite_size].copy()
            
            # Générer le reste de la nouvelle population
            while len(new_population) < self.POP_SIZE:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Résultat final
        final_population = sorted(population, key=self.fitness, reverse=True)
        best_individual = final_population[0]
        best_score = self.fitness(best_individual)
        
        print(f"\nOptimisation terminée!")
        print(f"Meilleur score final: {best_score:.4f}")
        
        return best_individual, best_score, best_scores_history
    
    def print_portfolio_summary(self, weights: np.ndarray, top_n: int = 10):
        """
        Affiche un résumé du portfolio
        """
        # Créer un DataFrame avec les poids
        portfolio_df = pd.DataFrame({
            'Ticker': self.tickers,
            'Weight': weights
        }).sort_values('Weight', ascending=False)
        
        print(f"\nTop {top_n} allocations:")
        print(portfolio_df.head(top_n).to_string(index=False, float_format='%.3f'))
        
        # Statistiques du portfolio
        portfolio_returns = (self.returns_data * weights).sum(axis=1).dropna()
        annual_return = portfolio_returns.mean() * 12
        annual_volatility = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = (annual_return - 0.02) / annual_volatility
        
        print(f"\nStatistiques du portfolio:")
        print(f"Rendement annuel: {annual_return:.2%}")
        print(f"Volatilité annuelle: {annual_volatility:.2%}")
        print(f"Ratio de Sharpe: {sharpe_ratio:.3f}")
        
    def plot_optimization_history(self, scores_history: List[float]):
        """
        Affiche l'évolution des scores au cours des générations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(scores_history)
        plt.title('Évolution du Score de Fitness (Ratio de Sharpe)')
        plt.xlabel('Génération')
        plt.ylabel('Score de Fitness')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def backtest_portfolio(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Effectue un backtest du portfolio optimisé
        """
        portfolio_returns = (self.returns_data * weights).sum(axis=1).dropna()
        
        # Calcul de la valeur cumulée (en partant de 1000€)
        cumulative_value = (1 + portfolio_returns).cumprod() * 1000
        
        # Créer un DataFrame avec les résultats
        backtest_df = pd.DataFrame({
            'Date': portfolio_returns.index,
            'Monthly_Return': portfolio_returns.values,
            'Portfolio_Value': cumulative_value.values
        })
        
        return backtest_df


def main():
    """Fonction principale pour tester l'optimiseur"""
    # Initialiser l'optimiseur
    optimizer = GeneticPortfolioOptimizer(
        returns_file="monthly_returns.csv",
        pop_size=100,
        mutation_rate=0.05
    )
    
    # Lancer l'optimisation
    best_portfolio, best_score, history = optimizer.optimize(
        generations=500,
        verbose=True
    )
    
    # Afficher le résultat final
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMAL TROUVÉ")
    print("="*50)
    optimizer.print_portfolio_summary(best_portfolio, top_n=15)
    
    # Backtest
    backtest_results = optimizer.backtest_portfolio(best_portfolio)
    print(f"\nPerformance sur la période:")
    print(f"Valeur initiale: 1000€")
    print(f"Valeur finale: {backtest_results['Portfolio_Value'].iloc[-1]:.2f}€")
    print(f"Rendement total: {(backtest_results['Portfolio_Value'].iloc[-1]/1000 - 1):.2%}")
    
    # Graphique d'évolution (optionnel - nécessite matplotlib)
    try:
        optimizer.plot_optimization_history(history)
    except ImportError:
        print("Matplotlib non disponible - pas de graphique généré")


if __name__ == "__main__":
    main()