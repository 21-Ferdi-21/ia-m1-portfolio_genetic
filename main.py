import random
import string

TARGET = "HELLO WORLD"
POP_SIZE = 100
MUTATION_RATE = 0.01

def random_individual():
    """Crée un individu aléatoire de la même longueur que TARGET"""
    return ''.join(random.choice(string.ascii_uppercase + " ") for _ in range(len(TARGET)))

population = [random_individual() for _ in range(POP_SIZE)]

def fitness(individual):
    """Calcule le score : plus c'est proche de TARGET, plus le score est élevé."""
    return sum(1 for i, j in zip(individual, TARGET) if i == j)

def select_parent(population):
    """Sélectionne le meilleur entre deux individus choisis au hasard."""
    ind1, ind2 = random.sample(population, 2)
    return ind1 if fitness(ind1) > fitness(ind2) else ind2

def crossover(parent1, parent2):
    """Créé un enfant en combinant les gènes des deux parents."""
    return ''.join(parent1[i] if random.random() > 0.5 else parent2[i] for i in range(len(TARGET)))

def mutate(individual):
    """Chaque caractère a une chance d'être remplacé par un caractère aléatoire."""
    return ''.join(
        c if random.random() > MUTATION_RATE else random.choice(string.ascii_uppercase + " ")
        for c in individual
    )

GENERATIONS = 1000

for generation in range(GENERATIONS):
    # Évaluation de la population
    population = sorted(population, key=fitness, reverse=True)
    
    # Affichage du meilleur individu
    best_individual = population[0]
    print(f"Génération {generation}: {best_individual} (Score: {fitness(best_individual)})")
    
    # Arrêt si on a trouvé la phrase cible
    if best_individual == TARGET:
        break
    
    # Création d'une nouvelle population
    new_population = []
    for _ in range(POP_SIZE // 2):
        parent1, parent2 = select_parent(population), select_parent(population)
        child1, child2 = crossover(parent1, parent2), crossover(parent1, parent2)
        new_population.extend([mutate(child1), mutate(child2)])
    
    population = new_population
