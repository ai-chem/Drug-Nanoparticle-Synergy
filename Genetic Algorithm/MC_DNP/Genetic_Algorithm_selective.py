import Compound_generation_MIC_synergy_single_bac_drug
import Crossing_selective
import pandas as pd
import time


mutation_rate = 0.05
cross_over_rate = 0.05

def new_generations(Gen, population_size):
    half = int((population_size * 0.5) + 1)
    selected = Gen.iloc[:half, :]
    new = [selected, Compound_generation_MIC_synergy_single_bac_drug.predict_MIC_of_drug_np(Compound_generation_MIC_synergy_single_bac_drug.population(half))]
    new_generation_input = pd.concat(new)
    new_generation_input.reset_index(drop=True, inplace=True)

    new_gen = Crossing_selective.perform_evolution(new_generation_input, cross_over_rate, mutation_rate)
    new_gen = new_gen.sort_values('Fitness', ascending=False)
    new_gen.reset_index(drop=True, inplace=True)
    return new_gen
# print(new_generations(Compound_generation.population(100), 100))
def Genetic_Algorithm(generation_number, population_size):
    Generation1 = Compound_generation_MIC_synergy_single_bac_drug.predict_MIC_of_drug_np(Compound_generation_MIC_synergy_single_bac_drug.population(population_size)).sort_values('Fitness', ascending=False)

    # Generation1 = V4_ga_compd_generation.fitness(V4_ga_compd_generation.population(population_size)).sort_values('Fitness', ascending=False)
    mean1 = Generation1['Fitness'].mean()
    max1 = Generation1['Fitness'].max()
    Generation1.to_csv('output/bacteria_sel/pop_size_' + str(population_size) + '_Generation_1.csv')
    Generation2 = Crossing_selective.perform_evolution(Generation1, cross_over_rate, mutation_rate).sort_values('Fitness', ascending=False)
    mean2 = Generation2['Fitness'].mean()
    max2 = Generation2['Fitness'].max()
    Generation2.to_csv('output/bacteria_sel/pop_size_' + str(population_size) + '_Generation_2.csv')
    Generation_next = Generation2
    means = [mean1, mean2]
    maxs = [max1, max2]
    g = 3
    while g in range(generation_number + 1):
        Generation_next = new_generations(Generation_next, population_size)
        mean = Generation_next['Fitness'].mean()
        max = Generation_next['Fitness'].max()
        Generation_next.to_csv('output/bacteria_sel/pop_size_' + str(population_size) + '_Generation_' + str(g) + '.csv')
        means.append(mean)
        maxs.append(max)
        g += 1

    genn = generation_number + 1
    gens = list(range(1, genn))
    summary = pd.DataFrame(list(zip(gens, means, maxs)), columns=['generations', 'mean', 'max'])
    print(summary)
    summary.to_csv('output/bacteria_sel/summary_pop_size_' + str(population_size) + '_gen_' + str(generation_number) + '.csv')
    return Generation_next

def final_loop():
    gen_col = []
    time_all = []
    gen = 70
    while gen <= 70:  # Corrected loop condition
        population_size = 10
        while population_size <= 100:  # Corrected loop condition
            st = time.time()
            Genetic_Algorithm(gen, population_size)
            gen_col.append(gen)
            escape_time = time.time() - st
            time_all.append(escape_time)
            print('Escape time:', escape_time)
            population_size += 10
        gen += 10
        et = pd.DataFrame(list(zip(gen_col, time_all)), columns=['Generation number', 'Time'])
        et.to_csv('output/bacteria_sel/Time_' + str(gen) + '.csv')

final_loop()
