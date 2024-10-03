import pandas as pd
import random
import Compound_generation_MIC_synergy_single_bac_drug

# Load the data
df_MIC_Drug_NP_all = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\GA_MC_drug_NP_data_all_features.csv')
df_MIC_Drug_NP_all =df_MIC_Drug_NP_all.drop(['Unnamed: 0','MIC_NP (μg/ml)', 'MIC_drug (μg/ml)', 'MIC_drug_NP (μg/ml)'], axis=1)
drug_descriptor = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_drug_descriptors_2024_version1.csv')
np_descriptor = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_nanoparticle_descriptors.csv')

random_individual = Compound_generation_MIC_synergy_single_bac_drug.predict_MIC_of_drug_np(Compound_generation_MIC_synergy_single_bac_drug.bacterial_population(Compound_generation_MIC_synergy_single_bac_drug.population(10))).loc[0].values
drug_col_dnp = df_MIC_Drug_NP_all.columns.intersection(drug_descriptor.columns)
np_col_dnp = df_MIC_Drug_NP_all.columns.intersection(np_descriptor.columns).append(pd.Index(['reference']))

# experimental columns
experimental_col = ['NP size_min (nm)', 'NP size_max (nm)', 'NP size_avg (nm)', 'shape', 'method','time (hr)','MDR']
experimental_col_indices = [df_MIC_Drug_NP_all.columns.get_loc(col) for col in experimental_col]


# Convert DataFrame to NumPy array for faster processing
data_array = df_MIC_Drug_NP_all.to_numpy()
columns = df_MIC_Drug_NP_all.columns

# indices of the drug and NP columns
drug_col_indices = [columns.get_loc(col) for col in drug_col_dnp]
np_col_indices = [columns.get_loc(col) for col in np_col_dnp]


# crossover function for NumPy arrays
def crossover_rows_np(row1, row2, cross_over_frequency):
    # Crossover for subset1
    if random.random() < cross_over_frequency:
        row1[np_col_indices], row2[np_col_indices] = row2[np_col_indices].copy(), row1[np_col_indices].copy()

    # Crossover for subset2
    if random.random() < cross_over_frequency:
        row1[drug_col_indices], row2[drug_col_indices] = row2[drug_col_indices].copy(), row1[drug_col_indices].copy()

    # Crossover for experimental columns
    for each in experimental_col_indices:
        if random.random() < cross_over_frequency:
            row1[each], row2[each] = row2[each], row1[each]

    return row1, row2


# Define mutation function for NumPy arrays
def mutation_row(row, mutation_frequency, random_individual):
    # Mutation for subset1
    if random.random() < mutation_frequency:
        row[np_col_indices] = random_individual[np_col_indices]

    # Mutation for subset2
    if random.random() < mutation_frequency:
        row[drug_col_indices] = random_individual[drug_col_indices]

    # Mutation for experimental columns
    for each in experimental_col_indices:
        if random.random() < mutation_frequency:
            row[each] = random_individual[each]

    return row


def perform_evolution(df_MIC_Drug_NP_all, cross_over_frequency, mutation_frequency):
    # Convert DataFrame to NumPy array for faster processing
    data_array = df_MIC_Drug_NP_all.to_numpy()
    columns = df_MIC_Drug_NP_all.columns

    # Perform crossover and mutation on NumPy array except for top 5
    for i in range(5,len(data_array)):
        row1 = data_array[i]

        # Select a random row from the remaining rows
        remaining_indices = list(range(len(data_array)))
        remaining_indices.remove(i)
        random_index = random.choice(remaining_indices)
        row2 = data_array[random_index]

        # Perform crossover
        row1, row2 = crossover_rows_np(row1, row2, cross_over_frequency)

        # Perform mutation
        row1 = mutation_row(row1, mutation_frequency, random_individual)
        row2 = mutation_row(row2, mutation_frequency, random_individual)

        # Update the NumPy array with the new values
        data_array[i] = row1
        data_array[random_index] = row2

    # Convert the modified NumPy array back to a DataFrame
    df_new = pd.DataFrame(data_array, columns=columns)
    df_new = df_new.sort_values('Fitness', ascending=False)
    return df_new
