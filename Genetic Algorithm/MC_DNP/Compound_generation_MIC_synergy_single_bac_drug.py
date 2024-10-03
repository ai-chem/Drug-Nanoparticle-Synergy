from Models import data_transform
import Model_new
import pandas as pd
import numpy as np
import random

population_size = 100

# Load data
df_MIC_Drug_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_drug_NP_filled_final.csv')
df_MIC_Drug_NP_all = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\GA_MC_drug_NP_data_all_features.csv')
df_MIC_Drug_NP_all =df_MIC_Drug_NP_all.drop(['Unnamed: 0','MIC_NP (μg/ml)', 'MIC_drug (μg/ml)', 'MIC_drug_NP (μg/ml)'], axis=1)
df_MIC_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_NP.csv')
df_MIC_Drug = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_drug.csv')
drug_class = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\drug_class.csv')
# df_MIC_Drug_NP_all =df_MIC_Drug_NP_all.drop([], axis=1)
bacterial_descriptor = pd.read_csv(
    r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_bacterial_descriptor.csv')
drug_descriptor = pd.read_csv(
    r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_drug_descriptors_2024_version1.csv')
np_descriptor = pd.read_csv(
    r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_nanoparticle_descriptors.csv')

# Merge drug_descriptor with drug_class
df_MIC_Drug = pd.merge(df_MIC_Drug, drug_class[['Drug', 'Drug_class']], on='Drug', how='left')
# df_MIC_Drug_NP_all = pd.merge(df_MIC_Drug_NP_all, drug_class[['Drug', 'Drug_class']], on='Drug', how='left')

# print(df_MIC_Drug_NP_all.columns.tolist())


# Identify common columns
drug_col_dnp = df_MIC_Drug_NP_all.columns.intersection(drug_descriptor.columns)
np_col_dnp = df_MIC_Drug_NP_all.columns.intersection(np_descriptor.columns).append(pd.Index(['reference']))
bac_col_dnp = df_MIC_Drug_NP_all.columns.intersection(bacterial_descriptor.columns)
# Drop target columns to create feature set X
X = df_MIC_Drug_NP_all.drop(['MIC_NP_log', 'MIC_drug_log', 'MIC_drug_NP_log'], axis=1)
# print(X.columns.tolist())
#unique bacteria
uniq_bacteria_data = X.drop_duplicates('Bacteria')
uniq_drug_data = X.drop_duplicates('Drug')
uniq_np_data = X.drop_duplicates('NP')

def population(size):
    # Sample size number of rows from X to create individuals
    indv = X.sample(n=size, replace=True).reset_index(drop=True)
    new_one = pd.DataFrame(data=indv, columns=X.columns)

    # Assign new random values for 'shape' and 'method' columns
    new_one['shape'] = [random.choice(X['shape'].unique()) for _ in range(size)]
    new_one['method'] = [random.choice(X['method'].unique()) for _ in range(size)]
    new_one['time (hr)'] = [random.choice(X['time (hr)'].unique()) for _ in range(size)]
    new_one['MDR'] = [random.choice(X['MDR'].unique()) for _ in range(size)]


    # Assign new random values for 'size' column
    min_values = np.random.randint(5, 101, size)
    avg_values = np.random.randint(min_values, 101, size)
    max_values = np.random.randint(avg_values, 101, size)

    # Assign the generated values to the DataFrame
    new_one['NP size_min (nm)'] = min_values
    new_one['NP size_avg (nm)'] = avg_values
    new_one['NP size_max (nm)'] = max_values

    # Randomly sample and assign values from original data for np_col_dnp and drug_col_dnp columns
    new_one[np_col_dnp] = X[np_col_dnp].sample(n=size, replace=True).reset_index(drop=True)
    new_one[drug_col_dnp] = X[drug_col_dnp].sample(n=size, replace=True).reset_index(drop=True)

    return new_one

# print(population(100))
# Function to adjust the population based on bacteria type
def bacterial_population(df_population):
    uniq_NP = 'Ag'
    indv_np =uniq_np_data[uniq_np_data['NP'] == uniq_NP]
    np_population = indv_np.sample(n=len(df_population),replace=True)
    np_population.reset_index(drop=True, inplace=True)

    pathogen_bacteria = 'Staphylococcus aureus' # Trichophyton mentagrophytes pathogenic > Escherichia coli, Staphylococcus aureus, Acinetobacter baumannii
    indv_pathogen = uniq_bacteria_data[uniq_bacteria_data['Bacteria'] == pathogen_bacteria]
    indv_population = indv_pathogen.sample(n=len(df_population), replace=True)
    indv_population.reset_index(drop=True, inplace=True)
    drug = 'Tetracycline' # Vancomycin Ampicillin, Amoxicillin, Tetracycline, Doxycycline
    indv_drug = uniq_drug_data[uniq_drug_data['Drug'] == drug]
    drug_population = indv_drug.sample(n=len(df_population),replace=True)
    drug_population.reset_index(drop=True, inplace=True)

    df_pathogen = df_population.copy()
    np_population.index = df_pathogen.index[:len(np_population)]
    indv_population.index = df_pathogen.index[:len(indv_population)]
    drug_population.index = df_pathogen.index[:len(df_population)]
    df_pathogen.loc[np_population.index, np_col_dnp] = np_population[np_col_dnp].values
    df_pathogen.loc[indv_population.index, bac_col_dnp] = indv_population[bac_col_dnp].values
    df_pathogen.loc[drug_population.index, drug_col_dnp] = drug_population[drug_col_dnp].values
    return df_pathogen

# df = bacterial_population(population(100))
# df.to_csv('b.csv')

def predict_MIC_of_drug_np(df):
    df_pathogen = bacterial_population(df)

    #predict MIC_NP
    df_x = df_MIC_NP.drop(['Unnamed: 0', 'MIC_log', 'MIC_NP (μg/ml)', 'reference'], axis=1)
    df_scaled, oe_dict, scaler = data_transform.df_fit_transformer(df_x)
    df_scaled_mic_np2 = data_transform.df_transformer(df_pathogen[df_scaled.columns], oe_dict, scaler)
    pathogen1 = Model_new.np_predict(df_scaled_mic_np2)
    df_p = df_pathogen.assign(MIC_NP_log=pathogen1)

    #predict MIC_Drug
    dfx = df_MIC_Drug.drop(['Unnamed: 0', 'MIC_drug_log'], axis=1)
    dg_scaled, oe_dict_dg, scaler_dg = data_transform.df_fit_transformer(dfx)
    df_scaled_mic_dg2 = data_transform.df_transformer(df_pathogen[dg_scaled.columns], oe_dict_dg, scaler_dg)
    pathogen2 = Model_new.drug_predict(df_scaled_mic_dg2)
    df_p = df_p.assign(MIC_drug_log=pathogen2)

    #predict MIC_Drug_NP
    df_x_dnp = df_MIC_Drug_NP.drop(['Unnamed: 0','MIC_drug_NP_log','reference'], axis=1)
    dnp_scaled, le_dict_dnp, scaler_dnp = data_transform.df_fit_transformer(df_x_dnp)
    df_scaled_mic_dnp2 = data_transform.df_transformer(df_p[dnp_scaled.columns], le_dict_dnp, scaler_dnp)

    pathogen3 = Model_new.drug_np_predict(df_scaled_mic_dnp2)

    #calculate fitness
    fitness = []
    for a in range(len(pathogen3)):
        p = pathogen3[a]
        dg = pathogen2[a]
        nanp = pathogen1[a]
        # fit = -(2*p -dg - np)
        fit =  2 * np.exp(p) - np.exp(dg) - np.exp(nanp)

        fitnn = fit.tolist()
        fitness.append(fitnn)
    df_pathogen[['p_MIC_NP_log','p_MIC_drug_log','p_MIC_drug_NP_log']] = pd.DataFrame({'p_MIC_NP_log':pathogen1,'p_MIC_drug_log':pathogen2,'p_MIC_drug_NP_log':pathogen3})
    df_pathogen = df_pathogen.assign(Fitness = fitness)

    return df_pathogen



# Generate population
new_population = population(population_size)
df_nonpathogen = predict_MIC_of_drug_np(new_population)
# df_nonpathogen.to_csv('new_gen_non.csv')

# print(df_nonpathogen)
