from Models import data_transform
import Model_new_ZOI
import pandas as pd
import numpy as np
import random

population_size = 100

# Load data
df_ZOI_Drug_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_drug_NP_filled.csv')
df_ZOI_Drug_NP_all = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\GA_ZOI_drug_NP_data_all_features.csv')
df_ZOI_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_NP.csv')
df_ZOI_Drug = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_drug.csv')
drug_class = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\drug_class.csv')

bacterial_descriptor = pd.read_csv(
    r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_bacterial_descriptor.csv')
drug_descriptor = pd.read_csv(
    r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_drug_descriptors_2024_version1.csv')
np_descriptor = pd.read_csv(
    r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\synergy_nanoparticle_descriptors.csv')

# Merge drug_descriptor with drug_class
df_ZOI_Drug = pd.merge(df_ZOI_Drug, drug_class[['Drug', 'Drug_class']], on='Drug', how='left')
# df_ZOI_Drug_NP_all = pd.merge(df_ZOI_Drug_NP_all, drug_class[['Drug', 'Drug_class']], on='Drug', how='left')

# Identify common columns
df_ZOI_Drug_NP_all= df_ZOI_Drug_NP_all.drop(['Unnamed: 0'], axis=1)

drug_col_dnp = df_ZOI_Drug_NP_all.columns.intersection(drug_descriptor.columns)
np_col_dnp = df_ZOI_Drug_NP_all.columns.intersection(np_descriptor.columns)
bac_col_dnp = df_ZOI_Drug_NP_all.columns.intersection(bacterial_descriptor.columns)
# Drop target columns to create feature set X
X = df_ZOI_Drug_NP_all.drop(['ZOI_drug (mm)', 'ZOI_NP (mm)','ZOI_Drug_NP (mm)'], axis=1)
print(X.columns.tolist())
#unique bacteria
uniq_bacteria_data = X.drop_duplicates('Bacteria')
uniq_drug_data = X.drop_duplicates('Drug')
uniq_np_data = X.drop_duplicates('NP')
def population(size):
    # Sample size number of rows from X to create individuals
    indv = X.sample(n=size, replace=True).reset_index(drop=True)
    new_one = pd.DataFrame(data=indv, columns=X.columns)
    new_one.columns.tolist()
    drug_conc = [0.5,1,2,5,10,20,30,40,50,100]
    # Assign new random values for 'shape' and 'method' columns
    new_one['Drug_dose (μg/disk)'] = np.random.choice(drug_conc, size)
    new_one['NP_concentration (μg/ml)'] = np.random.randint(1, 501, size)
    new_one['shape'] = [random.choice(X['shape'].unique()) for _ in range(size)]
    new_one['method'] = [random.choice(X['method'].unique()) for _ in range(size)]
    new_one['time (hr)'] = [random.choice(X['time (hr)'].unique()) for _ in range(size)]
    new_one['MDR'] = [random.choice(X['MDR'].unique()) for _ in range(size)]

    # Assign new random values for 'size' column
    min_values = np.random.randint(5, 101, size)
    avg_values = np.random.randint(min_values, 101, size)
    max_values = np.random.randint(avg_values, 101, size)
    new_one['NP size_min (nm)'] = min_values
    new_one['NP size_avg (nm)'] = avg_values
    new_one['NP size_max (nm)'] = max_values


    # Randomly sample and assign values from original data for np_col_dnp and drug_col_dnp columns
    new_one[np_col_dnp] = X[np_col_dnp].sample(n=size, replace=True).reset_index(drop=True)
    new_one[bac_col_dnp] = X[bac_col_dnp].sample(n=size, replace=True).reset_index(drop=True)
    new_one[drug_col_dnp] = X[drug_col_dnp].sample(n=size, replace=True).reset_index(drop=True)
    return new_one

# print(population(100))
# Function to adjust the population based on bacteria type
def bacterial_population(df_population):
    uniq_NP = 'ZnO'
    indv_np =uniq_np_data[uniq_np_data['NP'] == uniq_NP]
    np_population = indv_np.sample(n=len(df_population),replace=True)
    np_population.reset_index(drop=True, inplace=True)

    pathogen_bacteria = 'Salmonella typhi' # Trichophyton mentagrophytes pathogenic > Escherichia coli, Staphylococcus aureus, Acinetobacter baumannii
    indv_pathogen = uniq_bacteria_data[uniq_bacteria_data['Bacteria'] == pathogen_bacteria]
    indv_population = indv_pathogen.sample(n=len(df_population), replace=True)
    indv_population.reset_index(drop=True, inplace=True)
    drug = 'Ceftriaxone' # Vancomycin Ampicillin, Amoxicillin, Tetracycline, Doxycycline
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
def predict_ZOI_of_drug_np(df):
    df_pathogen = bacterial_population(df)
    #predict ZOI_NP
    df_x = df_ZOI_NP.drop(['Unnamed: 0','ZOI_NP (mm)','reference'], axis=1)
    df_scaled, le_dict, scaler = data_transform.df_fit_transformer(df_x)
    df_scaled_zoi_np2 = data_transform.df_transformer(df_pathogen[df_scaled.columns], le_dict, scaler)
    pathogen1 = Model_new_ZOI.np_predict(df_scaled_zoi_np2)
    df_p = df_pathogen.assign(**{'ZOI_NP (mm)': pathogen1})
    #predict ZOI_Drug
    dfx = df_ZOI_Drug.drop(['Unnamed: 0', 'ZOI_drug (mm)'], axis=1)
    dg_scaled, le_dict_dg, scaler_dg = data_transform.df_fit_transformer(dfx)
    df_scaled_zoi_dg2 = data_transform.df_transformer(df_pathogen[dg_scaled.columns], le_dict_dg, scaler_dg)
    pathogen2 = Model_new_ZOI.drug_predict(df_scaled_zoi_dg2)
    df_p = df_p.assign(**{'ZOI_drug (mm)': pathogen2})

    #predict ZOI_Drug_NP
    df_x_dnp = df_ZOI_Drug_NP.drop(['Unnamed: 0','ZOI_Drug_NP (mm)','reference'], axis=1)
    dnp_scaled, le_dict_dnp, scaler_dnp = data_transform.df_fit_transformer(df_x_dnp)
    df_scaled_zoi_dnp2 = data_transform.df_transformer(df_p[dnp_scaled.columns], le_dict_dnp, scaler_dnp)
    pathogen3 = Model_new_ZOI.drug_np_predict(df_scaled_zoi_dnp2)

    #calculate fitness
    fitness = []
    for a in range(len(pathogen3)):
        p = pathogen3[a]
        dg = pathogen2[a]
        np = pathogen1[a]
        fit = (2*p -dg - np)
        fitnn = fit.tolist()
        fitness.append(fitnn)
    df_pathogen[['np_ZOI_NP','np_ZOI_drug','np_ZOI_drug_NP']] = pd.DataFrame({'p_ZOI_NP':pathogen1,'p_ZOI_drug':pathogen2,'p_ZOI_drug_NP':pathogen3})
    df_pathogen = df_pathogen.assign(Fitness = fitness)

    return df_pathogen

# Generate population
new_population = population(population_size)
df_nonpathogen = predict_ZOI_of_drug_np(new_population)
# df_nonpathogen.to_csv('new_gen_non_zoi.csv')
# print(df_nonpathogen)
