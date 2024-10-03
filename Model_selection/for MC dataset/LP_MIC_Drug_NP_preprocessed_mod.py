import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from Models import data_transform
from Models import feature_selection
# from Models import data_transform
from sklearn.model_selection import train_test_split

# Load the dataset
# df = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_drug_NP_filled_mod.csv')
df = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\Models\filling_missing_values\Final\Final_synergy_MC_drug_NP_data_preprocessed_clean_fill_all.csv')
print(df.columns.tolist())
df['MIC_drug_NP_log'] = np.log(df['MIC_drug_NP (μg/ml)'])
df.to_csv('GA_MC_drug_NP_data_all_features.csv')
selector = feature_selection.feature_remover(variance_threshold=0.90 * (1 - 0.90), correlation_threshold=0.6)
df_filter = selector.fit_transform(df)
print('hh',df_filter.columns.tolist())


col = ['NP', 'NP_Synthesis', 'NP size_avg (nm)', 'shape', 'method', 'Bacteria', 'Drug', 'Drug_class', 'reference', 'time (hr)', 'Valance_electron', 'amw', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Superkingdom', 'Species', 'bac_type', 'gram', 'avg_Incub_period, h', 'growth_temp, C', 'isolated_from', 'MaxAbsEStateIndex', 'MinEStateIndex', 'SPS', 'MolWt', 'PEOE_VSA13', 'PEOE_VSA4', 'PEOE_VSA8', 'SMR_VSA2', 'SMR_VSA7', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'fr_alkyl_halide', 'fr_allylic_oxid', 'MIC_NP_log', 'MIC_drug_log', 'MIC_drug_NP_log']
df_clean = df[col]
# Split the data into features and target
X = df_clean.drop(['MIC_drug_NP_log', 'reference'], axis=1)
y = df_clean[['MIC_drug_NP_log']].copy()

print(X.columns.tolist())
# Transform the features
df_scaled, le_dict, scaler =  data_transform.df_fit_transformer(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size=0.2, random_state=42)

# Initialize and fit the LazyRegressor
clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
train_results, test_results = clf.fit(X_train, X_test, y_train, y_test)

# Save the test results to a CSV file
test_results.to_csv('Model_comparision_preprocessed_MIC_drug_NP_log_mod.csv')

print(test_results)
print(df.info())

df_clean.to_csv('preprocessed_MIC_drug_NP_filled_final.csv')