import pandas as pd
from lazypredict.Supervised import LazyRegressor
from Models import feature_selection
from Models.old import data_transformer_new
from sklearn.model_selection import train_test_split

df_clean = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\Models\filling_missing_values\Final\Final_synergy_ZOI_drug_NP_data_preprocessed_clean_fill_all.csv')
df_clean.to_csv('GA_ZOI_drug_NP_data_all_features.csv')
df_clean = df_clean.drop(['CID','doi', 'Canonical_smiles', 'IdList', 'Kingdom', 'Clade','smile'], axis=1)
selector = feature_selection.feature_remover(variance_threshold=0.05, correlation_threshold=0.6)
df_cleaned = selector.fit_transform(df_clean)



col = ['NP', 'NP_Synthesis', 'Drug_dose (μg/disk)', 'NP_concentration (μg/ml)','NP size_avg (nm)',
       'shape', 'method', 'Bacteria', 'Drug', 'reference', 'time (hr)',
       'Valance_electron', 'Phylum', 'Class', 'Order', 'Family', 'Genus',
       'Superkingdom', 'Species', 'bac_type', 'gram', 'min_Incub_period, h',
       'growth_temp, C', 'biosafety_level', 'isolated_from',
       'MaxAbsEStateIndex', 'MinEStateIndex', 'qed', 'BCUT2D_MWHI', 'AvgIpc',
       'BalabanJ', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA4', 'PEOE_VSA6',
       'PEOE_VSA8', 'SlogP_VSA4', 'SlogP_VSA8', 'EState_VSA6', 'EState_VSA7',
       'VSA_EState9', 'NumAliphaticCarbocycles', 'fr_ArN', 'fr_Imine',
       'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_unbrch_alkane', 'Drug_class','ZOI_NP (mm)','ZOI_drug (mm)', 'ZOI_Drug_NP (mm)']

df_clean = df_clean[col]
X = df_clean.drop(['reference', 'ZOI_Drug_NP (mm)'], axis=1)
# df_x = df_clean.drop(['reference', 'doi','Canonical_smiles', 'IdList','Clade','Unnamed: 0','smile','Kingdom','Genus','Species'], axis=1)

y = df_clean[['ZOI_Drug_NP (mm)']].copy()
df_scaled, le_dict, scaler = data_transformer_new.df_fit_transformer(X)


X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
# test.to_csv('Model_comparision_processed_ZOI_drug_NP.csv')
print(train)
