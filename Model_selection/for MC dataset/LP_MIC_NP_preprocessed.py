import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from Models import feature_selection
from Models import data_transform
from sklearn.model_selection import train_test_split

df_MIC_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\old\Data_MIC_NP_old\merged_MIC_NP_cleaned_final.csv')
df_MIC_NP.info()
df_MIC_NP = df_MIC_NP[df_MIC_NP['MIC_NP (μg/ml)'] > 0]
df_MIC_NP = df_MIC_NP.drop(['Unnamed: 0'], axis=1)
df_MIC_NP['MIC_log'] = np.log(df_MIC_NP['MIC_NP (μg/ml)'])
# print('he',df_MIC_NP.columns)

# df_model = df_MIC_NP.drop(['MIC_NP (μg/ml)','CID', 'Zeta_potential (mV)', 'Kingdom', 'Clade', 'Canonical_smiles','IdList','doi','Genus','Species','reference','NP size_min (nm)', 'NP size_max (nm)'], axis =1 )
selector = feature_selection.feature_remover(variance_threshold= 0.09, correlation_threshold=0.90)
df_clean = selector.fit_transform(df_MIC_NP)
df_clean =df_clean.drop_duplicates()
df_clean.reset_index(drop=True)
print('h',df_clean.columns)
df_x = df_clean.drop(['MIC_log','MIC_NP (μg/ml)','reference'], axis=1)
df_y = df_clean[['MIC_log']].copy()
df_scaled, le_dict, scaler  = data_transform.df_fit_transformer(df_x)
X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
test.to_csv('Model_comparision_preprocessed_MIC_NP_log.csv')
print(train)

order = ['NP', 'NP_Synthesis', 'shape', 'method', 'Bacteria', 'time (hr)',
       'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Superkingdom',
       'Species', 'bac_type', 'gram', 'isolated_from', 'NP size_min (nm)',
       'NP size_max (nm)', 'Valance_electron', 'amw', 'lipinskiHBA',
       'CrippenMR', 'chi0v', 'min_Incub_period, h', 'growth_temp, C',
       'biosafety_level','MIC_log','MIC_NP (μg/ml)','reference']
df = df_clean[order]
df.info()
# df.to_csv('preprocessed_MIC_NP.csv')
