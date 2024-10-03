import pandas as pd
from lazypredict.Supervised import LazyRegressor
from Models import feature_selection
from Models import data_transform
from sklearn.model_selection import train_test_split

df_ZOI_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\old\Data_ZOI_NP_old\merged_ZOI_NP_cleaned_final.csv')
df_ZOI_NP = df_ZOI_NP[df_ZOI_NP['ZOI_NP (mm)']>=5]
df_ZOI_NP = df_ZOI_NP.drop(['Unnamed: 0'], axis=1)
selector = feature_selection.feature_remover(variance_threshold= 0.85*(1-0.85), correlation_threshold=0.85)
df_clean = selector.fit_transform(df_ZOI_NP)
df_clean =df_clean.drop_duplicates()
df_clean.reset_index(drop=True)

print(df_clean.columns)
# df_ZOI_NP.reset_index()
df_x = df_clean.drop(['ZOI_NP (mm)','reference'], axis=1)
df_y = df_clean[['ZOI_NP (mm)']].copy()

df_scaled, le_dict, scaler  = data_transform.df_fit_transformer(df_x)
df_scaled.info()
print(df_scaled.columns)
X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
test.to_csv('Model_comparision_processed_ZOI_NP.csv')
print(train)

order = ['NP', 'NP_Synthesis', 'shape', 'method', 'Bacteria', 'time (hr)',
       'Phylum', 'Class', 'Order', 'Family', 'Superkingdom', 'bac_type',
       'gram', 'isolated_from', 'NP_concentration (μg/ml)', 'NP size_min (nm)',
       'NP size_max (nm)', 'Valance_electron', 'amw', 'lipinskiHBA', 'NumHBA',
       'chi0v', 'kappa1', 'min_Incub_period, h', 'growth_temp, C',
       'biosafety_level','reference','ZOI_NP (mm)']
df =df_clean[order]
# df.to_csv('preprocessed_ZOI_NP.csv')
df.info()