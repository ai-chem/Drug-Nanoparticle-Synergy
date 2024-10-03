import pandas as pd
from lazypredict.Supervised import LazyRegressor
from Models import feature_selection
from Models import data_transform
from sklearn.model_selection import train_test_split

df_ZOI_Drug = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\subset_clean\synergy_ZOI_Drug_data_preprocessed_clean.csv')

df = df_ZOI_Drug[df_ZOI_Drug['ZOI_drug (mm)'] > 5]
df = df.drop(['doi','IdList','Clade','smile','Kingdom','Genus','Species'], axis=1)
df = df.dropna(subset = ['Drug_dose (μg/disk)'])
df = df.drop_duplicates()
df.reset_index(drop= True)
print(df.info())

selector = feature_selection.feature_remover(variance_threshold= 0.80*(1-0.80), correlation_threshold=0.6)
df_clean = selector.fit_transform(df)

# Filter drugs and references with at least 10 occurrences
drug_counts = df_clean['Drug'].value_counts()
valid_drugs = drug_counts[drug_counts >= 15].index
df_clean = df_clean[df_clean['Drug'].isin(valid_drugs)].reset_index(drop=True)

ref_counts = df_clean['reference'].value_counts()
valid_ref = ref_counts[ref_counts >= 10].index
df_clean = df_clean[df_clean['reference'].isin(valid_ref)].reset_index(drop=True)


df_x = df_clean.drop(['ZOI_drug (mm)'], axis=1)
df_y = df_clean[['ZOI_drug (mm)']].copy()

df_scaled, le_dict, scaler  = data_transform.df_fit_transformer(df_x)

X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
test.to_csv('Model_comparision_processed_ZOI_drug.csv')
print(train)
print(df.columns.tolist())

df_clean.info()

common_columns = [col for col in df.columns if col in df_clean.columns]
df_clean = df_clean[common_columns]
df_clean.reset_index(drop=True)
# df_clean.to_csv('preprocessed_ZOI_drug.csv')
print(df_clean.columns)