import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from Models import feature_selection
from Models import data_transform
from sklearn.model_selection import train_test_split

df_MIC_Drug = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\subset_clean\synergy_MC_drug_data_preprocessed_clean.csv')
df = df_MIC_Drug[df_MIC_Drug['MIC_drug (μg/ml)'] >0]
df['MIC_drug_log'] = np.log(df['MIC_drug (μg/ml)'])

df = df.drop(['MIC_drug (μg/ml)', 'doi','smile'], axis=1)
# Filter drugs and references with at least 10 occurrences
drug_counts = df['Drug'].value_counts()
valid_drugs = drug_counts[drug_counts >= 25].index
df = df[df['Drug'].isin(valid_drugs)].reset_index(drop=True)

# Feature selection
selector = feature_selection.feature_remover(variance_threshold=0.1, correlation_threshold=0.60)
df_clean = selector.fit_transform(df)


# Split the data into features and target
df_x = df_clean.drop(['MIC_drug_log'], axis=1)
df_y = df_clean[['MIC_drug_log']].copy()

# Transform the features
df_scaled, le_dict, scaler = data_transform.df_fit_transformer(df_x)

X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
test.to_csv('Model_comparision_processed_MIC_drug.csv')
print(train)
print(df.columns.tolist())


common_columns = [col for col in df.columns if col in df_clean.columns]
df_clean = df_clean[common_columns]
df_clean.reset_index(drop=True)
# df_clean.to_csv('preprocessed_MIC_drug.csv')
print(df_clean.info())
