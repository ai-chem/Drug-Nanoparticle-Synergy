import pandas as pd
from lazypredict.Supervised import LazyRegressor
from Models.old_old import data_transformer
from sklearn.model_selection import train_test_split


df_ZOI_Drug_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\separate_datasets\synergy_ZOI_Drug_NP_data.csv')
df = df_ZOI_Drug_NP.drop_duplicates()
df.reset_index(drop= True)
print(df.columns.tolist())

df_x = df.drop(['ZOI_Drug_NP (mm)'], axis=1)
df_y = df[['ZOI_Drug_NP (mm)']].copy()
df_scaled, le_dict, scaler  = data_transformer.df_transfomer(df_x)
df_scaled.info()

X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
# test.to_csv('Model_comparision_raw_ZOI_drug_NP.csv')
print(train)