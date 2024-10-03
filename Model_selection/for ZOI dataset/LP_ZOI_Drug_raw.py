import pandas as pd
from lazypredict.Supervised import LazyRegressor
from Models import data_transform
from sklearn.model_selection import train_test_split


df_ZOI_Drug = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\separate_datasets\synergy_ZOI_Drug_data.csv')

df = df_ZOI_Drug.drop_duplicates()
df.reset_index(drop= True)

df_x = df.drop(['ZOI_drug (mm)'], axis=1)
df_y = df[['ZOI_drug (mm)']].copy()
df_scaled, le_dict, scaler  = data_transform.df_fit_transformer(df_x)
df_scaled.info()

X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
# test.to_csv('Model_comparision_raw_ZOI_drug.csv')
print(train)