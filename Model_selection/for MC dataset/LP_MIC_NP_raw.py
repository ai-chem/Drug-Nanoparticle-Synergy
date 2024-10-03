import pandas as pd
from lazypredict.Supervised import LazyRegressor
from Models import data_transform
from sklearn.model_selection import train_test_split

df_MIC_NP = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\old\Data_MIC_NP_old\merged_MIC_NP_raw.csv')
df_model = df_MIC_NP.drop_duplicates()
df_model.reset_index(drop= True)
df_x = df_model.drop(['MIC_NP (μg/ml)'], axis=1)
df_y = df_model[['MIC_NP (μg/ml)']].copy()
df_scaled, le_dict, scaler  = data_transform.df_fit_transformer(df_x)
df_scaled.info()

X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
# test.to_csv('Model_comparision_raw_MIC_NP.csv')
print(train)
