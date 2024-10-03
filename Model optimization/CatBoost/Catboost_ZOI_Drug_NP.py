import pandas as pd
import Model_visualization
import normal_model
import Hyperparameter_tuning

df = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_drug_NP_filled.csv')
# df = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\Models\filling_missing_values\Final\Final_synergy_ZOI_drug_NP_data_preprocessed_clean_fill_all.csv')

df = df[df['ZOI_Drug_NP (mm)'] < 45]
# df_x_preprocessed = df_clean.drop(['Unnamed: 0','ZOI_Drug_NP (mm)', 'reference'], axis=1)

df_x = df.drop(['Unnamed: 0','ZOI_Drug_NP (mm)', 'reference'], axis=1)

df_y = df[['ZOI_Drug_NP (mm)']].copy()


# best_params = Hyperparameter_tuning.optimization(df_x, df_y)
# best_params = {'verbose': 0}
best_params = {
    'depth': 6,
    'learning_rate': 0.05206538586090059,
    'n_estimators': 986,
    'min_child_samples': 8,
    'subsample': 0.660278813040351,
    'colsample_bylevel': 0.8905506554944481,
    'l2_leaf_reg': 6.015238186884968,
    'random_strength': 0.5023567815905859,
    'verbose': 0
}
# best_params = {
#     'depth': 5,
#     'learning_rate': 0.07882246462406996,
#     'n_estimators': 316,
#     'min_child_samples': 4,
#     'subsample': 0.46404318780080667,
#     'colsample_bylevel': 0.4446950472318217,
#     'l2_leaf_reg': 8.269434021834147,
#     'random_strength': 0.6394081104922198,
#     'verbose': 0
# }


# best_params = {
#     'depth': 5,
#     'learning_rate': 0.03352362191401824,
#     'n_estimators': 690,
#     'min_child_samples': 8,
#     'subsample': 0.4339350608542423,
#     'colsample_bylevel': 0.9100264617309242,
#     'l2_leaf_reg': 8.313782599434818,
#     'random_strength': 0.09736459147288601
# }
#


x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_catboost(df_x, df_y, best_params)
import joblib
# Save the model
joblib_file = r'C:\Users\user\Desktop\Synergy_project_2024\catboost_model_zoi_drug_np.pkl'
joblib.dump(model, joblib_file)

Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, title='Catboost Model for ZOI_Drug_NP prediction', xlim=(0, 55), ylim=(0, 55), save_path='model_catboost_ZOI_Drug_NP_optimized_scatter.png')
print(x_train.columns)
cols = x_train.columns
# cols = ['ZOI_drug (mm)', 'ZOI_NP (mm)', 'NP_concentration (μg/ml)', 'NP size_avg (nm)','time (hr)']
Model_visualization.feature_importance_plot(model, x_train, cols, title='Feature Importance of Catboost Model for ZOI_Drug_NP prediction', save_path='feature_importance_ZOI_Drug_NP.png')
Model_visualization.shap_summary_plot(model, x_train, cols, save_path='shap_summary_plot_catboost_model_ZOI_Drug_NP.png')
# model.save_model('CB_zoi_drug_np_optimized.cbm')
