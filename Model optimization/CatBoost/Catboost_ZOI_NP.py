import pandas as pd
import Model_visualization
import normal_model
import Hyperparameter_tuning

df_clean =pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_NP.csv')
# df_clean = df_clean[df_clean['ZOI_NP (mm)'] > 5 ]
df_x = df_clean.drop(['Unnamed: 0','ZOI_NP (mm)','reference'], axis=1)
df_y = df_clean[['ZOI_NP (mm)']].copy()
# best_params = Hyperparameter_tuning.optimization(df_x, df_y)

best_params = {
    'depth': 7,
    'learning_rate': 0.292062458240406,
    'n_estimators': 783,
    'min_child_samples': 8,
    'subsample': 0.6729221272861472,
    'colsample_bylevel': 0.5851269116309946,
    'l2_leaf_reg': 8.328233591505352,
    'random_strength': 2.7269252114485725,
    'verbose': 0
}
# best_params = {'verbose': 0}

x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_catboost(df_x, df_y, best_params)
import joblib
# Save the model
joblib_file = r'C:\Users\user\Desktop\Synergy_project_2024\catboost_model_zoi_np.pkl'
joblib.dump(model, joblib_file)

Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, title='Catboost Model for ZOI_NP prediction', xlim=(0, 45), ylim=(0, 45), save_path='model_catboost_ZOI_NP_optimized_scatter.png')
print(x_train.columns)
cols = x_train.columns
cols = ['NP_concentration (μg/ml)','NP size_max (nm)','NP size_min (nm)','amw','Valance_electron']
Model_visualization.feature_importance_plot(model, x_train, cols, title='Feature Importance of Catboost Model for ZOI_NP prediction', save_path='feature_importance_ZOI_NP.png')
Model_visualization.shap_summary_plot(model, x_train, cols, save_path='shap_summary_plot_catboost_model_ZOI_NP.png')
# model.save_model('CB_zoi_np_optimized.cbm')
