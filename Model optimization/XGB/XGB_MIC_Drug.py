import pandas as pd
import Model_visualization
import normal_model
import Hyperparameter_tuning

# Load and merge datasets
df_clean = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_drug.csv')
dc = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\drug_class.csv')
df_clean = pd.merge(df_clean, dc[['Drug', 'Drug_class']], on='Drug', how='left')

# Prepare features and target
df_x = df_clean.drop(['Unnamed: 0', 'MIC_drug_log'], axis=1)
df_y = df_clean[['MIC_drug_log']].copy()

# best_params = Hyperparameter_tuning.optimization(df_x, df_y)
best_params = {
    'max_depth': 7,
    'learning_rate': 0.1545267327215016,
    'n_estimators': 156,
    'min_child_weight': 3,
    'subsample': 0.7185170173647396,
    'colsample_bytree': 0.4737802437761452,
    'reg_alpha': 1.609747781105942,
    'reg_lambda': 3.63736910517587
}



x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_xgboost(df_x, df_y, best_params)
Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, title='XGB Model for MC_drug prediction', save_path='model_xgb_MC_drug_optimized_scatter.png')
print(x_train.columns)
cols = ['MaxAbsEStateIndex','time (hr)','MinEStateIndex', 'PEOE_VSA11', 'PEOE_VSA8']
# cols = x_train.columns
Model_visualization.feature_importance_plot(model, x_train, cols, title='Feature Importance of XGB Model for MC_drug prediction', save_path='feature_importance_XGB_drug.png')
Model_visualization.shap_summary_plot(model, x_train, cols, save_path='shap_summary_plot_XGB_model_MC_drug.png')
# model.save_model('XGB_mic_drug_optimized.cbm')