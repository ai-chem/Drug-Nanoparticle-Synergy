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
    'depth': 6,
    'learning_rate': 0.08900377614135022,
    'n_estimators': 344,
    'min_child_samples': 8,
    'subsample': 0.6560921099485709,
    'colsample_bylevel': 0.888576654122941,
    'l2_leaf_reg': 5.413523408965468,
    'random_strength': 9.182912625264738,
    'verbose': 0

}
# best_params = {'verbose': 0}
x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_catboost(df_x, df_y, best_params)
Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, title='Catboost Model for MC_Drug prediction', save_path='model_catboost_MC_Drug_optimized_scatter.png')
print(x_train.columns)
# cols = ['time (hr)','SPS','MinEStateIndex', 'BCUT2D_MWHI', 'PEOE_VSA11']
cols = x_train.columns
Model_visualization.feature_importance_plot(model, x_train, cols, title='Feature Importance of Catboost Model for MC_drug prediction', save_path='feature_importance_MC_drug.png')
Model_visualization.shap_summary_plot(model, x_train, cols, save_path='shap_summary_plot_catboost_model_MC_drug.png')
model.save_model('CB_mic_drug_optimized.cbm')