import pandas as pd
import Model_visualization
import normal_model
import Hyperparameter_tuning
import joblib


# Load and merge datasets
df_clean = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_drug.csv')
dc = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\drug_class.csv')
df_clean = pd.merge(df_clean, dc[['Drug', 'Drug_class']], on='Drug', how='left')

# # Filter drugs and references with at least 10 occurrences
# drug_counts = df_clean['Drug'].value_counts()
# valid_drugs = drug_counts[drug_counts >= 15].index
# df_clean = df_clean[df_clean['Drug'].isin(valid_drugs)].reset_index(drop=True)
#
# ref_counts = df_clean['reference'].value_counts()
# valid_ref = ref_counts[ref_counts >= 10].index
# df_clean = df_clean[df_clean['reference'].isin(valid_ref)].reset_index(drop=True)

# Prepare features and target
df_x = df_clean.drop(['Unnamed: 0', 'ZOI_drug (mm)'], axis=1)
df_y = df_clean[['ZOI_drug (mm)']].copy()

# best_params = Hyperparameter_tuning.optimization(df_x, df_y)

best_params = {
    'depth': 7,
    'learning_rate': 0.12033389024969879,
    'n_estimators': 388,
    'min_child_samples': 4,
    'subsample': 0.6996862792968911,
    'colsample_bylevel': 0.765519802937542,
    'l2_leaf_reg': 8.931555869665246,
    'random_strength': 0.8805633875696363,
    'verbose': 0
}


# best_params = {'verbose': 0}
x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_catboost(df_x, df_y, best_params)
# Save the model
joblib_file = r'C:\Users\user\Desktop\Synergy_project_2024\catboost_model_zoi_drug.pkl'
joblib.dump(model, joblib_file)

Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, title='Catboost Model for ZOI_Drug prediction', xlim=(0, 55), ylim=(0, 55), save_path='model_catboost_ZOI_Drug_optimized_scatter.png')
print(x_train.columns)
cols = x_train.columns
cols = ['Drug_dose (μg/disk)','time (hr)','MinEStateIndex','SPS','BCUT2D_MWHI']
Model_visualization.feature_importance_plot(model, x_train, cols, title='Feature Importance of Catboost Model for ZOI_Drug prediction', save_path='feature_importance_ZOI_Drug.png')
Model_visualization.shap_summary_plot(model, x_train, cols, save_path='shap_summary_plot_catboost_model_ZOI_Drug.png')
model.save_model('CB_zoi_drug_optimized.cbm')
