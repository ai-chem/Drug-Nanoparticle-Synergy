import pandas as pd
import Model_visualization
import normal_model
import Hyperparameter_tuning

# Load data
df_clean = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_NP.csv')

# Prepare features and target
df_x = df_clean.drop(['Unnamed: 0', 'MIC_log', 'MIC_NP (μg/ml)', 'reference'], axis=1)
df_y = df_clean[['MIC_log']].copy()
# best_params = Hyperparameter_tuning.optimization(df_x, df_y)

# best_params = {'verbose': 0}


best_params = {
    'depth': 3,
    'learning_rate': 0.2642309508248977,
    'n_estimators': 885,
    'min_child_samples': 7,
    'subsample': 0.6556334514254635,
    'colsample_bylevel': 0.6085684250148287,
    'l2_leaf_reg': 9.704582439848151,
    'random_strength': 7.999703462124868,
    'verbose': 0
}




x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_catboost(df_x, df_y, best_params)
Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, title='Catboost Model for MC_NP prediction', save_path='model_catboost_MC_NP_optimized_scatter.png')
print(x_train.columns)
cols = x_train.columns
cols = ['NP size_max (nm)','NP size_min (nm)','Valance_electron','time (hr)','amw']
Model_visualization.feature_importance_plot(model, x_train, cols, title='Feature Importance of Catboost Model for MC_NP prediction', save_path='feature_importance_MC_NP.png')
Model_visualization.shap_summary_plot(model, x_train, cols, save_path='shap_summary_plot_catboost_model_MC_NP.png')
model.save_model('CB_mic_np_optimized.cbm')