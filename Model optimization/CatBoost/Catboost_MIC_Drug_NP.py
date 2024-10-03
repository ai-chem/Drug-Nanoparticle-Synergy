import pandas as pd
import Model_visualization
import normal_model
import Hyperparameter_tuning


df = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_drug_NP_filled_final.csv')
df_x = df.drop(['Unnamed: 0','MIC_drug_NP_log','reference'], axis=1)
df_y = df[['MIC_drug_NP_log']].copy()

# best_params = Hyperparameter_tuning.optimization(df_x, df_y)
best_params = {
    'depth': 4,
    'learning_rate': 0.06917460360259758,
    'n_estimators': 734,
    'min_child_samples': 8,
    'subsample': 0.7707133543831904,
    'colsample_bylevel': 0.8179802164703527,
    'l2_leaf_reg': 8.784429474739879,
    'random_strength': 1.0939467490981523
}
x_train, y_train, model, y_test, test_preds, train_preds = normal_model.model_catboost(df_x, df_y, best_params)

Model_visualization.scatter_plot(y_train, train_preds, y_test, test_preds, xlim=(-5, 7), ylim=(-5, 7), title='Catboost Model for MC_Drug_NP prediction', save_path='model_catboost_MC_Drug_NP_optimized_scatter.png')
print(x_train.columns)
cols = ['MIC_drug_log','time (hr)','MIC_NP_log','NP size_avg (nm)','Family']
Model_visualization.feature_importance_plot(model, x_train, x_train.columns, title='Feature Importance of Catboost Model for MC_Drug_NP prediction', save_path='feature_importance_MC_Drug_NP.png')
Model_visualization.shap_summary_plot(model, x_train, x_train.columns, save_path='shap_summary_plot_catboost_model_MC_Drug_NP.png')
# model.save_model('CB_mic_drug_np_optimized.cbm')



