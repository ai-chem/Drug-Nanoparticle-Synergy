import pandas as pd
import numpy as np
from Models import data_transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess data
df = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_ZOI_drug_NP_filled.csv')
df = df[df['ZOI_Drug_NP (mm)'] < 45]
# df_validation = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\Models\filling_missing_values\Final\Final_synergy_ZOI_drug_NP_data_preprocessed_clean_validation_set_fill_with_pred_zoi_np+drug.csv')
df_validation = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\Models\filling_missing_values\Final\Final_synergy_ZOI_drug_NP_data_preprocessed_clean_validation_set_fill.csv')

dc = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\raw\descriptors\drug_class.csv')
df_validation = pd.merge(df_validation, dc[['Drug', 'Drug_class']], on='Drug', how='left')
df_validation = df_validation[df_validation['ZOI_Drug_NP (mm)'] > 5]
df_validation = df_validation.reset_index(drop=True)

# df_validation = df_validation[df.columns]
# print(df_validation['NP'].count())
df_x = df.drop(['ZOI_Drug_NP (mm)', 'reference'], axis=1)
df_y = df[['ZOI_Drug_NP (mm)']].copy()

# dfx_val = df_validation.drop(['ZOI_Drug_NP (mm)', 'reference'], axis=1)
dfy_val = df_validation[['ZOI_Drug_NP (mm)']].copy()

# Transform the data
df_scaled, le_dict, scaler = data_transform.df_fit_transformer(df_x)
df_val_scaled = data_transform.df_transformer(df_validation[df_x.columns], le_dict, scaler)
print('Columns in scaled DataFrame:', df_scaled.columns)

# Best parameters

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

x_train, x_test, y_train, y_test = train_test_split(df_scaled, df_y, test_size=0.2, random_state=42)
model = CatBoostRegressor(**best_params)
model.fit(x_train, y_train)
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)
val_preds = model.predict(df_val_scaled)

# Save the model
# joblib_file = r'C:\Users\user\Desktop\Synergy_project_2024\catboost_model_zoi_drug_np.pkl'
# joblib.dump(model, joblib_file)


def scatter_plot(y_train, y_train_preds, y_test, y_test_preds, y_val, y_val_preds, title='Catboost Model', xlim=(0, 55),
                 ylim=(0, 55), save_path='scatter_plot_catboost.png', val_label='Test'):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    f, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(y_train, y_train_preds, color='#2d4d85', s=30, label='Train ', alpha=0.2)
    plt.scatter(y_test, y_test_preds, color='#951d6d', s=30, label='Validation', alpha=0.2)
    plt.scatter(y_val, y_val_preds, color='#d60000', s=30, label=val_label, alpha=0.8)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='#444444', linewidth=2)

    # Customize axis
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(3)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['left'].set_linewidth(3)

    plt.title(title)
    plt.xlabel('Actual data')
    plt.ylabel('Predicted data')
    plt.legend(loc='upper left')
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Save the figure with transparency
    plt.savefig(save_path, transparent=True)
    plt.show()


scatter_plot(y_train, train_preds, y_test, test_preds, dfy_val, val_preds)


def print_metrics(y_true, y_preds, dataset_name):
    r2 = r2_score(y_true, y_preds)
    mse = mean_squared_error(y_true, y_preds)
    mae = mean_absolute_error(y_true, y_preds)
    rmse = np.sqrt(mse)
    print(f'{dataset_name} Metrics:')
    print(f'R² Score: {r2:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}\n')


# Print metrics for train, test, and validation sets
print_metrics(y_train, train_preds, 'Train')
print_metrics(y_test, test_preds, 'Validation')
print_metrics(dfy_val, val_preds, 'unseen test')

def scatter_plot_by_column(df_validation, df_val_scaled, dfy_val, val_preds, column, save_path_prefix):
    unique_values = df_validation[column].unique()
    for value in unique_values:
        subset_indices = df_validation[column] == value
        subset_df_validation = df_validation[subset_indices]
        subset_val_scaled = df_val_scaled[subset_indices]
        subset_val_preds = val_preds[subset_indices]
        subset_dfy_val = dfy_val[subset_indices]

        scatter_plot(y_train, train_preds, y_test, test_preds, subset_dfy_val, subset_val_preds,
                     title=f'Catboost Model - {value}', save_path=f'{save_path_prefix}_{value}.png',
                     val_label=f'{value}')

        # Print metrics for each value in the column
        print_metrics(subset_dfy_val, subset_val_preds, f'Validation ({value})')

# Create scatter plots for each unique Bacteria
# scatter_plot_by_column(df_validation, df_val_scaled, dfy_val, val_preds, 'Bacteria', 'scatter_plot_catboost_bacteria')

# Create scatter plots for each unique Drug
# scatter_plot_by_column(df_validation, df_val_scaled, dfy_val, val_preds, 'Drug', 'scatter_plot_catboost_drug')

# NP
# scatter_plot_by_column(df_validation, df_val_scaled, dfy_val, val_preds, 'NP', 'scatter_plot_catboost_np')

# Add the predicted values as a new column in the df_validation DataFrame
df_validation['pred_ZOI_drug_NP'] = val_preds

# Save the DataFrame to a CSV file
output_file_path = r'C:\Users\user\Desktop\Synergy_project_2024\Models\filling_missing_values\Final\Final_synergy_ZOI_drug_NP_data_preprocessed_clean_validation_set_fill_with_pred_zoi_np+drug+dnp.csv'
df_validation.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")

def scatter_plot_with_rmse(y_train, y_train_preds, y_test, y_test_preds, y_val, y_val_preds,
                           title='CatBoost Model with RMSE', xlim=(0, 55), ylim=(0, 55), save_path='scatter_plot_CB_with_RMSE_ZOI_dnp_final.png', val_label='Test'):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    f, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(y_train, y_train_preds, color='#2d4d85', s=30, label='Train', alpha=0.2)
    plt.scatter(y_test, y_test_preds, color='#951d6d', s=30, label='Validation', alpha=0.2)
    plt.scatter(y_val, y_val_preds, color='#d60000', s=30, label=val_label, alpha=0.8)

    # Extract scalar values for min and max
    min_val = min(y_train.min(), y_test.min(), y_val.min())
    max_val = max(y_train.max(), y_test.max(), y_val.max())
    line = np.linspace(min_val, max_val, 100)
    plt.plot(line, line, color='#444444', linewidth=2)

    # Calculate RMSE for test dataset
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_preds))

    # Plot RMSE bands
    plt.fill_between(line, line - test_rmse, line + test_rmse, color='#951d6d', alpha=0.1)

    # Customize axis
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(3)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['left'].set_linewidth(3)

    plt.title(title)
    plt.xlabel('Actual data')
    plt.ylabel('Predicted data')
    plt.legend(loc='upper left')
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Save the figure with transparency
    plt.savefig(save_path, transparent=True)
    plt.show()

# Call the function with appropriate parameters
scatter_plot_with_rmse(
    y_train.values.flatten(),
    train_preds,
    y_test.values.flatten(),
    test_preds,
    dfy_val.values.flatten(),
    val_preds
)
