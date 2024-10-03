import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Load data
df_clean = pd.read_csv(r'C:\Users\user\Desktop\Synergy_project_2024\data\LP\preprocessed_MIC_NP.csv')

# Prepare features and target
df_x = df_clean.drop(['Unnamed: 0', 'MIC_log', 'MIC_NP (μg/ml)', 'reference'], axis=1)
df_y = df_clean[['MIC_log']].copy()

def optimization(df_x, df_y):
    # Convert categorical values into numerical
    oe = OrdinalEncoder()
    categorical_features = df_x.select_dtypes(include=['object']).columns
    df_x[categorical_features] = oe.fit_transform(df_x[categorical_features].astype(str))

    # Scale features
    sc = StandardScaler()
    df_x_scaled = pd.DataFrame(sc.fit_transform(df_x), columns=df_x.columns)

    def objective(trial, df_x, df_y):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        }
        model = XGBRegressor(**params, verbosity=0)
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = []
        # Cross-validation loop
        for train_indices, test_indices in cv.split(df_x, df_y):
            x_train, x_test = df_x.iloc[train_indices], df_x.iloc[test_indices]
            y_train, y_test = df_y.iloc[train_indices], df_y.iloc[test_indices]

            model.fit(x_train, y_train, verbose=0)

            pred_test = model.predict(x_test)
            cv_scores.append(r2_score(y_test, pred_test))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", study_name="XGB_Regressor")
    func = lambda trial: objective(trial, df_x_scaled, df_y)
    study.optimize(func, n_trials=100)

    print(f"\tBest value (r2 score): {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")

    return study.best_params
