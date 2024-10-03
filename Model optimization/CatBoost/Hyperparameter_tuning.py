import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
            'depth': trial.suggest_int('depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=1),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 8),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0),
        }
        model = CatBoostRegressor(**params, verbose=0)
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = []
        # Cross-validation loop
        for train_indices, test_indices in cv.split(df_x, df_y):
            x_train, x_test = df_x.iloc[train_indices], df_x.iloc[test_indices]
            y_train, y_test = df_y.iloc[train_indices], df_y.iloc[test_indices]
            model.fit(x_train, y_train, silent=True)
            pred_test = model.predict(x_test)
            mse = mean_squared_error(y_test, pred_test)
            # cv_scores.append(mse)
            cv_scores.append(r2_score(y_test, pred_test))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", study_name="CatBoost_Regressor")
    func = lambda trial: objective(trial, df_x_scaled, df_y)
    study.optimize(func, n_trials=100)

    print(f"\tBest value (MSE): {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")

    return study.best_params

# best_params = optimization(df_x, df_y)
