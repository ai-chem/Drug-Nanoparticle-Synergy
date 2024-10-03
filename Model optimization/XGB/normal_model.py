import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def model_xgboost(df_x, df_y, best_params):

    # Conversion of categorical values into numerical
    oe = OrdinalEncoder()
    categorical_features = df_x.select_dtypes(include=['object']).columns
    df_x[categorical_features] = oe.fit_transform(df_x[categorical_features].astype(str))

    # Initialize lists to store results
    train_R2_metric_results = []
    train_mse_metric_results = []
    train_mae_metric_results = []
    test_R2_metric_results = []
    test_mse_metric_results = []
    test_mae_metric_results = []

    # KFold cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize lists to store predictions
    y_train_all = np.array([])
    y_train_preds_all = np.array([])
    y_test_all = np.array([])
    y_test_preds_all = np.array([])

    # Cross-validation loop
    for idx, (train_indices, test_indices) in enumerate(cv.split(df_x)):
        x_train, x_test = df_x.iloc[train_indices], df_x.iloc[test_indices]
        y_train, y_test = df_y.iloc[train_indices], df_y.iloc[test_indices]
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        model = XGBRegressor(**best_params)

        model.fit(x_train, y_train)
        train_preds = model.predict(x_train)
        test_preds = model.predict(x_test)

        train_R2_metric_results.append(r2_score(y_train, train_preds))
        train_mse_metric_results.append(mean_squared_error(y_train, train_preds))
        train_mae_metric_results.append(mean_absolute_error(y_train, train_preds))

        test_R2_metric_results.append(r2_score(y_test, test_preds))
        test_mse_metric_results.append(mean_squared_error(y_test, test_preds))
        test_mae_metric_results.append(mean_absolute_error(y_test, test_preds))

        # Collecting all predictions for scatter plot
        y_train_all = np.append(y_train_all, y_train)
        y_train_preds_all = np.append(y_train_preds_all, train_preds)
        y_test_all = np.append(y_test_all, y_test)
        y_test_preds_all = np.append(y_test_preds_all, test_preds)

    # Print training metrics
    print('Train dataset')
    print('Train R-square:', np.mean(train_R2_metric_results))
    print('Mean Absolute Error:', np.mean(train_mae_metric_results))
    print('Mean Squared Error:', np.mean(train_mse_metric_results))
    print('Root Mean Squared Error:', np.mean(train_mse_metric_results) ** 0.5)

    # Print validation metrics
    print('Test dataset')
    print('10-fold cross-validation R-square:', np.mean(test_R2_metric_results))
    print('10-fold cross-validation MAE:', np.mean(test_mae_metric_results))
    print('10-fold cross-validation MSE:', np.mean(test_mse_metric_results))
    print('10-fold cross-validation RMSE:', np.mean(test_mse_metric_results) ** 0.5)

    # For visualization
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    return x_train, y_train, model, y_test, test_preds, train_preds
