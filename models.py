

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def all_features_no_seasonality(train_df, test_df):
    #Assigning features to the data frames
    X_train_df = train_df.drop(columns=['TREFMXAV_U', 'time'])
    y_train_df = train_df['TREFMXAV_U']
    X_test_df = test_df.drop(columns=['TREFMXAV_U', 'time'])
    y_test_df = test_df['TREFMXAV_U']

    # Splitting the train data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

    # Instantiate the Linear Regression model, Random Forest and XGBoost
    lr_model1 = LinearRegression()
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model1 = XGBRegressor(n_estimators=100, random_state=42)

    # Training the Models on training set
    lr_model1.fit(X_train, y_train)
    rf_model1.fit(X_train, y_train)
    xgb_model1.fit(X_train, y_train)

    # Predictions and Metrics on test_df
    r_squared_testdf_lr = lr_model1.score(X_test_df, y_test_df)
    predictions_test_df_lr = lr_model1.predict(X_test_df)
    mse_test_df_lr = mean_squared_error(y_test_df, predictions_test_df_lr)
    rmse_lr = np.sqrt(mse_test_df_lr)

    r_squared_testdf_rf = rf_model1.score(X_test_df, y_test_df)
    predictions_test_df_rf = rf_model1.predict(X_test_df)
    mse_test_df_rf = mean_squared_error(y_test_df, predictions_test_df_rf)
    rmse_rf = np.sqrt(mse_test_df_rf)

    r_squared_testdf_xgb = xgb_model1.score(X_test_df, y_test_df)
    predictions_test_df_xgb = xgb_model1.predict(X_test_df)
    mse_test_df_xgb = mean_squared_error(y_test_df, predictions_test_df_xgb)
    rmse_xgb = np.sqrt(mse_test_df_xgb)
    
    # Printing and returning metrics
    metrics = {
        'Linear Regression': {'R-squared': r_squared_testdf_lr, 'RMSE': rmse_lr},
        'Random Forest': {'R-squared': r_squared_testdf_rf, 'RMSE': rmse_rf},
        'XGBoost': {'R-squared': r_squared_testdf_xgb, 'RMSE': rmse_xgb}
    }

    # Checking the model with the highest R-squared and lowest RMSE, and giving the best model
    max_r_squared = max(r_squared_testdf_lr, r_squared_testdf_rf, r_squared_testdf_xgb)
    min_rmse = min(rmse_lr, rmse_rf, rmse_xgb)
    best_models = [model for model, metrics in metrics.items() if metrics['R-squared'] == max_r_squared and metrics['RMSE'] == min_rmse]

    print("Metrics:")
    for model, metric in metrics.items():
        print(f"{model}: R-squared = {metric['R-squared']}, RMSE = {metric['RMSE']}")
    
    print("\nBest Model(s):", best_models)

    return metrics

def all_features_with_seasonality(szn_train_df, szn_test_df):
    #Assigning features to the data frames
    X_train_df = szn_train_df.drop(columns=['TREFMXAV_U'])
    y_train_df = szn_train_df['TREFMXAV_U']
    X_test_df = szn_test_df.drop(columns=['TREFMXAV_U'])
    y_test_df = szn_test_df['TREFMXAV_U']

    # Splitting the train data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

    # Instantiate the Linear Regression model, Random Forest and XGBoost
    lr_model1 = LinearRegression()
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model1 = XGBRegressor(n_estimators=100, random_state=42)

    # Training the Models on training set
    lr_model1.fit(X_train, y_train)
    rf_model1.fit(X_train, y_train)
    xgb_model1.fit(X_train, y_train)

    # Feature Importance for Random Forest
    feature_importances_rf = rf_model1.feature_importances_
    columns_rf = X_train.columns
    feature_importance_df_rf = pd.DataFrame({'Feature': columns_rf, 'Importance': feature_importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)

    # Feature Importance for XGBoost
    feature_importances_xgb = xgb_model1.feature_importances_
    columns_xgb = X_train.columns
    feature_importance_df_xgb = pd.DataFrame({'Feature': columns_xgb, 'Importance': feature_importances_xgb})
    feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

     # Predictions and Metrics on test_df
    r_squared_testdf_lr = lr_model1.score(X_test_df, y_test_df)
    predictions_test_df_lr = lr_model1.predict(X_test_df)
    mse_test_df_lr = mean_squared_error(y_test_df, predictions_test_df_lr)
    rmse_lr = np.sqrt(mse_test_df_lr)

    r_squared_testdf_rf = rf_model1.score(X_test_df, y_test_df)
    predictions_test_df_rf = rf_model1.predict(X_test_df)
    mse_test_df_rf = mean_squared_error(y_test_df, predictions_test_df_rf)
    rmse_rf = np.sqrt(mse_test_df_rf)

    r_squared_testdf_xgb = xgb_model1.score(X_test_df, y_test_df)
    predictions_test_df_xgb = xgb_model1.predict(X_test_df)
    mse_test_df_xgb = mean_squared_error(y_test_df, predictions_test_df_xgb)
    rmse_xgb = np.sqrt(mse_test_df_xgb)
    
    # Printing and returning metrics
    metrics = {
        'Linear Regression': {'R-squared': r_squared_testdf_lr, 'RMSE': rmse_lr},
        'Random Forest': {'R-squared': r_squared_testdf_rf, 'RMSE': rmse_rf},
        'XGBoost': {'R-squared': r_squared_testdf_xgb, 'RMSE': rmse_xgb}
    }

    # Outputting the model with the highest R-squared and lowest RMSE
    max_r_squared = max(r_squared_testdf_lr, r_squared_testdf_rf, r_squared_testdf_xgb)
    min_rmse = min(rmse_lr, rmse_rf, rmse_xgb)
    best_models = [model for model, metrics in metrics.items() if metrics['R-squared'] == max_r_squared and metrics['RMSE'] == min_rmse]

    print("Metrics:")
    for model, metric in metrics.items():
        print(f"{model}: R-squared = {metric['R-squared']}, RMSE = {metric['RMSE']}")
    
    print("\nBest Model(s):", best_models)

    # Plotting feature importances for Random Forest and XGBoost
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(feature_importance_df_rf['Feature'], feature_importance_df_rf['Importance'], color='lightcoral')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from Random Forest')

    plt.subplot(1, 2, 2)
    plt.barh(feature_importance_df_xgb['Feature'], feature_importance_df_xgb['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from XGBoost')

    plt.tight_layout()
    plt.show()

    # Plotting actual versus predicted for each model

    #Linear Regression
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_lr, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted LR')
    plt.legend()

    # Plot for Random Forest
    
    plt.subplot(1, 3, 2)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_rf, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted RF')
    plt.legend()

    # Plot for XGBoost
    
    plt.subplot(1, 3, 3)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_xgb, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted XGB')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return metrics

def removing_prect_prsn(szn_train_df, szn_test_df): 
    #Assigning features to the data frames
    X_train_df = szn_train_df.drop(columns=['TREFMXAV_U', 'PRECT', "PRSN"])
    y_train_df = szn_train_df['TREFMXAV_U']
    X_test_df = szn_test_df.drop(columns=['TREFMXAV_U', 'PRECT', "PRSN"])
    y_test_df = szn_test_df['TREFMXAV_U']

    # Splitting the train data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

    # Instantiate the Linear Regression model, Random Forest and XGBoost
    lr_model1 = LinearRegression()
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model1 = XGBRegressor(n_estimators=100, random_state=42)

    #Training on the training set
    lr_model1.fit(X_train, y_train)
    rf_model1.fit(X_train, y_train)
    xgb_model1.fit(X_train, y_train)

    # Feature Importance for Random Forest
    feature_importances_rf = rf_model1.feature_importances_
    columns_rf = X_train.columns
    feature_importance_df_rf = pd.DataFrame({'Feature': columns_rf, 'Importance': feature_importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)

    # Feature Importance for XGBoost
    feature_importances_xgb = xgb_model1.feature_importances_
    columns_xgb = X_train.columns
    feature_importance_df_xgb = pd.DataFrame({'Feature': columns_xgb, 'Importance': feature_importances_xgb})
    feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

    # Predictions on test_df
    r_squared_testdf_lr = lr_model1.score(X_test_df, y_test_df)
    predictions_test_df_lr = lr_model1.predict(X_test_df)
    mse_test_df_lr = mean_squared_error(y_test_df, predictions_test_df_lr)
    rmse_lr = np.sqrt(mse_test_df_lr)

    r_squared_testdf_rf = rf_model1.score(X_test_df, y_test_df)
    predictions_test_df_rf = rf_model1.predict(X_test_df)
    mse_test_df_rf = mean_squared_error(y_test_df, predictions_test_df_rf)
    rmse_rf = np.sqrt(mse_test_df_rf)

    r_squared_testdf_xgb = xgb_model1.score(X_test_df, y_test_df)
    predictions_test_df_xgb = xgb_model1.predict(X_test_df)
    mse_test_df_xgb = mean_squared_error(y_test_df, predictions_test_df_xgb)
    rmse_xgb = np.sqrt(mse_test_df_xgb)
    
    # Printing and returning metrics
    metrics = {
        'Linear Regression': {'R-squared': r_squared_testdf_lr, 'RMSE': rmse_lr},
        'Random Forest': {'R-squared': r_squared_testdf_rf, 'RMSE': rmse_rf},
        'XGBoost': {'R-squared': r_squared_testdf_xgb, 'RMSE': rmse_xgb}
    }

    # Find the model with the highest R-squared and lowest RMSE
    max_r_squared = max(r_squared_testdf_lr, r_squared_testdf_rf, r_squared_testdf_xgb)
    min_rmse = min(rmse_lr, rmse_rf, rmse_xgb)
    best_models = [model for model, metrics in metrics.items() if metrics['R-squared'] == max_r_squared and metrics['RMSE'] == min_rmse]

    print("Metrics:")
    for model, metric in metrics.items():
        print(f"{model}: R-squared = {metric['R-squared']}, RMSE = {metric['RMSE']}")
    
    print("\nBest Model(s):", best_models)

    # Plotting feature importances for Random Forest and XGBoost
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(feature_importance_df_rf['Feature'], feature_importance_df_rf['Importance'], color='lightcoral')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from Random Forest')

    plt.subplot(1, 2, 2)
    plt.barh(feature_importance_df_xgb['Feature'], feature_importance_df_xgb['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from XGBoost')

    plt.tight_layout()
    plt.show()

    # Plotting actual versus predicted for each model
    #Linear Regression
    plt.figure(figsize=(18,6))
    plt.subplot(1, 3, 1)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_lr, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted LR')
    plt.legend()

    # Plot for Random Forest
    
    plt.subplot(1, 3, 2)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_rf, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted RF')
    plt.legend()

    # Plot for XGBoost
    
    plt.subplot(1, 3, 3)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_xgb, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted XGB')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return metrics

def removing_ubot_vbot(szn_train_df, szn_test_df):
    X_train_df = szn_train_df.drop(columns=['TREFMXAV_U', 'UBOT', "VBOT"])
    y_train_df = szn_train_df['TREFMXAV_U']
    X_test_df = szn_test_df.drop(columns=['TREFMXAV_U', 'UBOT', "VBOT"])
    y_test_df = szn_test_df['TREFMXAV_U']

    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

    lr_model1 = LinearRegression()
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model1 = XGBRegressor(n_estimators=100, random_state=42)

    lr_model1.fit(X_train, y_train)
    rf_model1.fit(X_train, y_train)
    xgb_model1.fit(X_train, y_train)

    # Feature Importance for Random Forest
    feature_importances_rf = rf_model1.feature_importances_
    columns_rf = X_train.columns
    feature_importance_df_rf = pd.DataFrame({'Feature': columns_rf, 'Importance': feature_importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)

    # Feature Importance for XGBoost
    feature_importances_xgb = xgb_model1.feature_importances_
    columns_xgb = X_train.columns
    feature_importance_df_xgb = pd.DataFrame({'Feature': columns_xgb, 'Importance': feature_importances_xgb})
    feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

    # Predictions on test_df
    r_squared_testdf_lr = lr_model1.score(X_test_df, y_test_df)
    predictions_test_df_lr = lr_model1.predict(X_test_df)
    mse_test_df_lr = mean_squared_error(y_test_df, predictions_test_df_lr)
    rmse_lr = np.sqrt(mse_test_df_lr)

    r_squared_testdf_rf = rf_model1.score(X_test_df, y_test_df)
    predictions_test_df_rf = rf_model1.predict(X_test_df)
    mse_test_df_rf = mean_squared_error(y_test_df, predictions_test_df_rf)
    rmse_rf = np.sqrt(mse_test_df_rf)

    r_squared_testdf_xgb = xgb_model1.score(X_test_df, y_test_df)
    predictions_test_df_xgb = xgb_model1.predict(X_test_df)
    mse_test_df_xgb = mean_squared_error(y_test_df, predictions_test_df_xgb)
    rmse_xgb = np.sqrt(mse_test_df_xgb)
    
    # Printing and returning metrics
    metrics = {
        'Linear Regression': {'R-squared': r_squared_testdf_lr, 'RMSE': rmse_lr},
        'Random Forest': {'R-squared': r_squared_testdf_rf, 'RMSE': rmse_rf},
        'XGBoost': {'R-squared': r_squared_testdf_xgb, 'RMSE': rmse_xgb}
    }

    # Find the model with the highest R-squared and lowest RMSE
    max_r_squared = max(r_squared_testdf_lr, r_squared_testdf_rf, r_squared_testdf_xgb)
    min_rmse = min(rmse_lr, rmse_rf, rmse_xgb)
    best_models = [model for model, metrics in metrics.items() if metrics['R-squared'] == max_r_squared and metrics['RMSE'] == min_rmse]

    print("Metrics:")
    for model, metric in metrics.items():
        print(f"{model}: R-squared = {metric['R-squared']}, RMSE = {metric['RMSE']}")
    
    print("\nBest Model(s):", best_models)

    # Plotting feature importances for Random Forest and XGBoost
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(feature_importance_df_rf['Feature'], feature_importance_df_rf['Importance'], color='lightcoral')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from Random Forest')

    plt.subplot(1, 2, 2)
    plt.barh(feature_importance_df_xgb['Feature'], feature_importance_df_xgb['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from XGBoost')

    plt.tight_layout()
    plt.show()

    # Plotting actual versus predicted for each model
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_lr, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted LR')
    plt.legend()

    # Plot for Random Forest
    
    plt.subplot(1, 3, 2)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_rf, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted RF')
    plt.legend()

    # Plot for XGBoost
    
    plt.subplot(1, 3, 3)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_xgb, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted XGB')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return metrics

def only_qbot_fsns(szn_train_df, szn_test_df):
    #Feature assignment
    X_train_df = szn_train_df.drop(columns=['TREFMXAV_U', 'UBOT', "VBOT", "PRECT", 'PRSN', 'season_enc', 'FSNS'])
    y_train_df = szn_train_df['TREFMXAV_U']
    X_test_df = szn_test_df.drop(columns=['TREFMXAV_U', 'UBOT', "VBOT", "PRECT", 'PRSN', 'season_enc', 'FSNS'])
    y_test_df = szn_test_df['TREFMXAV_U']

    #splitting
    X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

    #Instantiating
    lr_model1 = LinearRegression()
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model1 = XGBRegressor(n_estimators=100, random_state=42)

    #Training
    lr_model1.fit(X_train, y_train)
    rf_model1.fit(X_train, y_train)
    xgb_model1.fit(X_train, y_train)

    # Feature Importance for Random Forest
    feature_importances_rf = rf_model1.feature_importances_
    columns_rf = X_train.columns
    feature_importance_df_rf = pd.DataFrame({'Feature': columns_rf, 'Importance': feature_importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)

    # Feature Importance for XGBoost
    feature_importances_xgb = xgb_model1.feature_importances_
    columns_xgb = X_train.columns
    feature_importance_df_xgb = pd.DataFrame({'Feature': columns_xgb, 'Importance': feature_importances_xgb})
    feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

    # Predictions on test_df
    r_squared_testdf_lr = lr_model1.score(X_test_df, y_test_df)
    predictions_test_df_lr = lr_model1.predict(X_test_df)
    mse_test_df_lr = mean_squared_error(y_test_df, predictions_test_df_lr)
    rmse_lr = np.sqrt(mse_test_df_lr)

    r_squared_testdf_rf = rf_model1.score(X_test_df, y_test_df)
    predictions_test_df_rf = rf_model1.predict(X_test_df)
    mse_test_df_rf = mean_squared_error(y_test_df, predictions_test_df_rf)
    rmse_rf = np.sqrt(mse_test_df_rf)

    r_squared_testdf_xgb = xgb_model1.score(X_test_df, y_test_df)
    predictions_test_df_xgb = xgb_model1.predict(X_test_df)
    mse_test_df_xgb = mean_squared_error(y_test_df, predictions_test_df_xgb)
    rmse_xgb = np.sqrt(mse_test_df_xgb)
    
    # Printing and returning metrics
    metrics = {
        'Linear Regression': {'R-squared': r_squared_testdf_lr, 'RMSE': rmse_lr},
        'Random Forest': {'R-squared': r_squared_testdf_rf, 'RMSE': rmse_rf},
        'XGBoost': {'R-squared': r_squared_testdf_xgb, 'RMSE': rmse_xgb}
    }

    # Finding the model with the highest R-squared and lowest RMSE
    max_r_squared = max(r_squared_testdf_lr, r_squared_testdf_rf, r_squared_testdf_xgb)
    min_rmse = min(rmse_lr, rmse_rf, rmse_xgb)
    best_models = [model for model, metrics in metrics.items() if metrics['R-squared'] == max_r_squared and metrics['RMSE'] == min_rmse]

    print("Metrics:")
    for model, metric in metrics.items():
        print(f"{model}: R-squared = {metric['R-squared']}, RMSE = {metric['RMSE']}")
    
    print("\nBest Model(s):", best_models)

    # Plotting feature importances for Random Forest and XGBoost
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(feature_importance_df_rf['Feature'], feature_importance_df_rf['Importance'], color='lightcoral')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from Random Forest')

    plt.subplot(1, 2, 2)
    plt.barh(feature_importance_df_xgb['Feature'], feature_importance_df_xgb['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from XGBoost')

    plt.tight_layout()
    plt.show()

    # Plotting actual versus predicted for each model
    plt.figure(figsize=(18, 6))
    #Linear Regression
    plt.subplot(1, 3, 1)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_lr, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted LR')
    plt.legend()

    # Plot for Random Forest
    plt.subplot(1, 3, 2)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_rf, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted RF')
    plt.legend()

    # Plot for XGBoost
    plt.subplot(1, 3, 3)
    plt.plot(y_test_df.values, label='Actual Temperature', color='red')
    plt.plot(predictions_test_df_xgb, label='Predicted Temperature', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Actual vs. Predicted XGB')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return metrics

