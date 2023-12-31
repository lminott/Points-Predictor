# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Lemar Minott, William Chung 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


def load_and_split_data(file_path, target_column, test_size=0.2, random_state=42):
    data = pd.read_csv(file_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def calculate_pearson_correlation(X, y):
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    correlation_scores = []
    
    for column in numeric_columns:
        correlation = pearsonr(X[column], y)
        correlation_scores.append((column, correlation[0]))
    
    correlation_scores = sorted(correlation_scores, key=lambda x: abs(x[1]), reverse=True)
    return correlation_scores


def knn_cluster(X_train, X_test, y_train, y_test, top_features, n_clusters=15):
    kmeans = KMeans(n_clusters=n_clusters)
    X_train_clustered = kmeans.fit_predict(X_train[top_features])
    X_test_clustered = kmeans.predict(X_test[top_features])
    return X_train_clustered, X_test_clustered

def hyperparameter_tune_find_optimal_k(X_train, X_test, y_train, y_test, top_features, k_values):
    mae_values = [] 
    best_k = None
    best_mae = float('inf')  

    for k in k_values:
        X_train_clustered, X_test_clustered = knn_cluster(X_train, X_test, y_train, y_test, top_features, n_clusters=k)
        predictions = regression_per_cluster(X_train, X_train_clustered, X_test, X_test_clustered, y_train, top_features)
        mae = mean_absolute_error(y_test, predictions)
        mae_values.append(mae)  
        if mae < best_mae:
            best_mae = mae
            best_k = k

    print(f"Best k: {best_k} with Mean Absolute Error (MAE): {best_mae}")
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mae_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE vs. Number of Clusters (k)')
    plt.grid(True)
    plt.show()

def regression_per_cluster(X_train, X_train_clustered, X_test, X_test_clustered, y_train, top_features):
    unique_clusters = np.unique(X_test_clustered) 

    predictions = []

    for cluster in unique_clusters:
        indices_train = X_train_clustered == cluster
        indices_test = X_test_clustered == cluster

        X_train_cluster = X_train.loc[indices_train]
        X_test_cluster = X_test.loc[indices_test]
        y_train_cluster = y_train[indices_train]

        X_train_cluster = X_train_cluster[top_features]
        X_test_cluster = X_test_cluster[top_features]

        scaler = StandardScaler()
        X_train_cluster_scaled = scaler.fit_transform(X_train_cluster)
        X_test_cluster_scaled = scaler.transform(X_test_cluster)

        lm = LinearRegression()

        cluster_predictions = np.zeros(len(X_test_cluster))  

        kfold = KFold(n_splits=15, shuffle=True, random_state=42)

        for train_idx, val_idx in kfold.split(X_train_cluster_scaled):
            X_train_fold, X_val_fold = X_train_cluster_scaled[train_idx], X_train_cluster_scaled[val_idx]
            y_train_fold, y_val_fold = y_train_cluster.iloc[train_idx], y_train_cluster.iloc[val_idx]

            lm.fit(X_train_fold, y_train_fold)
            fold_predictions = lm.predict(X_test_cluster_scaled)
            cluster_predictions += fold_predictions
        cluster_predictions /= kfold.n_splits
        predictions.extend(cluster_predictions)

    return predictions


def main():
    X_train, X_test, y_train, y_test = load_and_split_data('all_seasons.csv', 'pts')
    correlation_scores = calculate_pearson_correlation(X_train, y_train)
    print(correlation_scores)
    top_features = [x[0] for x in correlation_scores[:12]] 
    
    X_train_clustered, X_test_clustered = knn_cluster(X_train, X_test, y_train, y_test, top_features, n_clusters= 10)
    print(X_train_clustered)
    print(X_test_clustered)
    
    predictions = regression_per_cluster(X_train, X_train_clustered, X_test,X_test_clustered, y_train, top_features)
    print(predictions)  
    k_values = [3, 5, 8, 10, 15] 
    hyperparameter_tune_find_optimal_k(X_train, X_test, y_train, y_test, top_features, k_values)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error (MAE): {mae}")
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error (MSE): {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    cv = np.std(y_test) / np.mean(y_test)
    print(f"Coefficient of Variation (CV): {cv}")
    r_squared = r2_score(y_test, predictions)
    print(f"R-squared: {r_squared}")


if __name__ == "__main__":
    main()
