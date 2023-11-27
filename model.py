import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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


def knn_cluster(X_train, X_test, y_train, y_test, top_features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    X_train_clustered = kmeans.fit_predict(X_train[top_features])
    X_test_clustered = kmeans.predict(X_test[top_features])
    return X_train_clustered, X_test_clustered

def regression_per_cluster(X_train, X_train_clustered, X_test, X_test_clustered, y_train, top_features):
    unique_clusters = set(X_train_clustered)
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
        lm.fit(X_train_cluster_scaled, y_train_cluster)
        cluster_predictions = lm.predict(X_test_cluster_scaled)
        predictions.extend(cluster_predictions)
    
    return predictions

def main():
    X_train, X_test, y_train, y_test = load_and_split_data('all_seasons.csv', 'pts')
    correlation_scores = calculate_pearson_correlation(X_train, y_train)
    print(correlation_scores)
    top_features = [x[0] for x in correlation_scores[:8]] 
    
    X_train_clustered, X_test_clustered = knn_cluster(X_train, X_test, y_train, y_test, top_features)
    print(X_train_clustered)
    print(X_test_clustered)
    
    predictions = regression_per_cluster(X_train, X_train_clustered, X_test, X_test_clustered, y_train, top_features)
    print(predictions)  
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error (MAE): {mae}")

if __name__ == "__main__":
    main()