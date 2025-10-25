# Part 2: Cluster Analysis
import os
os.environ['OMP_NUM_THREADS'] = '2' #this is added because of the memeory leak due to library issue.
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools



# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.

def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(columns=['Channel', 'Region'])  
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
	
def summary_statistics(df):
    # Compute summary statistics: mean, standard deviation, min, and max.
    # Round the mean and standard deviation to the nearest integer.
    means = df.mean().round(0).astype(int)
    stds  = df.std().round(0).astype(int)
    mins  = df.min()
    maxs  = df.max()
    
    # Construct a new DataFrame where each row corresponds to an attribute.
    summary_df = pd.DataFrame({
        'mean': means,
        'std': stds,
        'min': mins,
        'max': maxs
    })

    return summary_df

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    # Compute mean and standard deviation for each column
    means = df.mean()
    stds = df.std()

    # Replace zero standard deviation with 1 to avoid division errors
    stds = stds.replace(0, 1)

    # Standardize the dataset (subtract mean, divide by std)
    standardized_df = (df - means) / stds

    return standardized_df.copy()
# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.

def kmeans(df, k):
    # Ensure the input dataframe is already standardized before calling this function
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)

    # Return the cluster assignments as a pandas Series
    return pd.Series(kmeans.labels_, index=df.index)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
"""def kmeans_plus(df, k):
	pass"""

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.

def agglomerative(df, k):
    # Ensure input is already standardized before calling this function
    model = AgglomerativeClustering(n_clusters=k)
    
    # Fit and predict cluster labels
    labels = model.fit_predict(df)
    
    # Return the cluster assignments as a pandas Series
    return pd.Series(labels, index=df.index)
	
# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
    # Compute and return the Silhouette score for the given clustering.
    return silhouette_score(X, y)



# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    results = []
    
    # Assume df is already cleaned (Channel & Region removed)
    df_std = standardize(df)  # Standardized dataset

    for data, data_type in [(df, "Original"), (df_std, "Standardized")]:
        for k in [3, 5, 10]:  # Iterate over different cluster sizes
            # Repeat 10 executions for KMeans.
            for i in range(10):
                model = KMeans(n_clusters=k, n_init='auto', random_state=42 + i)
                labels = model.fit_predict(data)
                
                # Compute silhouette score only if k > 1
                if k > 1:
                    score = silhouette_score(data, labels)
                else:
                    score = None
                
                results.append({
                    'Algorithm': 'Kmeans',
                    'Data type': data_type,
                    'k': k,
                    'Silhouette Score': score
                })

            # One execution for Agglomerative Clustering.
            agg_model = AgglomerativeClustering(n_clusters=k)
            agg_labels = agg_model.fit_predict(data)

            if k > 1:
                agg_score = silhouette_score(data, agg_labels)
            else:
                agg_score = None
            
            results.append({
                'Algorithm': 'Agglomerative',
                'Data type': data_type,
                'k': k,
                'Silhouette Score': agg_score
            })
    
    return pd.DataFrame(results)


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    # Return the maximum Silhouette Score from the evaluation DataFrame.
    return rdf['Silhouette Score'].max()

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.




def scatter_plots(df):
    # Run KMeans with k=3 on the standardized data set.
    k = 3
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans_model.fit_predict(df)
    
   
    columns = df.columns.tolist()
    
    # Generate a scatter plot for each pair of attributes.
    for (col1, col2) in itertools.combinations(columns, 2):
        plt.figure()
        plt.scatter(df[col1], df[col2], c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter plot: {col1} vs {col2}')
        plt.show()

