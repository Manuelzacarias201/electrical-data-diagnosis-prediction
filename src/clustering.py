import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

from src.preprocessing import get_inverter_cols

def extract_inverter_features(df_daylight):
    """
    Extracts operational features for each of the 24 inverters
    from the daylight dataset.
    """
    print("Extracting features per inverter for clustering...")
    inverter_features = []
    
    for i in range(1, 25):
        cols = get_inverter_cols(df_daylight, i)
        
        dc_i = df_daylight[cols['dc_current']]
        ac_p = df_daylight[cols['ac_power']]
        eff = df_daylight[f"inv_{i:02d}_efficiency"]
        
        # Calculate features during daylight
        median_eff = float(eff.median())
        mean_eff = float(eff.mean())
        std_eff = float(eff.std())
        
        # Outage rate: panels generating but inverter outputting near zero
        # active daylight is when dc current > 1.0 A
        active_daylight = dc_i > 1.0
        outage_count = ((active_daylight) & (ac_p < 0.1)).sum()
        total_active = active_daylight.sum()
        outage_rate = float(outage_count / total_active) if total_active > 0 else 0.0
        
        # Max AC Power
        max_ac_power = float(ac_p.max())
        
        inverter_features.append({
            'inverter_id': i,
            'median_efficiency': median_eff,
            'mean_efficiency': mean_eff,
            'std_efficiency': std_eff,
            'outage_rate': outage_rate,
            'max_ac_power': max_ac_power
        })
        
    df_inv = pd.DataFrame(inverter_features)
    print(f"Extracted features for {df_inv.shape[0]} inverters.")
    return df_inv

def run_kmeans_analysis(df_inv, max_k=6):
    """
    Performs K-Means clustering for K from 2 to max_k.
    Calculates Inertia and Silhouette scores to find the optimal K.
    Scales features internally.
    """
    print("Running K-Means parameter search (Elbow & Silhouette)...")
    
    # Selected features for clustering
    features = ['median_efficiency', 'std_efficiency', 'outage_rate']
    X = df_inv[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {
        'k_values': [],
        'inertia': [],
        'silhouette': []
    }
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        results['k_values'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
        
    return results, X_scaled

def perform_final_clustering(df_inv, X_scaled, n_clusters=3):
    """
    Fits K-Means with the chosen number of clusters and adds labels to df_inv.
    Sorts and names clusters based on performance health:
    e.g. Cluster 0: Critical, Cluster 1: Sub-optimal, Cluster 2: Optimal.
    """
    print(f"Fitting final K-Means model with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_result = df_inv.copy()
    df_result['cluster'] = labels
    
    # Map clusters to meaningful names (Optimal, Sub-optimal, Critical)
    # We will compute the mean efficiency of each cluster to order them
    cluster_means = df_result.groupby('cluster')['median_efficiency'].mean().sort_values()
    
    # Mapping based on sorted order: lowest efficiency -> Critical, highest -> Optimal
    mapping = {}
    if n_clusters == 3:
        mapping[cluster_means.index[0]] = "Critical (Sensor/Falla)"
        mapping[cluster_means.index[1]] = "Sub-óptimo (Pérdidas/Estrés)"
        mapping[cluster_means.index[2]] = "Óptimo (Saludable)"
    else:
        # Fallback for other K values
        for rank, c_id in enumerate(cluster_means.index):
            mapping[c_id] = f"Nivel {rank}"
            
    df_result['health_status'] = df_result['cluster'].map(mapping)
    
    # Save the table
    base_dir = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
    tables_dir = os.path.join(base_dir, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df_result.to_csv(os.path.join(tables_dir, "inverter_health_clustering.csv"), index=False)
    print(f"Saved clustering results to {os.path.join(tables_dir, 'inverter_health_clustering.csv')}")
    
    return df_result
