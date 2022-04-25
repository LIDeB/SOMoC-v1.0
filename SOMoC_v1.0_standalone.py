#!/usr/bin/env python
# coding: utf-8
"""
@author: LIDeB UNLP
"""
# SOMoC is a clustering methodology based on the combination of molecular fingerprinting, 
# dimensionality reduction by the Uniform Manifold Approximation and Projection (UMAP) algorithm 
# and clustering with the Gaussian Mixture Model (GMM) algorithm.

##################################### Import packages ####################################
###########################################################################################
# The following packages are required: SKlearn, RDKit, UMAP, Molvs, validclust and Plotly.
# Please, meake sure you have them installed before running the program.

import pandas as pd
from array import array
import time
import os
from datetime import date
from pathlib import Path
import numpy as np
import random
import plotly.express as plx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from validclust import dunn
import umap
from rdkit import Chem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from molvs import Standardizer


###################################### CONFIGURATION ######################################
###########################################################################################

# Folder with files
# Input file is a .CSV file with one molecule per line in SMILES format.
# Molecules must be in the first column.
input_file = None

test_file = "test/focal_adhesion.csv"

# If you already know the number of clusters in your data, then set K to this value.
# Otherwise, set K=None to let SOMoC approaximate it by running a range of K values.
# Alternatively, you can use the generated elbowplot to find K yourself and re-reun SOMoC with a fixed K.
K = None            # Optimal number of clusters K

# Perform molecule standardization using the MolVS package
smiles_standardization = True       

### UMAP parameters ###
n_neighbors = 40    # The size of local neighborhood used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
min_dist = 0.0      # The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points.
random_state = 10   # Use a fixed seed for reproducibility.
metric = "jaccard"  # The metric to use to compute distances in high dimensional space.

### GMM parameters ###
max_K = 25                      # Max number of clusters to cosidering during the GMM loop
Kers = np.arange(2, max_K+1, 1) # Generate the range of K values to explore
iterations = 10                 # Iterations of GMM to run for each K
n_init = 5                      # Number of initializations on each GMM run, then just keep the best one.
init_params = 'kmeans'          # How to initialize. Can be random or K-means
covariance_type = 'tied'        # Type of covariance to consider: "spherical", "diag", "tied", "full"
warm_start = False


#################################### Helper functions #####################################
###########################################################################################

def Get_name(archive):
    """strip path and extension to return the name of a file"""
    return os.path.basename(archive).split('.')[0]


def Make_dir(dirName: str):
    """Create a directory and not fail if it already exist"""
    try:
        os.makedirs(dirName)
    except FileExistsError:
        pass


def Get_input_data():
    """Get data from user input or use test dataset"""

    if input_file is not None:
        name = Get_name(input_file)
        data = pd.read_csv(input_file, delimiter=',', header=None)
    else:
        name = Get_name(test_file)
        data = pd.read_csv(test_file, delimiter=',', header=None)

    return data, name


def Standardize_molecules(data):
    """Standardize molecules using the MolVS package https://molvs.readthedocs.io/en/latest/"""
    print('='*50)
    print("Standardize molecules")

    print('By default SOMoC will standardize molecules before fingerprint calculation. However, you can disable standardization by setting smiles_standardization=False.')
    time_start = time.time()
    data_ = data.copy()

    molec_clean = []
    s = Standardizer()

    list_of_smiles = data_.iloc[:,0]
    # list_of_smiles = data['SMILES']

    for i, molecule in enumerate(list_of_smiles, start = 1):
        try:
            mol = Chem.MolFromSmiles(molecule)
            estandardized = s.super_parent(mol)
            molec_clean.append(estandardized)
        except:
            print(f'Something went wrong with molecule number {i}')

    data_['mol'] = molec_clean
    print(f'{len(molec_clean)} molecules processed')
    print(f'Standardization took {round(time.time()-time_start)} seconds')

    return data_


def Fingerprints_calculator(data):
    """Calculate EState molecular fingerprints using the RDKit package"""
    print('='*50)
    print("Encoding")
    time_start = time.time()
    data_ = data.copy()
    if 'mol' not in data_:  # Check if already converted
        data_['mol'] = data_['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    try:
        _EState = [FingerprintMol(x)[0]
                   for x in data_['mol']]  # [0]EState1 [1]EState2
        EState = np.stack(_EState, axis=0)
    except:
        print("Oh no! There was a problem with Fingerprint calculation of some smiles")
        print("Try using our standarization tool before clustering")
        print("LIDeB Standarization tool: https://share.streamlit.io/capigol/lbb-game/main/juego_lbb.py")

    print("Calculating EState molecular fingerprints...")
    print(
        f'Fingerprints calculation took {round(time.time()-time_start)} seconds')
    print('='*50)
    return EState  # X data, fingerprints values as a np array


def UMAP_reduction(X):
    """Reduce feature space using the UMAP algorithm"""
    print("Reducing")
    print('Reducing feature space with UMAP...')

    time_start = time.time()

    if n_neighbors >= len(X):
        print('The number of neighbors must be smaller than the number of molecules to cluster')

    # Set a lower bound to the number of components
    n_components = max(int(len(X)*0.01), 3)

    UMAP_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                             random_state=random_state).fit(X)
    embedding = UMAP_reducer.transform(X)

    print(f'{embedding.shape[1]} features have been retained.')
    print(f'UMAP took {round(time.time()-time_start)} seconds')
    print('='*50)
    return embedding, n_components


def GMM_clustering_loop(embeddings):
    """Run GMM clustering for a range of K values, to get the K which maximizes the SIL score"""
    print("Clustering")
    print(f'Running GMM clustering for {max_K} iterations..')
    time_start = time.time()
    temp = []
    for n in Kers:
        temp_sil = []
        for x in range(iterations):
            gmm = GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                      warm_start=warm_start, random_state=x, verbose=0).fit(embeddings)
            labels = gmm.predict(embeddings)
            sil = round(silhouette_score(
                embeddings, labels, metric='cosine'), 4)
            temp_sil.append(sil)
        temp.append([n, np.mean(temp_sil), np.std(temp_sil)])

    results = pd.DataFrame(
        temp, columns=['Clusters', 'Silhouette', 'sil_stdv'])
    results_sorted = results.sort_values(['Silhouette'], ascending=False)
    K_loop = results_sorted.iloc[0]['Clusters']  # Get max Sil K
    print(f'GMM clustering loop took {round(time.time()-time_start)} seconds')
    print(' '*100)
    return results, int(K_loop)


def GMM_clustering_final(embeddings, K):
    """Cluster the dataset using the optimal K value, and calculate several CVIs"""
    print('='*50)
    print("Final clustering")
    print(f'Running GMM with K = {K}')
    time_start = time.time()

    GMM_final = GMM(K, n_init=n_init, init_params=init_params, warm_start=warm_start,
                    covariance_type=covariance_type, random_state=random_state, verbose=0)
    GMM_final.fit(embeddings)
    labels_final = GMM_final.predict(embeddings)

    if GMM_final.converged_:
        print('GMM converged.')
    else:
        print('GMM did not converge. Please check you input configuration.')

    sil_ok = round(float(silhouette_score(
        embeddings, labels_final, metric='cosine')), 4)
    db_score = round(davies_bouldin_score(embeddings, labels_final), 4)
    ch_score = round(calinski_harabasz_score(embeddings, labels_final), 4)
    dist_dunn = pairwise_distances(embeddings)
    dunn_score = round(float(dunn(dist_dunn, labels_final)), 4)

    valid_metrics = [sil_ok, db_score, ch_score, dunn_score]

    sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st = Cluster_random(
        embeddings)

    random_means = [sil_random, db_random, ch_random, dunn_random]
    random_sds = [sil_random_st, db_random_st, ch_random_st, dunn_random_st]

    table_metrics = pd.DataFrame([valid_metrics, random_means, random_sds]).T
    table_metrics = table_metrics.rename(index={0: 'Silhouette score', 1: "Davies Bouldin score",
                                         2: 'Calinski Harabasz score', 3: 'Dunn Index'}, columns={0: "Value", 1: "Mean Random", 2: "SD Random"})

    print(f'GMM clustering took {round(time.time()-time_start)} seconds')
    print('='*50)
    print("Validation metrics")
    print(table_metrics)

    cluster_final = pd.DataFrame({'cluster': labels_final}, index=data.index)
    data_clustered = data.join(cluster_final)

    if 'mol' in data_clustered.columns:  # Check if mol column from standardization is present
        try:
            data_clustered['SMILES_standardized'] = data_clustered['mol'].apply(
                lambda x: Chem.MolToSmiles(x))
            data_clustered.drop(['mol'], axis=1, inplace=True)
        except:
            print('Something went wrong converting standardized molecules back to SMILES code..')

    data_clustered.to_csv(f'results_SOMoC_{name}/{name}_Clustered_SOMoC.csv', index=True, header=True)
    table_metrics.to_csv(f'results_SOMoC_{name}/{name}_Validation_SOMoC.csv', index=True, header=True)

    return data_clustered


def Elbow_plot(results):
    """Draw the elbow plot of SIL score vs. K"""

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(x=results['Clusters'], y=results['Silhouette'],
                             mode='lines+markers', name='Silhouette',
                             error_y=dict(type='data', symmetric=True, array=results["sil_stdv"]),
                             hovertemplate="Clusters = %{x}<br>Silhouette = %{y}"),
                  secondary_y=False)

    fig.update_layout(title="Silhouette vs. K", title_x=0.5,
                      title_font=dict(size=28, family='Calibri', color='black'),
                      legend_title_text = "Metric",
                      legend_title_font = dict(size=18, family='Calibri', color='black'),
                      legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='Number of clusters (K)', range=[2 - 0.5, max_K + 0.5],
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=25, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Silhouette score',
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=25, family='Calibri', color='black'), secondary_y=False)

    fig.update_layout(margin=dict(t=60, r=20, b=20, l=20), autosize=True)

    fig.write_html(f'results_SOMoC_{name}/{name}_elbowplot_SOMoC.html')

    print('By dafault SOMoC uses the K which resulted in the highest Silhouette score.')
    print('However, you can check the Silhouette vs. K elbow plot to choose the optimal K, identifying an inflection point in the curve (elbow method)')
    print('Then, re-run SOMoC with a fixed K.')
    print("Note: Silhouette score is bounded [-1,1], the closer to 1 the better")


def Distribution_plot(data_clustered):
    """Plot the clusters size distribution"""
    sizes = data_clustered["cluster"].value_counts().to_frame()
    sizes.index.names = ['Cluster']
    sizes.columns = ['Size']
    sizes.reset_index(drop=False, inplace=True)
    sizes = sizes.astype({'Cluster': str, 'Size': int})

    fig = plx.bar(sizes, x=sizes.Cluster, y=sizes.Size, color=sizes.Cluster)

    fig.update_layout(legend_title="Cluster", plot_bgcolor='rgb(256,256,256)',
                      legend_title_font = dict(size=18, family='Calibri', color='black'),
                      legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='Cluster', showline=True, linecolor='black',
                     gridcolor='lightgrey', zerolinecolor='lightgrey',
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=20, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Size', showline=True, linecolor='black',
                     gridcolor='lightgrey', zerolinecolor='lightgrey',
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font=dict(size=20, family='Calibri', color='black'))

    fig.write_html(f'results_SOMoC_{name}/{name}_size_distribution_SOMoC.html')

    sizes.to_csv(f'results_SOMoC_{name}/{name}_Size_distribution_SOMoC.csv', index=True, header=True)  # Write the .CSV file

    return


def Cluster_random(embeddings: array):
    """Perform random clustering and calculate several CVIs"""
    SILs = []
    DBs = []
    CHs = []
    DUNNs = []

    for i in range(500):
        random.seed(a=i, version=2)
        random_clusters = []
        for x in list(range(len(embeddings))):
            random_clusters.append(random.randint(0, K-1))
        silhouette_random = silhouette_score(embeddings, np.ravel(random_clusters))
        SILs.append(silhouette_random)
        db_random = davies_bouldin_score(embeddings, np.ravel(random_clusters))
        DBs.append(db_random)
        ch_random = calinski_harabasz_score(embeddings, np.ravel(random_clusters))
        CHs.append(ch_random)
        dist_dunn = pairwise_distances(embeddings)
        dunn_random = dunn(dist_dunn, np.ravel(random_clusters))
        DUNNs.append(dunn_random)

    sil_random = round(float(np.mean(SILs)), 4)
    sil_random_st = round(np.std(SILs), 4)
    db_random = round(np.mean(DBs), 4)
    db_random_st = round(np.std(DBs), 4)
    ch_random = round(np.mean(CHs), 4)
    ch_random_st = round(np.std(CHs), 4)
    dunn_random = round(float(np.mean(DUNNs)), 4)
    dunn_random_st = round(np.std(DUNNs), 4)

    return sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st


def Setting_info():
    """Create a dataframe with current run setting"""
    today = date.today()
    fecha = today.strftime("%d/%m/%Y")
    settings = []
    settings.append(["Date: ", fecha])
    settings.append(["Setings:", ""])
    settings.append(["", ""])
    settings.append(["Fingerprint type:", "EState1"])
    settings.append(["", ""])
    settings.append(["UMAP", ""])
    settings.append(["n_neighbors:", str(n_neighbors)])
    settings.append(["min_dist:", str(min_dist)])
    settings.append(["n_components:", str(n_components)])
    settings.append(["random_state:", str(random_state)])
    settings.append(["metric:", str(metric)])
    settings.append(["", ""])
    settings.append(["GMM", ""])
    settings.append(["max Nº of clusters (K):", str(max_K)])
    settings.append(["Optimal K:", str(K)])
    settings.append(["iterations:", str(iterations)])
    settings.append(["n_init:", str(n_init)])
    settings.append(["init_params", str(init_params)])
    settings.append(["covariance_type", str(covariance_type)])
    settings.append(["", ""])
    settings.append(["Total running time : ", total_time])
    settings.append(["", ""])
    settings_df = pd.DataFrame(settings)
    settings_df.to_csv(f'results_SOMoC_{name}/{name}_Settings_SOMoC.csv', index=True, header=False)
    return

####################################### SOMoC main ########################################
###########################################################################################


if __name__ == '__main__':

    start = time.time()

    print('-'*50)

    # Get input data
    data_raw, name = Get_input_data()

    # Create output dir
    Make_dir(f'results_SOMoC_{name}')

    # Standardize molecules
    if smiles_standardization == True:
        data = Standardize_molecules(data_raw)
    else:
        print('Skipping molecules standardization..\n')
        data = data_raw

    # Calculate Fingerprints
    X = Fingerprints_calculator(data)

    # Reduce feature space with UMAP
    embedding, n_components = UMAP_reduction(X)

    # If K is not set, run the GMM clustering loop to get K
    if K is None:
        results, K = GMM_clustering_loop(embedding)
        Elbow_plot(results)

    # Run the final clustering and calculate all CVIs
    data_clustered = GMM_clustering_final(embedding, K)

    # Generate distribution plot and .CSV file
    Distribution_plot(data_clustered)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

    # Write the settings file
    settings = Setting_info()

    print('='*50)
    print('ALL DONE !')
    print(f'SOMoC run took {total_time}')
    print('='*50)


    



