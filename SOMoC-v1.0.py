#!/usr/bin/env python
# coding: utf-8
"""
@author Manu Llanos   
"""

# Import packages

import pandas as pd
from array import array
import time
import os
from datetime import date
import numpy as np
import random
from PIL import Image
from io import StringIO, BytesIO
import base64

import streamlit as st

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

# Page layout
## Page expands to full width
st.set_page_config(page_title='LIDEB Tools - SOMoC', layout='wide')

image = Image.open('cropped-header.png')
st.image(image)

st.write("&nbsp[![Website](https://img.shields.io/badge/website-LIDeB-blue)](https://lideb.biol.unlp.edu.ar)&nbsp[![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)](https://twitter.com/intent/follow?screen_name=LIDeB_UNLP)")
st.subheader(":pushpin:" "About Us")
st.markdown("We are a drug discovery team with an interest in the development of publicly available open-source customizable cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. Our research group is focused on computer-guided drug repurposing and rational discovery of new drug candidates to treat epilepsy and neglected tropical diseases.")

# Introduction
#---------------------------------#

st.write("""
# LIDeB Tools - SOMoC (beta)
 SOMoC is a clustering methodology based on the combination of molecular fingerprinting, dimensionality reduction by the Uniform Manifold Approximation and Projection (UMAP) algorithm and clustering with the Gaussian Mixture Model (GMM) algorithm.
The next workflow summarizes the steps performed by SOMoC:
""")

image = Image.open('Workflow_SOMoC_streamlit2022.png')
st.image(image, caption='Clustering Workflow')

###################################### CONFIGURATION ######################################
###########################################################################################

## SIDEBAR

st.sidebar.header('Upload your dataset') # Loading file

input_file = st.sidebar.file_uploader("Upload a .CSV file with one molecule per line in SMILES format. Molecules must be in the first column.", type=["csv"])

st.sidebar.markdown("-------------------")

run = st.sidebar.button("RUN")
st.sidebar.info('Please upload your dataset or press RUN to cluster a example dataset.')

advanced_setting = st.sidebar.checkbox('Advanced settings', help='Check to change the default configuration')

if advanced_setting == True:

    # Input
    st.sidebar.header('Input options')    
    clean = st.sidebar.checkbox('Standardize molecules', value=True, help='Standardize input molecules using the [MolVS package](https://molvs.readthedocs.io/en/latest/)')

    # UMAP https://umap-learn.readthedocs.io/en/latest/index.html
    st.sidebar.header('UMAP')
    st.sidebar.caption(':point_right: [Read the docs](https://umap-learn.readthedocs.io/en/latest/index.html)')

    n_neighbors = st.sidebar.slider('Nº of neighbors', 2, 100, 40, 1, help='The size of local neighborhood used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.')
    min_dist = st.sidebar.slider('Min distance', 0.0, 0.95, 0.0, 0.05, help='The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points.')
    random_state = st.sidebar.slider('Random state', 0, 100, 10, 1, help='Use a fixed seed for reproducibility')
    metric = st.sidebar.selectbox("Metric", ("euclidean", "manhattan","canberra", "mahalanobis","cosine", "hamming","jaccard"),6, help='The metric to use to compute distances in high dimensional space.')

    # GMM https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    st.sidebar.header('GMM')
    st.sidebar.caption(':point_right: [Read the docs](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)')
    max_K = st.sidebar.slider('Max nº of clusters', 2, 200, 25, 1, help='Set the maximum number of clusters to explore.')
    Kers =  np.arange(2, max_K+1,1)
    iterations = st.sidebar.slider('Iterations', 2, 25, 10, 1, help='The number of GMM iterations to run for each K')
    n_init = st.sidebar.slider('Nº initializations', 2, 20, 5, 1, help='The number of initializations to perform. The best results are kept.') 
    init_params = st.sidebar.selectbox("Init params", ("kmeans", "random"),0, help='The method used to initialize the weights, the means and the precisions')
    covariance_type = st.sidebar.selectbox("Covariance type", ("full", "tied","diag","spherical"),1, help='Type of covariance matrix to calculate.')
    warm_start =  False

# Default configuration
else:   
    clean = True
    n_neighbors = 40
    min_dist = 0.0
    random_state = 10
    metric = "jaccard"
    max_K = 25    
    Kers= np.arange(2,max_K+1,1) # Range of K values to explore
    iterations= 10 # Iterations of GMM to run for each K
    n_init = 5 # Number of initializations on each GMM run, then just keep the best one.
    init_params = 'kmeans' # How to initialize. Can be random or K-means
    covariance_type = 'tied' # Tipe of covariance to consider
    warm_start =  False

ready = st.sidebar.checkbox('I got the Key', help='Select the optimal number of clusters (K)')

if ready == True:
    K = st.sidebar.number_input('Optimal Nº of clusters (K)', 2, max_K, 2, 1, key='K')
else:
    K = None

st.sidebar.markdown("-------------------")

#################################### Helper functions #####################################
###########################################################################################

def Get_name(archive: str):
    """strip path and extension to return the name of a file"""
    return os.path.basename(archive).split('.')[0]

def Get_input_data():
    """Get data from user input or use test dataset"""

    if input_file is not None:
        name = input_file.name
        data = pd.read_csv(input_file, delimiter=',', header=None)
    else:
        name = Get_name("test/focal_adhesion.csv")
        data = pd.read_csv("test/focal_adhesion.csv", delimiter=',', header=None)
    return data, name

def Download_CSV(df, name:str, filetype:str):
    """Convert dataframe to csv and get URL to download it"""
    csv = df.to_csv(index=True, header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}_{filetype}_SOMoC.csv">Download CSV file</a>'
    return href

def Download_HTML(fig, name:str, filetype:str):
    """Convert image and get URL to download it"""
    mybuff = StringIO()
    fig.write_html(mybuff, include_plotlyjs='cdn')
    mybuff = BytesIO(mybuff.getvalue().encode())
    b64 = base64.b64encode(mybuff.read()).decode()
    href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="{name}_{filetype}.html">Download {filetype} plot</a>'
    return href

def Standardize_molecules(data):
    """Standardize molecules using the MolVS package https://molvs.readthedocs.io/en/latest/"""
    st.markdown("-------------------")
    st.markdown("**Standardize molecules**")
    st.info('By default SOMoC will standardize molecules before fingerprint calculation. You can disable standardization in the Advanced settings menu.')
    
    time_start = time.time()
    data_ = data.copy()
    list_of_smiles = data_.iloc[:, 0]
    molec_clean=[]
    s = Standardizer() 
    i = 0
    t = st.empty()

    for molecule in list_of_smiles:
        t.markdown(f"Processing molecule {i+1} / {len(list_of_smiles)}")
        i = i+1
        try:
            mol = Chem.MolFromSmiles(molecule)
            # estandarizada = s.super_parent(mol)
            mol_sin_fragmento = s.fragment_parent(mol) #Return the fragment parent of a given molecule, the largest organic covalent unit in the molecule
            mol_sin_estereo = s.stereo_parent(mol_sin_fragmento, skip_standardize= True) #Return The stereo parentof a given molecule, has all stereochemistry information removed from tetrahedral centers and double bonds.
            mol_sin_carga = s.charge_parent(mol_sin_estereo, skip_standardize= True) #Return the charge parent of a given molecule,  the uncharged version of the fragment parent
            estandarizada = s.isotope_parent(mol_sin_carga, skip_standardize= True) #Return the isotope parent of a given molecule, has all atoms replaced with the most abundant isotope for that element.
            molec_clean.append(estandarizada)
        except:
            st.write(f'Something went wrong with molecule number {i}')

    data_['mol'] = molec_clean
    
    st.write(f'Standardization took {round(time.time()-time_start)} seconds')
    st.markdown("-------------------")
    
    return data_

def Fingerprints_calculator(data):
    """Calculate EState molecular fingerprints using the RDKit package"""

    st.markdown("**Encoding**")
    time_start = time.time()
    data_ = data.copy()
    if 'mol' not in data_: # Check if already converted
        data_['mol'] = data_[0].apply(lambda x: Chem.MolFromSmiles(x))

    try:
        _EState = [FingerprintMol(x)[0] for x in data_['mol']] #[0]EState1 [1]EState2
        EState = np.stack(_EState, axis=0)
    except:
        st.error("**Oh no! There was a problem with Fingerprint calculation of some smiles.**  :confused:")
        st.markdown(" :point_down: **Try using our standarization tool before clustering **")
        st.write("[LIDeB Standarization tool](https://share.streamlit.io/capigol/lbb-game/main/juego_lbb.py)")
        st.stop()

    st.write("Calculating EState molecular fingerprints...")
    st.write(f'Fingerprints calculation took {round(time.time()-time_start)} seconds')

    return EState # X data, fingerprints values as a np array

def UMAP_reduction(X: array):
    """Reduce feature space using the UMAP algorithm"""
    st.markdown("**Reducing**")
    st.write('Reducing feature space with UMAP...')
    
    time_start = time.time()

    if n_neighbors >= len(X):
        print(f'The number of neighbors must be smaller than the number of molecules to cluster')

    n_components = max(int(len(X)*0.01),3) # Set a lower bound to the number of components

    UMAP_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                                random_state=random_state).fit(X)
    embedding = UMAP_reducer.transform(X)
    
    st.write(f'{embedding.shape[1]} features have been retained.')
    st.write(f'UMAP took {round(time.time()-time_start)} seconds')

    return embedding, n_components

def GMM_clustering_loop(embeddings: array):
    """Run GMM clustering for a range of K values, to get the K which maximizes the SIL score"""
    st.markdown("**Clustering loop**")
    st.write(f'Running GMM clustering for {max_K} iterations...')
    time_start = time.time()
    temp = []
    for n in Kers:
        temp_sil = []
        for x in range(iterations):
            gmm=GMM(n, n_init=n_init, init_params=init_params, covariance_type=covariance_type,
                        warm_start=warm_start, random_state=x, verbose=0).fit(embeddings) 
            labels=gmm.predict(embeddings)
            sil = round(silhouette_score(embeddings, labels, metric='cosine'),4)
            temp_sil.append(sil)  
        temp.append([n, np.mean(temp_sil), np.std(temp_sil)])

    results = pd.DataFrame(temp, columns=['Clusters','Silhouette','sil_stdv'])
    results_sorted = results.sort_values(['Silhouette'], ascending=False)
    K_loop = results_sorted.iloc[0]['Clusters'] # Get max Sil K

    st.write(f'GMM clustering loop took {round(time.time()-time_start)} seconds')

    return results, int(K_loop)

def GMM_clustering_final(embeddings: array, K:int):
    """Cluster the dataset using the optimal K value, and calculate several CVIs"""
    st.markdown("**Final clustering**")
    st.write(f'Running GMM with K = {K}')
    time_start = time.time()

    GMM_final = GMM(K, n_init=n_init, init_params=init_params, warm_start=warm_start, covariance_type=covariance_type, random_state=random_state, verbose=0)
    GMM_final.fit(embeddings) 
    labels_final = GMM_final.predict(embeddings)
    
    if GMM_final.converged_:
        st.write(f'GMM converged.')
    else:
        st.write(f'GMM did not converge. Please check you input configuration.')

    sil_ok = round(float(silhouette_score(embeddings, labels_final, metric='cosine')),4)
    db_score = round(davies_bouldin_score(embeddings, labels_final),4)
    ch_score = round(calinski_harabasz_score(embeddings, labels_final),4)
    dist_dunn = pairwise_distances(embeddings)
    dunn_score = round(float(dunn(dist_dunn, labels_final)),4)
    
    valid_metrics = [sil_ok, db_score, ch_score, dunn_score]

    sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st = Cluster_random(embeddings)
    
    random_means = [sil_random,db_random,ch_random,dunn_random]
    random_sds = [sil_random_st,db_random_st,ch_random_st,dunn_random_st]

    table_metrics = pd.DataFrame([valid_metrics,random_means,random_sds]).T
    table_metrics=table_metrics.rename(index={0: 'Silhouette score', 1:"Davies Bouldin score", 2: 'Calinski Harabasz score', 3:'Dunn Index'},columns={0:"Value",1:"Mean Random",2:"SD Random"})
    
    st.write(f'GMM clustering took {round(time.time()-time_start)} seconds')
    st.markdown("**Validation metrics**")
    st.write(table_metrics)

    cluster_final = pd.DataFrame({'cluster': labels_final}, index=data.index)
    data_clustered = data.join(cluster_final)

    if 'mol' in data_clustered.columns: # Check if mol column from standardization is present
        try:
            data_clustered['SMILES_standardized'] = data_clustered['mol'].apply(lambda x: Chem.MolToSmiles(x))
            data_clustered.drop(['mol'], axis=1, inplace=True)
        except:
            print('Something went wrong converting standardized molecules back to SMILES code..')    
    
    return data_clustered, table_metrics

def Elbow_plot(results):
    """Draw the elbowplot of SIL score vs. K"""
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(go.Scatter(x=results['Clusters'], y=results['Silhouette'], 
                            mode='lines+markers', name= 'Silhouette', 
                            error_y = dict( type='data', symmetric=True, array= results["sil_stdv"]),
                            hovertemplate = "Clusters = %{x}<br>Silhouette = %{y}"), 
                            secondary_y=False)
    
    fig.update_layout(title = "Silhouette vs. K", title_x=0.5,
                  title_font = dict(size=28, family='Calibri', color='black'),
                  legend_title_text = "Metric", 
                  legend_title_font = dict(size=18, family='Calibri', color='black'),
                  legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='Number of clusters (K)', range = [2 - 0.5, max_K + 0.5],
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font = dict(size=25, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Silhouette score', 
                     tickfont=dict(family='Arial', size=16, color='black'),
                     title_font = dict(size=25, family='Calibri', color='black'), secondary_y=False)

    fig.update_layout(margin = dict(t=60,r=20,b=20,l=20), autosize = True)
    
    st.info('By dafault SOMoC will cluster your dataset using the K which resulted in the highest Silhouette score. However, you can check the Silhouette vs. K elbow plot to choose the optimal K, identifying an inflection point in the curve (elbow method). Then, re-run SOMoC with a fixed K.')
    st.plotly_chart(fig)
    st.write("Note: The Silhouette score is bounded [-1,1], the closer to one the better")

    return fig

def Distribution_plot(data_clustered):
    """Plot the clusters size distribution"""
    sizes = data_clustered["cluster"].value_counts().to_frame()
    sizes.index.names = ['Cluster']
    sizes.columns = ['Size']
    sizes.reset_index(drop=False, inplace=True)
    sizes = sizes.astype({'Cluster':str, 'Size':int})

    fig = plx.bar(sizes, x = sizes.Cluster, y = sizes.Size, color = sizes.Cluster)

    fig.update_layout(legend_title="Cluster", plot_bgcolor = 'rgb(256,256,256)',
                        legend_title_font = dict(size=18, family='Calibri', color='black'),
                        legend_font = dict(size=15, family='Calibri', color='black'))
    fig.update_xaxes(title_text='Cluster', showline=True, linecolor='black', 
                        gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                        tickfont=dict(family='Arial', size=16, color='black'),
                        title_font = dict(size=20, family='Calibri', color='black'))
    fig.update_yaxes(title_text='Size', showline=True, linecolor='black', 
                        gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                        tickfont=dict(family='Arial', size=16, color='black'),
                        title_font = dict(size=20, family='Calibri', color='black'))
    
    st.markdown("**Clusters size distribution**")

    st.plotly_chart(fig)

    return sizes, fig


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
            random_clusters.append(random.randint(0,K-1))
        silhouette_random = silhouette_score(embeddings, np.ravel(random_clusters))
        SILs.append(silhouette_random)
        db_random = davies_bouldin_score(embeddings, np.ravel(random_clusters))
        DBs.append(db_random)
        ch_random = calinski_harabasz_score(embeddings, np.ravel(random_clusters))
        CHs.append(ch_random)
        dist_dunn = pairwise_distances(embeddings)
        dunn_random = dunn(dist_dunn, np.ravel(random_clusters))
        DUNNs.append(dunn_random)

    sil_random = round(float(np.mean(SILs)),4)
    sil_random_st = round(np.std(SILs),4)
    db_random = round(np.mean(DBs),4)
    db_random_st = round(np.std(DBs),4)
    ch_random = round(np.mean(CHs),4)
    ch_random_st = round(np.std(CHs),4)
    dunn_random = round(float(np.mean(DUNNs)),4)
    dunn_random_st = round(np.std(DUNNs),4)

    return sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st

def Setting_info():
    """Create a dataframe with current run setting"""
    today = date.today()
    fecha = today.strftime("%d/%m/%Y")
    settings = []
    settings.append(["Date: " , fecha])
    settings.append(["Setings:",""])
    settings.append(["",""])   
    settings.append(["Fingerprint type:","EState1"])    
    settings.append(["",""])   
    settings.append(["UMAP",""])        
    settings.append(["n_neighbors:", str(n_neighbors)])
    settings.append(["min_dist:", str(min_dist)])
    settings.append(["n_components:", str(n_components)])
    settings.append(["random_state:", str(random_state)])
    settings.append(["metric:", str(metric)])
    settings.append(["",""])       
    settings.append(["GMM",""])        
    settings.append(["max Nº of clusters (K):", str(max_K)])
    settings.append(["Optimal K:", str(K)])
    settings.append(["iterations:", str(iterations)])
    settings.append(["n_init:", str(n_init)])
    settings.append(["init_params",str(init_params)])
    settings.append(["covariance_type",str(covariance_type)])       
    settings.append(["",""])           
    settings_df = pd.DataFrame(settings)
    
    return settings_df

####################################### SOMoC main ########################################
###########################################################################################

if run == True:

    # Get input data
    data_raw, name = Get_input_data()
    
    # Standardize molecules
    if clean == True:
        data = Standardize_molecules(data_raw)
    else:
        data = data_raw

    # Calculate Fingerprints
    X = Fingerprints_calculator(data)
    
    st.markdown("-------------------")

    # Reduce feature space with UMAP
    embedding, n_components = UMAP_reduction(X)
    
    st.markdown("-------------------")

    # Cluster with GMM
    if K is None: # Run the loop to get optimal K
        results, K = GMM_clustering_loop(embedding)
        elbowplot = Elbow_plot(results)
        # st.markdown(":point_down: **Elbowplot**")
        st.markdown(Download_HTML(elbowplot, name, 'Elbowplot'), unsafe_allow_html=True)
    
    st.markdown("-------------------")

    data_clustered, validation_metrics = GMM_clustering_final(embedding, K)
    cluster_distrib, distribplot = Distribution_plot(data_clustered)
    st.markdown(Download_HTML(distribplot, name, 'Size_Distribution'), unsafe_allow_html=True)

    st.markdown("-------------------")
    st.subheader('Download results')

    st.markdown(":point_down: **Clustered dataset**")
    st.markdown(Download_CSV(data_clustered, name, 'clustered'), unsafe_allow_html=True)

    st.markdown(":point_down: **Clustering validation**")
    st.markdown(Download_CSV(validation_metrics, name,'validation'), unsafe_allow_html=True)
    
    st.markdown(":point_down: **Clusters distribution**")
    st.markdown(Download_CSV(cluster_distrib, name,'Size_Distribution'), unsafe_allow_html=True)

    settings_df = Setting_info()
    st.markdown(":point_down: **Clustering run settings**")
    st.markdown(Download_CSV(settings_df, name,'settings'), unsafe_allow_html=True)

# Footer edit
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; ' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img>.  Developed with ❤️ by <a style='display: ; text-align: center' href="https://linkedin.com/in/manuel-llanos" target="_blank">Manu Llanos</a> for <a style='display:; text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p></div>
"""
st.markdown(footer, unsafe_allow_html=True)