from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from flask import render_template
import sys
import scipy
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from decimal import Decimal
from sklearn import preprocessing

from sklearn import datasets
import math


application = Flask(__name__)

@application.route("/")
def d3():
    return render_template('index.html')

rs_data_frntend=[]
ss_data_frntend=[]
No_sampling_data=pd.read_csv(r"flaskDirectory\India_disabled_population.csv")
data_frame=pd.DataFrame(No_sampling_data)
data_frame.drop('Area_Name',axis = 1,inplace=True)
data_frame.drop('State_Code',axis = 1,inplace=True)
# data_frame.drop('Total_disabled_population_Persons',axis = 1,inplace=True)
# data_frame.drop('Main_worker_Persons',axis = 1,inplace=True)
# data_frame.drop('Marginal_worker_Less_than_3_months_Persons',axis = 1,inplace=True)
# data_frame.drop('Marginal_worker_3to6_months_Persons',axis = 1,inplace=True)
# data_frame.drop('Non_worker_Persons',axis = 1,inplace=True)
df_dummies=pd.get_dummies(data_frame,columns=['Total_Rural_Urban','Disability', 'Age_group'])
data_frame.drop(['Total_Rural_Urban','Disability', 'Age_group'],axis = 1,inplace=True)
attributes=['Total_disabled_population_Persons','Total_disabled_population_Males', 'Total_disabled_population_Females',
'Main_worker_Persons', 'Main_worker_Males', 'Main_worker_Females', 'Marginal_worker_Less_than_3_months_Persons',
'Marginal_worker_Less_than_3_months_Males', 'Marginal_worker_Less_than_3months_Females', 
'Marginal_worker_3to6_months_Persons','Marginal_worker_3to6_months_Males', 'Marginal_worker_3to6_months_Females',
'Non_worker_Persons', 'Non_worker_Males','Non_worker_Females', 'Total_Rural_Urban_Rural','Total_Rural_Urban_Total','Total_Rural_Urban_Urban',
'Disability_Any_Other','Disability_In_Hearing','Disability_In_Movement','Disability_In_Seeing',
'Disability_In_Speech','Disability_Mental_Illness','Disability_Mental_Retardation',
'Disability_Multiple_Disability','Disability_Total','Age_group_0to14','Age_group_15to59',
'Age_group_60+','Age_group_AgeNotStated','Age_group_Total']
print("attributes ka length", len(attributes))
data_frame.to_csv(r"flaskDirectory\dataframe.csv", index = False)
df_dummies.to_csv(r"flaskDirectory\dfdummies.csv", index = False)
#scaling using standard scaling technique
ss = StandardScaler()
mm_scaler = preprocessing.MinMaxScaler()
scaled_df = mm_scaler.fit_transform(df_dummies)
# scaled_df = ss.fit_transform(df_dummies)
scaled_df=pd.DataFrame(scaled_df, columns=attributes, dtype='Float32')
scaled_df.to_csv(r"flaskDirectory\scaleddf.csv", index = False)

print("--------------Scaled Data Frame-------------------------")
print(scaled_df.head())
print("---------------DF DUMMIES-------------------------------")
# print(df_dummies.head())
data_len=len(data_frame)
stratasample=int(0.25*data_len)

def randomSample():
    global scaled_df
    global rs_data_frntend
    rs_data_frame=scaled_df.sample(frac=0.25)
    rs_data_frame.to_csv(r"flaskDirectory\rs_India_disabled_population.csv", index = False)
    print("------DATA FRAME RS---",rs_data_frame.head())
    # rs_data_frntend=rs_data_frame.values.tolist()
    rs_data_frntend=rs_data_frame
    # print("front end==", rs_data_frntend[0])
    print("--------------------------Random sample front end--------------------------")
    print(rs_data_frntend)   

def plotElbow():
    distortions = []
    K = range(1,11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data_frame)
        distortions.append(kmeanModel.inertia_)
        print("Interia error===",kmeanModel.inertia_)

   
    plt.figure(figsize=(8,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('optimal k by elbow method')
    plt.show()

def clustering():
    kmeans = KMeans(n_clusters=3)
    y_predict=kmeans.fit_predict(scaled_df)
    scaled_df['cluster']=y_predict
    # scaled_df.round(3)
    scaled_df.to_csv(r"flaskDirectory\modified_India_disabled_population.csv", index = False)

def stratifiedSample(): 
    global ss_data_frntend
    cluster0 = scaled_df[scaled_df['cluster'] == 0]
    cluster1 = scaled_df[scaled_df['cluster'] == 1]
    cluster2 = scaled_df[scaled_df['cluster'] == 2]
    # cluster3 = df_dummies[df_dummies['cluster'] == 3]

    cluster0len= int(len(cluster0) * (stratasample / data_len))
    cluster1len = int(len(cluster1) * (stratasample / data_len))
    cluster2len = int(len(cluster2) * (stratasample / data_len))
    # cluster3len = int(len(cluster3) * (stratasample / data_len))

    print("----------------cluster lengths--------------------")
    print("cluster 0 == ", cluster0len)
    print("cluster 1 == ", cluster1len)
    print("cluster 2 == ", cluster2len)
    # print("cluster 3 == ", cluster3len)
    li0=sample(list(cluster0.index), cluster0len)
    li1=sample(list(cluster1.index), cluster1len)
    li2=sample(list(cluster2.index), cluster2len)
    # li3=sample(list(cluster3.index), cluster3len)

    sample0 = cluster0.loc[li0]
    sample1 = cluster1.loc[li1]
    sample2 = cluster2.loc[li2]
    # sample3 = cluster3.loc[li3]

    stratified_samples_df=pd.concat([sample0, sample1, sample2])
    # stratified_samples_df.drop('cluster',axis = 1,inplace=True)
    stratified_samples_df.to_csv(r"flaskDirectory\strataSample_India_disabled_population.csv", index = False)
    ss_data_frntend=stratified_samples_df
    print("--------------------------Stratified sample front end--------------------------")
    print(ss_data_frntend) 
    print("--------------------------Stratified sample data frame--------------------------")
    print(stratified_samples_df.head()) 

def calculate_eigenValues(scaled_df):
  
    #calculating eigen values and vectors using covariance matrix
    # mean_vec = np.mean(saled_df, axis=0)
    # covariance_matrix = (df_dummies - mean_vec).T.dot((df_dummies - mean_vec)) / (df_dummies.shape[0]-1)
    # covariance_matrix = np.cov(scaled_df.T)
    # print("covariance matrix==",covariance_matrix)
    # eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    pca = PCA()
    pca.fit(scaled_df)
    eigen_vectors=pca.components_
    eigen_values=pca.explained_variance_ratio_
    #sorting eigen values

    eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])

    # eigen_sort = eigen_values.argsort()[::-1]
    # eigen_values = eigen_values[eigen_sort]
    # eigen_vectors = eigen_vectors[:, eigen_sort]
    print("----------Eigen Values-------------")
    print(eigen_values)
    print("----------Eigen Vectors-------------")
    print(eigen_vectors)
    return eigen_values, eigen_vectors


def plot_intrinsic_dimensionality_pca(scaled_df, k):
    [eigenValues, eigenVectors] = calculate_eigenValues(scaled_df)
    squaredLoadings = []
    ftrCount = len(eigenVectors)
    for ftrId in range(0, ftrCount):
        loadings = 0
        for compId in range(0, k):
            loadings = loadings + eigenVectors[compId][ftrId] * eigenVectors[compId][ftrId]
        squaredLoadings.append(loadings)

    # print("squaredLoadings==", squaredLoadings)
    plt.plot(eigenValues)
    plt.xlabel("PCA Components")
    plt.ylabel("Eigen Values")
    plt.show()
    return squaredLoadings


@application.route("/pca_scree")
def find_intrinsic_dimensionality():
    print("Inside scree")                           
    [eigenValues, eigenVectors] = calculate_eigenValues(scaled_df)
    print(eigenValues)
    return json.dumps(np.real(eigenValues).tolist())

@application.route("/pca_scree_rs")
def find_intrinsic_dimensionality_random():
    print("Inside scree")                           
    [eigenValues, eigenVectors] = calculate_eigenValues(rs_data_frntend)
    print(eigenValues)
    return json.dumps(np.real(eigenValues).tolist())

@application.route("/pca_scree_ss")
def find_intrinsic_dimensionality_stratified():
    print("Inside scree")                           
    [eigenValues, eigenVectors] = calculate_eigenValues(ss_data_frntend)
    print(eigenValues)
    return json.dumps(np.real(eigenValues).tolist())        

@application.route("/original_Scree")
def scree_plot_original():
    items_return = dict()
    pca_data = PCA(n_components=8)
    pca_data.fit_transform(scaled_df)
    variance = pca_data.explained_variance_ratio_ #calculate variance ratios
    variance=np.round(variance, 3)*100

    var=np.cumsum(np.round(variance, 3))
    variance=np.sort(variance)
    print("-------------Explained Variance-------------------------")
    print(variance)
    items_return['components'] = 8
    items_return['explained_variance']=json.dumps(variance.tolist())
    return jsonify(items_return)


@application.route("/random_scree")
def scree_plot_rs():
    items_return = dict()
    pca_data = PCA(n_components=8)
    pca_data.fit_transform(rs_data_frntend)
    variance = pca_data.explained_variance_ratio_ #calculate variance ratios
    variance=np.round(variance,3)*100

    var=np.cumsum(np.round(variance, 3))
    variance=np.sort(variance)
    print("-------------Explained Variance for random-------------------------")
    print(variance)
    items_return['components'] = 8
    items_return['explained_variance']=json.dumps(variance.tolist())
    return jsonify(items_return)

@application.route("/stratified_scree")
def scree_plot_ss():   
    items_return = dict()
    pca_data = PCA(n_components=8)
    pca_data.fit_transform(ss_data_frntend)
    variance = pca_data.explained_variance_ratio_ #calculate variance ratios
    variance=np.round(variance,3)*100
    var=np.cumsum(np.round(variance,3))
    variance=np.sort(variance)
    print("-------------Explained Variance for stratified-------------------------")
    print(variance)
    items_return['components'] = 8
    items_return['explained_variance']=json.dumps(variance.tolist())
    return jsonify(items_return)


def findThreeAtributesLoadingsForOriginal():
    global imp_ftrs_original
    pca_data = PCA(n_components=3)
    datafrm = pd.DataFrame(scaled_df)
    datafrm.drop('cluster',axis = 1,inplace=True)
    pca_data.fit_transform(datafrm) 
    loadings = pd.DataFrame(pca_data.components_.T, columns=['PC1', 'PC2','PC3'], index=datafrm.columns)
    x=loadings['PC1'].values
    y=loadings['PC2'].values
    z=loadings['PC3'].values
    # print("each pc1 loading",x[2])
    sum_sq=[]
    for i in range(0,len(loadings)):
        sum_sq.append(math.sqrt(math.pow(x[i],2)+math.pow(y[i],2)+math.pow(z[i],2)))
    print("-----------Sum Square Original---------")
    print(sum_sq)
    loadings['Sum_of_Square_Loadings']=sum_sq
    imp_ftrs_original = sorted(range(len(sum_sq)-1), key=lambda k: sum_sq[k], reverse=True)
    print("---------------------Important Features For Original--------------------------")
    print(imp_ftrs_original)
    print("---------------LOADINGS Original--------------")  
    print(loadings)
    return imp_ftrs_original

def findThreeAtributesLoadingsForRandom():
    global imp_ftrs_random
    random_data=pd.read_csv(r"flaskDirectory\rs_India_disabled_population.csv")
    pca_data = PCA(n_components=3)
    pca_data.fit_transform(random_data)
    datafrm = pd.DataFrame(random_data)
    loadings = pd.DataFrame(pca_data.components_.T, columns=['PC1', 'PC2','PC3'], index=random_data.columns)
    x=loadings['PC1'].values
    y=loadings['PC2'].values
    z=loadings['PC3'].values
    # print("each pc1 loading",x[2])
    sum_sq=[]
    for i in range(0,len(loadings)):
        sum_sq.append(math.sqrt(math.pow(x[i],2)+math.pow(y[i],2)+math.pow(z[i],2)))
    print("-----------Sum Square Random---------")
    print(sum_sq)
    loadings['Sum_of_Square_Loadings']=sum_sq
    imp_ftrs_random = sorted(range(len(sum_sq)-1), key=lambda k: sum_sq[k], reverse=True)
    print("---------------------Important Features Random--------------------------")
    print(imp_ftrs_random)
    print("---------------LOADINGS Random--------------")  
    print(loadings)
    return imp_ftrs_random

def findThreeAtributesLoadingsForStratified():
    global imp_ftrs_strata
    strata_data=pd.read_csv(r"flaskDirectory\strataSample_India_disabled_population.csv")
    pca_data = PCA(n_components=3)
    pca_data.fit_transform(strata_data)
    datafrm = pd.DataFrame(strata_data)
    loadings = pd.DataFrame(pca_data.components_.T, columns=['PC1', 'PC2','PC3'], index=strata_data.columns)
    x=loadings['PC1'].values
    y=loadings['PC2'].values
    z=loadings['PC3'].values
    # print("each pc1 loading",x[2])
    sum_sq=[]
    for i in range(0,len(loadings)):
        sum_sq.append(math.sqrt(math.pow(x[i],2)+math.pow(y[i],2)+math.pow(z[i],2)))
    print("-----------Sum Square Stratified---------")
    print(sum_sq)
    loadings['Sum_of_Square_Loadings']=sum_sq
    imp_ftrs_strata = sorted(range(len(sum_sq)-1), key=lambda k: sum_sq[k], reverse=True)
    print("---------------------Important Features Stratified--------------------------")
    print(imp_ftrs_strata)
    print("---------------LOADINGS Stratified--------------")  
    print(loadings)
    return imp_ftrs_strata



@application.route('/pca_random')
def pca_random():
    global imp_ftrs
    random_data=pd.read_csv(r"flaskDirectory\rs_India_disabled_population.csv")
    imp_ftrs = imp_ftrs_random
    modidata=pd.read_csv(r"flaskDirectory\modified_India_disabled_population.csv")
    modidf=pd.DataFrame(modidata)
    rdf=pd.DataFrame(random_data)
    print("-----------------------RANDOM_DATA--------------------------------")
    print(random_data.head())
    print("-----------------------MODIDF--------------------------------")
    print(modidf.head())
    data_columns = []
    samplesize=len(rdf)
    pca_data = PCA(n_components=2)
    rdf=pca_data.fit_transform(rdf)
    modidf = modidf.fillna(0)
    data_columns = pd.DataFrame(rdf)
    # print(data_columns.head())
    print("---------------------------RS DATA FRONTEND------------------------")
    print(rs_data_frntend)
    for i in range(0,2):
        print("Important features i = ", imp_ftrs[i])
        print("attributes at this = ", attributes[imp_ftrs[i]])
        data_columns[attributes[imp_ftrs[i]]] = random_data[attributes[imp_ftrs[i]]][:samplesize]
        print("rs data front going to data columns = ", random_data[attributes[imp_ftrs[i]]][:samplesize])
    data_columns['clusterid'] = modidata['cluster'][:samplesize]
    print("-----------------------DATA COLUMNS--------------------------------")
    print(data_columns.head(samplesize))
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns
   
@application.route('/pca_adaptive')
def pca_adaptive():
    data_columns = []
    strata_data=pd.read_csv(r"flaskDirectory\strataSample_India_disabled_population.csv")
    global imp_ftrs
    imp_ftrs = imp_ftrs_strata
    print("-----------raghav ne kaha------------------")
    print(imp_ftrs_strata)
    sdf=pd.DataFrame(strata_data)
    pca_data = PCA(n_components=2)
    sdf=pca_data.fit_transform(sdf)
    data_columns = pd.DataFrame(sdf)
    samplesize=len(sdf)
    for i in range(0, 2):
        data_columns[attributes[imp_ftrs[i]]] = strata_data[attributes[imp_ftrs[i]]][:samplesize]
    data_columns['clusterid'] = np.nan
    x = 0
    for index, row in strata_data.iterrows():
        data_columns['clusterid'][x] = row['cluster']
        x = x + 1
    print("------------------STRATA DATA COLUMNS---------------------------")
    print(data_columns.head(samplesize))   
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns


@application.route('/mds_euclidean_random')
def mds_euclidean_random():
    data_columns = []
    global imp_ftrs
    imp_ftrs = imp_ftrs_random
    random_data=pd.read_csv(r"flaskDirectory\rs_India_disabled_population.csv")
    modidata=pd.read_csv(r"flaskDirectory\modified_India_disabled_population.csv")
    rdf=pd.DataFrame(random_data)
    modidf=pd.DataFrame(modidata)
    samplesize=len(rdf)
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(random_data, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(X)
    for i in range(0, 3):
        data_columns[attributes[imp_ftrs[i]]] = random_data[attributes[imp_ftrs[i]]][:samplesize]
    data_columns['clusterid'] = modidata['cluster'][:samplesize]
    print("-----------------------RANDOM DATA COLUMNS FOR MDS EUCLIDEAN--------------------------------")
    print(data_columns.head(samplesize))
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns

@application.route('/mds_euclidean_adaptive')
def mds_euclidean_adaptive():
    strata_data=pd.read_csv(r"flaskDirectory\strataSample_India_disabled_population.csv")
    data_columns = []
    global imp_ftrs
    imp_ftrs = imp_ftrs_strata
    sdf=pd.DataFrame(strata_data)
    samplesize=len(sdf)
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')    
    similarity = pairwise_distances(strata_data, metric='euclidean')
    sdf = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(sdf)
    for i in range(0, 3):
        data_columns[attributes[imp_ftrs[i]]] = strata_data[attributes[imp_ftrs[i]]][:samplesize]
    data_columns['clusterid'] = np.nan
    x = 0
    for index, row in strata_data.iterrows():
        data_columns['clusterid'][x] = row['cluster']
        x = x + 1
    print("------------------EUCLIDEAN STRATA DATA COLUMNS---------------------------")
    print(data_columns.head(samplesize))   
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns

@application.route('/mds_correlation_random')
def mds_correlation_random():
    data_columns = []
    random_data=pd.read_csv(r"flaskDirectory\rs_India_disabled_population.csv")
    global imp_ftrs
    imp_ftrs = imp_ftrs_random
    rdf=pd.DataFrame(random_data)
    samplesize=len(rdf)
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(random_data, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(X)
    for i in range(0, 2):
        data_columns[attributes[imp_ftrs[i]]] = random_data[attributes[imp_ftrs[i]]][:samplesize]
    data_columns['clusterid'] = random_data['cluster'][:samplesize]
    print("-----------------------RANDOM DATA COLUMNS FOR MDS CORRELATION--------------------------------")
    print(data_columns.head(samplesize))
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns

@application.route('/mds_correlation_adaptive')
def mds_correlation_adaptive():
    data_columns = []
    strata_data=pd.read_csv(r"flaskDirectory\strataSample_India_disabled_population.csv")
    global imp_ftrs
    imp_ftrs = imp_ftrs_strata
    sdf=pd.DataFrame(strata_data)
    samplesize=len(sdf)
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    X = strata_data
    similarity = pairwise_distances(X, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_columns = pd.DataFrame(X)
    for i in range(0, 2):
        data_columns[attributes[imp_ftrs[i]]] = strata_data[attributes[imp_ftrs[i]]][:samplesize]

    data_columns['clusterid'] = np.nan
    x = 0
    for index, row in strata_data.iterrows():
        data_columns['clusterid'][x] = row['kcluster']
        x = x + 1
    print("------------------EUCLIDEAN STRATA DATA COLUMNS---------------------------")
    print(data_columns.head(samplesize))   
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns


@application.route('/scatter_matrix_random')
def scatter_matrix_random():
    global imp_ftrs
    modidata=pd.read_csv(r"flaskDirectory\modified_India_disabled_population.csv")
    random_data=pd.read_csv(r"flaskDirectory\rs_India_disabled_population.csv")
    imp_ftrs = imp_ftrs_random
    data_columns = pd.DataFrame()
    rdf=pd.DataFrame(random_data)
    samplesize=len(rdf)
    for i in range(0, 3):
        data_columns[attributes[imp_ftrs[i]]] = random_data[attributes[imp_ftrs[i]]][:samplesize]
    data_columns['clusterid'] = modidata['cluster'][:samplesize]
    print("------------------SCATTER PLOT MATRIX RANDOM DATA COLUMNS---------------------------")
    print(data_columns.head(samplesize))
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns

@application.route('/scatter_matrix_adaptive')
def scatter_matrix_adaptive():
    strata_data=pd.read_csv(r"flaskDirectory\strataSample_India_disabled_population.csv")
    global imp_ftrs
    imp_ftrs = imp_ftrs_strata
    data_columns = pd.DataFrame()
    sdf=pd.DataFrame(strata_data)
    samplesize=len(strata_data)
    print("size = ", samplesize)
    li=[9,3,5]
    for i in range(0, 3):
        # data_columns[attributes[imp_ftrs[i]]] = strata_data[attributes[imp_ftrs[i]]][:samplesize]
        data_columns[attributes[li[i]]] = strata_data[attributes[li[i]]][:samplesize]

    data_columns['clusterid'] = np.nan
    for index, row in strata_data.iterrows():
        data_columns['clusterid'][index] = row['cluster']
    data_columns = data_columns.reset_index(drop=True)
    print("------------------SCATTER PLOT MATRIX STRATIFIED DATA COLUMNS---------------------------")
    print(data_columns.head(20))
    data_columns=data_columns.to_dict(orient='records')
    data_columns=json.dumps(data_columns, indent=2)
    return data_columns

plotElbow()
randomSample()
clustering()
stratifiedSample()
# squared_loadings=plot_intrinsic_dimensionality_pca(scaled_df, 3)
imp_ftrs_strata = findThreeAtributesLoadingsForStratified()
imp_ftrs_random = findThreeAtributesLoadingsForRandom()
imp_ftrs_original = findThreeAtributesLoadingsForOriginal()
# imp_ftrs = sorted(range(len(squared_loadings)), key=lambda k: squared_loadings[k], reverse=True)
# print("=============The significant attributes are=======", significant_attr)
# pca_random()
# pca_adaptive()
# mds_euclidean_random()
# mds_euclidean_adaptive()
scatter_matrix_random()
scatter_matrix_adaptive()

if __name__ == "__main__": 
    application.run("localhost", 8080)