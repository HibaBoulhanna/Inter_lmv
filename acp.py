
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def get_acp(df):
    bbands=df.ta.bbands()
    BB=pd.DataFrame(bbands)
    BL=BB["BBB_5_2.0"]
    macd=df.ta.macd()
    macd=pd.DataFrame(macd)
    macd=macd["MACD_12_26_9"]
    RSI=df.ta.rsi()
    rsi=pd.DataFrame(RSI)
    data_IS=pd.concat([macd,RSI,BL],axis=1).dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_IS)
    data_IS_SCALED=pd.DataFrame(scaled)
    data_IS_SCALED
    scaler=StandardScaler()
    scaler.fit(data_IS)
    scaled_data=scaler.transform(data_IS)
    pca_out=PCA(n_components=3).fit(data_IS)
    loadings = pca_out.components_
    num_pc = pca_out.n_features_
    pc_list = ["Composante principale n°"+str(i) for i in (range(1, 7))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = data_IS.columns.values
    loadings_df = loadings_df.set_index('variable')
    var_axe_1=pca_out.explained_variance_ratio_[0]
    var_axe_2=pca_out.explained_variance_ratio_[1]
    var_axe_3=pca_out.explained_variance_ratio_[2]
    #COLONE 
    carre=loadings_df**2
    Somme_Composante_1=carre["Composante principale n°1"].sum()
    Somme_Composante_2=carre["Composante principale n°2"].sum()
    Somme_Composante_3=carre["Composante principale n°3"].sum()
    carre["PC1_SUR_somme_1"]=carre["Composante principale n°1"]/Somme_Composante_1
    carre["PC2_SUR_somme_2"]=carre["Composante principale n°2"]/Somme_Composante_2
    carre["PC3_SUR_somme_3"]=carre["Composante principale n°3"]/Somme_Composante_3
    carre["var_fois_pc_1"]=var_axe_1*carre["PC1_SUR_somme_1"]
    carre["var_fois_pc_2"]=var_axe_2*carre["PC2_SUR_somme_2"]
    carre["var_fois_pc_3"]=var_axe_3*carre["PC3_SUR_somme_3"]
    Tableau_pour_IS=carre[["var_fois_pc_1","var_fois_pc_2","var_fois_pc_3"]]
    IS = Tableau_pour_IS.sum(axis = 1)
    IS = pd.DataFrame(IS, columns=['Poid'])
    return IS

