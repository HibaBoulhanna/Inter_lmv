
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from Indicators import smm, stochastic, rate_of_change, momentum, emm, obv, williams, MFI, cho, nvi, pvi


def get_acp(df):
    bbands=df.ta.bbands()
    BB=pd.DataFrame(bbands)
    BL=BB["BBB_5_2.0"]
    ##################################
    macd=df.ta.macd()
    macd=pd.DataFrame(macd)
    macd=macd["MACD_12_26_9"]
    ##################################
    RSI=df.ta.rsi()
    rsi=pd.DataFrame(RSI)
    ################################## 
    try:
        roc1 = rate_of_change(df.close, 9)['ROC']
        sma = smm(df.close,9).SMM
        mom = momentum(df.close,9)['MOMsignal']
        stoch1 =  stochastic(df.close, df.high, df.low,9, 26)['%K']
        stoch2 =  stochastic(df.close, df.high, df.low,9, 26)['%D']
    except :
        roc1 = rate_of_change(df.Close, 9)['ROC']
        sma = smm(df.Close,9).SMM
        mom = momentum(df.Close,9)['MOMsignal']
        stoch1 =  stochastic(df.Close, df.High, df.Low,9, 26)['%K']
        stoch2 =  stochastic(df.Close, df.High, df.Low,9, 26)['%D'] 
        ###################################
    
    data_IS=pd.concat([macd,RSI,BL, roc1, sma, mom, stoch1, stoch2],axis=1).dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_IS)
    data_IS_SCALED=pd.DataFrame(scaled)
    data_IS_SCALED
    scaler=StandardScaler()
    scaler.fit(data_IS)
    scaled_data=scaler.transform(data_IS)
    pca_out=PCA(n_components=7).fit(data_IS)
    loadings = pca_out.components_
    num_pc = pca_out.n_features_
    pc_list = ["Composante principale n°"+str(i) for i in (range(1, 15))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = data_IS.columns.values
    loadings_df = loadings_df.set_index('variable')
    var_axe_1=pca_out.explained_variance_ratio_[0]
    var_axe_2=pca_out.explained_variance_ratio_[1]
    var_axe_3=pca_out.explained_variance_ratio_[2]
    var_axe_4=pca_out.explained_variance_ratio_[3]
    var_axe_5=pca_out.explained_variance_ratio_[4]
    var_axe_6=pca_out.explained_variance_ratio_[5]
    var_axe_7=pca_out.explained_variance_ratio_[6]

    #COLONE 
    carre=loadings_df**2
    Somme_Composante_1=carre["Composante principale n°1"].sum()
    Somme_Composante_2=carre["Composante principale n°2"].sum()
    Somme_Composante_3=carre["Composante principale n°3"].sum()
    Somme_Composante_4=carre["Composante principale n°4"].sum()
    Somme_Composante_5=carre["Composante principale n°5"].sum()
    Somme_Composante_6=carre["Composante principale n°6"].sum()
    Somme_Composante_7=carre["Composante principale n°7"].sum()
    
    carre["PC1_SUR_somme_1"]=carre["Composante principale n°1"]/Somme_Composante_1
    carre["PC2_SUR_somme_2"]=carre["Composante principale n°2"]/Somme_Composante_2
    carre["PC3_SUR_somme_3"]=carre["Composante principale n°3"]/Somme_Composante_3
    carre["PC4_SUR_somme_4"]=carre["Composante principale n°4"]/Somme_Composante_4
    carre["PC5_SUR_somme_5"]=carre["Composante principale n°5"]/Somme_Composante_5
    carre["PC6_SUR_somme_6"]=carre["Composante principale n°6"]/Somme_Composante_6
    carre["PC7_SUR_somme_7"]=carre["Composante principale n°7"]/Somme_Composante_7
    
    carre["var_fois_pc_1"]=var_axe_1*carre["PC1_SUR_somme_1"]
    carre["var_fois_pc_2"]=var_axe_2*carre["PC2_SUR_somme_2"]
    carre["var_fois_pc_3"]=var_axe_3*carre["PC3_SUR_somme_3"]
    carre["var_fois_pc_4"]=var_axe_4*carre["PC4_SUR_somme_4"]
    carre["var_fois_pc_5"]=var_axe_5*carre["PC5_SUR_somme_5"]
    carre["var_fois_pc_6"]=var_axe_6*carre["PC6_SUR_somme_6"]
    carre["var_fois_pc_7"]=var_axe_7*carre["PC7_SUR_somme_7"]

    Tableau_pour_IS=carre[["var_fois_pc_1","var_fois_pc_2","var_fois_pc_3","var_fois_pc_4","var_fois_pc_5","var_fois_pc_6",  "var_fois_pc_7"]]
    IS = Tableau_pour_IS.sum(axis = 1)
    IS = pd.DataFrame(IS, columns=['Poid'])
    IS=IS*100
    return IS
