import streamlit as st
import pandas as pd 
import requests
import investpy as py
from bs4 import BeautifulSoup
import plotly.express as px
from acp import get_acp
from _moving_average_convergence_divergence import MovingAverageConvergenceDivergence
from Indicators import  macd, sign_macd1, smm, stochastic, rate_of_change, momentum, emm, obv, williams, MFI, cho, nvi, pvi, bollinger, rsi, sign_momentum, sign_pvi, sign_bollinger, sign_rsi, sign_cho, sign_stochastique1, sign_roc
# st.set_option('deprecation.showPyplotGlobalUse', False)


def ticker_2_CodeValeur(ticker):
  ticker_2_CodeValeur = {"ADH" : "9000" , "AFM" : "12200" , "AFI" : "11700" , "GAZ" : "7100" , "AGM" : "6700" , "ADI" : "11200" , "ALM" : "6600" , "ARD" : "27" , "ATL" : "10300" , "ATW" : "8200" , "ATH" : "3200" , "NEJ" : "7000" , "BAL" : "3300" , "BOA" : "1100" , "BCP" : "8000" , "BCI" : "5100" , "CRS" : "8900" , "CDM" : "3600" , "CDA" : "3900" , "CIH" : "3100" , "CMA" : "4000" , "CMT" : "11000" , "COL" : "9200" , "CSR" : "4100" , "CTM" : "2200" , "DRI" : "8500" , "DLM" : "10800" , "DHO" : "10900" , "DIS" : "4200" , "DWY" : "9700" , "NKL" : "11300" , "EQD" : "2300" , "FBR" : "9300" , "HPS" : "9600" , "IBC" : "7600" , "IMO" : "12" , "INV" : "9500" , "JET" : "11600" , "LBV" : "11100" , "LHM" : "3800" , "LES" : "4800" , "LYD" : "8600" , "M2M" : "10000" , "MOX" : "7200" , "MAB" : "1600" , "MNG" : "7300" , "MLE" : "2500" , "IAM" : "8001" , "MDP" : "6500" , "MIC" : "10600" , "MUT" : "21" , "NEX" : "7400" , "OUL" : "5200" , "PRO" : "9900" , "REB" : "5300" , "RDS" : "12000" , "RISMA" : "8700" , "S2M" : "11800" , "SAH" : "11400" , "SLF" : "10700" , "SAM" : "6800" , "SMI" : "1500" , "SNA" : "10500" , "SNP" : "9400" , "MSA" : "12300" , "SID" : "1300" , "SOT" : "9800" , "SRM" : "2000" , "SBM" : "10400" , "STR" : "11500" , "TQM" : "11900" , "TIM" : "10100" , "TMA" : "12100" , "UMR" : "7500" , "WAA" : "6400" , "ZDJ" : "5800"}
  return ticker_2_CodeValeur[ticker]


def get_image(ticker):                                                           
  url = f"https://www.casablanca-bourse.com/bourseweb/img/societes_cote/{ticker}.gif"
  return url

apptitle = 'Projet hiba'

st.set_page_config(page_title=apptitle, page_icon=":chart_with_upwards_trend:")

# Title the app
st.title('Interface Stage PFE ')

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data


st.sidebar.markdown('<center><img src="https://www.lamarocainevie.com/bo/sites/default/files/2019-02/logo.png" width="300"  height="100" alt="Marocaine vie "></center>', unsafe_allow_html=True)

st.sidebar.markdown("## Sélectionner un indicateur et préciser une valeurs pour ses paramètres  ")
st.markdown('Travail realisé par: Boulhanna Hiba ')
st.markdown("Sous la direction de Mr.HOUMMANI Ayoub & Mr.BENABADJI Oualid")
st.markdown('__________________________________________________________')



dropdown = st.sidebar.selectbox("Choisir une action", pd.concat([pd.Series(["MASI"]), py.get_stocks(country='morocco').name]))
indicateur = st.sidebar.selectbox("Choisir un indicateur", ['BB','MACD','MOM','RSI','ROC','SMA','STOCHA','OBV', 'williams', 'EMM', 'MFI', 'CHO', 'NVI', 'PVI'])

start = st.sidebar.date_input('Debut', value =pd.to_datetime('01-01-2020'))
end = st.sidebar.date_input('Fin', value = pd.to_datetime('today'))

start = start.strftime('%d/%m/%Y')
end = end.strftime('%d/%m/%Y')

stocks = py.get_stocks(country='morocco')
stocks.set_index("name", inplace = True)


if dropdown != "MASI":
  ticker =  stocks.loc[dropdown,'symbol']
  df=py.get_stock_historical_data(stock=ticker, country='morocco', from_date=start, to_date=end)
  url = get_image(ticker)
  url = str(url)
else :
  df=py.get_index_historical_data(index='Moroccan All Shares', country='morocco', from_date=start, to_date=end)
  df.Volume = df.Close*10000 
  url = "https://static.lematin.ma/cdn/images/icobourse/indices/masi.png"


# st.markdown('<center><img src="'+url+'" alt="stock logo"></center>', unsafe_allow_html=True) # logo de l'action
st.markdown('![Alt Text]('+url+')') # logo de l'action

print(url)
st.markdown('__________________________________________________________')

if indicateur == 'MACD':
  c1, c2 = st.columns(2)
  with c1:
      ws = st.number_input('Ws', 9)
  with c2:
      wl = st.number_input('Wl', 26)

  macd = macd(df.Close, ws, wl)
  data = [df.Close, macd['MACD'], macd['MACDsignal']]
  headers = ["close", "macd", 'signal line']
  sign_fig = sign_macd1(df.Close, ws, wl)
  
elif indicateur == 'RSI':
    period = st.number_input('Period', 9)
    rsi=rsi(df.iloc[:,3], period)
    data = [rsi["COURS_CLOTURE"],rsi["RSI"]]
    headers = ["close", "rsi"]
    sign_fig = sign_rsi(df.iloc[:,3], period)


elif indicateur == 'BB':
    c1, c2 = st.columns(2)
    with c1:
        w = st.number_input('W', 12)
    with c2:
        k= st.number_input('K', 4)
    df1 = bollinger(df.iloc[:,3], w, k)
    data = [df1['COURS_CLOTURE'],df1["BBDOWN"],df1["BBMID"],df1["BBUP"]]
    headers = ["close",  "lower_band","middle_band","upper_band",]
    sign_fig = sign_bollinger(df.iloc[:,3], w, k)

elif indicateur == 'ROC':
    w = st.number_input('W', 9)
    roc = rate_of_change(df.Close, w)
    data = [roc['Close'], roc['ROC']]
    headers = ['Close', 'ROC']
    sign_fig = sign_roc(df.Close, w)

elif indicateur == 'OBV':
    obv = obv(df.Close, df.Volume)
    data = [obv['Close'], obv['OBV']]
    headers = ['Close', 'OBV'] 

elif indicateur == 'NVI':
    nvi = pd.DataFrame(nvi(df.Close, df.Volume), columns = ['NVI'])
    nvi.index = df.index
    data = [df['Close'], nvi['NVI']]
    headers = ['Close', 'NVI'] 

elif indicateur == 'PVI':
    n = st.number_input('n', 9)
    pvi = pd.DataFrame(pvi(df.Close, df.Volume), columns = ['PVI'])
    pvi.index = df.index
    data = [df['Close'], pvi['PVI']]
    headers = ['Close', 'PVI']
    sign_fig = sign_pvi(df.Close, df.Volume,n) 
  
elif indicateur == 'SMA':
    period = st.number_input('n', 9)
    sma = smm(df.Close,period)
    data = [sma['Close'], sma['SMM']]
    headers = ['Close', 'SMA']

elif indicateur == 'williams':
    period = st.number_input('n', 9)
    williams = williams(df.Close,period)
    data = [williams['Close'], williams['%R']]
    headers = ['Close', '%R']

elif indicateur == 'EMM':
    period = st.number_input('n', 9)
    emm = emm(df.Close,period)
    data = [df.Close, emm]
    headers = ['Close', 'EMM']

elif indicateur == 'MFI':
    period = st.number_input('n', 9)
    mfi = MFI(df.Close, df.Volume, df.High, df.Low, period)
    data = [mfi.Close, mfi.MFI]
    headers = ['Close', 'MFI']

elif indicateur == 'CHO':
    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.number_input('period', 1)
    with c2:
        ws = st.number_input('Ws', 1)
    with c3 :
        wl = st.number_input('Wl', 1)
    cho = cho(df.Close, df.Volume, df.High, df.Low, period, ws,wl)
    data = [cho['Close'], cho['CHO']]
    headers = ["close", 'CHO']
    sign_fig = sign_cho(df.Close, df.Volume, df.High, df.Low, period, ws,wl)

elif indicateur == 'MOM':
    c1, c2 = st.columns(2)
    with c1:
        w = st.number_input('W', 12)
    with c2:
        wsig = st.number_input('Wsig', 9)
    mom = momentum(df.Close,w, wsig)
    sign_fig = sign_momentum(df.Close,w, wsig)

    data = [mom['Close'], mom['MOM'], mom['MOMsignal']]
    headers = ["close", 'MOM','MOMsignal']

elif indicateur == 'STOCHA':
    c1, c2 = st.columns(2)
    with c1:
        period = st.number_input('Period', 12)
    with c2:
        w = st.number_input('W', 4)
    stoch = stochastic(df.Close, df.High, df.Low,period, w)
    data = [stoch['Close'],stoch['%K'], stoch['%D']]
    headers = ["close", "K", 'D']
    sign_fig = sign_stochastique1(df.Close, df.High, df.Low,period, w)

st.markdown('__________________________________________________________')
df3 = pd.concat(data, axis=1, keys=headers)
fig = px.line(df3, width=1200, height=700 )
st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
st.pyplot(sign_fig)

dt = get_acp(df)
st.markdown('ACP_Standard :')
st.dataframe(dt)
st.markdown('__________________________________________________________')
st.markdownt('________________________________________________________')
dtt=get_acp('ACP_Optimisé  :')
stt.dataframe(dt)
