import streamlit as st
import pandas as pd 
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.express as px
from acp import get_acp
from acp_opt import get_acp_opt
from _moving_average_convergence_divergence import MovingAverageConvergenceDivergence
from Indicators import  macd, sign_macd1, smm, stochastic, rate_of_change, momentum, emm, obv, williams, MFI, cho, nvi, pvi, bollinger, rsi, sign_momentum, sign_pvi, sign_bollinger, sign_rsi, sign_cho, sign_stochastique1, sign_roc, sign_nvi, sign_mfi, sign_mms1
# st.set_option('deprecation.showPyplotGlobalUse', False)


def get_logo(ticker):
    cookies = {
        '_sp_ses.cf1a': '*',
        '_sp_id.cf1a': 'fae78f9c-2ee8-4598-b504-df36037e640b.1646861479.3.1649003427.1647712930.a10c82a7-fc67-4f35-92a5-9b56646b5049',
    }
    headers = {
            'authority': 'www.tradingview.com',
            'cache-control': 'max-age=0',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-user': '?1',
            'sec-fetch-dest': 'document',
            'accept-language': 'en-US,en;q=0.9,fr;q=0.8',
        }

    try : 
        url = f'https://www.tradingview.com/symbols/NYSE-{ticker}/'
        r = requests.get(url, headers=headers, cookies=cookies)
        soup = BeautifulSoup(r.text, "html.parser")
        img = soup.find("div", {"class": "js-sticky-symbol-header-container tv-sticky-symbol-header"})
        img_url = img["data-logo-urls"]
        return img_url
    except :
        try: 
            url = f'https://www.tradingview.com/symbols/NASDAQ-{ticker}/'
            r = requests.get(url, headers=headers, cookies=cookies)
            soup = BeautifulSoup(r.text, "html.parser")
            img = soup.find("div", {"class": "js-sticky-symbol-header-container tv-sticky-symbol-header"})
            img_url = img["data-logo-urls"]
            return img_url
        except :
            return ''
       
apptitle = 'Projet hiba'

st.set_page_config(page_title=apptitle, page_icon=":chart_with_upwards_trend:")

# Title the app
st.title('Interface pour industrialisation de la solution')

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data




st.sidebar.markdown("## Sélectionner un indicateur et préciser une valeurs pour ses paramètres  ")
st.markdown('Travail realisé par: Boulhanna Hiba ')
st.markdown("I")
st.markdown('__________________________________________________________')



dropdown = st.sidebar.selectbox("Choisir une action", ['GOOG', 'AAPL'])
indicateur = st.sidebar.selectbox("Choisir un indicateur", ['BB','MACD','MOM','RSI','ROC','SMA','STOCHA','OBV', 'williams', 'EMM', 'MFI', 'CHO', 'NVI', 'PVI'])

start = st.sidebar.date_input('Debut', value =pd.to_datetime('01-01-2020'))
end = st.sidebar.date_input('Fin', value = pd.to_datetime('today'))
# 2017-04-30
start = start.strftime('%Y-%m-%d')
end = end.strftime('%Y-%m-%d')

df = yf.download(dropdown, start=start, end=end)


url = get_logo(dropdown)
url = str(url)

st.markdown('__________________________________________________________')
st.markdown('<center>  '+dropdown+'                <img src="'+url+'" alt="stock logo"></center>', unsafe_allow_html=True)
st.markdown('__________________________________________________________')

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
    n = st.number_input('n', 9)
    nvi = pd.DataFrame(nvi(df.Close, df.Volume), columns = ['NVI'])
    nvi.index = df.index
    data = [df['Close'], nvi['NVI']]
    headers = ['Close', 'NVI'] 
    sign_fig = sign_nvi(df.Close, df.Volume,n) 

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
    sign_fig = sign_mms1(df.Close,period)

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
    sign_fig = sign_mfi(df.Close, df.Volume, df.High, df.Low, period)

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
dtt=get_acp(df)
st.markdown('ACP_Optimisée :')
dtt=get_acp_opt(df)
st.dataframe(dtt)
dtt=get_acp_opt(df)

