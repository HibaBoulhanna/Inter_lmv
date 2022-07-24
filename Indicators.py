import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def pmv(close,signal):
	"""
		La Plus-ou-moins-value
	"""
	close=np.array(close)
	signal=np.array(signal.fillna(0))
	qt=signal.cumsum()
	achat= np.where(signal>=1,1,0)
	PDR=[]
	for t in range(len(close)):
		p=(close[:t+1]*achat[:t+1]).sum()
		if achat[:t+1].sum()==0 :
			val=0
		else :
			p=p/(achat[:t+1].sum())
			val=qt[t]*p 
		PDR.append(val)
	PMV_lat=close*qt-PDR
	PMV_re=[0]
	for i in range(1,len(close)):
		vl=PDR[i]-PDR[i-1]+PMV_re[i-1]-close[i]*signal[i]
		PMV_re.append(vl)
	PMV=PMV_lat+PMV_re
	return PMV

def adjustsignal(signal):
	sig=[]
	qtite=0
	for i in signal:
		if i > 0 :
			sig.append(i)
			qtite+=1
		elif i < 0:
			if qtite >= abs(i) :
				sig.append(i)
				qtite+=-i
			else:
				sig.append(0)
		else:
			sig.append(0)
	return sig

def smm(df,n):
	"""
	Simple Moving Average
	Inputs: 
			Input | Type                             | Description
			=================================================================================
			 df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	                 n    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ================================================================================= 
	               | pandas.DataFrame (2 columns)     | 1st column : the input df
	               |                                  | 2nd column(SMM): values of Moving average

	"""
	SMM = pd.Series(df.rolling(n, min_periods=n).mean(), name='SMM')
	df=pd.DataFrame(df)
	df=df.join(SMM)
	return df

def stochastic(df ,high,low, n , w):
	"""
	Stochastic oscillator : %K and %D
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |                                  | %D periods 
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (3 columns)     | 1st column : the input
	               |                                  | 2nd column(%K): values of %K
	               |                                  | 3rd column(%D): values of %D
	"""
	PH=high.rolling(n, min_periods=n).max()
	PB=low.rolling(n, min_periods=n).min()
	K=pd.Series(100*(df-PB)/(PH-PB), name='%K')
	D=pd.Series(K.rolling(w).mean(), name='%D')
	df=pd.DataFrame(df)
	df=df.join(K)
	df=df.join(D)
	return df

def rate_of_change(df,w):
	"""
	 Rate of change (ROC)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         w    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(ROC): values of ROC
	               |                                  |              

	"""
	ROC=pd.Series(100*(df.diff(w))/df.shift(w), name='ROC')
	df=pd.DataFrame(df)
	df=df.join(ROC)
	return df

def momentum(df,w,wsig=9):
	"""
	Momentum
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | The period 
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (3 columns)     | 1st column: The input
	               |                                  | 2nd column(MOM): Momentums
	               |                                  | 3rd column(MOMsignal) : The signal line 
	"""
	MOM=pd.Series(df.diff(w),name="MOM")
	MOMsignal=pd.Series(MOM.rolling(wsig, min_periods=wsig).mean(), name= "MOMsignal")
	df=pd.DataFrame(df)
	df=df.join(MOM)
	df=df.join(MOMsignal)
	return df

def emm(df,n):
	"""
	Exponential Moving Average
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input df
	               |                                  | 2nd column(EMM): values of Moving average             

	"""
	exp=[df[:n].mean()]
	lamda=2/(1+n)
	for i in range(1,len(df)-n+1):
		val=(1-lamda)*exp[i-1]+lamda*df[i+n-1]
		exp.append(val)
	MME=pd.Series(index=df.index)
	MME[n-1:]=exp 
	MME.name="EMM"
	return MME

def obv(df,vol):
	"""
	On Balance Volume (OBV)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame                  | Prices
	         vol  |pandas.DataFrame                  | Volumes
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(OBV): values of OBV

	"""
	prix=df.diff(1)/np.abs(df.diff(1))
	vec= vol*prix
	vec.iloc[0]=vol.iloc[0]
	OBV=pd.Series(vec.cumsum(), name= 'OBV')
	df=pd.DataFrame(df)
	df=df.join(OBV)
	return df

def williams(df,n):
	"""
	Williams %R
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Periods
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(%R): values of %R
	               |                                  |              
	"""
	PH=df.rolling(n, min_periods=n).max()
	PB=df.rolling(n, min_periods=n).min()
	R=pd.Series(-100*(PH-df)/(PH-PB), name='%R')
	df=pd.DataFrame(df)
	df=df.join(R)
	return df

def MFI(close,volume,high,low,n):
	"""
	 Money Flow Index (MFI)
	Inputs: 
	        Input   | Type                             | Description
	       =========================================================================================
	         close  |pandas.DataFrame or pandas.Series | Prices
	         volume |pandas.DataFrame or pandas.Series | Volumes
	         High   |pandas.DataFrame or pandas.Series | Highest high 
	         low    |pandas.DataFrame or pandas.Series | Lowest low
	         n      |int                               | Periods 
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input (only prices)
	               |                                  | 2nd column(MFI): values of MFI
	"""
	ptyp=(close+high+low)/3
	PMF=[0]
	for i in range(1,len(ptyp)):
		if ptyp[i] > ptyp[i-1]:
			PMF.append(ptyp[i]*volume[i])
		else:
			PMF.append(0)
	PMF=pd.Series(PMF, name=' PMF',index=ptyp.index)
	MF=ptyp*volume
	ratio=pd.Series(100*PMF/MF)
	MFI=pd.Series(ratio.rolling(n,min_periods=n).mean(), name="MFI")
	df=pd.DataFrame(close)
	df=df.join(MFI)
	return df

def cho(close,volume,high,low,n,ws,wl):
	"""
	Chaikin Oscillator
	Inputs: 
	        Input   | Type                             | Description
	       =========================================================================================
	         close  |pandas.DataFrame or pandas.Series | Prices
	         volume |pandas.DataFrame or pandas.Series | Volumes
	         High   |pandas.DataFrame or pandas.Series | Highest high 
	         low    |pandas.DataFrame or pandas.Series | Lowest low
	         n      |int                               | Periods
	         ws     |int                               | The period of the shorter moving average
	         wl     |int                               | The periode if the longer moving average
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input (only prices)
	               |                                  | 2nd column(CHO): values of CHO
	"""
	N=(2*close-low-high)/(high-low)
	adl=N*volume
	ADL=pd.Series(adl.rolling(n,min_periods=n).sum(), name='ADL')
	CHOL=pd.Series(ADL.ewm(ws,min_periods=ws).mean(),name='CHOL')
	CHOH=pd.Series(ADL.ewm(wl,min_periods=wl).mean(),name='CHOH')
	CHO=pd.Series(CHOL-CHOH, name="CHO")
	df=pd.DataFrame(close)
	df=df.join(CHO)
	return df

def nvi(close,volume):
	"""
	Negative Volume Index (NVI)
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.Series                    | NVI 
	"""
	roc=pd.Series(close.diff(1)/close.shift(1), name="ROC")
	nv=[np.nan,roc.iloc[1]]
	for i in range(2,len(volume)):
		if volume[i]< volume[i-1]:
			nv.append(nv[i-1]*(1+roc.iloc[i]))
		else:
			nv.append(nv[i-1])
	NVI=pd.Series(nv,name='NVI')
	return NVI

def pvi(close,volume):
	"""
	  Positive volume index (PVI)
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.Series                    | PVI 
	"""
	roc=pd.Series(close.diff(1)/close.shift(1), name="ROC")
	pv=[np.nan,roc.iloc[1]]
	for i in range(2,len(volume)):
		if volume[i] > volume[i-1]:
			pv.append(pv[i-1]*(1+roc.iloc[i]))
		else:
			pv.append(pv[i-1])
	PVI=pd.Series(pv,name='PVI')
	return PVI

def bollinger(df,w,k):
 	"""
 	Bandes de Bollinger
 	 Paramètre: df: pandas.DataFrame ou pandas.Series: vecteur des prix
 	            w : ordre de la moyenne mobile 
 	            k : 
 	 Retour:  BBDOWN: bande inférieure
 	          BBUP  : bande supérieure
 	          BBMID : bande au milieu
 	"""
 	BBMID=df.rolling(w, min_periods=w).mean()
 	sigma=df.rolling(w, min_periods=w).std()
 	BBUP=BBMID + k*sigma
 	BBDOWN=BBMID - k*sigma
 	BBDOWN=pd.Series(BBDOWN,name='BBDOWN')
 	BBMID=pd.Series(BBMID,name='BBMID')
 	BBUP=pd.Series(BBUP,name='BBUP')
 	df=pd.DataFrame(df)
 	df=df.join(BBDOWN)
 	df=df.join(BBMID)
 	df=df.join(BBUP)
 	df.columns=['COURS_CLOTURE',"BBDOWN","BBMID","BBUP"]
 	return df

def rsi(df,n):
 	"""
 	 Relative Strength index
 	  :Paramètre:
 	   df: pandas.DataFrame
 	   n : ordre
 	  :return:
 	   pandas.DataFrame
 	"""
 	diff=df.diff(1)
 	t=[]
 	for i in diff.values :
 		if i > 0:
 			t.append(i)
 		else :
 			t.append(0)
 	pos=pd.DataFrame(t,index=df.index)
 	diff=np.abs(pd.DataFrame(diff))
 	RSI=pos.rolling(n,min_periods=n).sum()/np.array((diff.rolling(n,min_periods=n).sum()))
 	df=pd.DataFrame(df)
 	df=df.join(RSI)
 	df.columns=["COURS_CLOTURE","RSI"] 
 	return df

def macd(df,ws,wl, wsig=9):
	"""
	Moving Average Convergence Divegence
	  :Paramère:
	    df: 
	    ws: ordre de court terme
	    wl: ordre de long terme
	    wsig: ordre pour le signal line
	  :return:
	   pndas.DataFrame  contient les valeurs des MACD et le Signal line
	"""
	MMECOURT = pd.Series(df.ewm(span=ws, min_periods=ws,adjust=False).mean())
	MMELONG = pd.Series(df.ewm(span=wl, min_periods=wl,adjust=False).mean())
	MACD = pd.Series(MMECOURT - MMELONG, name='MACD' )
	MACDsign = pd.Series(MACD.ewm(wsig, min_periods=wsig).mean(), name='MACDsignal')
	MACD=pd.DataFrame(MACD)
	MACD = MACD.join(MACDsign)
	return MACD

def sign_momentum(df,w,wsig=9):
	MOM=momentum(df,w,wsig=9)[["MOM","MOMsignal"]]
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where(MOM["MOM"][w:] > MOM["MOMsignal"][w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2=fig.add_subplot(312, ylabel='MOM')
	MOM.plot(ax=ax2, legend=True, grid=True)
	plt.title("MOM Strategy")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig

def sign_pvi(close,volume,n):
	"""
	"""
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	pv=pvi(close,volume)
	pvis=pd.Series(pv.rolling(n).mean(), name="PVIsignal")
	signal["compa"][n:]=np.where(pv[n:] > pvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","SMM","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Negative Volume Index')
	pv.plot(ax=ax2,lw=2., legend=True, grid=True)
	pvis.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig 

def sign_rsi(df,n):
	"""
	"""
	signal=pd.DataFrame(index=df.index)
	rs=rsi(df,n)["RSI"]
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(rs[n:] > 0.3,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(rs[n:] < 0.7,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	signal["signal"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal ==-1], 'v', markersize=7, color='r')
	plt.title("RSI Strategy")
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='RSI')
	rs.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(0.7,color="green")
	plt.axhline(0.3,color="red")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig

def sign_bollinger(df,w,k):
	"""
	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa2"]=np.nan
	bb=bollinger(df,w,k)[["BBDOWN","BBUP"]]
	signal["compa"][w:] = np.where( (df[w:] > bb["BBUP"][w:] ) ,1,0)
	signal["compa2"][w:] = np.where( (df[w:] < bb["BBDOWN"][w:] ) ,4,2)
	signal["signal"]=signal["compa"].diff()
	signal["signal2"]=signal["compa2"].diff()
	sig=np.where(signal["signal"]==1,-1,0)+np.where(signal["signal2"]==2,1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	signal["sig"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.fill_between(df.index,bb["BBUP"],bb["BBDOWN"],facecolor='red', alpha=0.2)
	bb.plot(ax=ax1, lw=.5)
	ax1.plot(signal.loc[signal.sig==1].index ,df[signal.sig==1],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.sig==-1].index ,df[signal.sig==-1],'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","BBDOWN","BBUP","Achat",  "Vente"])
	plt.title("Bondes de Bollinger Trading Strategy")
	ax2=fig.add_subplot(212, ylabel='PMV')
	ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig 

def sign_stochastique1(df,high,low ,n , w):
	"""
	"""
	signal=pd.DataFrame(index=df.index)
	k=stochastic(df ,high,low, n , w)["%K"]
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(k[n:] > 20,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(k[n:] < 80,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	signal["signal"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Stochastique')
	k.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig

def sign_roc(df,w):
	"""
	"""
	signal=pd.DataFrame(index=df.index)
	roc=rate_of_change(df,w)["ROC"]
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where( roc[w:]> 0,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Rate of change')
	roc.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig 

def sign_cho(close,volume,high,low,n,ws,wl):
	"""
	"""
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	ch=cho(close,volume,high,low,n,ws,wl)["CHO"]
	signal["compa"][ws+wl:]=np.where(ch[ws+wl:] > 0, 1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","SMM","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Chaikin Oscillator')
	ch.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig 

def sign_macd1(df,ws,wl):
	"""
	"""
	signal=pd.DataFrame(index=df.index)
	MACD=macd(df,ws,wl)["MACD"]
	signal["compa"]=np.nan
	signal["compa"][wl:]=np.where(MACD[wl:]>0 ,1 ,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.title("MACD Strategy")
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD.plot(ax=ax2, color='black', lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig

def sign_nvi(close,volume,n):
	"""
	"""
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	nv=nvi(close,volume)
	nvis=pd.Series(nv.rolling(n).mean(), name="NVIsignal")
	signal["compa"][n:]=np.where(nv[n:] > nvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","SMM","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Negative Volume Index')
	nv.plot(ax=ax2,lw=2., legend=True, grid=True)
	nvis.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig 

def sign_mfi(close,volume,high,low,n):
	"""
	"""
	signal=pd.DataFrame(index=close.index)
	mfi=MFI(close,volume,high,low,n)['MFI']
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(mfi[n:] > 80,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(mfi[n:] < 20,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=close.index)
	signal["signal"]=sig
	pmval=pmv(close,sig)
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='MFI')
	mfi.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig 

def sign_mms1(df,n):
	"""
		R1
	"""
	signal=pd.DataFrame(index=df.index)
	signal['signal']=0
	signal["compa"]=0
	SMM=smm(df,n)["SMM"]
	signal["compa"].loc[n:]=np.where(df[n:] > SMM[n:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='g', lw=.5)
	SMM.plot(ax=ax1, lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","SMM","Achat","Vente"])  
	ax2 = fig.add_subplot(212, ylabel='PMV')
	ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
	return fig