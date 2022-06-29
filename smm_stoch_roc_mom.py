import pandas as pd
import numpy as np 



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
	MMS = pd.Series(df.rolling(n, min_periods=n).mean(), name='SMM')
	df=pd.DataFrame(df)
	df=df.join(MMS)
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


