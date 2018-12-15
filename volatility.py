PR = 20
START='2005-02-01'
END='2017-03-27'
SYMBOL="dez.de"
SYMBOL2="eur=x"
from pandas_datareader.data import get_data_yahoo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import convolve
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
def EOM(df, n): 
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM') 
    df = df.join(Eom_ma) 
    return df
def ACCDIST(df, n): 
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume'] 
    M = ad.diff(n - 1) 
    N = ad.shift(n - 1) 
    ROC = M / N 
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC') 
    df = df.join(AD) 
    return df
def ROC(df, n): 
    M = df['Close'].diff(n - 1) 
    N = df['Close'].shift(n - 1) 
    ROC = pd.Series(M / N*100, name = 'ROC') 
    df = df.join(ROC) 
    return df
def STDDEV(df, n): 
    df = df.join(pd.Series(pd.rolling_std(df['ROC'], n)*15.87450786638754, name = 'STD')) 
    return df

datahighlowclosevolume = get_data_yahoo(SYMBOL, start = START, end = END)[['Close','Volume','High','Low']]
"""datahighlowclosevolume2 = get_data_yahoo(SYMBOL2, start = START, end = END)[['Close','Volume','High','Low']]
datahighlowclosevolume['Close']=datahighlowclosevolume['Close']/datahighlowclosevolume2['Close']"""
sma = movingaverage(np.asarray(datahighlowclosevolume['Close']), 20)
beginningzeros=np.repeat(0, 20-1)
fullsma = np.concatenate([beginningzeros,sma])
acc = ACCDIST((datahighlowclosevolume), PR)
datahighlowclosevolume['sma']=fullsma
roc=ROC((datahighlowclosevolume), PR)
rocdaily=ROC((datahighlowclosevolume), 2)
std=STDDEV((rocdaily), PR)
plt.figure(1,figsize=(16, 24))
plt.subplot(211)
plt.plot(datahighlowclosevolume['Close']);
plt.plot(datahighlowclosevolume['sma']);
plt.ylim(datahighlowclosevolume['Close'].min()*0.98,datahighlowclosevolume['Close'].max()*1.02)

plt.figure(2,figsize=(16, 14))
plt.subplot(211) 
y_pos = np.arange(len(datahighlowclosevolume['Volume']))
plt.bar(y_pos, datahighlowclosevolume['Volume'], align='center', alpha=0.5)
plt.ylim(acc['Volume'].quantile(.05),acc['Volume'].quantile(.95))    
'''plt.plot(datahighlowclosevolume['Volume'])'''
plt.subplot(212)       
plt.plot(acc['Acc/Dist_ROC'])
plt.ylim(acc['Acc/Dist_ROC'].quantile(.05),acc['Acc/Dist_ROC'].quantile(.95))

plt.figure(3,figsize=(16, 14))
plt.subplot(211)
plt.plot(roc['ROC'])
plt.ylim(roc['ROC'].quantile(.05),roc['ROC'].quantile(.95))
plt.axhline(roc['ROC'].mean())
rocquantp=list(filter(lambda x: x < roc['ROC'].quantile(.05), roc['ROC']))
rocquantn=list(filter(lambda x: x > roc['ROC'].quantile(.95), roc['ROC']))
plt.subplot(212)
plt.plot(std['STD'])
plt.axhline(std['STD'].quantile(.25))
plt.axhline(std['STD'].quantile(.50))
plt.axhline(std['STD'].quantile(.75))
plt.ylim(std['STD'].quantile(.05),std['STD'].quantile(.95))
