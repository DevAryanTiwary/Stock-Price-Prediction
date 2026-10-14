import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

FEATURES = ['Close','Volume','Return','SMA_10','SMA_20','EMA_10','RSI_14','Volatility_10']

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
    if df.empty: raise ValueError('No market data returned. Check ticker/date range.')
    return df[['Open','High','Low','Close','Volume']].dropna()

def add_features(df):
    x=df.copy(); c=x['Close']
    x['Return']=c.pct_change(); x['SMA_10']=c.rolling(10).mean(); x['SMA_20']=c.rolling(20).mean()
    x['EMA_10']=c.ewm(span=10, adjust=False).mean(); delta=c.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    rs=gain.rolling(14).mean()/loss.rolling(14).mean().replace(0,np.nan); x['RSI_14']=100-(100/(1+rs))
    x['Volatility_10']=x['Return'].rolling(10).std(); x['Target']=c.shift(-1)
    return x.dropna()

def split(df, ratio=.8):
    n=int(len(df)*ratio); return df.iloc[:n].copy(),df.iloc[n:].copy()

def metrics(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':mean_squared_error(y,p)**.5,'MAPE':np.mean(np.abs((np.asarray(y)-np.asarray(p))/np.asarray(y)))*100,'R2':r2_score(y,p)}

def traditional(train,test,tune=False):
    Xtr,ytr=train[FEATURES],train.Target; Xte,yte=test[FEATURES],test.Target
    models={'Linear Regression':LinearRegression(),'Random Forest':RandomForestRegressor(n_estimators=250,max_depth=10,random_state=42,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(random_state=42)}
    if tune:
        cv=TimeSeriesSplit(n_splits=3)
        search=RandomizedSearchCV(RandomForestRegressor(random_state=42,n_jobs=-1),{'n_estimators':[100,200,300,500],'max_depth':[5,10,15,None],'min_samples_split':[2,5,10]},n_iter=8,cv=cv,scoring='neg_mean_squared_error',random_state=42,n_jobs=-1)
        search.fit(Xtr,ytr); models['Tuned Random Forest']=search.best_estimator_
    out={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); out[name]={'model':m,'pred':p,'metrics':metrics(yte,p)}
    return out

def sequences(train,test,lookback=60):
    scaler=MinMaxScaler(); tr=scaler.fit_transform(train[['Close']]); combined=pd.concat([train[['Close']],test[['Close']]])
    all_scaled=scaler.transform(combined); Xtr=[]; ytr=[]
    for i in range(lookback,len(tr)): Xtr.append(tr[i-lookback:i]); ytr.append(tr[i,0])
    Xte=[]; yte=[]; offset=len(train)
    for i in range(offset, len(all_scaled)): Xte.append(all_scaled[i-lookback:i]); yte.append(all_scaled[i,0])
    return np.array(Xtr),np.array(ytr),np.array(Xte),np.array(yte),scaler

def deep_model(kind, input_shape, units=64, dropout=.2, lr=.001):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    Layer=LSTM if kind=='LSTM' else GRU
    m=Sequential([Layer(units,return_sequences=True,input_shape=input_shape),Dropout(dropout),Layer(max(16,units//2)),Dropout(dropout),Dense(16,activation='relu'),Dense(1)])
    m.compile(optimizer=Adam(learning_rate=lr),loss='mse'); return m

def train_deep(train,test,kind='LSTM',lookback=60,epochs=15,units=64,dropout=.2,lr=.001):
    from tensorflow.keras.callbacks import EarlyStopping
    Xtr,ytr,Xte,yte,scaler=sequences(train,test,lookback)
    if len(Xtr)<20 or len(Xte)<2: raise ValueError('Not enough observations for selected lookback.')
    m=deep_model(kind,(lookback,1),units,dropout,lr)
    h=m.fit(Xtr,ytr,validation_split=.15,epochs=epochs,batch_size=32,shuffle=False,verbose=0,callbacks=[EarlyStopping(patience=4,restore_best_weights=True)])
    ps=m.predict(Xte,verbose=0).reshape(-1,1); p=scaler.inverse_transform(ps).ravel(); actual=scaler.inverse_transform(yte.reshape(-1,1)).ravel()
    return {'model':m,'pred':p,'actual':actual,'metrics':metrics(actual,p),'history':h.history,'scaler':scaler}

def next_close_deep(result, full, lookback):
    vals=full[['Close']].tail(lookback); s=result['scaler'].transform(vals).reshape(1,lookback,1)
    pred=result['model'].predict(s,verbose=0).reshape(-1,1); return float(result['scaler'].inverse_transform(pred)[0,0])
