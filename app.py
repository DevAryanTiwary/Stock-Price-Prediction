import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from src.core import load_data, add_features, split, traditional, train_deep, next_close_deep

st.set_page_config(page_title='Stock Price Prediction Lab',page_icon='📈',layout='wide')
st.title('📈 Stock Price Prediction Lab')
st.caption('Compare traditional ML with LSTM/GRU on chronological market data. Educational forecasting only — not investment advice.')

with st.sidebar:
    st.header('Experiment')
    ticker=st.text_input('Ticker','AAPL').strip().upper()
    start=st.date_input('Start date',date.today()-timedelta(days=365*5))
    end=st.date_input('End date',date.today())
    model=st.selectbox('Deep model',['LSTM','GRU'])
    lookback=st.slider('Lookback window',20,120,60,10)
    epochs=st.slider('Epochs',5,60,15,5)
    units=st.select_slider('Hidden units',[32,64,96,128],value=64)
    dropout=st.slider('Dropout',0.0,0.5,0.2,0.05)
    lr=st.select_slider('Learning rate',[0.0003,0.0005,0.001,0.003],value=0.001)
    tune=st.checkbox('Tune Random Forest',True)
    run=st.button('Train & Predict',type='primary',use_container_width=True)

if run:
    try:
        with st.spinner('Downloading, preprocessing and training models...'):
            raw=load_data(ticker,start,end); data=add_features(raw); train,test=split(data)
            trad=traditional(train,test,tune=tune); deep=train_deep(train,test,model,lookback,epochs,units,dropout,lr)
            nxt=next_close_deep(deep,data,lookback)
        latest=float(raw.Close.iloc[-1]); delta=(nxt/latest-1)*100
        a,b,c,d=st.columns(4); a.metric('Latest close',f'${latest:,.2f}'); b.metric('Next-close forecast',f'${nxt:,.2f}',f'{delta:.2f}%'); c.metric('Training rows',len(train)); d.metric('Test rows',len(test))

        tab1,tab2,tab3,tab4=st.tabs(['Market','Predictions','Model Comparison','Training'])
        with tab1:
            fig=go.Figure(data=[go.Candlestick(x=raw.index,open=raw.Open,high=raw.High,low=raw.Low,close=raw.Close,name=ticker)])
            fig.update_layout(title=f'{ticker} price history',xaxis_rangeslider_visible=False,height=520); st.plotly_chart(fig,use_container_width=True)
            st.dataframe(raw.tail(20),use_container_width=True)
        with tab2:
            idx=test.index[-len(deep['pred']):]; fig=go.Figure(); fig.add_scatter(x=idx,y=deep['actual'],name='Actual'); fig.add_scatter(x=idx,y=deep['pred'],name=model)
            fig.update_layout(title=f'Actual vs {model} prediction',height=500,yaxis_title='Price'); st.plotly_chart(fig,use_container_width=True)
        with tab3:
            rows=[]
            for name,r in trad.items(): rows.append({'Model':name,**r['metrics']})
            rows.append({'Model':model,**deep['metrics']}); score=pd.DataFrame(rows).sort_values('RMSE')
            st.dataframe(score.style.format({'MAE':'{:.3f}','RMSE':'{:.3f}','MAPE':'{:.2f}%','R2':'{:.3f}'}),use_container_width=True)
            fig=go.Figure(go.Bar(x=score.Model,y=score.RMSE)); fig.update_layout(title='Model RMSE comparison',yaxis_title='RMSE'); st.plotly_chart(fig,use_container_width=True)
        with tab4:
            hist=pd.DataFrame(deep['history']); st.line_chart(hist[['loss','val_loss']]); st.write({'model':model,'lookback':lookback,'epochs_requested':epochs,'units':units,'dropout':dropout,'learning_rate':lr})
            st.info('Hyperparameters can be adjusted in the sidebar. Random Forest uses time-series cross-validation when tuning is enabled.')
    except Exception as e: st.error(f'Experiment failed: {e}')
else:
    st.info('Choose a ticker and settings, then click **Train & Predict**.')
    st.markdown('''### Included features
- Time-series preprocessing and chronological train/test split
- LSTM and GRU recurrent neural networks
- Linear Regression, Random Forest and Gradient Boosting comparison
- Random Forest hyperparameter tuning with TimeSeriesSplit
- MAE, RMSE, MAPE and R² evaluation
- Candlestick, prediction and training-loss visualizations
- Interactive web prediction interface
''')
