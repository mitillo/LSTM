# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:11:30 2017

@author: jaime
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
from pandas import datetime
import pickle
import math, time,itertools
#import itertools
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from functools import reduce
import time
#import keras
import pandas_datareader.data as web
#import h5py

dicc_directory='C:\\Users\\jaime\\MitoProyect\\Trading\\Variables\\Daily\\objetos\\dic_full.tkl'
diccionario_full=pickle.load(open(dicc_directory,'rb'))

#%% get data from diferent sources
def get_stock_data(stock_name,use_dates=True,inicio='2013',final='2017', normalize=True):
    
#    start = datetime.datetime(2010, 1, 1)
#    end = datetime.date.today()
    
    df=pd.read_csv(stock_name,names=['Date','Open','High','Low','Close'],usecols=[0,2,3,4,5],index_col=[0])
    if use_dates== True:
        df=df.loc[inicio:final]
#    print(df.head(),df.tail())
#    df = web.DataReader(stock_name, "yahoo", start, end)
#    df.drop(['Date'], 1, inplace=True)
    df.reset_index(inplace=True,drop=True)
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df



def year_data(stock_name,use_dates=True,inicio='2013',final='2017', normalize=True):
    
#    start = datetime.datetime(2010, 1, 1)
#    end = datetime.date.today()
    #esta formula la uso para cuando tenga datos como formato EURUSD_2017.csv
#    stock_name='EURUSD_2017.csv'
    df=pd.read_csv(stock_name,usecols=[1,2,3,4,5],header=0,index_col=[0])
    if use_dates== True:
        df=df.loc[inicio:final]
#    print(df.head(),df.tail())
#    df = web.DataReader(stock_name, "yahoo", start, end)
#    df.drop(['Date'], 1, inplace=True)
    df.reset_index(inplace=True,drop=True)
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df


def load_data(stock, seq_len,train_split=0.9):
#    stock=df
#    type(stock)
#    type(data)
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0 esto se hace ya que el indice empieza por 0 entonces quedaria
    #de 0 a 21 mientras que con 22+1 terminaria con 22
    result = []
    # TOMA SEQ len +1 proque el valor +1 es el que toma como y_test es decir en seq_length=23, 22 son train y el ultimo es test
    # la resta se hace para tomar los grupos bien y no llegar con problemas al final.
    # en nuestro caso la long que queremos predecir es len=22 por ello la la length de result es 22+1
    # sinos fijamos luego en X_train y_train lo que hace transformar result a train y posterirmente toma
#    X_train=[:,:-1] o lo que es lo mismo todos menos el ultimo y_train[:,-1] solo el ultimo
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days 
    result = np.array(result)
    row = round(train_split * result.shape[0]) # 90% split
    
    train = result[:int(row), :] # 90% date
    X_train = train[:, :-1] # all data until day m
    y_train = train[:, -1][:,-1] # day m + 1 adjusted close price OJO HACE ESTO DEBIDO A QUE TENEMOS VARIAS FEATURES estar al tanto
#    que siempre en las features tenemos que tener one dimension array
    
#    y_train.shape

    
    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1] 

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    return [X_train, y_train, X_test, y_test]

def dic_year_data(divisa,diccio,use_dates=False,inicio='2013',final='2017', normalize=True):
    df=diccio[divisa][['Open','High','Low','Close']].loc[final]
    print(df.iloc[-1])
    if use_dates== True:
        df=df.loc[inicio:final]
#    print(df.head(),df.tail())
#    df = web.DataReader(stock_name, "yahoo", start, end)
#    df.drop(['Date'], 1, inplace=True)
    df.reset_index(inplace=True,drop=True)
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df

#%%Building models


def build_model2(layers, neurons, d):
#    layers=shape = [4, seq_len, 1] # feature, window, output
#    neuros=neurons = [128, 128, 32, 1]
    
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.summary()
    return model



def build_model4(layers, neurons, d,optimizador):
#    layers=shape = [4, seq_len, 1] # feature, window, output
#    neuros=neurons = [128, 128, 32, 1]
    
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    
    model.add(Dense(4,kernel_initializer="uniform",activation='linear'))
    
    
    model.compile(loss='mse',optimizer=optimizador, metrics=['accuracy'])
    
    
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.summary()
    start=time.time()
    print("> Compilation Time : ", time.time() - start)
    return model


def build_model5(layers, neurons, d,optimizador):
#    layers=shape = [4, seq_len, 1] # feature, window, output
#    neuros=neurons = [128, 128, 32, 1]
#    clave es poner 
    model = Sequential()
    
    model.add(LSTM(250, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(70, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))  
      
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    
    
    model.add(Dense(layers[0],kernel_initializer="uniform",activation='linear'))
    
    model.compile(loss='mse',optimizer=optimizador, metrics=['accuracy'])
    
    
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.summary()
    start=time.time()
    print("> Compilation Time : ", time.time() - start)
    return model


def build_model6(layers, neurons, d,optimizador):
#    layers=shape = [4, seq_len, 1] # feature, window, output
#    neuros=neurons = [128, 128, 32, 1]
#    clave es poner 
    model = Sequential()
    # el basico en vez de 250 es neurons[0]
    model.add(LSTM(250, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(neurons[2], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))  
      
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    
    
    model.add(Dense(layers[0],kernel_initializer="uniform",activation='linear'))
    
    model.compile(loss='mse',optimizer=optimizador, metrics=['accuracy'])
    
    
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.summary()
    start=time.time()
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    
    prediction_seqs = []
#    data=df_predictions
##    len(X_test)
#    window_size=22
#    prediction_len=22
#    model=model4
#    data.shape
    # Lo que hacemos inicialmente es hacer grupos de len(data)/ la longitud de prediccion que uqeremos
    # tener en cuenta que partidmos de un array de 3 d por lo tanto cambiamos de este a un array de 2d
    #ene ste caso seria len(data=756)/prediction_len=22 COMO RESULTADO nos da #####34####
# =============================================================================
#     1º Bloque vemos los grupos que poademos hacer y extraemos los grupos pasando de 3d a 2d
    #data=3d                  curr_frame=2d
    for i in range(int(len(data)/prediction_len)):
#        i=1
        curr_frame = data[i*prediction_len]
        print(curr_frame)
        print(curr_frame.shape)
        print(len(curr_frame))
        predicted = []
# =============================================================================
        # 2º bloque pasamos de 2d a 3 d y hacemos predicciones
        #aqui cambiamos el array de 2 a uno de 3 d para llevar la prediccion
        for j in range(prediction_len):
#            j=1
#            x=model.predict(curr_frame) si lo ponemos asi no funciona ya que el modelo
#            espera un array de 3 dimensiones y le damos solo 2 por ello usamos 
#            print(j)
            curr_frame2=curr_frame[np.newaxis,:,:] # creamos curr_frame3 dimension
            print(curr_frame2.shape)
            print(curr_frame2)
            x=model.predict(curr_frame2)
            print(x.shape) # vemos que de un input de 3d (curr frame)
            print(x)
#           obtenemos un output 2d y solo queremos el valor inside 
            x=x[0,0] # extraemos valor inside)
            predicted.append(x) # añadimos valor inside a la lista de predicciones
            
#            Aqui empezamos con actualizacion de curr_frame
            
#            ==========================================
#           3º bloque quitamos el 1º valor de la lista y metemos el valor predicho en el ultimo lugar                 
            curr_frame = curr_frame[1:]  # lo que hacemos es selecciones todos menos el primero
#            haciendo esto se nos que da un curr_frame de len=49
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0) 
#            En el alinea de arriba lo que hacemos es basicamente insertar el ultimo valor predicho (predicted[-1])
#            en el ultimo lugar del array y de esa forma volvemos a obtener un array de len 50 y seguimos prediciendo
            
#            len(curr_frame)
        prediction_seqs.append(predicted)
#        len(prediction_seqs[0])
    return prediction_seqs




def denormalize(stock_name, normalized_value):
#    start = datetime.datetime(2000, 1, 1)
#    end = datetime.date.today()
    # Tener en cuenta que me confundi mucho al llevar a cabo esto ya que denormalize sobre valores que ya estaban normalizados es decir
    #use invert_transform usando como base una columna normalizada y no una columna normal (sin normalizar) por ello he creado df_denormalize
    # asi y todo nose si deberia inverse transform por columna es deicr fit_transform('open')/ denormalize(open) y asi sucesivamente
#    stock_name = df_denormalize
#    normalized_value=y_test
  ############################################################################  
#    stock_name=model_data_denorm
#    normalized_value=predictions
#    normalized_value[:,-1].reshape(-1,1)
#    df2 = stock_name.values
#    normalized_value =normalized_value
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df2)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new




def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
    normalized_value_p=predictions
    normalized_value_y_test=y_test
    stock_name=model_data_denorm
    
    newp = denormalize(stock_name, normalized_value_p)
    newp_close=newp[:,-1].reshape(-1,1)
    newy_test = denormalize(stock_name, normalized_value_y_test)
    newy_close= newy_test[:,-1].reshape(-1,1)
    plt2.figure('pdiff_ytest')
    plt2.plot(newp_close, color='red', label='Prediction')
    plt2.plot(newy_close,color='blue', label='Actual')
    plt2.legend(loc='best')
    plt2.title('The test result for {}'.format(stock_name))
    plt2.xlabel('Days')
    plt2.ylabel('Adjusted Close')
    plt2.show()


# cogida del otro modelo albertos brother
def plot_results_multiple(predicted_data, true_data, prediction_len):
#    predicted_data=prediction_seqs
#    true_data=y_test
#    prediction_len=22
    fig = plt.figure('multiple_result',facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        print(i)
        print(data)
        padding = [None for p in range(i * prediction_len)]
        print('Padding%n',padding,'%n')
#        print(padding)
        print('Padding %n {}'.format(padding+data))
        plt.plot(padding + data, label='Prediction')
#        plt.title(divisa)
        plt.legend()
    plt.show()
    



def plotting_prediction_plus(divisa,diccionario_full,model4,num_pred,seq_len,final='2017',title='prueba'):
    
    datos_predecir_norm=dic_year_data(divisa,diccionario_full,final=final) 
    vertical_line=datos_predecir_norm.iloc[seq_len].name
    vertical_line2=datos_predecir_norm.iloc[-seq_len].name
    datos_predecir_denorm=dic_year_data(divisa,diccionario_full,normalize=False,final=final) 
    indice=diccionario_full[divisa].loc[final].index
    indice=indice.strftime('%Y-%m-%d')
    
    
    #datos_predecir_norm=datos_predecir_norm.iloc[:100] para usar los ultimos 100 valores
    
    datos_predecir=load_predict_data4(datos_predecir_norm,seq_len,train_split=1) # Ojo solo tengo X_test 
#    len(datos_predecir)
#    len(datos_predecir_norm)

    predicciones_yearly=model4.predict(datos_predecir)
#    len(predicciones_yearly)
#    len(datos_predecir_norm)

    
    #%%
    #Adicion de ultimas predicciones y addicion de 5 valores mas a los valores actuales
    
    
    
    last_data_predict=datos_predecir[-1] # tomo el ultimo valor de last_data predcit, shape(2dimension )

    
    last_data_predict=np.reshape(last_data_predict,(1,last_data_predict.shape[0],last_data_predict.shape[1])) # el 1 me sirve para añadir una dimesion a los datos
    #y lo paso de 2 a 3 dimensiones para meterlo en modelo
    
    lista_datos_predecir=[]
    for i in range(num_pred):
        predicciones_last_data=model4.predict(last_data_predict)
        
        last_data_predict_plus1=last_data_predict[:,1:] #cojo todos los valores menos el 1º quedandon con una len de seq_len-1 
    #    print(last_data_predict_plus1)
        last_data_predict_plus1=np.append(last_data_predict_plus1,[predicciones_last_data],axis=1) #añado el valor predicho al final
        #obteniendo len22 con un shape de 3d (1,22,4) y con esto vuelvo 
        last_data_predict=last_data_predict_plus1 # igualo para que con ello mi nueva matrix de len 22 a la cual he añadido mi valor
        #cuyo inicio es xsub0+1 y cuyo final es xt+valorpredicho
    #    lista_datos_predecir.append(predicciones_last_data) #creo una lista con los valores predichos
        lista_datos_predecir.append(predicciones_last_data[-1,:].tolist())# lo hago para facilitarme a la hora de pasarlo a lista
    
    df_predictions_yearly=pd.DataFrame(predicciones_yearly)
    list_predict=pd.DataFrame(lista_datos_predecir[1:])
    
    df_both=pd.concat([df_predictions_yearly,list_predict],ignore_index=True)
    
    
    #%%plottting results
    
    Cierres_pred_plus=df_both[3]
#    print(len(Cierres_pred_plus))
#    print(len(datos_predecir_norm))
    
    Cierres_pred_yearly=predicciones_yearly[:,-1]
    rango_nulo=[None for x in range(seq_len)]
    rango_nulo=np.array(rango_nulo)
    Cierres_pred_yearly=np.append(rango_nulo,Cierres_pred_yearly)
    Cierres_pred_plus=np.append(rango_nulo,Cierres_pred_plus)
    fig,ax1=plt.subplots(1,1)
    
#    ax1.plot(predicciones_yearly)
    #plt.plot(Cierres_pred_yearly)
    ax1.plot(datos_predecir_norm['Close'],color='green',label='normales_full',linewidth=2.5)
    ax1.plot(datos_predecir_norm['Close'].iloc[:-20],color='magenta',label='normales-20')
    ax1.plot(Cierres_pred_yearly,color='blue',label='pred_yearly',linewidth=6)
    ax1.plot(Cierres_pred_plus,color='grey',label='pred_yearly_plus',linewidth=4,linestyle='--')
    ax1.axvline(vertical_line,color='mediumblue')
    ax1.axvline(vertical_line2,color='mediumblue')
    ax1.axvline(Cierres_pred_yearly.shape[0],color='mediumblue')
#    ax1.axvline(180,color='green')
    plt.title(divisa)
    # getting xaxes
    tickers=ax1.get_xticks()
    lista_tickers=tickers.tolist()
    lista_tickers=list(filter(lambda x: x>=0,lista_tickers))
    lista_tickers_set=set(lista_tickers)
    set_indice=set(range(len(indice)))
    datos_no_indice=lista_tickers_set-set_indice
    label_ticker=[]
    
    for value in lista_tickers:
        if value not in datos_no_indice:
#            print(value)
            label_ticker.append(indice[int(value)]) # ojo siempre poner int a veces parece que lo tienes pero en la lista hay float
        else:
            label_ticker.append(value)
    
    ax1.set_xticks(lista_tickers)
    ax1.set_xticklabels(label_ticker)
    plt.legend()
    plt.show()
    fig.show()
    
    # Calculo %difference
    def_per_change=pd.DataFrame(Cierres_pred_yearly[22:])
    
    df_datos_predecir_norm=pd.DataFrame(datos_predecir_norm[22:])
    df_datos_predecir_norm.reset_index(inplace=True,drop=True)
    df_datos_predecir_norm['predicted']=def_per_change
    df_calulo_per_change=df_datos_predecir_norm[['Close','predicted']]
    df_calulo_per_change['change']=(((df_calulo_per_change['predicted']-df_calulo_per_change['Close'])/df_calulo_per_change['predicted'])*100)






def model_score(model, X_train, y_train, X_test, y_test):
    
#    X_test,y_test=test_data_X,test_data_y
    
    trainScore = model.evaluate(X_train, y_train, verbose=1)
    print(' /n Train Score\n  MSE= %.5f\n RMSE= %.2f  \n' % (trainScore[0], sqrt(trainScore[0]))) #[0] MSE  [1] RMSE
    trainScore_mse_rmse=[trainScore[0],sqrt(trainScore[0])] #lista con valores train MSE and RMSE
    
    testScore = model.evaluate(X_test, y_test, verbose=1) #lista con valores test MSE and RMSE
    print(' /n Test Score \n  MSE= %.5f \n RMSE= %.2f  \n' % (testScore[0], sqrt(testScore[0])))
    
    testScore_mse_rmse=[testScore[0],sqrt(testScore[0])]
    return trainScore[0], testScore[0],trainScore_mse_rmse,testScore_mse_rmse


def percentage_difference(model, X_test, y_test):
    
 
    percentage_diff=[]

    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u

        percentage_diff.append((pr-y_test[u]/pr)*100)
    return p

def predictions_per_diff4(model, X_test, y_test):
    # OJO ESTA FORMULA NOS DA PREDICCIONES A PESAR DE SU NOMBRE,
#    model=model5
#    X_test=test_data_X
#    y_test=test_data_y
    percentage_diff=[]

    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u
        pr2=(pr-y_test[u]/pr)*100
        pr2=pr2.mean()
        percentage_diff.append(pr2)
#    print(percentage_diff)
    percentage_diff2=reduce(lambda x,y: x+y,percentage_diff)
    percentage_diff2=percentage_diff2/len(percentage_diff)
    print('#############################\n')
    print('Percengate difference: {}'.format(percentage_diff2))
    print('\n#############################\n')
    return p,percentage_diff2



def data_predict4(file='EURUSD_2017.csv',normalise=True):
    df=pd.read_csv(file,usecols=[1,2,3,4,5],index_col=[0])
    df.reset_index(inplace=True,drop=True)  
    if normalise==True:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df



def load_model_data4(stock, seq_len,train_split=0.9):
#    stock=predict_data_norm
#    train_split=0.9
#    type(stock)
#    type(data)
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0 esto se hace ya que el indice empieza por 0 entonces quedaria
    #de 0 a 21 mientras que con 22+1 terminaria con 22
    result = []
    # TOMA SEQ len +1 proque el valor +1 es el que toma como y_test es decir en seq_length=23, 22 son train y el ultimo es test
    # la resta se hace para tomar los grupos bien y no llegar con problemas al final.
    # en nuestro caso la long que queremos predecir es len=22 por ello la la length de result es 22+1
    # sinos fijamos luego en X_train y_train lo que hace transformar result a train y posterirmente toma
#    X_train=[:,:-1] o lo que es lo mismo todos menos el ultimo y_train[:,-1] solo el ultimo
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days 
    result = np.array(result)
    
    row = round(train_split * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date
    X_train = train[:, :-1] # all data until day m
    y_train = train[:, -1] # day m + 1 adjusted close price OJO HACE ESTO DEBIDO A QUE TENEMOS VARIAS FEATURES estar al tanto
#    que siempre en las features tenemos que tener one dimension array
    
#    y_train.shape

    
    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1] 

#    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
#    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    return [X_train, y_train, X_test, y_test]



def load_predict_data4(stock, seq_len,train_split=0.9):
    #This formula is used to get predict data outside the model and get the proper format to model.predict
    
#    stock=datos_predecir_norm
#    seq_len=22
#    train_split=0.9
#    type(stock)
#    type(data)
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    len(data)
#    sequence_length = seq_len + 1 # index starting from 0 esto se hace ya que el indice empieza por 0 entonces quedaria
    sequence_length = seq_len  # index starting from 0 esto se hace ya que el indice empieza por 0 entonces quedaria
    #de 0 a 21 mientras que con 22+1 terminaria con 22
    result = []
#    result2=[]
#    for index in range(len(data)): # maxmimum date = lastest date - sequence length
#        result2.append(data[index: index + sequence_length]) # index : index + 22days 
#    result2 = np.array(result2)
#    result2
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days 
    result = np.array(result)    
    X_train=result
    len(X_train[0])
    len(X_train)
    len(X_train)
#    len(X_train[0])
    y_test=result[:,-1]
#    '''Mejor no quitar nada y dejarlo ya que si lo quito, lso datos se me retrasan ya que realmente el test forma
#    parte del predict y si vemos van a la misma altura tanto el test como el predict, si lo quito el +1 el predict
#    se me queda retrasado'''

    return X_train
            


def plotting_history(history,epochs,divisa='eurusdd1'):
#    history=history3
#    directorio='optimizadores/Adam'
#    fig=plt.figure()
    plt.figure()
    plt.title('model_train vs val_loss')
#    plt.title('%s epoch= %s' % (title_optimizador,epochs))
    plt.plot(history.history['val_loss'])
    plt.xticks(np.arange(len(history.history['loss'])))
    #    plt.xticks(np.arange(len([history.history['loss']))
    #    plt.plot(history.history['acc'])  
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper right')
    plt.plot(history.history['loss'])

    plt.show()
    
    
    
    
    
#%% FORMAULAS OPTIMIZACION
    
    
def quick_measure( stock_name,seq_len, dropout, shape, neurons, epochs):
    df = get_stock_data(stock_name,inicio='2013', normalize=True)
    df_denormalize = get_stock_data(stock_name,inicio='2013',normalize=False)
    
    X_train, y_train, X_test, y_test = load_data(df, seq_len)
    model = build_model4(shape, neurons, dropout)
    history=model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)
    plotting_history(history,epochs)
    #    plt.savefig('model_param/mtrain_vs_validationloss.png',bbox_inches= 'tight')
    #    predictions2=predict_sequences_multiple(model,X_test,seq_len,seq_len) # da una list of list de una len[i]=prediction_len, en este caso 22
    #    plt.figure('multipleresult')
    #    plot_results_multiple(predictions2,y_test,22)
    model_score(model, X_train, y_train, X_test, y_test)
    
    predictions,per_diff = predictions_per_diff4(model, X_test, y_test)
#    plot_result(df_denormalize, predictions, y_test)
    # model.save('LSTM_Stock_prediction-20170429.h5')
    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)
    return trainScore, testScore,history

def quick_measure2(stock_name, seq_len, d, shape, neurons, epochs, decay):
    df = get_stock_data(stock_name)
    df_denormalize = get_stock_data(stock_name,inicio='2013',normalize=False)
    X_train, y_train, X_test, y_test = load_data(df, seq_len)
    model = build_model4(shape, neurons, d, decay)
    history=model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)
#    
    plotting_history(history,epochs)
#    plt.savefig('model_param/mtrain_vs_validationloss.png',bbox_inches= 'tight')
#    predictions2=predict_sequences_multiple(model,X_test,seq_len,seq_len) # da una list of list de una len[i]=prediction_len, en este caso 22
#    plt.figure('multipleresult')
#    plot_results_multiple(predictions2,y_test,22)
    model_score(model, X_train, y_train, X_test, y_test)
    predictions, per_diff = predictions_per_diff4(model, X_test, y_test)
#    plot_result(df_denormalize, p, y_test)
    # model.save('LSTM_Stock_prediction-20170429.h5')
    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)
    return trainScore, testScore,history
    

def quick_measure4( stock_name,seq_len, dropout, shape, neurons, epochs,optimizador,divisa_test,fecha_inicio_modelo='2010',fecha_final_modelo='2016'):
    
    model_data_norm=get_stock_data(stock_name,inicio=fecha_inicio_modelo,final=fecha_final_modelo,use_dates=True)
    model_data_denorm=get_stock_data(stock_name,inicio=fecha_inicio_modelo,final=fecha_final_modelo,use_dates=True, normalize=False)
    X_train, y_train, X_test, y_test = load_model_data4(model_data_norm, seq_len,train_split=1)
    
    
    test_data_norm=year_data(divisa_test,inicio='2015',final='2017',use_dates=False)
    test_data_denorm=year_data(divisa_test,inicio='2015',final='2017',use_dates=False,normalize=False)
    train_data_X,train_data_y,test_data_X,test_data_y=load_model_data4(test_data_norm,seq_len, train_split=0.0) #estos datos los uos pa test model
    
    model5=build_model4(shape,neurons,dropout,optimizador='adam') #en un futuro esto tendre que modificarlo para optimizar los activadores
    history=model5.fit(X_train,y_train,batch_size=512,epochs=epochs,validation_split=0.1,verbose=1, shuffle=False)
    
#    plotting_history(history,epochs)
    
    predictions,per_diff=predictions_per_diff4(model5,test_data_X,test_data_y)
    
    trainScore, testScore = model_score(model5, X_train, y_train, test_data_X, test_data_y)
    return trainScore, testScore,history,per_diff,model5

def quick_measure5( formula_modelo,divisa,stock_name,seq_len, dropout, shape, neurons, epochs,optimizador,divisa_test,fecha_inicio_modelo='2000',fecha_final_modelo='2016'):
#    formula_modelo=build_model6
    
    
    model_data_norm=get_stock_data(stock_name,inicio=fecha_inicio_modelo,final=fecha_final_modelo,use_dates=True)
    model_data_denorm=get_stock_data(stock_name,inicio=fecha_inicio_modelo,final=fecha_final_modelo,use_dates=True, normalize=False)
    X_train, y_train, X_test, y_test = load_model_data4(model_data_norm, seq_len,train_split=1)
    
    
    test_data_norm=dic_year_data(divisa,diccionario_full) 
    test_data_denorm=dic_year_data(divisa,diccionario_full,normalize=False)
    train_data_X,train_data_y,test_data_X,test_data_y=load_model_data4(test_data_norm,seq_len, train_split=0.0) #estos datos los uos pa test model
    
    model5=formula_modelo(shape,neurons,dropout,optimizador) #en un futuro esto tendre que modificarlo para optimizar los activadores
    history=model5.fit(X_train,y_train,batch_size=512,epochs=epochs,validation_split=0.1,verbose=1,shuffle=False)
    
#    plotting_history(history,epochs)
    
    predictions,per_diff=predictions_per_diff4(model5,test_data_X,test_data_y)
    
    trainScore, testScore,trainScore_mse_rmse,testScore_mse_rmse = model_score(model5, X_train, y_train, test_data_X, test_data_y)
    return trainScore, testScore,history,per_diff,model5,trainScore_mse_rmse,testScore_mse_rmse




def get_better_value(diccio_result):
#    diccio_result=dropout_result
    
    min_val=min(diccio_result.values())
    min_val_key= [k for k ,v in diccio_result.items() if v==min_val]
    print(min_val_key)
    print(diccio_result[min_val_key[0]])
#    diccio_result[0.3]
    lists = sorted(diccio_result.items())
    x,y = zip(*lists)
    plt.plot(x,y)
    plt.title('Finding the best hyperparameter')
    plt.xlabel('Dropout')
    plt.ylabel('Mean Square Error')
    plt.show()
    return(diccio_result,min_val_key[0])