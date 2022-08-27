from operator import index
from tkinter import E
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split

#cointainers creation
title=st.container()
data_cleaning=st.container()
model_prep=st.container()
user_prediction=st.container()


with title:
    st.title('Heart Disease Prediction App')
    st.text('Dataset: Heart Disease took  from UCI Machine Learning Repository')
    st.header('Basic Info about the Dataset')
    df=pd.read_csv('heart.csv')
    st.write('First 5 rows of the dataset')
    st.write(df.head())
    st.write('Shape of the dataset')
    st.write(df.shape)
    st.write('Summary of the dataset')
    st.write(df.describe())
    st.write('Data Types')
    st.write('Total number of Columns:',len(df.columns))
    a=df['HeartDisease'].unique()
with data_cleaning:
    df=df.drop([109, 241, 365, 399, 449, 592, 732, 759],axis=0)
    df=df.drop([76,149,616],axis=0)
    df=pd.get_dummies(df,columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],sparse=False)
    st.header(' Data Set after Encoding')
    st.write(df.head(10))
with model_prep:
    from sklearn.model_selection import train_test_split
    y=df['HeartDisease']
    X=df.drop('HeartDisease',axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    # SVM
    
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import sklearn.metrics as metrics
    st.markdown(''' # SVM model Results ''')
    svm=SVC(kernel='linear')
    svm.fit(X_train_scaled,y_train)
    predi_svm=svm.predict(X_test_scaled)
    st.write('Accuracy:',accuracy_score(predi_svm,y_test))

    #K nearest neighbours
    st.header('K-Nearest Neighbours')
    st.markdown(''' # Recommended Value of K is 20 ''')
    value_k=st.slider('Select Neighbours For KNN',min_value=1,max_value=30,step=1,value=5)
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=value_k)
    knn.fit(X_train_scaled,y_train)
    predi_knn=knn.predict(X_test_scaled)
    st.write('Accuracy:',accuracy_score(predi_knn,y_test))
    st.write('Classification report ',metrics.classification_report(y_test,predi_knn))

with user_prediction:
    age=st.number_input('Enter Age',min_value=15,max_value=100)
    st.subheader('Please use M for Male and F for Female')
    sex=st.text_input('Enter Gender',max_chars=1)
    chest_pain=st.text_input('''Enter Chest Pain type  USE "ATA NAP ASY TA" ONLY ''',max_chars=3)
    resting_bp=st.number_input('Enter Resting BP',min_value=0,max_value=200)
    st.write('Use   1 for Yes if you have taken Blood Sugar Test and "0" for No')
    cholesterol=st.number_input('Enter Cholesterol Level',min_value=0,max_value=600)
    fasting_bs=st.number_input('Enter Fasting Blood Sugar test',min_value=0,max_value=1)
    resting_ecg=st.text_input('''Enter Resting ECG  Result  USE "Normal ST and LVH" ONLY ''',max_chars=6)
    max_hr=st.number_input('Enter Maximum Heart Rate',min_value=60,max_value=202)
    st.subheader('''Please  input Exercise Angina use  'N' for No and 'Y' for Yes''')
    exercise_angina=st.text_input('',max_chars=1)
    old_peak=st.number_input('Enter old peak value between -2 to 6',min_value=-2.6,max_value=6.0,value=0.0)
    st_slope=st.text_input('''Enter ST Slope  USE "Up Flat Down" ONLY ''',max_chars=4)
    Age=age
    RestingBP=resting_bp
    Cholesterol=cholesterol
    FastingBS=fasting_bs
    MaxHR=max_hr
    Oldpeak=old_peak
    Sex_F=0
    Sex_M=0
    ChestPainType_ASY=0
    ChestPainType_ATA=0
    ChestPainType_NAP=0
    ChestPainType_TA=0
    RestingECG_LVH=0
    RestingECG_Normal=0
    RestingECG_ST=0
    ExerciseAngina_N=0
    ExerciseAngina_Y=0
    ST_Slope_Down=0
    ST_Slope_Flat=0
    ST_Slope_Up=0
    def if_checker():
        Age=age
        RestingBP=resting_bp
        Cholesterol=cholesterol
        FastingBS=fasting_bs
        MaxHR=max_hr
        Oldpeak=old_peak
        Sex_F=0
        Sex_M=0
        ChestPainType_ASY=0
        ChestPainType_ATA=0
        ChestPainType_NAP=0
        ChestPainType_TA=0
        RestingECG_LVH=0
        RestingECG_Normal=0
        RestingECG_ST=0
        ExerciseAngina_N=0
        ExerciseAngina_Y=0
        ST_Slope_Down=0
        ST_Slope_Flat=0
        ST_Slope_Up=0
        if sex=='F':
            Sex_F=1
        elif sex=='M':
            Sex_M=1
        if chest_pain=='ASY':
            ChestPainType_ASY=1
        elif chest_pain=='ATA':
            ChestPainType_ATA=1
        elif chest_pain=='NAP':
            ChestPainType_NAP=1
        elif chest_pain=='TA':
            ChestPainType_TA=1
        if resting_ecg=='Normal':
            RestingECG_Normal=1
        elif resting_ecg=='ST':
            RestingECG_ST=1
        elif resting_ecg=='LVH':
            RestingECG_LVH=1
        if exercise_angina=='N':
            ExerciseAngina_N=1
        elif exercise_angina=='Y':
            ExerciseAngina_Y=1
        if st_slope=='Down':
            ST_Slope_Down=1
        elif st_slope=='Flat':
            ST_Slope_Flat=1
        elif st_slope=='Up':
            ST_Slope_Up=1
        dict_instance={'Age':age, 'RestingBP':resting_bp, 'Cholesterol':cholesterol, 'FastingBS':fasting_bs, 'MaxHR':max_hr, 'Oldpeak':old_peak, 'Sex_F':Sex_F,
       'Sex_M':Sex_M, 'ChestPainType_ASY':ChestPainType_ASY, 'ChestPainType_ATA':ChestPainType_ATA, 'ChestPainType_NAP':ChestPainType_NAP,
       'ChestPainType_TA':ChestPainType_TA, 'RestingECG_LVH':RestingECG_LVH, 'RestingECG_Normal':RestingECG_Normal,
       'RestingECG_ST':RestingECG_ST, 'ExerciseAngina_N':ExerciseAngina_N,'ExerciseAngina_Y':ExerciseAngina_Y,
       'ST_Slope_Down':ST_Slope_Down, 'ST_Slope_Flat':ST_Slope_Flat, 'ST_Slope_Up':ST_Slope_Up}
        return dict_instance
    a_dict=if_checker()
    new_df=pd.DataFrame(a_dict,index=[0])
    st.subheader('Your Input')
    st.write(new_df)
    st.subheader('Heart Disease Predicted' if svm.predict(new_df)[0]==1 else 'No Heart Disease Predicted')