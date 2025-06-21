import streamlit as st
import numpy as np 
import pandas as pd 
import pickle


st.title('Heart Disease Prediction')
tab1,tab2,tab3=st.tabs(['Predict','File Predict','Model Info'])

with tab1:
    # ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    # 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
    # 'HeartDisease']
    age = st.number_input("Age")
    sex = st.selectbox("Sex",("Male","Female"))
    # ['ATA', 'NAP', 'ASY', 'TA']
    ChestPainType = st.selectbox("Chest Pain Type",['ATA', 'NAP', 'ASY', 'TA'])
    RestingBP = st.number_input('Resting Blood Pressure',min_value=0,max_value=160)
    Cholesterol = st.number_input('Cholesterol',max_value=300,min_value=0)
    FastingBS = st.selectbox('Fasting Blood Suger',['<= 120','> 120'])
    # ['Normal', 'ST', 'LVH']
    RestingECG = st.selectbox('RestingECG',['Normal', 'ST', 'LVH'])
    MaxHR = st.number_input('Max Heart Rate',min_value=60,max_value=202,value=150)
    ExerciseAngina = st.selectbox('Exercise-Induced Angina',['Y','N'])
    Oldpeak =  st.number_input('Oldpeak',min_value=0,max_value=10)
    ST_Slope = st.selectbox('St slope',['Up', 'Flat', 'Down'])

    # Convert all cat values "selectboxs"
    # Sex ['M' 'F'] [0, 1]
    # ChestPainType ['ATA' 'NAP' 'ASY' 'TA'] [0, 1, 2, 3]
    # RestingECG ['Normal' 'ST' 'LVH'] [0, 1, 2]
    # ExerciseAngina ['N' 'Y'] [0, 1]
    # ST_Slope ['Up' 'Flat' 'Down'] [0, 1, 2]
    sex = 0 if sex == 'Male' else 1
    ChestPainType = ['ATA', 'NAP', 'ASY', 'TA'].index(ChestPainType)
    FastingBS = 0 if FastingBS == '<= 120' else 1
    RestingECG = ['Normal', 'ST', 'LVH'].index(RestingECG)
    ExerciseAngina = 0 if ExerciseAngina == 'N' else 1
    ST_Slope = ['Up', 'Flat', 'Down'].index(ST_Slope)


    # ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    # 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
    # 'HeartDisease']
    input_df = pd.DataFrame(
        {
            'Age':[age],
            'Sex':[sex],
            'ChestPainType':[ChestPainType],
            'RestingBP':[RestingBP],
            'Cholesterol':[Cholesterol],
            'FastingBS':[FastingBS],
            'RestingECG':[RestingECG],
            'MaxHR':[MaxHR],
            'ExerciseAngina':[ExerciseAngina],
            'Oldpeak':[Oldpeak],
            'ST_Slope':[ST_Slope]
        }
    )

    algo_name = ['Logistic Regrission','SVM']
    models = ['LogisticR.pkl','svm.pkl']

    predictions = []
    def heart_prediction(data):
        for m_name in models:
            model = pickle.load(open(m_name,'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions


    if st.button("Submet"):
        st.subheader('Resuls')
        st.markdown('--------------------')
        
        results = heart_prediction(input_df)
        for i in range(len(predictions)):
            st.subheader(algo_name[i])
            if results[i][0] == 0:
                st.write('Not heart diseace detected')
            else : 
                st.write('Heart diseace detected')
                

with tab2:
    st.subheader('File......')
    uploaded_file = st.file_uploader('Csv File',type='csv')
    
    if uploaded_file is not None:
        model_Lr = pickle.load(open('LogisticR.pkl','rb'))
        model_Svm = pickle.load(open('svm.pkl','rb'))
        exopected_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',]
        
        input_data = pd.read_csv(uploaded_file)
        if set(exopected_cols).issubset(input_data.columns):
            predicTions_LR = []
            predicTion_SVM = []
            
            for i in range(len(input_data)):
                row = input_data.loc[i].values.reshape(1,-1)
                prediction_lr = model_Lr.predict(row)[0]
                prediction_svm = model_Svm.predict(row)[0]
                predicTions_LR.append(prediction_lr)
                predicTion_SVM.append(prediction_svm)
            input_data['pridectedLr'] = predicTions_LR
            input_data['pridectedSvm'] = predicTion_SVM
            input_data.to_csv('pridected_Lr.csv')
            
            st.subheader('Predictions:')
            st.write(input_data)
        else:
            st.warning('please make sure the uploaded file has the correct columns')
            
    else :
        st.warning('Please upload a csv file')
        
with tab3:
    import plotly.express as px
    data_dect = {
        'Logistic Regrission' : 85.86,
        'SVM' : 85.85
    }
    
    models_names = list(data_dect.keys())
    model_accurices = list(data_dect.values())
    
    fig = px.bar(x=models_names,y=model_accurices)
    st.plotly_chart(fig)