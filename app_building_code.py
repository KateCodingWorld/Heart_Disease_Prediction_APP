### import libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# stay at widemode
st.set_page_config(layout='wide')

#######################################################################################################################################
### Create a title  
st.title('Heart Disease Prediction APP')

# divide the page into 3 sections
col1, col2, col3 = st.columns([2,2,3])

# left section showing a image
with col1:
    st.image('image/HEART.jpg', width=250)

# middle section showing text
with col2:
    st.subheader("In every 36 seconds,")
    st.markdown("one person dies in the United States from **cardiovascular disease**.")
    st.markdown("About 659,000 people in the United States die from heart disease each year—that's **1 in every 4 deaths**!")
    st.markdown("This application is build based on 32k responses from CDC questionaire, and applied optimal machine learning model, in order to help you predict the possibility of getting heart disease more accurately.")
    st.markdown('*(CDC: Centers for Disease Control and Prevention)*')


#######################################################################################################################################
### Set up side bar and user input features

# create a header
st.sidebar.header('💖 Current Health Status 💖')


# define user inout function
def user_input_features():
    Age = st.sidebar.slider('How old are you?', 18,80,45)
    Height = st.sidebar.slider('How tall are you? (cm)', 150,200,175)
    Weight = st.sidebar.slider('How much do you weight? (kg)', 40,120,80)
    SleepTime = st.sidebar.slider('Average sleep time', 4,12,8)
    PhysicalHealth = st.sidebar.slider('Days in a month feel physical health is NOT good', 0,31)
    MentalHealth = st.sidebar.slider('Days in a month feel mental health is NOT good', 0,31)
    Sex = st.sidebar.selectbox('What sex are you?', ('Male','Female'))
    Race = st.sidebar.selectbox('What is your ancestral background?', ('White','Hispanic','Black','Asian','American Indian/Alaskan Native','Other'))
    Stroke = st.sidebar.selectbox('If you have ever had a stroke?', ('No','Yes'))
    Diabetic = st.sidebar.selectbox('Do you have diabetic?', ('No','Yes'))
    Asthma = st.sidebar.selectbox('Do you have Asthma?', ('No','Yes'))
    KidneyDisease = st.sidebar.selectbox('Do you have kidney disease?', ('No','Yes'))
    SkinCancer = st.sidebar.selectbox('Do you have skin cancer?', ('No','Yes'))
    Smoking = st.sidebar.selectbox('If you have smoked at least 100 cigarettes in entire life?', ('No','Yes'))
    AlcoholDrinking = st.sidebar.selectbox('If you have more than 14 drinks of alcohol (men) or more than 7 (women) in a week?', ('No','Yes'))
    DiffWalking = st.sidebar.selectbox('If you feel serious difficulty walking or climbing stairs?', ('No','Yes'))
    PhysicalActivity = st.sidebar.selectbox('If you have physical activity or exercise during the past 30 days other than the regular job?', ('No','Yes'))
    GenHealth = st.sidebar.selectbox('What is your overall health situation?', ('Excellent','Very good','Good','Fair','Poor'))

    data = {
        'BMI':Weight/(Height**2),
        'PhysicalHealth':PhysicalHealth,
        'MentalHealth':MentalHealth,
        'SleepTime':SleepTime,
        'Sex':Sex,
        'Smoking':Smoking,
        'AlcoholDrinking':AlcoholDrinking,
        'Stroke':Stroke,
        'DiffWalking':DiffWalking,
        'PhysicalActivity':PhysicalActivity,
        'Asthma':Asthma,
        'KidneyDisease':KidneyDisease,
        'SkinCancer':SkinCancer,
        'Age':Age,
        'Diabetic':Diabetic,
        'Race':Race,
        'GenHealth':GenHealth

    }
    features = pd.DataFrame(data, index=[0])

    # convert binary column into numeric
    binary_column = ['Smoking', 'AlcoholDrinking','Stroke','DiffWalking','Diabetic','PhysicalActivity','Asthma','KidneyDisease','SkinCancer']
    for col in binary_column:
        # convert to numeric and add to numeric list 
        features[col] = np.where(features[col] == "Yes", 1, 0)
    
    # convert 'Sex' column to numeric 1/0
    features['Sex'] = np.where(features['Sex'] == "Female", 1, 0)

    # convert 'GenHealth' column to numeric series
    features.loc[features['GenHealth']=='Excellent','GenHealth']=4
    features.loc[features['GenHealth']=='Very good','GenHealth']=3
    features.loc[features['GenHealth']=='Good','GenHealth']=2
    features.loc[features['GenHealth']=='Fair','GenHealth']=1
    features.loc[features['GenHealth']=='Poor','GenHealth']=0

    return features


# use function to get user input and transformed into optimal format
input_df = user_input_features()


# combine existing dataframe with new input data
heartdisease = pd.read_csv('data\cleaned_dataset.csv')
heartdisease_X = heartdisease.drop(columns = ['HeartDisease'])
df = pd.concat([input_df, heartdisease_X], axis=0)


# convert multi-class 'Race' column
dummy = pd.get_dummies(df['Race'], drop_first=True)
df = pd.concat([df,dummy],axis=1)
del df['Race']

# get user input data (the newest one in the df)
df_test = df[:1]

#######################################################################################################################################
### MODEL INFERENCE

# A. Load the model using joblib
import joblib
load_pipe = joblib.load(open('optimal_model.pkl','rb'))



# B. Use the model to predict sentiment & write result
import time
# show up in block 3
with col3:
    # show click button then predict the result
    if st.button("💝 Click Here to Test 💝"):
        
        # Temporarily displays a message while executing 
        with st.spinner('Wait for it...🤔'):
            time.sleep(2)

        # show up the probability of prediction
        prediction_proba = load_pipe.predict_proba(df_test)
        for num in prediction_proba[:,1]:
                final = round(num*100,2)
        # show the content
        st.markdown('**Possibility of Getting Heart Disease is:**')
        st.subheader(f'{final}%')
        st.progress(final/100)

        # based on the predicted result 1/0, show up different messages and image
        prediction = load_pipe.predict(df_test)
        if prediction == 0:
            st.success('You have been taken care of your heart well!! 💖')
            st.image('image/GOOD.jpg', width=300) 
            st.balloons()
        else:
            st.warning('You should take better care of your heart... 💔')
            st.image('image/BAD.jpg', width=300)
            st.snow()

                # add a retry button
        if st.button('retry it'):
            st.empty()

