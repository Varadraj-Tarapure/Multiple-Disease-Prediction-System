import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import random
import joblib
import streamlit.components.v1 as components

#Setting background image for app
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]  {
background-image: url('https://wallpaper.dog/large/20356391.jpg');
background-size: cover;
}
[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

#Importing all the models of diseases     
diabetes_model = pickle.load(open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Models\diabetes_disease_model.sav", 'rb'))

heart_disease_model = pickle.load(open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Models\heart_disease_model.sav",'rb'))

parkinsons_model = pickle.load(open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Models\parkinson_disease_model.sav", 'rb'))

liver_model = pickle.load(open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Models\liver_disease_model",'rb'))

pneumonia_model=tf.keras.models.load_model(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Models\pneumonia model.hdf5")
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Pneumonia Prediction',
                           'Liver Prediction',
                          ],
                          icons=['activity','heart','person'],
                          default_index=0)
    

    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # inserting image of diabetes in its prediction page
    image = Image.open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Images\diabetes img.jpeg")
    st.image(image)
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input(label="",placeholder='Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input(label="",placeholder='Glucose Level')
    
    with col3:
        BloodPressure = st.text_input(label="",placeholder='Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input(label="",placeholder='Skin Thickness value')
    
    with col2:
        Insulin = st.text_input(label="",placeholder='Insulin Level')
    
    with col3:
        BMI = st.text_input(label="",placeholder='BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input(label="",placeholder='Pedigree Function value')
    
    with col2:
        Age = st.text_input(label="",placeholder='Age of the Person')
    
    
    
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
           #Applying html,css for showing information of treatment for diabetes and providing link for consulting doctors
           components.html("""
    
           <style>
           .danger{
           background-color:red;
           }</style>
           <h4 class=danger>You have a high chance of having Diabetes, Consult a doctor!</h4>
    
           
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1>Treatment for Diabetes</h1>

Depending on what type of diabetes you have, blood sugar monitoring, insulin and oral drugs may be part of your treatment. Eating a healthy diet, staying at a healthy weight and getting regular physical activity also are important parts of managing diabetes.

Treatments for all types of diabetes-
An important part of managing diabetes — as well as your overall health — is keeping a healthy weight through a healthy diet and exercise plan:

Healthy eating. Your diabetes diet is simply a healthy-eating plan that will help you control your blood sugar. You'll need to focus your diet on more fruits, vegetables, lean proteins and whole grains. These are foods that are high in nutrition and fiber and low in fat and calories. You'll also cut down on saturated fats, refined carbohydrates and sweets. In fact, it's the best eating plan for the entire family. Sugary foods are sufficient once in a while. They must be counted as part of your meal plan.

Understanding what and how much to eat can be a challenge. A registered dietitian can help you create a meal plan that fits your health goals, food preferences and lifestyle. This will likely include carbohydrate counting, especially if you have type 1 diabetes or use insulin as part of your treatment.

Physical activity. Everyone needs regular aerobic activity. This includes people who have diabetes. Physical activity lowers your blood sugar level by moving sugar into your cells, where it's used for energy. Physical activity also makes your body more sensitive to insulin. That means your body needs less insulin to transport sugar to your cells.

Get your provider's ready to exercise. Then choose activities you enjoy, such as walking, swimming or biking. What's most important is making physical activity part of your daily routine.

Aim for at least 30 minutes or more of moderate physical activity most days of the week, or at least 150 minutes of moderate physical activity a week. Bouts of activity can be a few minutes during the day. If you haven't been active for a while, start slowly and build up slowly. Also avoid sitting for too long. Try to get up and move if you've been sitting for more than 30 minutes.

Treatments for type 1 and type 2 diabetes
<li>Treatment for type 1 diabetes involves insulin injections or the use of an insulin pump, frequent blood sugar checks, and carbohydrate counting. For some people with type 1 diabetes, pancreas transplant or islet cell transplant may be an option.

<li>Treatment of type 2 diabetes mostly involves lifestyle changes, monitoring of your blood sugar, along with oral diabetes drugs, insulin or both.
        </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed">
             <a href="https://www.healthdoc.in/blog/top-10-diabetologist-diabetes-specialists-and-hospitals-in-pune-maharashtra#:~:text=Top%2010%20Diabetologist%2C%20Diabetes%20Specialists%20and%20Hospitals%20in,Hemant%20Kulkarni%20%28Samarth%20Speciality%20Clinic%29%20...%20More%20items" target="_blank">CLICK HERE FOR CONSULTING BEST DOCTORS FOR TREATMENT</a>
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          
        </div>
      </div>
    </div>
    """,
    height=1000,
)
        else:
           #Applying html,css for showing information of prevention for diabetes
           components.html(
    """
    <style>
           .safe{
           background-color:#00ff00;
           }</style>
           <h4 class=safe>You are not having Diabetes, Nothing to worry!</h4>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1 class="prevent">Prevention for Diabetes</h1>
            

Preventing diabetes involves adopting a healthy lifestyle that includes a balanced diet, regular physical activity, and maintaining a healthy weight. Here are some key strategies for preventing diabetes:

<br>1)Healthy Eating:

Focus on a diet that includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit the intake of processed foods, sugary drinks, and high-fat foods.
Be mindful of portion sizes to avoid overeating, which can contribute to weight gain.
<br>2)Regular Physical Activity:

Aim for at least 150 minutes of moderate-intensity aerobic exercise or 75 minutes of vigorous-intensity exercise per week, along with muscle-strengthening activities on two or more days a week.
Activities can include brisk walking, jogging, swimming, cycling, or other exercises that get your heart rate up.
<br>3)Maintain a Healthy Weight:

Achieving and maintaining a healthy weight is crucial for preventing diabetes. Losing even a small amount of weight can have a significant impact on reducing diabetes risk.
Combine a healthy diet with regular physical activity to help manage weight.
<br>4)Limit Sugar and Refined Carbohydrates:

Reduce the intake of added sugars and refined carbohydrates. Choose whole grains over refined grains and opt for natural sources of sweetness, such as fruits, instead of sugary snacks.
<br>5)Stay Hydrated:

Drink plenty of water throughout the day. Limit the consumption of sugary drinks, including sodas and sweetened beverages.
<br>6)Quit Smoking:

Smoking is associated with an increased risk of diabetes, particularly type 2 diabetes. Quitting smoking can improve overall health and reduce diabetes risk.
<br>7)Limit Alcohol Intake:

If you choose to drink alcohol, do so in moderation. Excessive alcohol consumption can contribute to weight gain and increase the risk of developing diabetes.
<br>8)Regular Health Check-ups:

Schedule regular check-ups with your healthcare provider to monitor your overall health, including blood sugar levels, cholesterol, and blood pressure.
<br>9)Manage Stress:

Chronic stress can contribute to unhealthy lifestyle habits. Practice stress-reducing techniques such as meditation, yoga, deep breathing, or other activities that help you relax.
<br>10)Genetic Counseling:

If you have a family history of diabetes or other risk factors, consider genetic counseling to understand your risk and take proactive measures.
It's important to note that individual factors, including genetics and family history, can also contribute to diabetes risk. Therefore, it's advisable to consult with a healthcare professional for personalized advice based on your specific health profile.

    """,
    height=1000,
)
        
   




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
   
    image = Image.open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Images\heart img.jpeg")
    st.image(image)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    
    
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
            components.html("""
    
           <style>
           .danger{
           background-color:red;
           }</style>
           <h4 class=danger>You have a high chance of having Heart Disease, Consult a doctor!</h4>
    
           
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1>Treatment for Heart Disease</h1>

The treatment for heart disease depends on the specific type and severity of the condition. Here are common approaches and interventions used in the treatment of heart disease:

<br>1)Lifestyle Modifications:

Healthy Diet: Adopting a heart-healthy diet low in saturated and trans fats, cholesterol, and sodium. Emphasize fruits, vegetables, whole grains, and lean proteins.
Regular Exercise: Engaging in regular physical activity to improve cardiovascular health. This includes aerobic exercises like walking, jogging, swimming, and cycling.
<br>2)Medications:

Cholesterol-lowering Medications: Statins and other medications may be prescribed to lower cholesterol levels.
Blood Pressure Medications: Antihypertensive medications help control high blood pressure.
Antiplatelet Drugs: Aspirin and other antiplatelet medications may be recommended to reduce the risk of blood clots.
Beta-Blockers: These medications can help control heart rate and reduce the workload on the heart.
<br>3)Interventional Procedures:

Angioplasty and Stenting: In cases of coronary artery disease, angioplasty may be performed to open narrowed or blocked arteries. A stent may be inserted to keep the artery open.
Coronary Artery Bypass Grafting (CABG): In more severe cases, especially if multiple arteries are blocked, CABG surgery may be recommended to reroute blood around blocked arteries.
<br>4)Device Therapy:

Implantable Cardioverter-Defibrillator (ICD): For individuals at risk of life-threatening arrhythmias, an ICD may be implanted to monitor and regulate the heart's rhythm.
Cardiac Resynchronization Therapy (CRT): CRT devices are used for certain heart failure patients to improve the coordination of the heart's contractions.
<br>5)Heart Failure Management:

Medications: Diuretics, ACE inhibitors, and other medications may be prescribed to manage heart failure symptoms.
Heart Failure Rehabilitation: Exercise and lifestyle programs tailored to individuals with heart failure can improve overall well-being.
<br>6)Heart Transplant:

For individuals with end-stage heart failure, a heart transplant may be considered.
<br>7)Risk Factor Management:

Diabetes Management: If present, diabetes needs to be carefully managed to reduce cardiovascular risk.
Smoking Cessation: Quitting smoking is crucial for heart health.
Weight Management: Achieving and maintaining a healthy weight is important for overall cardiovascular health.
<br>8)Regular Monitoring and Follow-up:

Regular check-ups with healthcare providers to monitor heart health and adjust treatment plans as needed.
        </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed">
             <a href="https://www.clinicspots.com/cardiologist/pune" target="_blank">CLICK HERE FOR CONSULTING BEST DOCTORS FOR TREATMENT</a>
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          
        </div>
      </div>
    </div>
    """,
    height=1000,)
        else:
           components.html(
    """
    <style>
           .safe{
           background-color:#00ff00;
           }</style>
           <h4 class=safe>You are not having Heart Disease, Nothing to worry!</h4>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1 class="prevent">Prevention for Heart Disease</h1>
            


Preventing heart disease involves adopting a heart-healthy lifestyle and managing risk factors. Here are key strategies for heart disease prevention:

<br>1)Healthy Diet:

Eat a Balanced Diet: Include a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats in your diet. Limit the intake of saturated and trans fats, cholesterol, and sodium.
Control Portion Sizes: Be mindful of portion sizes to avoid overeating.
<br>2)Regular Exercise:

Aim for at least 150 minutes of moderate-intensity aerobic exercise or 75 minutes of vigorous-intensity exercise per week, along with muscle-strengthening activities on two or more days a week.
Activities can include walking, jogging, swimming, cycling, or other exercises that increase your heart rate.
<br>3)Maintain a Healthy Weight:

Achieve and maintain a healthy weight through a combination of a balanced diet and regular physical activity.
<br>4)Quit Smoking:

Smoking is a major risk factor for heart disease. Quitting smoking significantly reduces the risk of heart-related problems.
<br>5)Limit Alcohol Intake:

If you choose to drink alcohol, do so in moderation. Excessive alcohol consumption can contribute to high blood pressure and other heart-related issues.
<br>6)Manage Stress:

Chronic stress can contribute to heart disease. Practice stress-reducing techniques such as meditation, deep breathing, yoga, or other relaxation methods.
<br>7)Regular Health Check-ups:

Schedule regular check-ups with your healthcare provider to monitor blood pressure, cholesterol levels, and overall cardiovascular health.
Discuss with your healthcare provider how often you should have screenings for conditions such as diabetes.
<br>8)Control Blood Pressure:

Monitor and manage high blood pressure through lifestyle changes and, if necessary, medication as prescribed by your healthcare provider.
<br>9)Manage Diabetes:

If you have diabetes, work closely with your healthcare team to manage blood sugar levels and reduce the risk of complications, including heart disease.
<br>10)Get Enough Sleep:

Aim for 7-9 hours of quality sleep per night. Poor sleep patterns are associated with an increased risk of heart disease.
<br>11)Limit Processed Foods and Added Sugars:

Reduce the intake of processed foods, sugary snacks, and beverages. Choose whole, unprocessed foods whenever possible.
<br>12)Know Your Family History:

Be aware of your family history of heart disease and discuss it with your healthcare provider. Genetics can play a role in cardiovascular health.
<br>13)Stay Hydrated:

Drink plenty of water throughout the day. Limit the consumption of sugary drinks.
    """,
    height=1000,
)
        
    
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    
    image = Image.open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Images\parkinson img.jpg")
    st.image(image)
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input(label='',placeholder="MDVP:Fo(Hz)")
        
    with col2:
        fhi = st.text_input(label="",placeholder='MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input(label="",placeholder='MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input(label="",placeholder='MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input(label="",placeholder='MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input(label="",placeholder='MDVP:RAP')
        
    with col2:
        PPQ = st.text_input(label="",placeholder='MDVP:PPQ')
        
    with col3:
        DDP = st.text_input(label="",placeholder='Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input(label="",placeholder='Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input(label="",placeholder='Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input(label="",placeholder='Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input(label="",placeholder='Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input(label="",placeholder='MDVP:APQ')
        
    with col4:
        DDA = st.text_input(label="",placeholder='Shimmer:DDA')
        
    with col5:
        NHR = st.text_input(label="",placeholder='NHR')
        
    with col1:
        HNR = st.text_input(label="",placeholder='HNR')
        
    with col2:
        RPDE = st.text_input(label="",placeholder='RPDE')
        
    with col3:
        DFA = st.text_input(label="",placeholder='DFA')
        
    with col4:
        spread1 = st.text_input(label="",placeholder='spread1')
        
    with col5:
        spread2 = st.text_input(label="",placeholder='spread2')
        
    with col1:
        D2 = st.text_input(label="",placeholder='D2')
        
    with col2:
        PPE = st.text_input(label="",placeholder='PPE')
        
    
    
   
    
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
            components.html("""
    
           <style>
           .danger{
           background-color:red;
           }</style>
           <h4 class=danger>You have a high chance of having Parkinson Disease, Consult a doctor!</h4>
    
           
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1>Treatment for Parkinson Disease</h1>

Parkinson's disease is a neurodegenerative disorder that affects movement control. While there is no cure for Parkinson's disease, various treatments are available to manage symptoms and improve the quality of life for individuals with the condition. Treatment plans are typically tailored to each person's specific symptoms and needs. Here are common approaches to the treatment of Parkinson's disease:

<br>1)Medications:

Levodopa (L-DOPA): This is one of the most effective medications for managing Parkinson's symptoms. It is converted into dopamine in the brain, helping to compensate for the dopamine deficiency in individuals with Parkinson's.
Dopamine Agonists: These medications mimic the effects of dopamine in the brain.
MAO-B Inhibitors: Monoamine oxidase type B (MAO-B) inhibitors help prevent the breakdown of dopamine in the brain.
COMT Inhibitors: Catechol-O-methyltransferase (COMT) inhibitors prolong the effects of levodopa by preventing its breakdown.
<br>2)Deep Brain Stimulation (DBS):

DBS involves surgically implanting electrodes into specific areas of the brain, usually the subthalamic nucleus or globus pallidus. These electrodes are connected to a stimulator device, which helps regulate abnormal electrical signals and alleviate motor symptoms.
<br>3)Physical Therapy:

Physical therapy can help improve mobility, balance, and flexibility. Therapists work with individuals to develop exercises and strategies that address specific movement challenges associated with Parkinson's disease.
<br>4)Occupational Therapy:

Occupational therapists assist individuals in adapting their daily activities to cope with the effects of Parkinson's disease. This may include strategies for improving fine motor skills and managing activities of daily living.
<br>5)Speech Therapy:

Speech therapy can be beneficial for individuals experiencing speech and swallowing difficulties, which can be common in Parkinson's disease.
        </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed">
             <a href="https://www.practo.com/pune/treatment-for-parkinson-s-disease" target="_blank">CLICK HERE FOR CONSULTING BEST DOCTORS FOR TREATMENT</a>
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          
        </div>
      </div>
    </div>
    """,
    height=900,)
        else:
          components.html(
    """
    <style>
           .safe{
           background-color:#00ff00;
           }</style>
           <h4 class=safe>You are not having Parkinson Disease, Nothing to worry!</h4>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1 class="prevent">Prevention for Parkinson Disease</h1>
            


<br>1)Regular Exercise:

Engaging in regular physical activity has been linked to a lower risk of developing Parkinson's disease. Activities such as walking, jogging, swimming, and other forms of exercise may be beneficial.

<br>2)Healthy Diet:

Eating a nutritious and well-balanced diet, rich in fruits, vegetables, whole grains, and lean proteins, may contribute to overall health. Some studies suggest that diets high in antioxidants and certain nutrients, such as vitamin E, may have a protective effect.

<br>3)Avoiding Environmental Toxins:

While the evidence is not conclusive, some studies suggest that exposure to certain environmental toxins, such as pesticides and herbicides, may be associated with an increased risk of Parkinson's disease. Minimizing exposure to these substances when possible may be prudent.

<br>4)Caffeine Consumption:

Some research suggests that caffeine intake may be associated with a lower risk of Parkinson's disease. This includes coffee and tea, which contain antioxidants and other potentially beneficial compounds.

<br>5)Maintaining a Healthy Weight:

Obesity has been suggested as a potential risk factor for Parkinson's disease. Maintaining a healthy weight through a combination of a balanced diet and regular exercise may be beneficial.

<br>6)Avoiding Head Trauma:

There is some evidence to suggest that head injuries may increase the risk of Parkinson's disease. Taking precautions to avoid head injuries, such as wearing helmets during certain activities, may be important.
    """,
    height=800,
)
        
    


if (selected == "Pneumonia Prediction"):
    image2 = Image.open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Images\pneumonia img.jpg")
    st.image(image2)
    
    file=st.sidebar.file_uploader("Please upload your X-Ray image and Nothing Else", type= ["png","JPG","jpeg"])
    def predict(image_path):
        image1 = image.load_img(image_path, target_size=(150, 150))
        image1 = image.img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        img_array= image1/255
        prediction = pneumonia_model.predict(img_array)
        if prediction[0][0]>.6:
           components.html("""
    
           <style>
           .danger{
           background-color:red;
           }</style>
           <h4 class=danger>You have a high chance of having Pneoumonia, Consult a doctor!</h4>
    
           
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1>Treatment for Pneoumonia</h1>
Your treatment will depend on the type of pneumonia you have, how severe it is, and your general health.


Your doctor may prescribe a medication to help treat your pneumonia. What you’re prescribed will depend on the specific cause of your pneumonia.

Oral antibiotics can treat most cases of bacterial pneumonia. Always take your entire course of antibiotics, even if you begin to feel better. Not doing so can prevent the infection from clearing, and it may be harder to treat in the future.

Antibiotic medications don’t work on viruses. In some cases, your doctor may prescribe an antiviral. However, many cases of viral pneumonia clear on their own with at-home care.

Antifungal medications are used to treat fungal pneumonia. You may have to take this medication for several weeks to clear the infection.

OTC medications
Your doctor may also recommend over-the-counter (OTC) medications to relieve your pain and fever, as needed. These may include:

aspirin
ibuprofen (Advil, Motrin)
acetaminophen (Tylenol)
Your doctor may also recommend cough medicine to calm your cough so you can rest. Keep in mind coughing helps remove fluid from your lungs, so you don’t want to eliminate it entirely.

Home remedies
Although home remedies don’t actually treat pneumonia, there are some things you can do to help ease symptoms.

Coughing is one of the most common symptoms of pneumonia. Natural ways to relieve a cough include gargling salt water or drinking peppermint tea.

Cool compresses can work to relieve a fever. Drinking warm water or having a nice warm bowl of soup can help with chills. Here are more home remedies to try.

You can help your recovery and prevent a recurrence by getting a lot of rest and drinking plenty of fluids.

Although home remedies can help ease symptoms, it’s important to stick to your treatment plan. Take any prescribed medications as directed.

Hospitalization
If your symptoms are very severe or you have other health problems, you may need to be hospitalized. At the hospital, doctors can keep track of your heart rate, temperature, and breathing. Hospital treatment may include:

antibiotics injected into a vein
respiratory therapy, which involves delivering specific medications directly into the lungs, or teaching you to perform breathing exercises to maximize your oxygenation
oxygen therapy to maintain oxygen levels in your bloodstream (received through a nasal tube, face mask, or ventilator, depending on severity)

Your doctor may do this test if your initial symptoms are severe, or if you’re hospitalized and not responding well to antibiotics.
        </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed">
             <a href="https://www.365doctor.in/pulmonologist-in-pune" target="_blank">CLICK HERE FOR CONSULTING BEST DOCTORS FOR TREATMENT</a>
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          
        </div>
      </div>
    </div>
    """,
    height=1000,
)
        
        else:
           
           components.html(
    """
    <style>
           .safe{
           background-color:#00ff00;
           }</style>
           <h4 class=safe>You have a very low chance of having Pneoumonia, Nothing to worry!</h4>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1 class="prevent">Prevention for Pneoumonia</h1>
            Prevention
In many cases, pneumonia can be prevented.

Vaccination
The first line of defense against pneumonia is to get vaccinated. There are several vaccines that can help prevent pneumonia.

Prevnar 13 and Pneumovax 23
These two pneumonia vaccines help protect against pneumonia and meningitis caused by pneumococcal bacteria. Your doctor can tell you which one might be better for you.

Prevnar 13 is effective against 13 types of pneumococcal bacteria. The CDCTrusted Source recommends this vaccine for:

children under age 2
people between ages 2 and 64 with chronic conditions that increase their risk of pneumonia
adults ages 65 and older, on the recommendation of their doctor
Pneumovax 23 is effective against 23 types of pneumococcal bacteria. The CDCTrusted Source recommends it for:

adults ages 65 and older
adults ages 19 to 64 who smoke
people between ages 2 and 64 with chronic conditions that increase their risk of pneumonia
Flu vaccine
Pneumonia can often be a complication of the flu, so be sure to also get an annual flu shot. The CDCTrusted Source recommends that everyone ages 6 months and older get vaccinated, particularly those who may be at risk of flu complications.

Hib vaccine
This vaccine protects against Haemophilus influenzae type b (Hib), a type of bacterium that can cause pneumonia and meningitis. The CDCTrusted Source recommends this vaccine for:

all children under 5 years old
unvaccinated older children or adults who have certain health conditions
people who’ve gotten a bone marrow transplant
According to the National Heart, Lung, and Blood InstituteTrusted Source, pneumonia vaccines won’t prevent all cases of the condition.

But if you’re vaccinated, you’re likely to have a milder and shorter illness as well as a lower risk of complications.

Other prevention tips
In addition to vaccination, there are other things you can do to avoid pneumonia:

If you smoke, try to quit. Smoking makes you more susceptible to respiratory infections, especially pneumonia.
Regularly wash your hands with soap and water for at least 20 seconds.
Cover your coughs and sneezes. Promptly dispose used tissues.
Maintain a healthy lifestyle to strengthen your immune system. Get enough rest, eat a balanced diet, and get regular exercise.
Together with vaccination and additional prevention steps, you can help reduce your risk of getting pneumonia.


    """,
    height=900,
)
        
        
    
    
    if file is not None:
        img=Image.open(file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        predict(file)
        

     
# Liver prediction page
if (selected == "Liver Prediction"):  
    image = Image.open(r"C:\Users\Varadraj\Desktop\Multiple-Disease-Prediction-System\Images\liver img.jpg")
    st.image(image)
    
   
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col2:
        age = st.number_input("Enter your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Enter your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Enter your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Enter your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Enter your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Enter your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Enter your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Enter your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Enter your Albumin_and_Globulin_Ratio") # 10 
   
    
     # creating a button for Prediction 
    if st.button("Liver test result"):
        
        liver_prediction=[[]]
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

        
        if liver_prediction[0] == 1:
            components.html("""
    
           <style>
           .danger{
           background-color:red;
           }</style>
           <h4 class=danger>You are having Liver Disease, Consult a doctor!</h4>
    
           
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1>Treatment for Liver Disease</h1>

The treatment for liver disease depends on the specific type and cause of the liver condition. Liver diseases can range from viral infections to alcohol-related liver disease, non-alcoholic fatty liver disease (NAFLD), cirrhosis, and others. Here are some general approaches to the treatment of liver disease:

<br>1)Lifestyle Changes:

For certain types of liver disease, lifestyle modifications are crucial. This includes abstaining from alcohol for alcohol-related liver diseases and adopting a healthy diet and exercise routine for conditions like NAFLD.
<br>2)Medications:

Medications may be prescribed to manage symptoms, control the progression of the disease, or treat underlying causes. Examples include antiviral medications for viral hepatitis, medications to control blood sugar levels for individuals with liver disease related to diabetes, and drugs to manage symptoms of autoimmune liver diseases.
<br>3)Antiviral Therapy:

For viral hepatitis (such as hepatitis B or C), antiviral medications may be prescribed to suppress the virus, reduce liver inflammation, and prevent further damage.
<br>4)Corticosteroids and Immunomodulators:

In cases of autoimmune liver diseases, corticosteroids or other immunomodulating medications may be used to suppress the immune system's response and reduce inflammation.
<br>5)Treatment of Underlying Conditions:

Addressing and managing underlying conditions, such as diabetes, high blood pressure, or high cholesterol, is essential to prevent further liver damage.
<br>6)Hepatic Encephalopathy Management:

For individuals with advanced liver disease and hepatic encephalopathy (a condition affecting brain function due to liver dysfunction), medications like lactulose may be prescribed to reduce ammonia levels in the blood.
<br>7)Diuretics:

Diuretics may be used to manage fluid retention (ascites) and reduce swelling in individuals with liver cirrhosis.
<br>8)Nutritional Support:

In cases of malnutrition or weight loss associated with liver disease, nutritional support may be provided, which can include dietary changes, supplements, or, in severe cases, tube feeding.
<br>9)Liver Transplant:

In cases of severe liver disease or failure, a liver transplant may be considered as a treatment option. Liver transplantation involves replacing a damaged or diseased liver with a healthy liver from a donor.
<br>10)Regular Monitoring and Follow-up:

Regular check-ups and monitoring of liver function are essential to assess the progression of the disease and make adjustments to the treatment plan as needed.
        </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed">
             <a href="https://www.practo.com/pune/treatment-for-liver-disease" target="_blank">CLICK HERE FOR CONSULTING BEST DOCTORS FOR TREATMENT</a>
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          
        </div>
      </div>
    </div>
    """,
    height=1000,
    
)
        else:
             components.html(
    """
    <style>
           .safe{
           background-color:#00ff00;
           }</style>
           <h4 class=safe>You are not having Liver Disease, Nothing to worry!</h4>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
     
          <div class="card-body">
            <h1 class="prevent">Prevention for Liver Disease</h1>
            

Preventing liver disease involves adopting a healthy lifestyle and making choices that support liver health. Liver diseases can be caused by various factors, including viral infections, excessive alcohol consumption, obesity, and certain medications. Here are some general guidelines for preventing liver disease:

<br>1)Limit Alcohol Intake:

Excessive alcohol consumption is a leading cause of liver disease. Limit alcohol intake and follow recommended guidelines for moderate drinking, which is up to one drink per day for women and up to two drinks per day for men.

<br>2)Maintain a Healthy Weight:

Obesity is a risk factor for non-alcoholic fatty liver disease (NAFLD). Adopt a healthy diet and engage in regular physical activity to maintain a healthy weight.
<br>3)Healthy Diet:

Eat a well-balanced diet that includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit the intake of processed foods, saturated fats, and added sugars.
<br>4)Be Cautious with Medications:

Some medications can cause liver damage. Follow your healthcare provider's instructions when taking medications, including over-the-counter drugs and supplements. Inform your healthcare provider of any existing liver conditions.
<br>5)Practice Good Hygiene:

Hepatitis A and E can be contracted through contaminated food and water. Wash your hands thoroughly, especially after using the bathroom and before handling food.
<br>6)Get Vaccinated:

Vaccination is available for hepatitis A and B. Check with your healthcare provider to ensure that you are up-to-date on your vaccinations.
<br>7)Avoid Intravenous Drug Use:

Sharing needles or using contaminated needles for drug injection increases the risk of hepatitis B, hepatitis C, and other infections. Seek help if you have a substance abuse problem.
<br>8)Protect Against Toxins:

Limit exposure to chemicals and toxins in the workplace and at home. Follow safety guidelines when handling hazardous substances.
<br>9)Manage Chronic Conditions:

Conditions such as diabetes, high blood pressure, and high cholesterol can contribute to liver disease. Manage these conditions through lifestyle changes and medication as prescribed by your healthcare provider.
<br>10)Regular Exercise:

Engage in regular physical activity, as it can help reduce the risk of obesity and improve overall health.
<br>11)Screen for Viral Hepatitis:

Regular screenings for hepatitis B and C can help detect infections early and facilitate timely intervention.
    """,
    height=800,
)

#Hidding streamlit setting option and watermark of streamlit        
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
