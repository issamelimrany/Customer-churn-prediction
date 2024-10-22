import pandas as pd
import streamlit as st 
import pickle 
import numpy as np
from openai import OpenAI
import os
import plotly.graph_objects as go
import plotly.express as px

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

def explain_prediction(probability, input_dict, surname): 
    prompt = f"The customer {surname} has a {probability:.2f} probability of churning. The customer has the following features: {input_dict}. Explain why the customer is likely to churn or not. make it short and to the point"
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def personalized_email(probability, input_dict, surname, explanation) : 
    prompt = f"The customer {surname} has a {probability:.2f} probability of churning. The customer has the following features: {input_dict}. considering this explanation {explanation} create a personalized email for this particular user including the name."
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def load_model(file_name): 
    with open(file_name, "rb") as file: 
        model = pickle.load(file)
    return model

decision_tree_model = load_model("dt_model.pkl")
naive_bayes_model = load_model("nb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
voting_classifier_model = load_model("vcf_model.pkl")
xgboost_model = load_model("xgb_model.pkl")
support_vector_machine_model = load_model("svc_model.pkl")

def prepare_input(credit_score, location, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary): 
    input_dict = {
        "CreditScore": credit_score, 
        "Age": age, 
        "Tenure": tenure, 
        "Balance": balance, 
        "NumOfProducts": num_of_products, 
        "HasCrCard": 1 if has_credit_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0, 
        "EstimatedSalary": estimated_salary, 
        "Geography_France": 1 if location == "France" else 0, 
        "Geography_Germany": 1 if location == "Germany" else 0, 
        "Geography_Spain": 1 if location == "Spain" else 0, 
        "Gender_Male": 1 if gender == "Male" else 0, 
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Female1": 1 if gender == "Female" else 0,
        "Gender_Female2": 1 if gender == "Female" else 0
    }

    input_df = pd.DataFrame([input_dict])

    return input_df, input_dict

def get_prediction(input_df, input_dict): 
    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1], 
        "DecisionTree": decision_tree_model.predict_proba(input_df)[0][1], 
        "RandomForest": random_forest_model.predict_proba(input_df)[0][1], 
    }

    avg_probabilities = np.mean(list(probabilities.values()))

    st.markdown("### Models Probabilities")

    # Bar chart for model probabilities
    fig_bar = go.Figure(data=[go.Bar(
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        text=[f"{prob:.2f}" for prob in probabilities.values()],
        textposition='auto',
    )])
    fig_bar.update_layout(title_text='Churn Probability by Model', xaxis_title='Model', yaxis_title='Probability')
    st.plotly_chart(fig_bar, key="bar_chart")

    # Gauge chart for average probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_probabilities,
        title = {'text': "Average Churn Probability"},
        gauge = {'axis': {'range': [0, 1]},
                 'bar': {'color': "darkblue"},
                 'steps' : [
                     {'range': [0, 0.3], 'color': "green"},
                     {'range': [0.3, 0.7], 'color': "yellow"},
                     {'range': [0.7, 1], 'color': "red"}],
                 'threshold': {
                     'line': {'color': "red", 'width': 4},
                     'thickness': 0.75,
                     'value': avg_probabilities}}))
    st.plotly_chart(fig_gauge, key="gauge_chart")

    # Radar chart for input features
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    values = [input_dict[feature] for feature in features]
    fig_radar = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=features,
      fill='toself'
    ))
    fig_radar.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, max(values)]
        )),
      showlegend=False,
      title='Customer Profile'
    )
    st.plotly_chart(fig_radar, key="radar_chart")

    return avg_probabilities



st.title("Customer Churn Prediction")


df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer = st.selectbox("Select Customer", customers)

if selected_customer: 
    selected_customer_id = int(selected_customer.split(" - ")[0])
    selected_customer_name = selected_customer.split(" - ")[1]

    #st.write(f"Selected Customer: {selected_customer_name}")
    #st.write(f"Customer ID: {selected_customer_id}")

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id]
    #st.write("Selected Customer Data:", selected_customer)

    col1, col2 = st.columns(2)

    with col1: 
        credit_score = st.number_input("Credit Score", value=int(selected_customer["CreditScore"].values[0]), min_value=100, max_value=850)
        location = st.selectbox("Location", ["France", "Spain", "Germany"], index=["France", "Spain", "Germany"].index(selected_customer["Geography"].values[0]))
        gender = st.radio("Gender", ['Male', 'Female'], index = 0 if selected_customer["Gender"].values[0] == "Male" else 1)
        age = st.number_input("Age", value=int(selected_customer["Age"]), min_value=18, max_value=100)
        tenure = st.number_input("Tenure", value=int(selected_customer["Tenure"]), min_value=0, max_value=10)
        
    with col2: 
        balance = st.number_input("Balance", value=int(selected_customer["Balance"]), min_value=0)
        num_of_products = st.number_input("Number of Products", value=int(selected_customer["NumOfProducts"]), min_value=1, max_value=4)
        has_credit_card = st.radio("Has Credit Card", ['Yes', 'No'], index = 1 if selected_customer["HasCrCard"].values[0] == 1 else 0)
        is_active_member = st.radio("Is Active Member", ['Yes', 'No'], index = 1 if selected_customer["IsActiveMember"].values[0] == 1 else 0)
        estimated_salary = st.number_input("Estimated Salary", value=int(selected_customer["EstimatedSalary"]), min_value=0)


    predict = st.button("Predict")

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary)

if predict and input_df is not None: 
    avg_probability = get_prediction(input_df, input_dict)
    st.write("### Explanation of the prediction")
    st.write(explain_prediction(avg_probability, input_dict, selected_customer_name))
    st.write("### Personalized email : ")
    st.write(personalized_email(avg_probability, input_dict, selected_customer_name, explain_prediction(avg_probability, input_dict, selected_customer_name)))


