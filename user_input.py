#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

def get_user_input():
    # Collect user inputs
    Cholesterol = float(input("Enter your Cholesterol level (mg/dL): "))
    Hemoglobin = float(input("Enter your Hemoglobin level (g/dL): "))
    Vitamin_B12 = float(input("Enter your Vitamin B12 level (pg/mL): "))
    Vitamin_D = float(input("Enter your Vitamin D level (ng/mL): "))
    Glucose = float(input("Enter your Glucose level (mg/dL): "))
    Platelet_Count = float(input("Enter your Platelet Count (x10^3/µL): "))
    WBC = float(input("Enter your WBC level (x10^3/µL): "))
    RBC = float(input("Enter your RBC level (million/µL): "))
    Calcium = float(input("Enter your Calcium level (mg/dL): "))

    # Create a DataFrame
    user_input = pd.DataFrame({
        "Cholesterol (mg/dL)": [Cholesterol],
        "Hemoglobin (g/dL)": [Hemoglobin],
        "Vitamin B12 (pg/mL)": [Vitamin_B12],
        "Vitamin D (ng/mL)": [Vitamin_D],
        "Glucose (mg/dL)": [Glucose],
        "Platelet Count (x10^3/µL)": [Platelet_Count],
        "WBC (x10^3/µL)": [WBC],
        "RBC (million/µL)": [RBC],
        "Calcium (mg/dL)": [Calcium]
    })

    return user_input


# In[5]:


user_data=get_user_input()


# In[6]:


user_data


# In[12]:


def map_deficiencies(prediction):
    deficiency_labels = [
        "High Glucose, Low Platelet Count",  # Index 0
        "Low Platelet Count",                # Index 1
        "High Glucose",                      # Index 2
        "High Cholesterol",                  # Index 3
        "High Cholesterol, High Glucose",    # Index 4
        "No Deficiency",                     # Index 5
        "High Cholesterol, High Glucose, Low Platelet Count",  # Index 6
        "High Cholesterol, Low Platelet Count"                 # Index 7
    ]
    
    deficiencies = []
    for i, val in enumerate(prediction[0]):  # Since prediction is 2D array
        if val == 1:
            deficiencies.append(deficiency_labels[i])
    
    return deficiencies



# In[13]:


from joblib import load

# Load the best model
loaded_best_model = load('best_model.joblib')


# In[14]:


# Assuming user_input is the DataFrame with the user's input data
predictions = loaded_best_model.predict(user_data)

# Optionally, map the predicted binary labels to actual deficiency labels
actual_deficiencies = map_deficiencies(predictions)

print("Predicted Deficiencies:", actual_deficiencies)


# In[ ]:




