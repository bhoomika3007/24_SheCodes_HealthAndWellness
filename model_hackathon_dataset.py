#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[1]:


import pandas as pd
import numpy as np

# Function to generate synthetic data for medical reports with specific deficiency labels
def generate_synthetic_data_with_labels(num_rows=1000):
    # Define normal ranges for each medical marker
    cholesterol_range = (180, 240)  # Total cholesterol (normal range)
    hemoglobin_range_women = (12.1, 15.1)  # Hemoglobin for women
    hemoglobin_range_men = (13.8, 17.2)  # Hemoglobin for men
    vitamin_b12_range = (200, 900)  # Vitamin B12 (normal range)
    vitamin_d_range = (20, 100)  # Vitamin D (normal range)
    glucose_range = (70, 100)  # Glucose fasting (normal range)
    platelet_range = (150, 450)  # Platelet count (normal range)
    wbc_range = (4.0, 11.0)  # WBC count (normal range)
    rbc_range_women = (4.2, 5.4)  # RBC count for women
    rbc_range_men = (4.7, 6.1)  # RBC count for men
    calcium_range = (8.5, 10.2)  # Calcium (normal range)

    # Deficiency criteria for each test
    def is_deficient(value, min_val, max_val):
        return value < min_val or value > max_val
    
    # Generating synthetic data
    data = []
    for _ in range(num_rows):
        # Randomly choose gender (for hemoglobin and RBC ranges)
        gender = np.random.choice(['Male', 'Female'])
        
        # Generate medical markers
        cholesterol = np.random.randint(180, 270)
        hemoglobin = np.random.uniform(hemoglobin_range_women[0], hemoglobin_range_women[1]) if gender == 'Female' else np.random.uniform(hemoglobin_range_men[0], hemoglobin_range_men[1])
        vitamin_b12 = np.random.uniform(vitamin_b12_range[0], vitamin_b12_range[1])
        vitamin_d = np.random.uniform(vitamin_d_range[0], vitamin_d_range[1])
        glucose = np.random.randint(70, 130)
        platelet_count = np.random.randint(150, 500)
        wbc = np.random.uniform(wbc_range[0], wbc_range[1])
        rbc = np.random.uniform(rbc_range_women[0], rbc_range_women[1]) if gender == 'Female' else np.random.uniform(rbc_range_men[0], rbc_range_men[1])
        calcium = np.random.uniform(calcium_range[0], calcium_range[1])
        
        # Check for deficiencies and label them specifically
        deficiencies = []
        if is_deficient(cholesterol, cholesterol_range[0], cholesterol_range[1]): deficiencies.append('High Cholesterol')
        if is_deficient(hemoglobin, hemoglobin_range_women[0], hemoglobin_range_women[1]) if gender == 'Female' else is_deficient(hemoglobin, hemoglobin_range_men[0], hemoglobin_range_men[1]): deficiencies.append('Low Hemoglobin')
        if is_deficient(vitamin_b12, vitamin_b12_range[0], vitamin_b12_range[1]): deficiencies.append('Low Vitamin B12')
        if is_deficient(vitamin_d, vitamin_d_range[0], vitamin_d_range[1]): deficiencies.append('Low Vitamin D')
        if is_deficient(glucose, glucose_range[0], glucose_range[1]): deficiencies.append('High Glucose')
        if is_deficient(platelet_count, platelet_range[0], platelet_range[1]): deficiencies.append('Low Platelet Count')
        if is_deficient(wbc, wbc_range[0], wbc_range[1]): deficiencies.append('Abnormal WBC')
        if is_deficient(rbc, rbc_range_women[0], rbc_range_women[1]) if gender == 'Female' else is_deficient(rbc, rbc_range_men[0], rbc_range_men[1]): deficiencies.append('Low RBC')
        if is_deficient(calcium, calcium_range[0], calcium_range[1]): deficiencies.append('Low Calcium') if calcium < calcium_range[0] else deficiencies.append('High Calcium')
        
        # Append the record
        data.append([
            cholesterol, hemoglobin, vitamin_b12, vitamin_d, glucose, platelet_count, 
            wbc, rbc, calcium, ', '.join(deficiencies)
        ])
    
    # Create a DataFrame from the synthetic data
    columns = [
        'Cholesterol (mg/dL)', 'Hemoglobin (g/dL)', 'Vitamin B12 (pg/mL)', 'Vitamin D (ng/mL)', 
        'Glucose (mg/dL)', 'Platelet Count (x10^3/µL)', 'WBC (x10^3/µL)', 'RBC (million/µL)', 
        'Calcium (mg/dL)', 'Deficiencies'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate 100 rows of synthetic data with specific deficiency labels
synthetic_data_with_labels = generate_synthetic_data_with_labels(1000)

# Save to CSV
file_path_with_labels = 'medical_reports_with_labels.csv'
synthetic_data_with_labels.to_csv(file_path_with_labels, index=False)

print(f"Dataset saved to {file_path_with_labels}")


# In[2]:


data=pd.read_csv("medical_reports_with_labels.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




