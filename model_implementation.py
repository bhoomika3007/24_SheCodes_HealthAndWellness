#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("medical_reports_with_labels.csv")


# In[3]:


data['Deficiencies']=data['Deficiencies'].fillna('No Deficiency')


# In[4]:


data.isna().sum()


# In[5]:


data.shape


# In[6]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split


# In[7]:


df=pd.get_dummies(data,columns=['Deficiencies'])


# In[8]:


df


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[10]:


X = df.drop(columns=['Deficiencies_High Cholesterol', 'Deficiencies_High Cholesterol, High Glucose', 
                     'Deficiencies_High Cholesterol, High Glucose, Low Platelet Count', 
                     'Deficiencies_High Cholesterol, Low Platelet Count',
                     'Deficiencies_High Glucose', 'Deficiencies_High Glucose, Low Platelet Count',
                     'Deficiencies_Low Platelet Count', 'Deficiencies_No Deficiency'])


# In[11]:


df.columns


# In[12]:


print("X (Features):")
X.head()


# In[13]:


y = df[['Deficiencies_High Cholesterol', 'Deficiencies_High Cholesterol, High Glucose', 
        'Deficiencies_High Cholesterol, High Glucose, Low Platelet Count', 
        'Deficiencies_High Cholesterol, Low Platelet Count',
        'Deficiencies_High Glucose', 'Deficiencies_High Glucose, Low Platelet Count',
        'Deficiencies_Low Platelet Count', 'Deficiencies_No Deficiency']]


# In[14]:


y.shape


# In[15]:


print("\n Y( Target)")
y.head()


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# In[17]:


scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


# In[19]:


rfc=MultiOutputClassifier(RandomForestClassifier(random_state=42))


# In[20]:


rfc.fit(x_train_scaled,y_train)


# In[21]:


y_pred=rfc.predict(x_test_scaled)


# In[22]:


y.shape


# In[23]:


y_pred[0]


# In[24]:


from sklearn.metrics import accuracy_score, classification_report


# In[25]:


print("Accuracy for MultiOutput Classifier:", accuracy_score(y_test, y_pred))


# In[27]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier


# In[28]:


dtc=DecisionTreeClassifier(random_state=42)


# In[29]:


multi_target_clf = MultiOutputClassifier(dtc, n_jobs=-1)

# Train the model
multi_target_clf.fit(x_train_scaled, y_train)

# Predict on test data
y_pred = multi_target_clf.predict(x_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model: {:.2f}%".format(accuracy * 100))

# Print the predictions for the test set
print("\nPredictions for the test set:")
print(y_pred)


# In[30]:


new_patient = pd.DataFrame({
    'Cholesterol (mg/dL)': [225],
    'Hemoglobin (g/dL)': [15.019743],
    'Vitamin B12 (pg/mL)': [473.103410],
    'Vitamin D (ng/mL)': [34.935414],
    'Glucose': [107],
    'Platelet Count (x10^3/µL)': [275],
    "WBC (x10^3/µL)":[10.221975],
    "RBC (million/µL)":[5.486683],
    "Calcium (mg/dL)":[8.903136],
})

# Standardizing the new input data
new_patient_scaled = scaler.transform(new_patient)

# Predicting the deficiencies
new_prediction = rfc.predict(new_patient_scaled)

# Converting binary predictions back to deficiencies

print("Predicted Deficiencies for New Patient:", new_prediction)


# In[31]:


new_patient = pd.DataFrame({
    'Cholesterol (mg/dL)': [225],
    'Hemoglobin (g/dL)': [15.019743],
    'Vitamin B12 (pg/mL)': [473.103410],
    'Vitamin D (ng/mL)': [34.935414],
    'Glucose': [107],
    'Platelet Count (x10^3/µL)': [275],
    "WBC (x10^3/µL)":[10.221975],
    "RBC (million/µL)":[5.486683],
    "Calcium (mg/dL)":[8.903136],
})

# Standardizing the new input data
new_patient_scaled = scaler.transform(new_patient)

# Predicting the deficiencies
new_prediction = multi_target_clf.predict(new_patient_scaled)

# Converting binary predictions back to deficiencies

print("Predicted Deficiencies for New Patient:", new_prediction)


# In[32]:


new_data_2=pd.DataFrame([X.iloc[2]])


# In[33]:


new_patient_scaled_2 = scaler.transform(new_data_2)

# Predicting the deficiencies
new_prediction_2 = multi_target_clf.predict(new_patient_scaled_2)

# Converting binary predictions back to deficiencies

print("Predicted Deficiencies for New Patient:", new_prediction_2)


# In[41]:


def map_deficiencies(prediction):
    deficiency_labels=data['Deficiencies'].unique()
    
    deficiencies = []
    for i, val in enumerate(prediction[0]):  # Since prediction is 2D array
        if val == 1:
            deficiencies.append(deficiency_labels[i])
    
    return deficiencies

# Example prediction
predicted_deficiency = [[0, 0, 0, 0, 1, 0, 0, 0]]

# Get actual deficiency names
actual_deficiencies = map_deficiencies(new_prediction_2)

print("Predicted Deficiencies:", actual_deficiencies)


# In[51]:


deficiency_labels=data['Deficiencies'].unique()


# In[52]:


deficiency_labels


# In[44]:


from sklearn.ensemble import BaggingClassifier


# In[47]:


#best decision tree classifier
base_estimator = DecisionTreeClassifier(random_state=42)
bagging_clf = MultiOutputClassifier(BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators=10,
    random_state=42
))

# Fit the bagging classifier
bagging_clf.fit(x_train_scaled, y_train)

# Make predictions
y_pred = bagging_clf.predict(x_test_scaled)

# Evaluate accuracy for multi-output classification
# (subset accuracy: all labels must match exactly)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (subset): {accuracy:.2f}")


# In[46]:


# beat rf estimator
base_estimator_rf = RandomForestClassifier(random_state=42)

# Create the MultiOutputClassifier with BaggingClassifier and Random Forest as the base estimator
bagging_clf_rf = MultiOutputClassifier(BaggingClassifier(
    base_estimator=base_estimator_rf,
    n_estimators=10,
    random_state=42
))

# Fit the bagging classifier with the training data
bagging_clf_rf.fit(x_train_scaled, y_train)

# Make predictions on the test data
y_pred_rf = bagging_clf_rf.predict(x_test_scaled)

# Evaluate accuracy for multi-output classification
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy (subset) for Random Forest: {accuracy_rf:.2f}")


# In[48]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

# Create the base models for Bagging
base_estimator_dt = DecisionTreeClassifier(random_state=42)
base_estimator_rf = RandomForestClassifier(random_state=42)

# Create Bagging classifiers with DTC and RFC as base estimators
bagging_clf_dt = MultiOutputClassifier(BaggingClassifier(base_estimator=base_estimator_dt, n_estimators=10, random_state=42))
bagging_clf_rf = MultiOutputClassifier(BaggingClassifier(base_estimator=base_estimator_rf, n_estimators=10, random_state=42))

# Fit both classifiers
bagging_clf_dt.fit(x_train_scaled, y_train)
bagging_clf_rf.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred_dt = bagging_clf_dt.predict(x_test_scaled)
y_pred_rf = bagging_clf_rf.predict(x_test_scaled)

# Calculate accuracy scores
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy of Bagging with Decision Tree: {accuracy_dt:.2f}")
print(f"Accuracy of Bagging with Random Forest: {accuracy_rf:.2f}")


# In[49]:


# Choose the best model based on accuracy
if accuracy_dt > accuracy_rf:
    best_model = bagging_clf_dt
    print("Decision Tree Classifier has the highest accuracy.")
else:
    best_model = bagging_clf_rf
    print("Random Forest Classifier has the highest accuracy.")


# In[50]:


from joblib import dump

# Save the best model
dump(best_model, 'best_model.joblib')


# In[ ]:


# Assuming user_input is the DataFrame with the user's input data
predictions = loaded_best_model.predict(user_input)

# Optionally, map the predicted binary labels to actual deficiency labels
actual_deficiencies = map_deficiencies(predictions)

print("Predicted Deficiencies:", actual_deficiencies)


# In[ ]:





# In[ ]:




