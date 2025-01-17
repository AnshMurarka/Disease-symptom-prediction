#Importing various necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix



#Importing all the available csv file
df1=pd.DataFrame()
df1=pd.read_csv('/kaggle/input/disease-symptom-description-dataset/Symptom-severity.csv')
df2=pd.DataFrame()
df2=pd.read_csv('/kaggle/input/disease-symptom-description-dataset/dataset.csv')
df3=pd.DataFrame()
df3=pd.read_csv('/kaggle/input/disease-symptom-description-dataset/symptom_Description.csv')
df4=pd.DataFrame()
df4=pd.read_csv('/kaggle/input/disease-symptom-description-dataset/symptom_precaution.csv')




#Visualizing each of the dataframe 
df1.head()
df1.tail()

df2.head()
df2.tail()

df3.head()
df3.tail()

df4.head()
df4.tail()



#Getting the overall discription of the model
df1.info()
df2.info()
df3.info()
df4.info()

len(df2['Disease'].unique()) #Total number of unique disease





# Cleaning and processing of data
#DataFrame with name: dataset.csv or df2 (as in code included)
for col in df2.columns:
    df2[col] = df2[col].str.replace('_',' ') #Replacing all the underscore with blank space in all the columns
df2.head()

# Plotting a graph for NaN values
null_count = df2.isnull().sum()

plt.figure(figsize=(10, 5))
plt.plot(null_count.index, null_count.values, marker='o')
plt.xticks(rotation=45, horizontalalignment='right')
plt.title('Before Removing Null Values')
plt.xlabel('Column Names')
plt.ylabel('Count of Null Values')
plt.margins(0.1)
plt.tight_layout()
plt.show()

# Removing trailing space
cols = df2.columns
data = df2[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df2.shape)

df = pd.DataFrame(s, columns=df2.columns)
df.head()


#Replacing all the nan values with 0
df = df.fillna(0)
df.head()




# Sympton severity rank
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
df1.head()

df1['Symptom'].unique()
len(df1['Symptom'].unique()) #Unique number of Symptoms present in dataset



#Enconding symptoms in data with symptoms rank
cols = df.columns
vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)
d.head()
d.tail()

#Assign symptoms with no rank to zero

d = d.replace('dischromic  patches', 0)
d = d.replace('spotting  urination',0)
df = d.replace('foul smell of urine',0)
df.head(10)



print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
print("Number of diseases that can be identified ",len(df['Disease'].unique()))
data = df.drop(columns=['Disease']).to_numpy()
labels= df['Disease'].to_numpy()



#Train test split
# Training set 80% and test set 20%
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
predicts=model.predict(x_test)

print('Accuracy =', accuracy_score(y_test, predicts))
print('Precision =', precision_score(y_test, predicts,average='macro'))
print('Recall socre= ',recall_score(y_test,predicts,average='macro'))
print('F1-score =', f1_score(y_test, predicts, average='macro'))
probas_ = model.predict_proba(x_test)  # predicted probabilities for all classes
print('ROC-AUC Score =', roc_auc_score(y_test, probas_, average='macro', multi_class='ovr'))

# Printing confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test,predicts))




#Code for individual input
def predd(S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17):
    #Combining symptoms into list
    psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
    print("Input symptoms:", psymptoms)

    #Map symptoms to weights
    symptom_to_weight = dict(zip(df1["Symptom"], df1["weight"]))
    psymptoms = [symptom_to_weight.get(symptom, 0) for symptom in psymptoms]
    print("Mapped symptoms to weights:", psymptoms)

    # Ensure input is 2D for the model
    psy = np.array([psymptoms])  # Ensure 2D array
    
    #Predict disease
    pred = model.predict(psy)
    probas_ = model.predict_proba(psy)
    
    print("Predicted disease:", pred[0])
    print("Predicted probabilities:", probas_)
    disease = np.array(df3["Disease"])
    if pred[0] in disease:
        j = np.where(disease == pred[0])[0][0]
        class_index = np.where(model.classes_ == pred[0])[0][0]
        confidence = probas_[0][class_index]
        
        print("Confidence level for prediction:", confidence)
        precaution = df4.iloc[j, 1:]
        print(precaution)
    else:
        print("Predicted disease not found in the disease list.")





# Printing all the available symptoms
sympList=df1["Symptom"].to_list()
print(sympList)




predd(sympList[7],sympList[5],sympList[2],sympList[80],0,0,0,0,0,0,0,0,0,0,0,0,0) #Manual Input 1
predd(sympList[8],sympList[5],sympList[2],sympList[80],0,0,0,0,0,0,0,0,0,0,0,0,0) #Manual input 2
