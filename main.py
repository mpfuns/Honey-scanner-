#section 1
import pandas as pd
df = pd.read_csv('honey_purity_dataset.csv')
df.shape

# section 2
#converting Pollen_analysis value into string into a ingter that Represent the pollen 
# 0= Clover 1= Wildflower 2= Orange Blossom  3= Alfalfa 4= Acacia 5= Lavender
# 6= Eucalyptus 7= Buckwheat 8= Manuka 9=Sage 10= Sunflower 11=Borage 
# 12= Rosemary 13= Thyme 14=Heather 15=Tupelo 16=Blueberry 17=Chestnut 18= Avocado
pollenList=('Clover', 'Wildflower', 'Orange Blossom', 'Alfalfa', 'Acacia', 'Lavender', 'Eucalyptus', 'Buckwheat', 'Manuka', 'Sage', 'Sunflower', 'Borage', 'Rosemary', 'Thyme', 'Heather', 'Tupelo', 'Blueberry', 'Chestnut', 'Avocado')
#function
def addnumpollen():
    for index, pollen in enumerate(pollenList): 
     df.loc[df['Pollen_analysis']== pollen, 'Pollen_type']= index
     print(index,pollen)
addnumpollen()

# section 3
df.describe()

# section 4
clean_data = df.dropna()
clean_data.drop_duplicates
clean_data.shape

# section 5
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
model= LinearRegression()

 
X= clean_data.drop(columns= ['Price','Pollen_analysis'])
y= clean_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model.fit(X_train , y_train)
predictions = model.predict(X_test)
predictions

# section 6
mse= mean_squared_error(y_test, predictions)
r2= r2_score(y_test, predictions)
print('Mean squared error for test set:', mse , ' R-squared score:', r2)
