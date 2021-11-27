import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("D:\maktabkhooneh\ML\housePrice.csv")

df = df.dropna()

df['Area'] = df[~df['Area'].str.contains(',', na=False)]['Area']

df['Area'] = pd.to_numeric(df['Area'])

df = df[~(df['Room']==0)]

df = df.dropna()

df.Parking = df.Parking.astype(int)
df.Warehouse = df.Warehouse.astype(int)
df.Elevator = df.Elevator.astype(int)
df.Area = df.Area.astype(int)

df['sort'] = df.groupby('Address').transform('mean')['Price(USD)']/df.groupby('Address').transform('mean')['Area']

Address_df = df.groupby('Address').mean()['sort'].reset_index()
Address_df = Address_df.sort_values(by=['sort'], ascending=True).reset_index()
Address_df.insert(0, 'codedAddress', range(1, 193))
df1 = pd.Series(Address_df.codedAddress.values,index=Address_df.Address).to_dict()
df["Address_n"] = df["Address"].map(df1)
df = df.drop(['sort'], axis=1)

bf = df.groupby('Address').max()['Address_n']
address_dict = bf.to_dict()

# Features:
X = df.drop(['Price(USD)'  , 'Price', 'Address'] , axis = 1)
# Label:
y = df['Price(USD)']

from sklearn.linear_model import LinearRegression

model = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

polynomial_converter = PolynomialFeatures(degree=2 , include_bias= False)
polynomial_features = polynomial_converter.fit_transform(X)
polynomial_features.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(polynomial_features, y, test_size=0.2, random_state=100)

model.fit(X_train, y_train)

coef = model.coef_
intercept = model.intercept_

from sklearn.preprocessing import PolynomialFeatures

# x = [Area, Room, Parking, Warehouse, Elevator, Address]
def predict_price(x):
    address_code = address_dict[x[-1]]
    x[-1] = address_code
    polynomial_converter = PolynomialFeatures(degree=2 , include_bias= False)
    xx2 = polynomial_converter.fit_transform([x2])
    return (intercept + np.dot(xx2, coef))*30000

x2 = [140, 3, 1, 1, 1, 'Seyed Khandan']
y2 = predict_price(x2)
print(y2)
