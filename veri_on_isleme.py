# 1) kütüphanelerin yüklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2) veri ön işleme

# 2.1) veri yükleme

#pd.read_csv('veriler.csv')
veriler = pd.read_csv('veriler.csv')


boy = veriler[['boy']]
print(boy)

boy_kilo = veriler[['boy', 'kilo']]
print(boy_kilo)


# x = 10

# class insan:
#     boy = 180
#     def kosmak(self, b): 
#         return b + 10
    
# ali = insan()
# print(ali.boy)
# print(ali.kosmak(90))


# 3) eksik veriler

from sklearn.impute import SimpleImputer

eksik_veriler = pd.read_csv('eksikveriler.csv')

#imputer SimpleImputer sınıfından oluşturulan bir nesne(object) 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# eksikveriler.csv den 1,2 ve 3. kolonlar alınır
yas = eksik_veriler.iloc[:,1:4].values
print(yas)

#yaşın 1den 4e kadar olan kolonlarını öğrenir. öğrendiği şey kolonların ortalama değerleri
imputer = imputer.fit(yas[:,1:4])

#öğrendikten sonra nan değerleri ortalama değere dönüşecek
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)


#kategorik verileri sayısal forma dönüştürme

ulke = veriler.iloc[:,0:1].values
print(ulke)


from sklearn import preprocessing

#LabelEncoder, kategorik string verileri sayısal değerlere dönüştürür.
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke) #numpy dizisi

#OneHotEncoder, sayısal etiketleri vektörlere dönüştürür.
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


print(list(range(22)))

#numpy dizileri dataframe dönüştürme

sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = yas, index = range(22), columns = ["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])
print(sonuc3)


#dataframe birleştirme

s1 = pd.concat([sonuc, sonuc2], axis=1)
print(s1)
s2 = pd.concat([s1, sonuc3], axis = 1)
print(s2)


#veri kümesinin bölünmesi

from sklearn.model_selection import train_test_split

# x -> bağımsız değişken
# y -> bağımlı değişken

x_train, x_test, y_train, y_test = train_test_split(s1, sonuc3, test_size=0.33, random_state=0)


# ölçeklendirme(scaling)

from sklearn.preprocessing import StandardScaler

#Bu nesne ile fit() ve transform() işlemleri yapılacak.
sc  = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


