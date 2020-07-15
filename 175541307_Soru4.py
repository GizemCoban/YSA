# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#verisetinin yüklenmesi
veri=pd.read_csv("diyabet.csv")

#sınıf sayısını belirle
label_encoder=LabelEncoder().fit(veri["class"])
labels=label_encoder.transform(veri["class"])
classes=list(label_encoder.classes_)  #52 tane çıktı

x=veri.drop(["class"], axis=1)
y=labels





#verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)



#eğitim ve test verilerinin hazırlanması
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2) #%20sini testte kullanma


#çıktı değerlerinin kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)




#YSA Modelinin oluşturulması
from tensorflow.keras.models import Sequential

#Dense ysa oluşturacağımız kısımdır.
from tensorflow.keras.layers import Dense

#16 tane nöron olsun. 20 tane alanımız bulunmakta. activasyonumuzda relu olsun
model=Sequential()
#girdi katmanı
model.add(Dense(16,input_dim=8,activation="relu"))
#ara katman
model.add(Dense(12,activation="relu"))

#çıktı katmanı
#çıktı sayısı kaçsa o kadar nöron olmak zorunda. 
#activasyonu softmax olmak zorunda çünkü sınıflandırma yapılmaktadır.
model.add(Dense(2,activation="softmax"))
model.summary()



#Modelin Derlenmesi
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

#modelin eğitilmesi
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50)


#Eğitim ve Doğrulama Başarımlarının Gösterilmesi
import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show


#Eğitim ve Doğrulama Kayıplarının Gösterilmesi
import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show





