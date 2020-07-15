# -- coding: utf-8 --
"""
Created on Sat Apr 25 01:37:08 2020

@author: Gizem ÇOBAN
"""

# -- coding: utf-8 --
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

def build_model():
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
    return model

ns_probs = [0 for _ in range(len(y_test))]

from keras.wrappers.scikit_learn import KerasClassifier
keras_model = build_model()
keras_model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1)

lr_probs = keras_model.predict_proba(x_test)
lr_probs = lr_probs[:, 1]


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

ns_auc = roc_auc_score(y_test[:,1], ns_probs)
lr_auc = roc_auc_score(y_test[:,1], lr_probs)

ns_fpr, ns_tpr, _ = roc_curve(y_test[:,1], ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test[:,1], lr_probs)


import matplotlib.pyplot as pyplot
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()