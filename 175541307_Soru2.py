import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#verisetinin yüklenmesi
veri=pd.read_csv("telefon_fiyat_değişimi.csv")


#sınıf sayısını belirle
label_encoder=LabelEncoder().fit(veri.price_range)
labels=label_encoder.transform(veri.price_range)
classes=list(label_encoder.classes_)

x=veri.drop(["price_range"], axis=1)
y=labels


#verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

#çıktı değerlerinin kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)

#YSA Modelinin oluşturulması
from tensorflow.keras.models import Sequential

#Dense ysa oluşturacağımız kısımdır.
from tensorflow.keras.layers import Dense

#veriyi k kadar bölme
k=5  
val_data_samples =  len(x)//k
all_scores = []

for i in range(k):
    print("işlem ", i)
    #test ve eğitim verilerini çapraz doğrulama ile ayırma
    val_data = x[i * val_data_samples: (i+1) * val_data_samples]
    val_targets = y[i * val_data_samples: (i+1) * val_data_samples]
    
    part_train_data = np.concatenate([x[:i * val_data_samples], x[(i + 1) * val_data_samples:]],axis=0)
    part_train_targets = np.concatenate([y[:i * val_data_samples], y[(i + 1) * val_data_samples:]],axis=0)
    

    #16 tane nöron olsun. 20 tane alanımız bulunmakta. activasyonumuzda relu olsun
    model=Sequential()
    #girdi katmanı
    model.add(Dense(16,input_dim=20,activation="relu"))
    #ara katman
    model.add(Dense(12,activation="relu"))
    
    #çıktı katmanı
    #çıktı sayısı kaçsa o kadar nöron olmak zorunda. 
    #activasyonu softmax olmak zorunda çünkü sınıflandırma yapılmaktadır.
    model.add(Dense(4,activation="softmax"))
    model.summary()
    
    #Modelin Derlenmesi
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    
    #modelin eğitilmesi
    val_accuracy = model.fit(part_train_data,part_train_targets,epochs=50)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

toplam=0
for i in range(len(all_scores)):
  toplam=toplam+all_scores[i]
ortalamabasari=toplam/(len(all_scores))
print("Ortalama Basari:",ortalamabasari)