Porównanie dwóch wersji modelów MLP dla zbioru wheat_seeds_dataset.csv

### Architektura modeli


**Model mniejszy (`wheat_seeds_model`):** 
* warstwa wejściowa (7 cech)
* warstwa ukryta (64 neurony)
* warstwa ukryta (32 neurony)
* warstwa wyjściowa (3 klasy, softmax)

**Model większy (`wheat_seeds_model_two`):**
* warstwa wejściowa (7 cech)
* warstwa ukryta (64 neuronów) + Dropout (0.3)
* warstwa ukryta (32 neurony) + Dropout (0.2)
* warstwa ukryta (16 neurony) + Dropout (0.1)
* warstwa wyjściowa (3 klasy, softmax)



### Wynik modelu mniejszego
Dokładność: 0.8810

Raport klasyfikacji:
              precision    recall  f1-score   support

     Class 1       0.80      0.73      0.76        11
     Class 2       0.93      1.00      0.97        14
     Class 3       0.88      0.88      0.88        17

    accuracy                           0.88        42
   macro avg       0.87      0.87      0.87        42
weighted avg       0.88      0.88      0.88        42

### Wynik modelu większego
Dokładność: 0.9048

Raport klasyfikacji:
              precision    recall  f1-score   support

     Class 1       0.89      0.73      0.80        11
     Class 2       0.93      1.00      0.97        14
     Class 3       0.89      0.94      0.91        17

    accuracy                           0.90        42
   macro avg       0.90      0.89      0.89        42
weighted avg       0.90      0.90      0.90        42

### Wnioski z porównania
* Lepsza generalizacja, model poszerzony o dodatke warstwy ukryte i droputu, osiągnął wyższą celność na zbiorze testowym o ponad 2 punkty procentowe. 
* Większy model znacznie lepiej radzi sobie z identyfikacją nasion z Klasy 1, która była najtrudniejsza do sklasyfikowania w mniejszym modelu.
* Dodanie warstw Dropout w większym modelu pomogło zredukować przeuczenie, co jest widoczne w poprawie wyników na zbiorze testowym.
* Zbiór testowy wheat_seeds_dataset.csv jest stosunkowo mały, co ma wpływ na wyniki, ponieważ pojedyncze zmiana klasyfikacji może znacząco wpłynąć na metryki oceny.
