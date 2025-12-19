Porównanie wyników klasyfikacji zbioru wheat_seeds_dataset.csv uzyskanych przy użyciu Sieci Neuronowych oraz algorytmu SVM (z poprzedni zajęć).

### SVM vs Sieć neuronowa - dokładność klasyfikacji
Wyniki klasyfikacji nasion pszenicy uzyskane przy pomocy klasycznego algorytmu **SVM** (z poprzednich zajęć) oraz **Sieci Neuronowych**.
Do porównania wybrano najlepsze konfiguracje obu podejść.

| Model   | Konfiguracja                                                                                                                                                                                              | Dokładność |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| **SVM** | **Linear (C=1.0)**                                                                                                                                                                                        | **0.9048** |
| **SVM** | **Linear, (C=0.01)**                                                                                                                                                                                      | **0.9048** |
| **SVM** | **Poly, (degree=3)**                                                                                                                                                                                      | **0.8333** |
| **SVM** | **RBF, (gamma=scale)**                                                                                                                                                                                    | **0.9048** |
| **SVM** | **Sigmoid, (C=1.0)**                                                                                                                                                                                      | **0.9048** |
| **MLP** | **warstwa wejściowa (7 cech) <br/>warstwa ukryta (128 neuronów) + Dropout (0.3) <br/>warstwa ukryta (64 neurony) + Dropout (0.2) <br/>warstwa ukryta (32 neurony) + Dropout (0.1) <br/>warstwa wyjściowa (3 klasy, softmax)** | **0.9048** |

### Wnioski z porównania:
* Zarówno większość modeli SVM, jak i Model MLP większy osiągnęły identyczny wynik. Każdy z tych modeli pomylił się tylko w 4 przypadkach, z wyjątkiem SVM z jądrem wielomianowym (Poly), który popełnił aż 7 błędów.
* Wynik modelu SVM z jądrem liniowym (Linear) na poziomie 90.48% jednoznacznie pokazuje, że cechy nasion (area, perimeter itp.) można skutecznie oddzielić prostymi granicami. Sieć neuronowa, dzięki swojej złożoności, była w stanie tą liniowość idealnie odwzorować.
* Czas trenowania - Modele SVM, zwłaszcza z jądrem liniowym, trenowały się znacznie szybciej niż większy model MLP. Dla małych zbiorów danych, takich jak wheat_seeds_dataset.csv, SVM jest często bardziej efektywny pod względem czasu obliczeń.
* Złożoność modelu - Mimo że większy model MLP osiągnął ten sam wynik co SVM, jego złożoność (wielowarstwowa architektura z Dropoutem) może być nadmierna dla tak prostego problemu. W praktyce, dla małych i liniowo separowalnych zbiorów danych, prostsze modele jak SVM są często bardziej efektywne.
* Skalowanie - Zbiór *Wheat Seeds* jest mały. W takim przypadku **SVM Linear** wygrywa prostotą i szybkością trenowania, o czym wspomnieliśmy wyżej. Jednak Sieć neuronowa wydaje się o wiele lepszą opcją, jeśli zbiór danych bardzo by się powiększył.
* Łatwość dostosowania - Sieci neurone są o wiele bardziej elastyczne, ponieważ nieliniowość dzieje się sama. Natomiast jeśli chodzi o SVM, to musimy dobrze wybrać jądro oraz jego parametry na podstawie natury danych
* Przyszłościowe użycie - Taki wytrenowany model sieci, może odrazu posłuzyć do klasyfikacji np. zbioru użytego w poprzednich zajęciach - apple_quality_dataset.csv, gdzie dane są nieliniowe. W przypadku SVM, konifguracja liniowa już nie zadziała, i trzeba by ręcznie dobrać konfigurację.
