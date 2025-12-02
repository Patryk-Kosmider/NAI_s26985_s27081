# Wpływ wyboru kernela i jego parametrów dla SVM

Na przykładzie zbioru wheat seeds dataset.<br>
Analiza klasyfikacji zbioru danych wykazała, że najlepszą wydajność osiągnięto dzięki modelom SVM, które generują proste, liniowe granice decyzyjne (linear i sigmoid).

### Wyniki:

*  Najwyższa dokładność (0.9048) została osiągnięta przez modele SVM z jądrem i parametrami - **Linear (C=1.0)**,  **RBF (gamma='scale')** oraz **Sigmoid (C=1.0)**. Jądro Liniowe jest preferowane, ponieważ widać, że dane są liniowo separowalne. Mimo to również bardzo dobrze poradził sobie model RBF, który jest jądrem gaussowskim więc uniwersalnym.
* **Wpływ Parametrów:**
    * **C i Gamma:** Najlepsze wyniki uzyskano przy **bazowych/domyślnych** ustawieniach parametrów. Zmiana parametru **C** (kara za błąd) lub **gamma** (zasięg wpływu) w większości przypadków **obniżała** dokładność (np. Linear C=0.01 $\rightarrow$ 0.8810).
    * **Degree (Stopień):** Użycie jądra **Polynomial (Poly)** ze zwiększoną złożonością (**degree=5**) doprowadziło do **najniższej dokładności (0.7619)**, co jest wyraźnym sygnałem **overfittingu** (nadmiernego dopasowania) na tym prostym zbiorze.

**Wniosek:** Zbiór jest **liniowo separowalny**, dlatego złożoność nieliniowa w modelach SVM jest zbędna i wprowadza błędy.


Na przykładzie zbioru apple quality dataset. <br>

Analiza klasyfikacji zbioru danych wykazała, że najlepszą wydajność osiągnięto dzięki modelom SVM, które generują nieliniowe granice decyzyjne.

### Wyniki:

* Najwyższa dokładność (0.9038), została osiągnięta przez model SVM z jądrem **RBF (gamma='scale')**. Jądra Liniowe **(Linear i Sigmoid)** okazały się nieskuteczne (dokładność poniżej 0.76). Jądro **RBF** (Gaussowskie) było optymalne, dzięki swojej uniwersalności i zdolności do modelowania złożonych zależności.

### Wpływ Parametrów:

* **C i Gamma:** Najlepsze wyniki uzyskano przy **bazowych/domyślnych** ustawieniach parametrów. Zmiana parametru **C** (kara za błąd) lub **gamma** (zasięg wpływu) w większości przypadków **obniżała** dokładność (np. RBF gamma=0.01 $\rightarrow$ 0.8150).
* **Degree (Stopień):** Użycie jądra **Polynomial (Poly)** ze zwiększoną złożonością (**degree=5**) doprowadziło do obniżenia dokładności (**0.7712**).

**Wniosek:** Zbiór jest **nieliniowo separowalny i wymieszany**. Wymaga to zastosowania **jądra RBF** z odpowiednio dobraną wartością **gamma**, aby stworzyć złożoną granicę decyzyjną i osiągnąć **najwyższą dokładność**.


Dokładne wyniki i screenshoty są zamieszczone w folderach - wyniki_apple_quality oraz wyniki_wheat_seeds