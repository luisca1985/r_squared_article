# Comprendiendo R-squared

## ¿Por Qué R-squared?
El Coeficiente de Determinación, conocido como R-squared, es una métrica ampliamente utilizada en estadística para medir la fuerza de la relación entre variables. Aunque frecuentemente se compara con el Coeficiente de Correlación de Pearson r, R-squared ofrece una interpretación más intuitiva y directa.

Pero ¿Por qué la necesidad del Coeficiente R-squared cuando ya se dispone del Coeficiente de Correlación r?

## La Relación entre R-squared y r

El R-squared, aunque similar a r, se distingue por su interpretación más directa. En el contexto de una regresión lineal, R-squared corresponde al cuadrado del Coeficiente de Correlación de Pearson r. Por ejemplo, un r estadísticamente significativo de 0,8 al cuadrado es 0,64, lo cual indica que la relación entre las dos variables explica el 64% de la variación en los datos. De forma similar, un r de 0,6 al cuadrado, que es 0,36, sugiere que la relación explica el 36% de la variación en los datos.

El Coeficiente R-squared es frecuentemente preferido sobre r por su claridad interpretativa. Consideremos un r de 0,8 en comparación con uno de 0,6. Al cuadrar estos valores para obtener R-squared, resulta que un R-squared de 0,64 explica el 64% de la variación original, mientras que un R-squared de 0,36 explica el 36%. Así, R-squared facilita la comprensión de que la primera correlación es aproximadamente 1,8 veces más explicativa que la segunda.

## Interpretando los Valores de R-squared

En un escenario donde se presenta un R-squared estadísticamente significativo de 0,85, se podría interpretar que la relación entre las dos variables explica el 85% de la variación en los datos. Por otro lado, un R-squared significativo de sólo 0,02 indica que, aunque la relación sea estadísticamente significativa, apenas explica el 2% de la variación en los datos, sugiriendo la influencia de otros factores en el restante 98%.

## Aplicando R-squared
Para ejemplificar la utilidad y el uso de R-scuared, se examina un conjunto de datos que ilustran la identificación, peso, altura y tiempo de traslado al trabajo de un grupo de individuos.


```python
import pandas as pd
import numpy as np

np.random.seed(0)

n = 30
id = range(1, n + 1)
# https://www.eltiempo.com/archivo/documento/CMS-13128617
height = np.random.normal(172, 8, n)
# Body mass index (BMI)
# https://en.wikipedia.org/wiki/Body_mass_index
# weight = BMI * height^2
bmi = 26 + np.random.normal(0, 2.5, n)
weight = bmi * ((height/100) ** 2)
commute_time = np.random.uniform(15, 60, n)

data = pd.DataFrame({'ID': id, 'Height': height, 'Weight': weight, 'BMI': bmi, 'Commute Time': commute_time})
data.round(1)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
      <th>Commute Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>186.1</td>
      <td>91.4</td>
      <td>26.4</td>
      <td>28.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>175.2</td>
      <td>82.7</td>
      <td>26.9</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>179.8</td>
      <td>76.9</td>
      <td>23.8</td>
      <td>29.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>189.9</td>
      <td>75.9</td>
      <td>21.0</td>
      <td>33.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>186.9</td>
      <td>87.8</td>
      <td>25.1</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>164.2</td>
      <td>71.1</td>
      <td>26.4</td>
      <td>46.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>179.6</td>
      <td>93.8</td>
      <td>29.1</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>170.8</td>
      <td>84.6</td>
      <td>29.0</td>
      <td>26.9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>171.2</td>
      <td>73.3</td>
      <td>25.0</td>
      <td>38.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>175.3</td>
      <td>77.6</td>
      <td>25.2</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>173.2</td>
      <td>70.1</td>
      <td>23.4</td>
      <td>40.9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>183.6</td>
      <td>75.7</td>
      <td>22.4</td>
      <td>56.8</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>178.1</td>
      <td>68.9</td>
      <td>21.7</td>
      <td>29.3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>173.0</td>
      <td>92.4</td>
      <td>30.9</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>175.6</td>
      <td>76.2</td>
      <td>24.7</td>
      <td>20.9</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>174.7</td>
      <td>76.0</td>
      <td>24.9</td>
      <td>47.2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>184.0</td>
      <td>77.4</td>
      <td>22.9</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>170.4</td>
      <td>81.1</td>
      <td>27.9</td>
      <td>23.2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>174.5</td>
      <td>66.9</td>
      <td>22.0</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>165.2</td>
      <td>69.5</td>
      <td>25.5</td>
      <td>15.9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>151.6</td>
      <td>54.6</td>
      <td>23.8</td>
      <td>52.3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>177.2</td>
      <td>84.7</td>
      <td>27.0</td>
      <td>15.2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>178.9</td>
      <td>79.1</td>
      <td>24.7</td>
      <td>45.5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>166.1</td>
      <td>63.6</td>
      <td>23.0</td>
      <td>27.2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>190.2</td>
      <td>93.8</td>
      <td>25.9</td>
      <td>48.1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>160.4</td>
      <td>69.6</td>
      <td>27.1</td>
      <td>58.3</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>172.4</td>
      <td>77.7</td>
      <td>26.2</td>
      <td>26.2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>170.5</td>
      <td>77.8</td>
      <td>26.8</td>
      <td>40.9</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>184.3</td>
      <td>82.9</td>
      <td>24.4</td>
      <td>41.6</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>183.8</td>
      <td>84.7</td>
      <td>25.1</td>
      <td>40.8</td>
    </tr>
  </tbody>
</table>
</div>

Primero, se grafica el peso de los individuos en el eje Y y los números de identificación en el eje X. Es posible calcular la media de los pesos y representarla como una línea horizontal en el gráfico. La variación de los datos alrededor de esta media se calcula sumando las diferencias al cuadrado (SS<sub>tot</sub>) entre el peso de cada individuo y la media. Las diferencias se elevan al cuadrado para asegurar que los valores por debajo de la media no contrarresten a los que están por encima


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
weight_mean = data['Weight'].mean()
sns.scatterplot(x='ID', y='Weight', data=data, color='red')
plt.axhline(y=weight_mean, color='gray')
for i in range(n):
    plt.plot([data['ID'][i], data['ID'][i]], [data['Weight'][i], weight_mean], color='gray', linestyle=':')
plt.title('Weight vs. ID')
plt.xlabel('ID')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_3_0.png)

Ahora, si en lugar de ordenar por el número de identificación, se ordenan por su altura, con el más bajo a la izquierda y el más alto a la derecha, la media y la variación siguen siendo las mismas que antes.


```python
sns.scatterplot(x='Height', y='Weight', data=data, color='red')
plt.axhline(y=weight_mean, color='gray')
for i in range(n):
    plt.plot([data['Height'][i], data['Height'][i]], [data['Weight'][i], weight_mean], color='gray', linestyle=':')
plt.title('Weight vs. Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_5_0.png)

De esta manera, se plantea la siguiente cuestión: dados la altura y el peso de un individuo, ¿representa la media el predictor más eficaz del peso? La respuesta es negativa. Una predicción más precisa del peso se logra mediante la aplicación de una **línea** de ajuste a los datos.

## Mejorando la Predicción con Líneas de Ajuste
La aplicación de una línea de ajuste a los datos conlleva una mejora significativa en la predicción del peso de un individuo basándose en su altura. Por ejemplo, conociendo la altura de una persona, se puede emplear la línea de ajuste para estimar su peso con mayor exactitud.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(data[['Height']], data['Weight'])
weight_pred = model.predict(data[['Height']])

sns.scatterplot(x='Height', y='Weight', data=data, color='red')
sns.lineplot(x=data['Height'], y=weight_pred, color='blue')
plt.axhline(y=weight_mean, color='gray')
plt.title('Weight vs. Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_7_0.png)

De esta manera, se plantea una pregunta fundamental: ¿Supera el ajuste de la línea azul, recientemente trazada, al de la media en términos de precisión? Y de ser así, ¿en qué medida?

A primera vista, la línea azul parece ofrecer un mejor ajuste a los datos que la media. Para cuantificar esta mejora, se emplea R-squared.

$$ R^2 = \frac{\text{SS}_\text{tot} - \text{SS}_\text{res}}{\text{SS}_\text{tot}} $$

La ecuación de R-squared se formula como la proporción de la variación total menos la variación residual sobre la variación total. La primera parte de esta ecuación refleja la variación alrededor de la media, calculada como la suma de las diferencias al cuadrado entre los valores reales de los datos y la media de estos.

$$ \text{SS}_\text{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 $$

Donde $y_i$ es el valor real del peso y $\bar{y}$ es la media del peso.

```python
sns.scatterplot(x='Height', y='Weight', data=data, color='red')
sns.lineplot(x=data['Height'], y=weight_pred, color='blue', alpha=0.1)
plt.axhline(y=weight_mean, color='gray')
for i in range(n):
    plt.plot([data['Height'][i], data['Height'][i]], [data['Weight'][i], weight_mean], color='gray', linestyle=':')
plt.title('Weight vs. Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_9_0.png)
    
La segunda componente de la ecuación representa la variación alrededor de la línea azul. Esta parte de la ecuación cuantifica la variación residual, que es la suma de las diferencias al cuadrado entre los valores reales de los datos y los valores predichos por la línea ajustada.

$$ \text{SS}_\text{res} = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 $$

Aquí, $y_i$ es el valor real del peso y $\hat{y_i}$ es el valor predicho por la línea azul.

```python
sns.scatterplot(x='Height', y='Weight', data=data, color='red')
sns.lineplot(x=data['Height'], y=weight_pred, color='blue')
plt.axhline(y=weight_mean, color='gray', alpha=0.1)
for i in range(n):
    plt.plot([data['Height'][i], data['Height'][i]], [data['Weight'][i], weight_pred[i]], color='blue', linestyle=':')
plt.title('Weight vs. Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_11_0.png)

## Calculando R-squared
El cálculo de R-squared implica el numerador, que representa la diferencia entre la variación total y la variación residual. Esta diferencia se divide por la variación total. Dicha operación hace que R-squared oscile entre cero y uno, reflejando que la variación alrededor de la línea ajustada nunca excede la variación total y siempre es un valor no negativo. Este procedimiento también transforma R-squared en una métrica porcentual.

Ahora, veamos el ejemplo.

```python
sns.regplot(x='Height', y='Weight', data=data, color='blue', scatter_kws={'s':50})
plt.title('Weight vs. Height')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_13_0.png)
    
```python
ss_tot = ((data['Weight'] - weight_mean) ** 2).sum()
ss_res = ((data['Weight'] - weight_pred) ** 2).sum()
r_squared = (ss_tot - ss_res) / ss_tot
ss_tot.round(2), ss_res.round(2), r_squared.round(3)
```
    (2426.42, 1371.45, 0.435)


- La variación total alrededor de la media (suma de las diferencias al cuadrado entre el peso y la media) es aproximadamente 2.426,2.
- La variación alrededor de la línea azul (suma de las diferencias al cuadrado entre el peso y las predicciones de la línea de regresión) es 1.371,45, lo que confirma que se ajusta mucho mejor a los datos.
- Al aplicar estos valores en nuestra fórmula para R-squared obtenemos 0,435 (o 43,5%).

Esto indica que la línea azul explica aproximadamente el 43,5% de la variación en el peso, basado en su altura. Esto sugiere una correlación fuerte entre el peso y el altura.

## Comparando Variables Desvinculadas
Se analiza un ejemplo adicional comparando dos variables potencialmente no correlacionadas: el peso de un individuo en el eje Y y su tiempo de traslado al trabajo en el eje X.


```python
sns.regplot(x='Commute Time', y='Weight', data=data, color='blue', scatter_kws={'s':50})
plt.title('Weight vs. Commute Time')
plt.xlabel('Time spent traveling to office (s)')
plt.ylabel('Weight (kg)')
plt.show()
```

![png](README_files/README_16_0.png)

```python
model_commute = LinearRegression()
model_commute.fit(data[['Commute Time']], data['Weight'])
weight_pred_commute = model_commute.predict(data[['Commute Time']])

ss_res_commute = ((data['Weight'] - weight_pred_commute) ** 2).sum()
r_squared_commute = (ss_tot - ss_res_commute) / ss_tot
ss_tot.round(2), ss_res_commute.round(2), r_squared_commute.round(3)
```
    (2426.42, 2360.25, 0.027)

- Al igual que antes, la variación total alrededor de la media es aproximadamente 2.426,42.
- Pero esta vez, la variación alrededor de la línea azul es mucho mayor, aproximadamente 2.360,25.
- Al introducir estos valores en la fórmula de R-squared es aproximadamente 0,027 (o 2,7%), lo que indica que la relación entre el tiempo de viaje al trabajo y el peso sólo explica el 2,7% de la variación total.

Este resultado sugiere que la relación entre el peso y el tiempo dedicado a viajar al trabajo explica sólo el 2,7% de la variación en el peso. Esto indica una correlación muy débil o inexistente entre estas dos variables en los datos simulados, lo cual es coherente con el análisis.

## Conclusiones

El Coeficiente de Determinación R-squared se destaca en estadística por su habilidad para expresar qué proporción de la variación en una variable dependiente es explicada por las variables independientes en un modelo. Esta capacidad se traduce en una medida porcentual, proporcionando una comprensión clara y directa del grado en que las variables están correlacionadas. En comparación con otras métricas, como el Coeficiente de Correlación de Pearson r, R-squared aporta una visión más tangible y cuantificable sobre las relaciones entre variables.
