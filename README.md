# TFM – Contribuciones a la computación cuántica adiabática en hardware de átomos neutros

> Trabajo de Fin de Máster – Máster en Ciencia de Datos

> Autor: Aitor Arribas  

> Tutor: José David Martín Guerrero

> Cotutora: Yolanda Vives Gilabert

> Universidad: Universidad de Valencia

> Año: 2025

---

## 📋 Descripción

Este repositorio contiene el código, datos y documentación asociados al Trabajo de Fin de Máster titulado:

> *Contribuciones a la computación cuántica adiabática en hardware de átomos neutros*

##### Resúmen:

Este trabajo investiga la resolución del problema del conjunto independiente máximo ponderado (\textit{Maximum Weighted Independent Set}, MWIS), un problema de optimización combinatoria en la teoría de grafos, mediante el uso de ordenadores cuánticos basados en átomos neutros. Para abordar este problema, se ha empleado un simulador cuántico donde se representa el grafo mediante una disposición de átomos en una cuadrícula. En esta configuración, los enlaces entre los nodos del grafo se reflejan en las interacciones entre los átomos. La solución se obtiene mediante una evolución adiabática del sistema, comenzando desde un estado inicial sencillo hasta alcanzar un estado final cuya configuración corresponde a la solución óptima del MWIS.

Para maximizar la probabilidad de obtener dicha solución óptima, se han aplicado tres algoritmos de optimización diferentes con el propósito de identificar aquellos protocolos que ofrecen el mejor rendimiento en la mayoría de los casos estudiados.

Además, el problema se ha analizado en profundidad, identificando características específicas de los grafos que dificultan su resolución. Se propone un nuevo parámetro (\textit{Hardness Parameter}) basado en la estructura espectral de la función de coste del MWIS, que, combinado con otras variables espectrales, permite entrenar un modelo predictivo capaz de anticipar situaciones en las que el simulador podría fallar.

Finalmente, se evalúa el rendimiento de la metodología propuesta sobre un conjunto diverso de grafos y se analiza su escalabilidad al aplicarse a grafos de mayor tamaño.

**Palabras clave:** Optimización de _drivings_ · _Quantum annealing_ · Conjunto Independiente Máximo Ponderado · Computación Cuántica · Átomos neutros

---

## Instalación y entorno

Este proyecto ha sido desarrollado con **Python 3.10** en un entorno **conda** llamado `TFM_Aitor`. Sigue los pasos para reproducir el entorno en tu equipo.

##### 1. Clona el repositorio

```bash
git clone https://github.com/AitorArribas/TFM_Aitor_Arribas.git
cd TFM_Aitor_Arribas
```

##### 2. Crea y activa el entorno

```bash
conda create --name TFM_Aitor python=3.10
conda activate TFM_Aitor

```

##### 3. Instalar la librería de amazon braket

```bash
git clone https://github.com/amazon-braket/amazon-braket-sdk-python.git
cd amazon-braket-sdk-python
pip install .
```
Se ha observado en algunos casos que este paso produce un error por la librería `sympy`. Para solucionarlo, se recomienda, una vez clonado el repositorio, entrar dentro del archivo `setup.py` y comentar la línea donde pone `sympy`. Después, repite la instalación mediante `pip install .`

```bash
cd ../
```


##### 4. Instalar el resto de librerías

```bash
pip install -r requirements.txt
```

##### 4. Instala los utils

```bash
pip install -e .
```

