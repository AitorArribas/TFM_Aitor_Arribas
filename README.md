# TFM – Quantum Optimization Using Rydberg Atom Simulators

> Trabajo de Fin de Máster – Máster en Ciencia de Datos

> Autor: Aitor Arribas  

> Tutor: José David Martín Guerrero

> Cotutora: Yolanda Vives Gilabert

> Universidad: Universidad de Valencia

> Año: 2025

---

## 📋 Descripción

Este repositorio contiene el código, datos y documentación asociados al Trabajo de Fin de Máster titulado:

> *"Título"*

El objetivo del proyecto es...

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

## Estructura del repositorio


## Resultados principales