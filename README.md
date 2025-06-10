# Clasificador TFM

Este repositorio contiene varios scripts y experimentos para la clasificación de textos. Se han reorganizado los archivos para facilitar su mantenimiento.

## Estructura

- `src/`
  - `scraping/`: utilidades para descarga y extracción de datos.
  - `preprocessing/`: código para limpiar y preparar los datos.
  - `models/`: implementaciones de diferentes modelos de clasificación.
  - `analysis/`: código de análisis de resultados.
  - `utils/`: funciones auxiliares.
- `data/`: archivos de datos de ejemplo.
- `docs/`: documentos y diagramas del proyecto.

Para ejecutar cualquier script se recomienda usar el modo módulo de Python, por ejemplo:

```bash
python -m src.preprocessing.prepare_data
```

## Entorno de trabajo

Se recomienda crear un entorno virtual para aislar las dependencias del proyecto.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
