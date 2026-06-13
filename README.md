# PrediccionBursatilG3
# Sistema Web para Apoyo en Decisiones de Inversión usando IA
## Instrucciones de Ejecución de Interfaces (Frontend)

Este repositorio contiene la **Capa de Salida / Interfaces Gráficas (GUI)** desarrolladas para el trabajo de la Semana 10 del curso **iDeSo**. Las interfaces fueron construidas mediante el paradigma de Programación Asistida por IA, usando puramente HTML, CSS y JavaScript nativo.

### Requisitos Previos
Dado que todo el desarrollo se realizó para el frontend del sistema utilizando tecnologías web estándar, **no se requiere instalar dependencias complejas de Node.js, Python o bases de datos** para previsualizar los diseños. Necesitarás:
- Un navegador web moderno actualizado (Google Chrome, Mozilla Firefox, Microsoft Edge, o Safari).
- *(Opcional pero recomendado)* Una extensión de servidor local como **Live Server** en VSCode, o levantar un servidor local en tu terminal para evitar restricciones de seguridad restrictivas (CORS).

### Archivos del Proyecto
Asegúrate de tener los siguientes archivos en tu carpeta de trabajo:
- `01_autenticacion .html`
- `09_dashboard_completo .html`
- `Análisis de Noticias con NLP.html`
- `consola_modelos_ia.html`
- `dashboard_bursatil.html`
- `lstm_regressor_prediccion.html`
- `reporte_backtesting.html`
- `señales_broker.html`
- `svc_clasificador_tendencia.html`
- `HTML Gestion de portafolio.txt`
- `Generacion de estrategias.txt`

### Pasos para Ejecutar las Interfaces

#### 1. Preparación de Archivos `.txt`
Algunas interfaces fueron exportadas en formato de texto. Antes de abrirlas en el navegador, **debes cambiar su extensión a `.html`**:
- Renombra `HTML Gestion de portafolio.txt` a `gestion_portafolio.html`
- Renombra `Generacion de estrategias.txt` a `generacion_estrategias.html`

#### 2. Ejecución Directa
Haz **doble clic** en cualquiera de los archivos `.html`. El archivo se abrirá en tu navegador predeterminado y podrás interactuar directamente con los elementos visuales.

#### 3. Ejecución recomendada (Servidor HTTP Local)
Para compatibilidad total de los gráficos, abre tu terminal en la carpeta de los archivos y ejecuta (requiere Python):
`python -m http.server 8000`
Luego, accede a http://localhost:8000 en tu navegador.
