# Buscador de Eventos de turismo PROCOLOMBIA


## Tabla de Contenidos

- [Instalación](#instalación)
- [Uso](#uso)
- [Componentes](#características)

## Instalación

Para instalar localmente el aplicativo realiza los siguientes pasos:

### Copia el repositorio en la carpeta local

```bash
git clone https://github.com/SergioRojasM/app-eventos-procolombia.git
```

### Crea el entorno virtual de python, compatible con python 3.9
```bash
cd nombre-del-proyecto
python -m venv .venv
```
### Activa el entorno virtual
```bash
# En Windows:
.\.venv\Scripts\activate
# En Linux:
source .venv/bin/activate 
```

### Instala las librerias necesarias
```bash
pip install -r requirements.txt
```


## Componentes

A continuacion la descripcion de los directorios del proyecto:

```bash
	app-eventos-procolombia/
	│
	├── .streamlit/				# Directorio oculto que contiene configuracion streamlit y llaves secretas 
	│   ├── config.toml/        	    # Archivo Configuracion Streamlit
	│   └── secrets.toml/       	    # Archivo con llaves de APIs a cargar en Streamlit
	├── .venv/				# Entorno virtual Python
	├── .gitignore				# Archivo con detallaes de los archivos a cargar y no cargar en GIT
	├── pages/				# Directorio oculto que contiene los datos y metadatos de Git
	│   ├── lib/        			# Librerias desarrolladas en python
	│   	├── config.py        	      	# Archivo local con configuracion de la app, generado automaticamente por la app
	│   	└── funciones_db.py    	    	# Libreria para conexion y administracion de bases de datos
	│  		└── funciones_llm.py    # Libreria para manejo de LLMs
	│   	└── funciones.py    	      	# Funciones generales del aplicativo
	│   └── 1-busqueda_auto.py       	# Pagina Streamlit con script aplicacion para busqueda automatica
	│   └── 2-busqueda_manual.py   	 	# Pagina Streamlit con script aplicacion para busqueda manual
	│   └── 3-busqueda_recursiva.py   	#  Pagina Streamlit con script aplicacion para busqueda recursiva
	│   └── 4-dashboard.py	    	    	# Pagina Streamlit con script aplicacion con Dashboard
	│   └── 5-Configuracion.py    	  	# Pagina Streamlit para configuracion de aplicativo
	│   └── 6-Dashboard_uso.py    	  	# Pagina Streamlit con script para el dashboard de uso
	├── src/				# Informacion fuente con archivos compementarios
	│   ├── data/        			# Imagenes fuente para Streamlit
	│   └── app_config.json/    	    	# Archivo con configuracion de aplicativo
	├── app_eventos_procolombia.py	  	# Pagina de entrada aplicativo Streamlit
	├── menu.py				# Script para pagina de Menu
	├── readme.txt				# Documentacion GIT 
	├── requirements.txt			# Librerias necesarias del proyecto 
```
