
RUNNING TIS Predictor UI

### Create environment
Firstly, we will create a conda environment called *rag_env*
```sh
python -m venv venv
```
Secondly, we will login to the *rag_env* environement
```sh
source ./venv/bin/activate
```
### Install prerequisite libraries

```sh
pip install -r requirements.txt
```

###  Launch the app
```
streamlit run ui/app.py
```

###  Launch the api
```
uvicorn --app-dir api api:app --reload
uvicorn --app-dir api api:app --host 10.52.88.33 --port 8000 --reload
python start.py 
```
