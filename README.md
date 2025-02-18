## Benchmarck Table
<img src="./ui/static/TIS_vs_Prodigal.png"/>

## Web tools guidelines
<img src="./ui/static/app.png"/>

#### 1. Create environment
Firstly, we will create a python environment called
```sh
python -m venv venv
```
Secondly, we will login to the environement
```sh
source ./venv/bin/activate
```
#### 2. Install prerequisite libraries

```sh
pip install -r requirements.txt
```

####  3. Launch the web tool ui
```
streamlit run ui/app.py
```

####  4. Launch the api
```
uvicorn --app-dir api api:app --host 127.0.0.1 --port 8000 --reload
```

#### 5. Start annotation

Go on the web tool page and submit a fasta/fna file containing your full genome sequence:

<img src="./ui/static/task.png"/>

The results should look like this: 

<img src="./ui/static/results.png"/>


<!-- 

uvicorn --app-dir api api:app --host 10.52.88.33 --port 8000 --reload 
python start.py 

-->
