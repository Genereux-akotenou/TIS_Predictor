## Benchmarck Table
<!-- <img src="./ui/static/TIS_vs_Prodigal.png"/> -->

| Method  | Bacteria                                      | Total Verified CDS | Prodigal Matched | Prodigal Missed | Prodigal Total Found | TIS_Annotator Matched | TIS_Annotator Missed | TIS_Annotator Total Found |
|---------|----------------------------------------------|--------------------|------------------|----------------|------------------|------------------|----------------|------------------|
|         | **Escherichia coli K-12 MG1655**             | 769                | 338              | 431            | 4347             | 744              | 25             | 4213             |
|         | **Halobacterium salinarum R1**               | 530                | 243              | 287            | 2851             | 438              | 92             | 2659             |
|         | **Mycobacterium tuberculosis H37Rv**         | 701                | 311              | 390            | 4204             | 626              | 75             | 3853             |
|         | **Natronomonas pharaonis DSM 2160**          | 315                | 169              | 146            | 2873             | 248              | 67             | 2737             |
|         | **Roseobacter denitrificans Och114**         | 526                | 0                | 526            | 4120             | 492              | 34             | 4006             |


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
