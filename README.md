# ELiDi
This repository includes all the code and data for the paper ELiDi (End-to-end Entity Linking and Disambiguation leveraging Word and Knowledge Graph Embeddings)

![](https://github.com/rashad101/ELiDi/blob/main/elidi-demo.gif)
### Installation:
* Python 3.8 (required)

Run the following commands to install the required libraries
```python
conda create -n elidi -y python=3.8 && conda activate elidi
pip install -r requirements.txt
```
Now download the required files by running:
```
python utils/download.py
```

### Interactive CLI:
To use the interactive command line interface, run:

In CPU:
```python
python e2e_cli.py
```
In GPU:
```python
python e2e_cli.py --gpu 1
```

### Web Demo:
First run the following command in your terminal:
```
python app.py
```
Now, open your browser and go to the following address:
```
http://localhost:3355/elidi
```
### Training
```
python train_e2e.py
```

### Evaluation
```
python utils/eval.py
```