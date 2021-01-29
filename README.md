# ELiDi  
This repository includes all the code and data for the paper ELiDi ([End-to-end Entity Linking and Disambiguation leveraging Word and Knowledge Graph Embeddings](https://openreview.net/pdf/cae4393d0817ad50aee6065e11a4d7487f8c4344.pdf)).

### Abstract: 
Entity linking – connecting entity mentions in a natural language utterance to knowledge graph (KG) entities is a crucial step for question answering over KGs. It is often based on measuring the string similarity between the entity label and its mention in the question. The relation referred to in the question can help to disambiguate between entities with the same label. This can be misleading if an incorrect relation has been identified in the relation linking step. However, an incorrect relation may still be semantically similar to the relation in which the correct entity forms a triple within the KG; which could be captured by the similarity of their KG embeddings. Based on this idea, we propose the first end-to-end neural network approach that employs KG as well as word embeddings to perform joint relation and entity classification of simple questions while implicitly performing entity disambiguation with the help of a novel gating mechanism. An empirical evaluation shows that the proposed approach achieves a performance comparable to state-of-the-art entity linking while requiring less post-processing. Moreover, this model trained on a question answering dataset can be directly applied to other datasets without any further training, in a zero-shot setting. The pre-trained model along with the corresponding software for entity linking is the main contribution of this work.

![](https://github.com/rashad101/ELiDi/blob/main/elidi-demo.gif)
### 🔧 Installation:
It is really simple to setup ELiDi !!

ELiDi is developed using python 3.8. Other version of python>=3.6 might also work. However, it is recommended to use python 3.8 to avoid unwanted bugs.
* Python 3.8 (required)
* [Anaconda](https://www.anaconda.com/products/individual) 

If you don't have Anaconda installed, then install it and make sure that the installation path is added in the system environment.
Now, run the following commands to install the required libraries
```shell
conda create -n elidi -y python=3.8 && conda activate elidi
pip install -r requirements.txt
```
Now download the required files (precessed SimpleQuestion data and Freebase file) for running the system by executing:
```python
python utils/download.py
```
After executing the command the ```/data``` directory should look like this:
```
ELiDi
├── data/
    ├── FB2M/
    │     ├── ent2emb_id.json
    │     ├── entity_1hop_buba.npy
    │     ├── relation2vec.vec
    │     └── reln2id.json
    ├── freebase/
    │     └── names.trimmed.2M.txt
    ├── processed_simplequestions_dataset/
    │      │  └── webqsp/
    │      │       └── webqsp_wd-test.json
    │      ├── sq_wikidata.txt
    │      ├── test100.txt
    │      ├── train100.txt
    │      └── valid100.txt
    ├── fb2w_id.pkl
    └── sq_glove300d.pt
```

### 🌐 Quick start: Demo
In order to run the web demo, first, complete the [installation step](https://github.com/rashad101/ELiDi#-installation) . Then, simply run the following command in your terminal:
```python
python app.py
```
Now, open your browser and go to the following address:
```
http://localhost:3355/elidi
```



### 💻 Interactive CLI:
An interactive command line interface (CLI) also comes with this repo. To use the interactive CLI, run:

In CPU:
```python
python e2e_cli.py 
```
With GPU:
```python
python e2e_cli.py --gpu 1
```
Running the CLI will prompt the user with a question ```Please type your question (type q to quit):  ```. Asking a question will return the user with a WikiData ID with it's corresponding entity label. Press ```q``` and then ENTER to exit from the CLI.

### 🏋️ Training
[Installation]((https://github.com/rashad101/ELiDi#-installation)) of ELiDi will equip the project with ready to use pre-trained model. In order to train the model from the scratch run the following command:
```python
python train_e2e.py --dataset <DATASET_NAME>
```
Available options for ```<DATASET_NAME>``` are ```sq``` and ```webqsp```. Trained models are saved inside the ```/models``` directory.
### 🎯 Testing
In order the test the trained model run the following command. Running the command will load the trained model automatically and run it over the test set of the given dataset.
```python
python train_e2e.py --eval_only --dataset <DATASET_NAME>
```
Running the command  will also save the predicted outputs in a JSON file inside the directory ```/predictions```.
### ⚖️ Evaluation
The following command evaluates the predicted output and return the evaluated score in Precision, Recall and F1-score.
```python
python utils/eval.py --dataset <DATASET_NAME>
```

### 🐳 Docker
Run the program in Docker:
```dockerfile
sudo docker-compose up --build
```
Alternatively, try the following commands:
```dockerfile
sudo docker build -t "elidi:Dockerfile" .
sudo docker run -d -p 3355:3355 elidi:Dockerfile
```

### 📝 Citation
Please cite the paper if you use any of the part of this work.
### 📜 License  <a href='https://opensource.org/licenses/MIT'><img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/></a>
