<h1 align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Technion_logo.svg" alt="Technion Logo" height="100">
  <br>
  Advanced Information Retrieval - Final Project
</h1>

<p align="center">
  <em>
    Query Adaptive Contextual Document Embeddings
  </em>
</p>

<p align="center">
  <strong>Technion - Israel Institute of Technology</strong> <br>
  Faculty of Data Science and Decisions
</p>

<h1 align="center">
  <img src="https://github.com/shoshosho3/query_adaptive_contextual_document_embedding/blob/main/pictures_QACDE/IR_Logo_1.png" alt="IR Logo 1" height="200">
  <img src="https://github.com/shoshosho3/query_adaptive_contextual_document_embedding/blob/main/pictures_QACDE/IR_Logo_2.png" alt="IR Logo 2" height="200">
</h1>

---

<details open>
<summary><strong>Table of Contents</strong> ⚙️</summary>

1. [About the Article](#link-of-the-article)
2. [Project Overview](#project-overview)  
3. [About the code](#about-the-code)
4. [Running Instructions](#running-instructions)  
5. [Results & Comments](#results-&-comments)  

</details>

---

## About the article
Our project is an extension on the following article:
<blockquote>
  <a href="https://arxiv.org/abs/2410.02525">Contextual Document Embeddings</a> by John X. Morris and Alexander M. Rush.</blockquote>

The article "Contextual Document Embeddings" by John X. Morris and Alexander M. Rush from Cornell University proposes a method to improve document embeddings by incorporating context from surrounding documents. Traditional embeddings, which rely solely on the individual document, can be insufficient for highly specific information retrieval tasks. This paper addresses this by incorporating neighboring documents into the embedding process, making the representation more context-aware. The authors introduce a contextual training objective and an architecture that explicitly encodes information from neighboring documents, demonstrating improved performance in various scenarios.

## Project Overview
In this project, we aimed to extend the "Contextual Document Embeddings" model by introducing a query-adaptive approach. While the original model effectively contextualizes document embeddings by considering neighboring documents, it does not account for the user's needs expressed in the query, which is essential for information retrieval tasks.

To achieve this, we developed two models. The first model, called Query Adaptive Contextual Document Embedder (QACDE), makes the document embedding query-adaptive, meaning that the document embedding is influenced by the query's embedding, following the same embedding method proposed in the original paper. However, upon reflection, we realized that the document embedding greatly depends on how the query is represented. This led us to develop a second model called Multi-Embeddings-Query Adaptive Contextual Document Embedder (MEQCDE), that computes the document embedding using multiple query embeddings, such as BERT, TF-IDF and query embedding generated by the model of the article. The goal of this approach is to combine the strengths of different query embedding methods to create more robust and informative document embeddings. By leveraging multiple representations of the query, we aim to capture various aspects of the document-query relationship, improving the relevance of the information retrieval process.


## About the code
The code for this project is built upon the original implementation from the paper. We first use the pre-trained model from the article to generate document and query embeddings. These embeddings are then utilized to train our own models. Our code handles the training and evaluation of both our query-adaptive models as well as the evaluation of the original model from the paper (which serves as our baseline). Additionally, it manages the query embedding generation using TF-IDF and BERT methods. The evaluation is conducted using Mean Average Precision (MAP), with the goal of comparing the performance of our models and the baseline across various BeIR datasets such as SciFact, FiQA, and NFCorpus. The training procedure includes contrastive negative sampling and a custom Loss function tailored to the specific training conditions, ensuring a thorough and meaningful comparison between the models.


## Running Instructions
This repository contains the code and instructions to run the code. It is preferable to have acces to a Virtual Machine with a GPU because of the computational cost of the models.

1. If you have not VM, skip this step and go directly to step 2.
Connect to the VM.
Start the VM and run the following code in the terminal:
```python
ssh <user-name>@<host-name>
```
where <user-name> is your username and <host-name> is the public adress IP of the VM. Write your password to finish connection. 

2. Install all the dependencies via the requirements.txt file. Run this command on the terminal:
```python
pip install requirements.txt
```

3. Enter in the project file via the following command on the terminal:
```python
cd query_adaptive_contextual_document_embedding
```

4. Run the module Python run_pre_trained_cde.py to save into pickle files the document and queries embeddings by the model of the article (CDE):
```python
python run_pre_trained_cde.py --dataset <dataset_name>
```
where <dataset_name> is a correct name of a dataset present in the Benchmark BEIR.

5. Run the module compute_query_embeddings.py to save into pickle files the query embeddings for different methods of embeddings.
```python
python compute_query_embeddings.py --dataset <dataset_name> --tfidf_max_dim <max_dim_emb>
```
where <dataset_name> is a correct name of a dataset present in the Benchmark BEIR and <max_dim_emb> represents the dimension maximal of the TF-IDF query embedding.

6. Run the module baseline_run.py to obtain MAP results for the chosen dataset for the model of the article (CDE).
```python
python baseline_run.py --dataset <dataset_name> --index <index_num>
```
where <dataset_name> is a correct name of a dataset present in the Benchmark BEIR and <index_num> is the index of the embeddings that we want to use. For each run on the same dataset, we increases the index by one. Run the code with the higher index (number of runs on the same dataset to avoid overwritings) to run the simulation that you begin.

7. Run the run_query_adaptive_layer.py to obtain MAP results for the chosen dataset for the QACDE and MEQACDE models.
```python
python run_query_adaptive_layer.py --dataset <dataset_name> --index <index_num> --hidden_dim <hidden_dim> --epochs <num_epochs> --seed <seed>
```
where <dataset_name> is a correct name of a dataset present in the Benchmark BEIR, <index_num> is the index of the embeddings that we want to use, <hidden_dim> is the dimension of the hidden layer of the two models, <num_epochs> is the number of training epochs and <seed> is the seed of the program.

## Results & Comments
For this part, we ran the code for different datasets with hidden_dim = 1024, num_epochs = 20 and seed = 42.

