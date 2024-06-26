# stImpute

stImpute is a novel method that leverages single-cell data to predict authentic spatial transcriptomics data by integrating gene relationships. stImpute initiates the process using an autoencoder to create joint embeddings of spatial and single-cell transcriptomics data. These embeddings are then used to identify the nearest neighboring cells in scRNA-seq data for each cell in the spatial transcriptomics dataset. Subsequently, stImpute predicts spatial gene expression for each spatial cell using the nearest neighboring cells through a graph neural network (GNN), where nodes represent genes. The gene-to-gene relationships in the GNN are based on cosine similarity, utilizing the pre-trained embeddings of the gene-encoding proteins extracted from the protein language model ESM-2. In addition, stImpute is capable of identifying genes that efficiently predict imputation uncertainty, allowing the method to select genes that are reliably imputed.

## Installation  

We recommend using Anaconda to create a new Python environment and activate it via

```
conda env create -f stImpute_env.yaml
conda activate stImpute_env
```

## Quick Start

#### Input

* **spatial_df:**   [pandas dataframe] normalized and logarithmized original spatial data (cell by gene)
* **scrna_df:**    [pandas dataframe] normalized and logarithmized reference scRNA-seq data (cell by gene)
* **train_gene:**   [numpy array] genes for training
* **test_gene:**    [numpy array] genes for predicting
* **seed:**      [int] random seed in torch
* **emb_file:**    [str] a file containing the results of gene's coding by ESM-2 (or None)

#### Output

* **stImpute_res:**   [pandas dataframe] predicted spatial data (cell by gene)
* **reliable_score:** [numpy array] predicted gene's reliable score

#### For calling stImpute programmatically

```python
stImpute_res, reliable_score = stImpute(spatial_df, scrna_df, train_gene, test_gene, seed=seed, emb_file=emb_file)
```

## Reproduce the result of the paper

#### Main result

1. **Change lines 17-19 of train.py** to the address of the dataset you downloaded(The download address for the dataset will be provided later)

```python
st_adata = sc.read_h5ad('dataset/st-seq/osmFISH.h5ad')
sc_adata = sc.read_h5ad('dataset/scRNA-seq/Zeisel.h5ad')
emb_file = 'embed/osmFISH_emb.pkl'
```

2. Run train.py

```python
python train.py
```

If you want to accurately reproduce the results in the paper, **use NVIDIA GeForce RTX 4090**

#### The other result

We provide the full reproduction process [here]() of the other result in the paper(The address will be provided later)

## Tutorial

#### Spatial transcriptomic data impute

1. Install the required environment according to [stImpute Installation](https://github.com/cquzys/stImpute/?tab=readme-ov-file#installation).
   
2. Get the ESM-2 embedding for the genes associated with your dataset.
   
   If you don't want to get the embedding, you can skip this step and set `emb_file = None` in the next step.

4. **Change lines 17-19 of train.py** to the address and the embedding of your dataset.

```python
st_adata = sc.read_h5ad('st_seq_data.h5ad')
sc_adata = sc.read_h5ad('scRNA_seq_data.h5ad')
emb_file = 'emb.pkl' # or set emb_file = None
```

4. Run train.py

```python
python train.py
```

You can also run the tutorial [tutorial.ipynb](https://github.com/cquzys/stImpute/blob/main/tutorial.ipynb) for more detailed information.


