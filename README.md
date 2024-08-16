README
===============================
SpaNFM:A method for spatial transcriptomics data clustering based on sparse attention hierarchical node representation and multi-view contrastive learning 
This document mainly introduces the python code of SpaNFM algorithm.

# Requirements
- pytorch==2.2.1
- numpy==1.26.4
- scipy==1.11.4 
- scikit-learn=1.2.2
- torch-geometric==2.5.1
- torch-sparse ==0.6.18
- scanpy==1.9.8
- python==3.11.8
- h5py==3.9.0 
- anndata==0.10.6
- entmax==1.3

# Instructions
This project includes all the codes for the SpaNFM algorithm experimented on the dataset (DLPFC). We only introduce the algorithm proposed in our paper, SpaNFM, and the introduction of other algorithms can be found in the corresponding paper.

# Model composition and meaning
SpaNFM is composed of common modules and experimental modules.

## Common module
- Data defines the data used by the model
	- data
		- 151671
			- filtered_feature_bc_matrix.h5
			- metadata.tsv
			- spatial
				- scalefactors_json.json
				- tissue_hires_image.png
				- tissue_lowres_image.png
				- tissue_positions_list.csv
- augment.py defines the augmentation of the model			
- get_data.py defines the data loading of the model.
- model_hierar.py defines the complete Node Fusion Module model.
- utils.py and untils_copy.py defines the tool functions needed by the entire algorithm during its operation.
- img_feature.py defines the method to extract image feature of the model.

## Experimental module
 main.py files are capable of conducting all data experiments within the same dataset. In subsequent statistical analyses, we examine the output of the main files. The utils.py file encompasses all tools necessary for the performance and analysis of the entire experiment, including calculations for ARI, NMI scores, and data transformations. All functions are developed using PyTorch and support CUDA.

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
