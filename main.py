from __future__ import print_function
import os
import argparse 
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from get_data import run
import scanpy as sc
from sklearn import metrics
from scipy.spatial import distance
from utils import *
from torch_geometric.nn import HypergraphConv
import anndata
from scipy.stats import zscore
from utils_copy import clustering
from sklearn.metrics import silhouette_score
from augment import *
from model_hierar import model_hierar
import torch.nn as nn
import torch.nn.functional as F
import gc

def contrastive_loss(mat1, mat2):
	mat1 = torch.relu(mat1)
	mat2 = torch.relu(mat2)
	z1 = F.normalize(mat1)
	z2 = F.normalize(mat2)
	z1 = z1.squeeze(0)
	z2 = z2.squeeze(0)
	emb_similarity = torch.mm(z1, z2.t())
	eye_matrix = torch.eye(emb_similarity.shape[0])
	eye_matrix=eye_matrix.to('cuda')
	numerator = emb_similarity.mul(eye_matrix)
	numerator = numerator.sum(dim=-1) + 1e-8
	denominator = torch.sum(emb_similarity, dim=-1) + 1e-8
	loss = -torch.log(numerator / denominator).mean()
	return loss	

def train_main(args, adata, X, X2, adj_pure, adj_aug, df_meta, n_domains):
	
    decoder_rec = nn.Sequential(
		# nn.Linear(7, 128),
		# nn.Linear(420,512),
		nn.Linear(512,1000),
		)
    
    model_0 = model_hierar(args)
    model_1 = model_hierar(args)
    
    print(str(model_0))
    print(str(model_1))

    if args.use_sgd == 0:
        print("Use SGD")
        opt_0 = torch.optim.SGD(
            model_0.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001
		)
        opt_1 = torch.optim.SGD(
            model_1.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001
        )
    else:
        print("Use Adam")
        opt_0 = torch.optim.Adam(
            model_0.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001
		)
        opt_1 = torch.optim.Adam(
        	model_1.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001
        )

    device = torch.device("cuda" if args.cuda else "cpu")
    
    model_0.to(device)
    model_1.to(device)

    X = X.to(device)
    X2 = X2.to(device)

    adj_aug = adj_aug.to(device)
    adj_pure = adj_pure.to(device)
    
    decoder_rec.to(device)
    mse_loss=nn.MSELoss()
    
    best_epoch = 0
    best_ari = 0.0
    best_features = None
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        model_0.train()
        model_1.train()
        
        opt_0.zero_grad()
        opt_1.zero_grad()
        
        z_X = X.unsqueeze(0)
        z_X2 = X2.unsqueeze(0)
        output_0, z_0 = model_0(z_X, adj_pure)
        output_1, z_1 = model_1(z_X2, adj_aug) 

        z0_rec = decoder_rec(z_0)
        z1_rec = decoder_rec(z_1)

        loss_gcn_rec0 = mse_loss(z0_rec, z_X)
        loss_gcn_rec1 = mse_loss(z1_rec, z_X2)
        loss_rec = loss_gcn_rec0 + loss_gcn_rec1
        
        loss_con = contrastive_loss(output_0, output_1)
		
        train_loss = loss_rec + loss_con
        train_loss.backward(retain_graph=True)
        opt_0.step()
        opt_1.step()
        
        outstr_all = 'Train %d, loss: %.6f, loss_rec: %.6f, loss_con: %.6f' % (epoch, train_loss, loss_rec, loss_con)
        outstr_rec = 'Train %d, loss_rec_z0: %.6f, loss_rec_z1: %.6f' % (epoch, loss_gcn_rec0, loss_gcn_rec1)

        if epoch % 10 == 0:
            print(outstr_all)
            print(outstr_rec)

        if epoch > 800:
            if epoch % 50 == 0:
                z = torch.cat((z_0, z_1), dim=-1)
                z = z.squeeze(0)
				
                z = z.cpu()
                z = z.detach().numpy()
                _,ARI,_=cluster(adata,z,df_meta,n_domains)
                
                if ARI > best_ari:
                    best_ari = ARI
                    best_features = z
				
    return best_features

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='SpaNFM')
	parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
	parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='number of episode to train ')
	parser.add_argument('--use_sgd', type=int, default=0,
                        help='Use SGD')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
	parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
	parser.add_argument('--dropout', type=float, default=0.5,
	                    help='dropout rate')
	parser.add_argument('--partroi', type=int, default=512)
	parser.add_argument('--Gbias', type=bool, default=True, help='if bias ')
	parser.add_argument('--num_pooling', type=int, default=1, help=' ')
	parser.add_argument('--embedding_dim', type=int, default=128, help=' ')
	parser.add_argument('--assign_ratio', type=float, default=1, help=' ')
	parser.add_argument('--assign_ratio_1', type=float, default=1, help=' ')
	parser.add_argument('--mult_num', type=int, default=8, help=' ')
    
	args = parser.parse_args()

	sample_list = ['151671'] # '151507','151508','151509','151510','151669','151670','151672','151673','151674','151675','151676'
	for sample in sample_list:
		data_path = "/home/pingzhihao/SpaNFM/data" 
		data_name = sample
		save_path = data_path+"/"+sample+"/" #### save path
		save_path_figure = Path(os.path.join(save_path, "Figure", data_name))
		save_path_figure.mkdir(parents=True, exist_ok=True)
		
		if data_name in ['151669','151670','151671','151672']:
			n_domains = 5
		else:
			n_domains = 7

		data = run(save_path = save_path, 
			platform = "Visium",
			pca_n_comps = 128,
			pre_epochs = 800,
			vit_type='vit_b',#'vit'
			)

		df_meta = pd.read_csv(data_path+'/'+data_name+'/metadata.tsv', sep='\t')         
		adata =data._get_adata(data_path, data_name)
		adata = anndata.read_h5ad(save_path+"/"+sample+".h5ad")

		adata = data._get_augment(adata, adjacent_weight = 1, neighbour_k =6)
		adata1=adata.copy()
		adata2=adata.copy()
		adata.X = adata.obsm["augment_gene_data"].astype(float)

		sc.pp.filter_genes(adata, min_cells=3)
		sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
		adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
		adata_X = sc.pp.log1p(adata_X)
		adata_X = sc.pp.scale(adata_X)
		inputs1 = sc.pp.pca(adata_X, n_comps=128)
		inputs = sc.pp.pca(adata_X, n_comps=1000)

		cluster_label,_,_=cluster(adata,inputs1,df_meta,n_domains)
		cluster_adj=create_adjacency_matrix(cluster_label)
		adj_augment=sim2adj(adata.obsm["weights_matrix_all"],6)
		adj_pure=sim2adj(adata.obsm["weights_matrix_nomd"],6)
		adj_pure=cluster_adj*adj_pure
		adata1.obsm['weights_matrix_all']=adj_pure
		adata1=find_adjacent_spot(
		adata1,
		use_data = "raw",
		neighbour_k = 4,
		weights='weights_matrix_all',
		verbose = False,
		)

		adata1=augment_gene_data(
		adata1,
		use_data = "raw",
		adjacent_weight = 1,
		)

		adata1.X = adata1.obsm["augment_gene_data"].astype(float)
		adata1_X = sc.pp.normalize_total(adata1, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
		adata1_X = sc.pp.log1p(adata1_X)
		adata1_X = sc.pp.scale(adata1_X)
		inputs2 = sc.pp.pca(adata1_X, n_comps=1000)

		X=inputs.copy()
		X2=inputs2.copy()

		X=torch.tensor(X,dtype=torch.float)
		X2=torch.tensor(X2,dtype=torch.float)
	
		adj_augment=torch.tensor(adj_augment,dtype=torch.float)
		adj_pure=torch.tensor(adj_pure,dtype=torch.float)

		print("done")
		args = parser.parse_args()
        
		args.cuda = not args.no_cuda and torch.cuda.is_available()
        
		if args.cuda:
			print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
		else:
			print('Using CPU')

		best_features = train_main(args, adata,X,X2,adj_pure,adj_augment,df_meta,n_domains)
		print(sample)
    
		_,ARI,NMI=cluster(adata,best_features,df_meta,n_domains,refined=True)

		data.plot_domains(adata, sample)
		adata.write(os.path.join(save_path, f'{data_name}.h5ad'),compression="gzip")