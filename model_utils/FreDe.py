import torch
from torch import nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx, pairwise_distance


def local_operator(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor-x, neighbor), dim=3).permute(0, 3, 1, 2)  # local and global all in

    return feature


def local_operator_withnorm(x, norm_plt, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    norm_plt = norm_plt.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    norm_plt = norm_plt.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_norm = norm_plt.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor-x, neighbor, neighbor_norm), dim=3).permute(0, 3, 1, 2)  # 3c

    return feature
  
def FreDe(x, M):
    
    k = 16  # number of neighbors to decide the range of j in Eq.(5)
   # k = 32  # number of neighbors to decide the range of j in Eq.(5)
    tau = 0.2  # threshold in Eq.(2)
    sigma = 2  # parameters of f (Gaussian function in Eq.(2))
    ###############
    """Graph Construction:"""
    device = torch.device('cuda')
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx, p = knn(x, k=k)  # p: -[(x1-x2)^2+...]

    # here we add a tau
    p1 = torch.abs(p)
    p1 = torch.sqrt(p1)
    mask = p1 < tau

    # here we add a sigma
    p = p / (sigma * sigma)
    w = torch.exp(p)  # b,n,n
    w = torch.mul(mask.float(), w)

    b = 1/torch.sum(w, dim=1)
    b = b.reshape(batch_size, num_points, 1).repeat(1, 1, num_points)
    c = torch.eye(num_points, num_points, device=device)
    c = c.expand(batch_size, num_points, num_points)
    D = b * c  # b,n,n

    A = torch.matmul(D, w)  # normalized adjacency matrix A_hat

    # Get Aij in a local area:
    idx2 = idx.view(batch_size * num_points, -1)
    idx_base2 = torch.arange(0, batch_size * num_points, device=device).view(-1, 1) * num_points
    idx2 = idx2 + idx_base2

    idx2 = idx2.reshape(batch_size * num_points, k)[:, 1:k]
    idx2 = idx2.reshape(batch_size * num_points * (k - 1))
    idx2 = idx2.view(-1)

    A = A.view(-1)
    A = A[idx2].reshape(batch_size, num_points, k - 1)  # Aij: b,n,k
    ###############
    """Disentangling Point Clouds into Sharp(xs) and Gentle(xg) Variation Components:"""
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(batch_size * num_points, k)[:, 1:k]
    idx = idx.reshape(batch_size * num_points * (k - 1))

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # b,n,c
    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k - 1, num_dims)  # b,n,k,c
    A = A.reshape(batch_size, num_points, k - 1, 1)  # b,n,k,1
    n = A.mul(neighbor)  # b,n,k,c
    n = torch.sum(n, dim=2)  # b,n,c

    pai = torch.norm(x - n, dim=-1).pow(2)  # Eq.(5)
    pais = pai.topk(k=M, dim=-1)[1]  # first M points as the sharp variation component
    paig = (-pai).topk(k=M, dim=-1)[1]  # last M points as the gentle variation component
    
    sorted_indices = torch.argsort(pai, dim=-1) ##
    paimh = sorted_indices[:, M:2*M] ##
#    paiml = sorted_indices[:, 2*M:3*M] ##

    pai_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points
    indices = (pais + pai_base).view(-1)
    indiceg = (paig + pai_base).view(-1)
    indicemh = (paimh + pai_base).view(-1)##
  #  indiceml = (paiml + pai_base).view(-1)##

    return indices, indicemh, indiceg ##

def select_top_20_percent_from_all12(orih, ori, batch_size):
    M = orih.size(1)
    d = orih.size(2)
    orih_global = torch.max(ori, 1, keepdim=True)[0]  
   # orih_global = torch.max(orih, 1, keepdim=True)[0] 

    with torch.no_grad():
        mask_sim_h_list = []
        orih_global = F.normalize(orih_global, dim=-1) 
        orih_norm = F.normalize(orih, dim=-1) 
        
        for bi in range(batch_size):
            token_sim_with_global = torch.matmul(orih_norm[bi], orih_global[bi].squeeze(0).t())  # [M, 1]
           # sort_token_idx = token_sim_with_global.argsort(dim=0) #most
            sort_token_idx = token_sim_with_global.argsort(dim=0, descending=True)
            index = sort_token_idx[:int(M * 0.5)]  # 
            mask_sim_h = orih[bi][index]
            mask_sim_h_list.append(mask_sim_h)  # b, M', c

    mask_sim_h_tensor = torch.stack(mask_sim_h_list)  # [b, M', c]
  #  print(mask_sim_h_tensor.size())
    mask_sim_h_tensor = torch.max(mask_sim_h_tensor, 1, keepdim=True)[0]            
    mask_sim_h_tensor = mask_sim_h_tensor.view(-1, d)
    
    return mask_sim_h_tensor
    
def select_top_20_percent_from_all34(orih, ori, batch_size):
    M = orih.size(1)
    d = orih.size(2)
    orih_global = torch.max(ori, 1, keepdim=True)[0]  

    with torch.no_grad():
        mask_sim_h_list = []
        orih_global = F.normalize(orih_global, dim=-1) 
        orih_norm = F.normalize(orih, dim=-1) 
        
        for bi in range(batch_size):
            token_sim_with_global = torch.matmul(orih_norm[bi], orih_global[bi].squeeze(0).t())  # [M, 1]
           # print(token_sim_with_global)
           # sort_token_idx = token_sim_with_global.argsort(dim=0) #most
            sort_token_idx = token_sim_with_global.argsort(dim=0, descending=True)
            index = sort_token_idx[:int(M * 0.5)]  # 
            mask_sim_h = orih[bi][index]
            mask_sim_h_list.append(mask_sim_h)  # b, M', c

    mask_sim_h_tensor = torch.stack(mask_sim_h_list)  # [b, M', c]
   # print(mask_sim_h_tensor.size())
   # mask_sim_h_tensor = torch.max(mask_sim_h_tensor, 1, keepdim=True)[0]            
   # mask_sim_h_tensor = mask_sim_h_tensor.view(-1, d)
    
    return mask_sim_h_tensor