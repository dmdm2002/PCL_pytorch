import numpy as np
import faiss

import torch
import torch.nn as nn


def run_kmeans(x, args):
    """
    < Faiss >
    Git: https://github.com/facebookresearch/faiss
    설명: https://velog.io/@nawnoes/Faiss-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0

    facebook research에서 개발한, dense vector들의 clustering과 similarity를 구할 때 사용하는 라이브러리이다. 
    C++로 작성되었으며 python에서도 지원된다. 
    그리고 GPU 상에서도 효율적으로 동작한다.

    nearest neighbor와 k-th nearest neghbor를 구할 수 있다.

    index 라는 개념을 사용 (index는 데이버베이스 벡터들의 집합을 캡슐화하고 효율적으로 검색하기 위해 선택적으로 전처리를 할 수도 있다.)
    idnex에는 여러가지 타입들이 있으며, 가장 단순하게 사용할 수 있는 알고리즘은 bruteforce L2 ditance 검색을 하는 IndexFlatL2이다.
    """
    print('Performing kmeans clustering...')
    results = {'img2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(args.num_cluster):
        d = x.shape[1]
        k = int(num_cluster)

        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments (D: distnace, I: indexes)
        img2cluster = [int(n[0]) for n in I] # distance가 가장 가까운 값 추출, search를 통해 나오는 index는 distance가 가까운 순으로 나온다!

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distance for each cluster
        Dcluster = [[] for c in range(k)] # cluster 갯수만큼 빈 list 생성
        for im, i in enumerate(img2cluster):
            Dcluster[i].append(D[im][0]) # im2cluster의 갯수만큼 img의 가장 가까운 distance를 Dcluster에 넣는다.

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90)) # clamp extreme values for stability
        density = args.temperature * density / density.mean() # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(args.gpu)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        img2cluster = torch.LongTensor(img2cluster).to(args.gpu)
        density = torch.Tensor(density).to(args.gpu)

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['img2cluster'].append(img2cluster)    
        
    return results