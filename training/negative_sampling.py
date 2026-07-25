import torch

def corrupt_edges(edge_index, edge_type, num_nodes, generator=None):
    E = edge_index.shape[1]
    device = edge_index.device

    corrupt_src_mask = torch.rand(E, generator=generator, device=device) < 0.5
    random_nodes = torch.randint(
        0, num_nodes, (E,), generator=generator, device=device
    )

    neg_src = torch.where(corrupt_src_mask, random_nodes, edge_index[0])
    neg_dst = torch.where(corrupt_src_mask, edge_index[1], random_nodes)

    neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
    neg_edge_type = edge_type.clone()

    return neg_edge_index, neg_edge_type