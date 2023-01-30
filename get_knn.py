import torch


def get_knn(
    queries: torch.Tensor,
    references: torch.Tensor,
    num_k: int,
    embeddings_come_from_same_source: bool,
) -> torch.Tensor:
    assert isinstance(embeddings_come_from_same_source, bool)

    num_k += embeddings_come_from_same_source

    scores = queries @ references.t()
    indices = torch.topk(scores, num_k).indices

    return indices[:, int(embeddings_come_from_same_source):]
