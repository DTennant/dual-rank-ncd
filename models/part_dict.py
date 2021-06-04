import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed


def global_pooling(x):
    assert x.dim() == 4
    return F.avg_pool2d(x, (x.size(2), x.size(3)))

class PartDict(nn.Module):
    def __init__(self, num_words,
        num_channels,
        inv_delta=15):
        
        super(PartDict, self).__init__()
        
        if inv_delta is not None:
            assert isinstance(inv_delta, (float, int))
            assert inv_delta > 0.0

        self._num_channels = num_channels
        self._num_words = num_words
        self._inv_delta = inv_delta
        self._decay = 0.99

        embedding = torch.randn(num_words, num_channels).clamp(min=0)
        self.register_buffer("_embedding", embedding)
        self.register_buffer("_embedding_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_track_num_batches", torch.zeros(1))
        self.register_buffer("_min_distance_mean", torch.ones(1) * 0.5)

    @torch.no_grad()
    def _update_dictionary(self, features):
        assert features.dim() == 4
        # features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=0)
        features = features.flatten(2)
        batch_size, _, num_locs = features.size()
        index = torch.randint(0, num_locs, (batch_size,), device=features.device)
        index += torch.arange(batch_size, device=features.device) * num_locs
        selected_features = features.permute(0,2,1).reshape(batch_size*num_locs, -1)
        selected_features = selected_features[index].contiguous()

        assert selected_features.dim() == 2

        assert self._num_words % selected_features.shape[0] == 0
        batch_size = selected_features.shape[0]

        ptr = int(self._embedding_ptr)
        self._embedding[ptr:(ptr + batch_size),:] = selected_features
        self._embedding_ptr[0] = (ptr + batch_size) % self._num_words

    @torch.no_grad()
    def get_dictionary(self):
        return self._embedding.detach().clone()

    def forward(self, features):
        features = features[:, :, 1:-1, 1:-1].contiguous()

        embeddings_b = self._embedding.pow(2).sum(1)
        embeddings_w = -2*self._embedding.unsqueeze(2).unsqueeze(3)
        # dist = ||features||^2 + |embeddings||^2 + conv(features, -2 * embedding)
        dist = (features.pow(2).sum(1, keepdim=True) +
                F.conv2d(features, weight=embeddings_w, bias=embeddings_b))
        min_dist, enc_indices = torch.min(dist, dim=1)
        mu_min_dist = min_dist.mean()

        if self.training:
            self._min_distance_mean.data.mul_(self._decay).add_(
                mu_min_dist, alpha=(1. - self._decay))
            self._update_dictionary(features)
            self._track_num_batches += 1

        inv_delta_adaptive = self._inv_delta / self._min_distance_mean
        codes = F.softmax(-inv_delta_adaptive * dist, dim=1)

        bow = global_pooling(codes).flatten(1)
        bow = F.normalize(bow, p=1, dim=1)
        return bow, codes




