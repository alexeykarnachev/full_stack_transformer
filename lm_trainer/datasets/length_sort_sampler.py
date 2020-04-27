import more_itertools
import numpy as np
import torch.utils.data


class LengthSortSampler(torch.utils.data.Sampler):

    def __init__(self, sample_lengths, batch_size, ):
        super().__init__(sample_lengths)
        batch_size = batch_size

        inds = np.argsort(sample_lengths)[::-1]

        chunks = list(more_itertools.chunked(inds, batch_size))
        chunk_inds = list(range(len(chunks) - 1))

        last_chunk_ind = len(chunk_inds)
        np.random.shuffle(chunk_inds)
        chunk_inds.append(last_chunk_ind)

        self.inds = list(
            more_itertools.flatten([chunks[i] for i in chunk_inds]))
        self.backsort_inds = np.argsort(self.inds)

    def __len__(self):
        return len(self.inds)

    def __iter__(self):
        it = iter(self.inds)
        return it
