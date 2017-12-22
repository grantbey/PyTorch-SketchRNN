import numpy as np
import torch

class sketchLoader():
    def __init__(self,datafile):

        def purify(self, strokes):
            data = []
            for seq in strokes:
                if len(seq[:, 0]) <= hyper.max_seq_length and len(seq[:, 0]) > 10:
                    seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype = np.float32)
                data.append(seq)
            return data
        def calculate_normalizing_scale_factor(self, strokes):
            """Calculate the normalizing factor explained in appendix of sketch-rnn."""
            data = []
            for i in range(len(strokes)):
                for j in range(len(strokes[i])):
                    data.append(strokes[i][j, 0])
                    data.append(strokes[i][j, 1])
            data = np.array(data)
            return np.std(data)
        def normalize(self, strokes):
            """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
            data = []
            scale_factor = calculate_normalizing_scale_factor(strokes)
            for seq in strokes:
                seq[:, 0:2] /= scale_factor
                data.append(seq)
            return data
        def max_size(self, strokes):
            """larger sequence length in the data set"""
            sizes = [len(seq) for seq in strokes]
            return max(sizes)

        self.data = np.load(datafile, encoding = 'latin1')
        self.data = purify(data)
        self.data = normalize(data)
        self.Nmax = max_size(data)

    def get_batch(self, batch_size):
        idxs = np.random.choice(len(self.data),batch_size)
        batch_strokes = [self.data[idx] for idx in idxs]
        strokes = []
        lengths = []
        for seq in batch_strokes:
            len_seq = len(seq[:, 0])  # I think this is how many lines in the image
            # Seq is always of shape (n,3) where the three dimensions
            # ∆x, ∆y, and a binary value representing whether the pen is lifted away from the paper
            new_seq = np.zeros((self.Nmax, 5))  # New seq of max length, all zeros
            new_seq[:len_seq, :2] = seq[:, :2]  # fill in x:y co-ords in first two dims
            new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
            new_seq[:len_seq, 3] = seq[:, 2]  # pen binary in fourth dim
            new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
            new_seq[len_seq - 1, 2:4] = 0  # zeros in last point for dims three and four
            lengths.append(len(seq[:, 0]))  # Record the length of the actual sequence
            strokes.append(new_seq)  # Record the sequence

        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).cuda().float())

        return batch, lengths

    def get_target(self, batch, lengths):
        eos = Variable(torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).cuda()).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.Nmax + 1, batch.size()[1])

        for id, length in enumerate(lengths)
            mask[:length,id] = 1
            mask = Variable(mask.cuda()).detach()

        dx = torch.stack([Variable(batch.data[:, :, 0])] * hp.M, 2).detach()
        dy = torch.stack([Variable(batch.data[:, :, 1])] * hp.M, 2).detach()
        p1 = Variable(batch.data[:, :, 2]).detach()
        p2 = Variable(batch.data[:, :, 3]).detach()
        p3 = Variable(batch.data[:, :, 4]).detach()
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p
