import os

import torch

from envs.data import Data


def test_dataset():
    data = Data(os.path.join(os.path.dirname(__file__), 'test_sample.tsv'))
    x = data.get_tensor('pick up the blue cube?')
    assert len(data.text_field.vocab) == 12
    assert x.equal(torch.LongTensor([[2, 4, 3, 0, 6, 0]]))
