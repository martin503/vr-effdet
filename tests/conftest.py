import pytest
import torch


@pytest.fixture(scope='module')
def device():
    d = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return d
