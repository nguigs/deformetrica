import unittest

import torch
from numpy import pi

from core.model_tools.deformations.exponential import Exponential
from support.kernels import torch_kernel


class TestPoleLadder(unittest.TestCase):
    def test_rk_inverse(self):
        n_landmarks = 10
        times = torch.tensor([float(k) * 2 * pi / n_landmarks for k in range(n_landmarks)])
        x = torch.stack([torch.cos(times), torch.sin(times)]).t()
        y = torch.stack([0.8 * torch.cos(times), 0.8 * torch.sin(times)]).t()
        kernel = torch_kernel.TorchKernel(kernel_width=0.5)
        mom = Exponential.rk4_inverse(kernel, x, y, 0.1)
        result = Exponential.rk4_step(kernel, x, mom, 0.1, return_mom=False)
        expected = Exponential._rk2_step(kernel, x, mom, 0.1, return_mom=False)
        self.assertTrue(torch.allclose(result, y, atol=1e-6))
        print(result - expected)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))


