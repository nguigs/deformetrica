import logging
import torch

from core import GpuMode, default
from support.kernels.abstract_kernel import AbstractKernel

logger = logging.getLogger(__name__)


class PolynomialKernel(AbstractKernel):
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, gpu_mode=default.gpu_mode, kernel_width=None, coef0=.1, degree=3, **kwargs):
        super().__init__('polynomial', gpu_mode, kernel_width)
        self.coef0 = coef0
        self.degree = degree

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def convolve(self, x, y, p, mode='gaussian'):
        res = None

        # move tensors with respect to gpu_mode
        x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
        assert x.device == y.device == p.device, 'x, y and p must be on the same device'

        sq = self._differences(x, y)
        res = torch.mm(torch.pow(sq * self.kernel_width + self.coef0, self.degree), p)
        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

    def convolve_gradient(self, px, x, y=None, py=None):
        if y is None:
            y = x
        if py is None:
            py = px

        # move tensors with respect to gpu_mode
        x, px, y, py = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, px, y, py])
        assert px.device == x.device == y.device == py.device, 'tensors must be on the same device'

        sq = self._differences(x, y)
        A = torch.pow(sq * self.kernel_width + self.coef0, self.degree - 1)

        B = self.degree * self.kernel_width * torch.einsum('ij,ik->kij', A, y)

        res = (self.degree * torch.sum(px * (torch.matmul(B, py)), 2) * self.kernel_width).t()
        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    @staticmethod
    def _differences(x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return torch.einsum('dmi, din->mn', x_col, y_lin)

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None:
            y = x
        assert (x.size(0) == y.size(0))
        sq = self._differences(x, y)
        return torch.pow(sq * self.kernel_width + self.coef0, self.degree)
