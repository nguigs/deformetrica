from core import default
from core.model_tools.deformations.geodesic import Geodesic
from core.model_tools.deformations.exponential import Exponential

from support import utilities

import logging
logger = logging.getLogger(__name__)


class Spline(Geodesic):
    """
    Control-point-based LDDMM geodesic.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dense_mode=default.dense_mode, use_rk2_for_flow=default.use_rk2_for_flow, geodesic_weight=.1,
                 kernel=default.deformation_kernel, shoot_kernel_type=None, **kwargs):
        super(Spline, self).__init__(
            dense_mode=dense_mode, use_rk2_for_flow=use_rk2_for_flow,
            kernel=kernel, shoot_kernel_type=shoot_kernel_type, **kwargs)

        self.external_forces = None
        self.geodesic_weight = geodesic_weight

        self.backward_exponential = SplineEvolutionModel(
            dense_mode=dense_mode,
            kernel=kernel, shoot_kernel_type=shoot_kernel_type,
            use_rk2_for_flow=use_rk2_for_flow)

        self.forward_exponential = SplineEvolutionModel(
            dense_mode=dense_mode,
            kernel=kernel, shoot_kernel_type=shoot_kernel_type,
            use_rk2_for_flow=use_rk2_for_flow)

    def set_external_force(self, force):
        self.external_forces = force
        self.shoot_is_modified = True

    ####################################################################################################################
    # Main methods:
    ####################################################################################################################

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """

        assert self.t0 >= self.tmin, "tmin should be smaller than t0"
        assert self.t0 <= self.tmax, "tmax should be larger than t0"

        if self.shoot_is_modified or self.flow_is_modified:

            device, _ = utilities.get_best_device(self.backward_exponential.kernel.gpu_mode)

            # Backward exponential -------------------------------------------------------------------------------------
            length = self.t0 - self.tmin
            self.backward_exponential.number_of_time_points = \
                max(1, int(length * self.concentration_of_time_points + 1.5))
            if self.shoot_is_modified:
                self.backward_exponential.set_initial_momenta(- self.momenta_t0 * length)
                self.backward_exponential.set_initial_control_points(self.control_points_t0)
            if self.flow_is_modified:
                self.backward_exponential.set_initial_template_points(self.template_points_t0)
            if self.backward_exponential.number_of_time_points > 1:
                self.backward_exponential.move_data_to_(device=device)
                self.backward_exponential.update()

            # Forward exponential --------------------------------------------------------------------------------------
            length = self.tmax - self.t0
            self.forward_exponential.number_of_time_points = \
                max(1, int(length * self.concentration_of_time_points + 1.5))
            if self.shoot_is_modified:
                self.forward_exponential.set_initial_momenta(self.momenta_t0 * length)
                self.forward_exponential.set_initial_control_points(self.control_points_t0)
                self.forward_exponential.set_external_forces(self.external_forces)
            if self.flow_is_modified:
                self.forward_exponential.set_initial_template_points(self.template_points_t0)
            if self.forward_exponential.number_of_time_points > 1:
                self.forward_exponential.move_data_to_(device=device)
                self.forward_exponential.update()

            self.shoot_is_modified = False
            self.flow_is_modified = False
            self.backward_extension = 0
            self.forward_extension = 0

        else:
            if self.backward_extension > 0:
                self.backward_exponential.extend(self.backward_extension)
                self.backward_extension = 0

            if self.forward_extension > 0:
                self.forward_exponential.extend(self.forward_extension)
                self.forward_extension = 0

    def get_norm_squared(self):
        """
        Get the norm of the geodesic.
        """
        geodesic_part = self.forward_exponential.scalar_product(
            self.control_points_t0, self.momenta_t0, self.momenta_t0)
        return geodesic_part + (self.external_forces ** 2).sum() / self.concentration_of_time_points


class SplineEvolutionModel(Exponential):

    def __init__(self, **kwargs):
        super(SplineEvolutionModel, self).__init__(**kwargs)
        self.external_forces = None

    def set_external_forces(self, force):
        self.shoot_is_modified = True
        self.external_forces = force

    def get_external_forces(self):
        return self.external_forces

    def shoot(self):
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"
        assert len(self.external_forces) == self.number_of_time_points - 1

        # Integrate the Hamiltonian equations.
        self.control_points_t = []
        self.momenta_t = []
        self.momenta_t.append(self.initial_momenta)
        self.control_points_t.append(self.initial_control_points)

        dt = 1.0 / float(self.number_of_time_points - 1)
        for i in range(self.number_of_time_points - 1):
            new_cp, new_mom = self.euler_step(
                self.shoot_kernel, self.control_points_t[i], self.momenta_t[i], self.external_forces[i], dt)
            self.control_points_t.append(new_cp)
            self.momenta_t.append(new_mom)

        # Correctly resets the attribute flag.
        self.shoot_is_modified = False

    @staticmethod
    def spline_grad(kernel):
        def vector(cp, mom, u):
            return kernel.convolve(cp, cp, mom), - kernel.convolve_gradient(mom, cp) + u
        return vector

    @classmethod
    def euler_step(cls, kernel, cp, mom, u, h, return_mom=True):
        assert cp.device == mom.device, 'tensors must be on the same device, cp.device=' + str(
            cp.device) + ', mom.device=' + str(mom.device)

        k1, l1 = cls.spline_grad(kernel)(cp, mom, u)
        cp_1 = cp + h * k1
        mom_1 = mom + h * l1

        return cp_1, mom_1 if return_mom else cp_1


