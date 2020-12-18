import warnings
from copy import deepcopy
import support.kernels as kernel_factory
import torch

from core import default
from in_out.array_readers_and_writers import *

from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from scipy.optimize import minimize

import logging

from support import utilities

logger = logging.getLogger(__name__)

class Exponential:
    """
    Control-point-based LDDMM exponential, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dense_mode=default.dense_mode,
                 kernel=default.deformation_kernel,
                 shoot_kernel_type=None,
                 use_svf=False,
                 number_of_time_points=None,
                 initial_control_points=None, control_points_t=None,
                 initial_momenta=None, momenta_t=None, preserve_volume=False,
                 initial_template_points=None, template_points_t=None, use_rk4_for_shoot=False, polydata=None,
                 shoot_is_modified=True, flow_is_modified=True, use_rk2_for_shoot=False, use_rk2_for_flow=False):

        self.dense_mode = dense_mode
        self.kernel = kernel

        if shoot_kernel_type is not None:
            self.shoot_kernel = kernel_factory.factory(shoot_kernel_type, gpu_mode=kernel.gpu_mode, kernel_width=kernel.kernel_width)
        else:
            self.shoot_kernel = self.kernel

        # logger.debug(hex(id(self)) + ' using kernel: ' + str(self.kernel))
        # logger.debug(hex(id(self)) + ' using shoot_kernel: ' + str(self.shoot_kernel))

        self.number_of_time_points = number_of_time_points
        # Initial position of control points
        self.initial_control_points = initial_control_points
        # Control points trajectory
        self.control_points_t = control_points_t
        # Initial momenta
        self.initial_momenta = initial_momenta
        # Momenta trajectory
        self.momenta_t = momenta_t
        # Initial template points
        self.initial_template_points = initial_template_points
        # Trajectory of the whole vertices of landmark type at different time steps.
        self.template_points_t = template_points_t
        # If the cp or mom have been modified:
        self.shoot_is_modified = shoot_is_modified
        # If the template points has been modified
        self.flow_is_modified = flow_is_modified
        # Wether to use a RK2 or a simple euler for shooting or flowing respectively.
        self.use_rk2_for_shoot = use_rk2_for_shoot
        self.use_rk4_for_shoot = use_rk4_for_shoot
        self.use_rk2_for_flow = use_rk2_for_flow

        self.use_svf = use_svf
        self.triangles = self._get_triangle_mask(polydata)
        self.preserve_volume = preserve_volume

        # Contains the inverse kernel matrices for the time points 1 to self.number_of_time_points
        # (ACHTUNG does not contain the initial matrix, it is not needed)
        self.cometric_matrices = {}
        # self.cholesky_matrices = {}

    def move_data_to_(self, device):
        if self.initial_control_points is not None:
            self.initial_control_points = utilities.move_data(self.initial_control_points, device)
        if self.initial_momenta is not None:
            self.initial_momenta = utilities.move_data(self.initial_momenta, device)

        if self.initial_template_points is not None:
            self.initial_template_points = {key: utilities.move_data(value, device) for key, value in
                                            self.initial_template_points.items()}

    def light_copy(self):
        light_copy = Exponential(self.dense_mode,
                                 deepcopy(self.kernel), self.shoot_kernel.kernel_type,
                                 self.number_of_time_points,
                                 self.initial_control_points, self.control_points_t,
                                 self.initial_momenta, self.momenta_t,
                                 self.initial_template_points, self.template_points_t,
                                 self.shoot_is_modified, self.flow_is_modified,
                                 self.use_rk2_for_shoot, use_rk2_for_flow=self.use_rk2_for_flow)
        return light_copy

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2_for_shoot(self, flag):
        self.shoot_is_modified = True
        self.use_rk2_for_shoot = flag

    def set_use_rk2_for_flow(self, flag):
        self.flow_is_modified = True
        self.use_rk2_for_flow = flag

    def get_kernel_type(self):
        return self.kernel.kernel_type

    def get_kernel_width(self):
        return self.kernel.kernel_width

    def set_kernel(self, kernel):
        # TODO which kernel to set ?
        self.kernel = kernel

    def set_initial_template_points(self, td):
        self.initial_template_points = td
        self.flow_is_modified = True

    def get_initial_template_points(self):
        return self.initial_template_points

    def set_initial_control_points(self, cps):
        self.shoot_is_modified = True
        self.initial_control_points = cps

    def get_initial_control_points(self):
        return self.initial_control_points

    def get_initial_momenta(self):
        return self.initial_momenta

    def set_initial_momenta(self, mom):
        self.shoot_is_modified = True
        self.initial_momenta = mom

    def scalar_product(self, cp, mom1, mom2):
        """
        returns the scalar product 'mom1 K(cp) mom 2'
        """
        return torch.sum(mom1 * self.kernel.convolve(cp, cp, mom2))

    def get_template_points(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if self.flow_is_modified:
            msg = "You tried to get some template points, but the flow was modified. " \
                  "The exponential should be updated before."
            warnings.warn(msg)
        if time_index is None:
            return {key: self.template_points_t[key][-1] for key in self.initial_template_points.keys()}
        return {key: self.template_points_t[key][time_index] for key in self.initial_template_points.keys()}

    def get_norm_squared(self):
        return self.scalar_product(self.initial_control_points, self.initial_momenta, self.initial_momenta)

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Flow the trajectory of the landmark and/or image points.
        """
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"

        # Initialization.
        dt = 1.0 / float(self.number_of_time_points - 1)
        self.template_points_t = {}

        # Flow both Hamiltonian system and template points
        self.control_points_t = [self.initial_control_points]
        self.momenta_t = [self.initial_momenta]
        landmark_points_t = [self.initial_template_points['landmark_points']]

        if self.use_rk2_for_shoot:
            step = self._rk2_step
        elif self.use_rk4_for_shoot:
            step = self.rk4_step
        else:
            step = self._euler_step

        for i in range(self.number_of_time_points - 1):
            new_cp, new_mom, new_x = step(landmark_points_t[i], self.control_points_t[i], self.momenta_t[i], dt)
            self.control_points_t.append(new_cp)
            self.momenta_t.append(new_mom)
            landmark_points_t.append(new_x)

        # Special case of the dense mode.
        if self.dense_mode:
            assert 'image_points' not in self.initial_template_points.keys(), 'Dense mode not allowed with image data.'
            self.template_points_t['landmark_points'] = self.control_points_t
            self.flow_is_modified = False
            return

        self.template_points_t['landmark_points'] = landmark_points_t
        assert len(self.template_points_t) > 0, 'That\'s unexpected'

        # Correctly resets the attribute flag.
        self.flow_is_modified = False

    def volume_gradient_flow(self, end_time=1.0):
        """
        Flow the trajectory of the landmark and/or image points.
        """
        # Initialization.
        dt = end_time / float(self.number_of_time_points - 1)
        self.template_points_t = {}

        # Special case of the dense mode.
        if self.dense_mode:
            assert 'image_points' not in self.initial_template_points.keys(), 'Dense mode not allowed with image data.'
            self.template_points_t['landmark_points'] = self.control_points_t
            self.flow_is_modified = False
            return

        # Flow landmarks points.
        if 'landmark_points' in self.initial_template_points.keys():
            landmark_points = [self.initial_template_points['landmark_points']]

            for i in range(self.number_of_time_points - 1):
                initial_shape = landmark_points[i].clone().detach().requires_grad_(True)
                deformed_volume = self._volume(tensor=initial_shape)
                deformed_volume.backward()
                grad_star = initial_shape.grad.detach() / deformed_volume.detach() * 100
                d_pos = - self.kernel.convolve(initial_shape, initial_shape, grad_star)

                landmark_points.append(landmark_points[i] + dt * d_pos)

                if self.use_rk2_for_flow:
                    # In this case improved euler (= Heun's method)
                    # to save one computation of convolve gradient per iteration.
                    if i < self.number_of_time_points - 2:
                        initial_shape = landmark_points[i + 1].clone().detach().requires_grad_(True)
                        deformed_volume = self._volume(tensor=initial_shape)
                        deformed_volume.backward()
                        grad_star = initial_shape.grad.detach()
                        d_pos_new = self.kernel.convolve(initial_shape, initial_shape, grad_star)

                        landmark_points[-1] = landmark_points[i] + dt / 2 * (d_pos_new + d_pos)
                    else:
                        final_cp, final_mom = self._rk2_step(
                            self.kernel, self.control_points_t[-1], self.momenta_t[-1], dt, return_mom=True)
                        landmark_points[-1] = landmark_points[i] + dt / 2 * (
                                self.kernel.convolve(landmark_points[i + 1], final_cp, final_mom) + d_pos)

            self.template_points_t['landmark_points'] = landmark_points
        self.flow_is_modified = False

    def parallel_transport(self, momenta_to_transport, initial_time_point=0, is_orthogonal=False):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        if is_orthogonal is on, then the momenta to transport must be orthogonal to the momenta of the geodesic.
        Note: uses shoot kernel
        """

        # Sanity checks ------------------------------------------------------------------------------------------------
        assert not self.shoot_is_modified, "You want to parallel transport but the shoot was modified, please update."
        assert self.use_rk2_for_shoot, "The shoot integration must be done with a second order numerical scheme in order to use parallel transport."
        assert (momenta_to_transport.size() == self.initial_momenta.size())

        # Special cases, where the transport is simply the identity ----------------------------------------------------
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        if (torch.norm(self.initial_momenta).detach().cpu().numpy() < 1e-6 or
                torch.norm(momenta_to_transport).detach().cpu().numpy() < 1e-6):
            parallel_transport_t = [momenta_to_transport] * (self.number_of_time_points - initial_time_point)
            return parallel_transport_t

        # Step sizes ---------------------------------------------------------------------------------------------------
        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # For printing -------------------------------------------------------------------------------------------------
        worst_renormalization_factor = 1.0

        # Optional initial orthogonalization ---------------------------------------------------------------------------
        norm_squared = self.get_norm_squared()
        if not is_orthogonal:
            sp = self.scalar_product(
                self.control_points_t[initial_time_point], momenta_to_transport,
                self.momenta_t[initial_time_point]) / norm_squared
            momenta_to_transport_orthogonal = momenta_to_transport - sp * self.momenta_t[initial_time_point]
            parallel_transport_t = [momenta_to_transport_orthogonal]
        else:
            sp = (self.scalar_product(
                self.control_points_t[initial_time_point], momenta_to_transport,
                self.momenta_t[initial_time_point]) / norm_squared).detach().cpu().numpy()
            assert abs(sp) < 1e-2, \
                'Error: the momenta to transport is not orthogonal to the driving momenta, ' \
                'but the is_orthogonal flag is active. sp = %.3E' % sp
            parallel_transport_t = [momenta_to_transport]

        # Then, store the initial norm of this orthogonal momenta ------------------------------------------------------
        initial_norm_squared = self.scalar_product(self.control_points_t[initial_time_point], parallel_transport_t[0],
                                                   parallel_transport_t[0])

        for i in range(initial_time_point, self.number_of_time_points - 1):
            # Shoot the two perturbed geodesics ------------------------------------------------------------------------
            cp_eps_pos = self._rk2_step(self.shoot_kernel, self.control_points_t[i],
                                        self.momenta_t[i] + epsilon * parallel_transport_t[-1], h, return_mom=False)
            cp_eps_neg = self._rk2_step(self.shoot_kernel, self.control_points_t[i],
                                        self.momenta_t[i] - epsilon * parallel_transport_t[-1], h, return_mom=False)

            # Compute J/h ----------------------------------------------------------------------------------------------
            approx_velocity = (cp_eps_pos - cp_eps_neg) / (2 * epsilon * h)

            # We need to find the cotangent space version of this vector -----------------------------------------------
            # If we don't have already the cometric matrix, we compute and store it.
            # TODO: add optional flag for not saving this if it's too large.
            # OPTIM: keep an eye on https://github.com/pytorch/pytorch/issues/4669
            if i not in self.cometric_matrices:
                kernel_matrix = self.shoot_kernel.get_kernel_matrix(self.control_points_t[i + 1])
                # self.kernel_matrices[i] = kernel_matrix
                # self.cholesky_matrices[i] = torch.potrf(kernel_matrix.t().matmul(kernel_matrix), upper=False)
                # self.cholesky_matrices[i] = torch.cholesky(kernel_matrix, upper=False)
                # self.cholesky_matrices[i] = torch.potrf(kernel_matrix, upper=True)
                self.cometric_matrices[i] = torch.inverse(kernel_matrix)
                # self.cometric_matrices[i] = torch.inverse(kernel_matrix.cuda()).cpu()

            # Solve the linear system.
            # rhs = approx_velocity.matmul(self.kernel_matrices[i])
            # z = torch.trtrs(rhs.t(), self.cholesky_matrices[i], transpose=False, upper=False)[0]
            # approx_momenta = torch.trtrs(z, self.cholesky_matrices[i], transpose=True, upper=False)[0]
            # approx_momenta = torch.potrs(approx_velocity, self.cholesky_matrices[i], upper=True)
            approx_momenta = torch.mm(self.cometric_matrices[i], approx_velocity)

            # We get rid of the component of this momenta along the geodesic velocity:
            scalar_prod_with_velocity = self.scalar_product(self.control_points_t[i + 1], approx_momenta,
                                                            self.momenta_t[i + 1]) / norm_squared

            approx_momenta = approx_momenta - scalar_prod_with_velocity * self.momenta_t[i + 1]

            # Renormalization ------------------------------------------------------------------------------------------
            approx_momenta_norm_squared = self.scalar_product(self.control_points_t[i + 1], approx_momenta,
                                                              approx_momenta)

            renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
            renormalized_momenta = approx_momenta * renormalization_factor

            if abs(renormalization_factor.detach().cpu().numpy() - 1.) > 0.1:
                raise ValueError('Absurd required renormalization factor during parallel transport: %.4f. '
                                 'Exception raised.' % renormalization_factor.detach().cpu().numpy())
            elif abs(renormalization_factor.detach().cpu().numpy() - 1.) > abs(worst_renormalization_factor - 1.):
                worst_renormalization_factor = renormalization_factor.detach().cpu().numpy()

            # Finalization ---------------------------------------------------------------------------------------------
            parallel_transport_t.append(renormalized_momenta)

        assert len(parallel_transport_t) == self.number_of_time_points - initial_time_point, \
            "Oops, something went wrong."

        # We now need to add back the component along the velocity to the transported vectors.
        if not is_orthogonal:
            parallel_transport_t = [parallel_transport_t[i] + sp * self.momenta_t[i]
                                    for i in range(initial_time_point, self.number_of_time_points)]

        if abs(worst_renormalization_factor - 1.) > 0.05:
            msg = ("Watch out, a large renormalization factor %.4f is required during the parallel transport. "
                   "Try using a finer discretization." % worst_renormalization_factor)
            logger.warning(msg)

        return parallel_transport_t

    def pole_ladder_transport(self, initial_shoot, initial_time_point=0):
        # Step sizes ---------------------------------------------------------------------------------------------------
        h = 1. / self.number_of_time_points
        shoot = initial_shoot
        main_geodesic = self.template_points_t['landmark_points']

        for i in range(initial_time_point, self.number_of_time_points):
            mom = self.rk4_inverse(main_geodesic[i], self.control_points_t[i], shoot, h)
            shoot = self.rk4_step(main_geodesic[i], self.control_points_t[i], -mom, h, return_mom=False)

        final_cp, _, final_shape = self.rk4_step(
            main_geodesic[-1], self.control_points_t[-1], self.momenta_t[-1], h / 2)

        transported_momenta = self.rk4_inverse(final_shape, final_cp, shoot, h)
        if self.number_of_time_points % 2 == 1:
            transported_momenta *= -1.

        return final_cp, transported_momenta

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    def _euler_step(self, x, cp, mom, h):
        """
        simple euler step of length h, with cp and mom. It always returns mom.
        """
        assert cp.device == mom.device, 'tensors must be on the same device, cp.device=' + str(
            cp.device) + ', mom.device=' + str(mom.device)

        k1, l1, v1 = self.symplectic_grad(x, cp, mom)
        return cp + h * k1, mom + h * l1, x + h * v1

    def _rk2_step(self, x, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
        also used in parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        assert cp.device == mom.device, 'tensors must be on the same device, cp.device=' + str(
            cp.device) + ', mom.device=' + str(mom.device)

        k1, l1, v1 = self.symplectic_grad(x, cp, mom)
        mid_cp = cp + h / 2. * k1
        mid_mom = mom + h / 2 * l1
        mid_x = x + h / 2. * v1

        k2, l2, v2 = self.symplectic_grad(mid_x, mid_cp, mid_mom)
        if return_mom:
            return cp + h * k2, mom + h * l2, x + h * v2
        else:
            return cp + h * k2

    def _hamiltonian_grad_helper(self, x, cp, mom):

        initial_shape = x.clone().detach().requires_grad_(True)
        deformed_volume = self._volume(tensor=initial_shape)
        deformed_volume.backward()
        dvol = initial_shape.grad.detach()

        grad = self.kernel.convolve(x, x, dvol)
        norm = torch.sum(grad * dvol)

        dq_orth = self.kernel.convolve(cp, x, dvol)
        dot_product = torch.sum(dq_orth * mom)

        return dot_product / norm, dq_orth, grad, dvol

    def symplectic_grad(self, x, cp, mom):
        coef, dq_orth, grad, dvol = self._hamiltonian_grad_helper(x, cp, mom)

        dq = self.kernel.convolve(cp, cp, mom)
        vel_cp = dq - coef * dq_orth

        d_pos = self.kernel.convolve(x, cp, mom)
        vel_landmarks = d_pos - coef * grad

        dp = self.kernel.convolve_gradient(mom, cp)
        dp_orth = coef * self.kernel.convolve_gradient(mom, cp, x, dvol)
        vel_mom = dp - dp_orth

        return vel_cp, - vel_mom, vel_landmarks

    def rk4_step(self, x, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk4 step on the geodesic equation with initial cp and mom.
        also used in pole ladder parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        assert cp.device == mom.device, 'tensors must be on the same device, cp.device=' + str(
            cp.device) + ', mom.device=' + str(mom.device)

        k1, l1, v1 = self.symplectic_grad(x, cp, mom)
        cp_1 = cp + h / 2. * k1
        mom_1 = mom + h / 2 * l1
        x_1 = x + h / 2. * v1

        k2, l2, v2 = self.symplectic_grad(x_1, cp_1, mom_1)
        cp_2 = cp + h / 2 * k2
        mom_2 = mom + h / 2 * l2
        x_2 = x + h / 2. * v1

        k3, l3, v3 = self.symplectic_grad(x_2, cp_2, mom_2)
        cp_3 = cp + h * k3
        mom_3 = mom + h * l3
        x_3 = x + h / 2. * v1

        k4, l4, v4 = self.symplectic_grad(x_3, cp_3, mom_3)
        cp_new = cp + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        mom_new = mom + h / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        x_new = x + h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        if return_mom:
            return cp_new, mom_new, x_new
        return cp_new

    def rk4_inverse(self, base, x, y, h):
        cp = x.double()
        target = y.double()
        initial_shape = base.double()

        def loss_and_grad(mom):
            if isinstance(mom, np.ndarray):
                mom = torch.from_numpy(mom)
            mom = mom.reshape(x.shape)
            mom_ = mom.clone().detach().requires_grad_(True)
            shoot = self.rk4_step(initial_shape, cp, mom_, h, return_mom=False)
            loss = torch.sum((shoot.contiguous().view(-1) - target.contiguous().view(-1)) ** 2)
            loss.backward()
            grad = mom_.grad.detach().cpu()
            return loss.detach().cpu(), grad.numpy().flatten()

        init_mom = torch.rand(*torch.flatten(x).shape).numpy()
        res = minimize(
            loss_and_grad, init_mom, method='L-BFGS-B', jac=True,
            options={'disp': False, 'maxiter': 50}, tol=1e-14)

        tangent_vec = torch.Tensor(res.x).reshape(x.shape)
        return tangent_vec

    @staticmethod
    def _get_triangle_mask(polydata):
        n = polydata.GetNumberOfCells()
        triangles = []
        for i in range(n):
            cell = polydata.GetCell(i)
            p0 = cell.GetPointId(0)
            p1 = cell.GetPointId(1)
            p2 = cell.GetPointId(2)
            triangle = torch.tensor([p0, p1, p2])
            triangles.append(triangle)
        return torch.stack(triangles)

    def _volume(self, tensor):
        triangles = self.triangles
        shape = torch.stack(
            [tensor[triangles[:, 0]], tensor[triangles[:, 1]],
             tensor[triangles[:, 2]]], dim=1)
        return torch.det(shape).sum() / 6

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write_flow(self, objects_names, objects_extensions, template, template_data, output_dir,
                   write_adjoint_parameters=False):

        assert not self.flow_is_modified, \
            "You are trying to write data relative to the flow, but it has been modified and not updated."

        for j in range(self.number_of_time_points):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for k, elt in enumerate(objects_names):
                names.append(elt + "__tp_" + str(j) + objects_extensions[k])

            deformed_points = self.get_template_points(j)
            deformed_data = template.get_deformed_data(deformed_points, template_data)
            template.write(output_dir, names,
                           {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})

            if write_adjoint_parameters:
                cp = self.control_points_t[j].detach().cpu().numpy()
                mom = self.momenta_t[j].detach().cpu().numpy()
                write_2D_array(cp, output_dir, elt + "__ControlPoints__tp_" + str(j) + ".txt")
                write_3D_array(mom, output_dir, elt + "__Momenta__tp_" + str(j) + ".txt")

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert not self.shoot_is_modified, \
            "You are trying to write data relative to the shooting, but it has been modified and not updated."
        assert len(self.control_points_t) == len(self.momenta_t), \
            "Something is wrong, not as many cp as momenta in diffeo"
        for j, (control_points, momenta) in enumerate(zip(self.control_points_t, self.momenta_t)):
            write_2D_array(control_points.detach().cpu().numpy(), name + "__control_points_" + str(j) + ".txt")
            write_2D_array(momenta.detach().cpu().numpy(), name + "__momenta_" + str(j) + ".txt")
