import torch

import support.kernels as kernel_factory
from core import default
from core.model_tools.deformations.spline import Spline
from core.models.geodesic_regression import GeodesicRegression
from in_out.array_readers_and_writers import *
from support import utilities

logger = logging.getLogger(__name__)


class SplineRegression(GeodesicRegression):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    # Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,

                 shoot_kernel_type=default.shoot_kernel_type,
                 concentration_of_time_points=default.concentration_of_time_points, t0=default.t0,
                 use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 freeze_external_forces=False, target_weights=None,
                 geodesic_weight=.1,

                 initial_momenta=default.initial_momenta,

                 gpu_mode=default.gpu_mode,

                 **kwargs):

        super(SplineRegression, self).__init__(
            gpu_mode=gpu_mode, dimension=dimension, tensor_scalar_type=tensor_scalar_type,
            tensor_integer_type=tensor_integer_type, dense_mode=dense_mode,
            number_of_processes=number_of_processes,
            template_specifications=template_specifications,

            deformation_kernel_type=deformation_kernel_type,
            deformation_kernel_width=deformation_kernel_width,

            shoot_kernel_type=shoot_kernel_type,
            concentration_of_time_points=concentration_of_time_points, t0=t0,
            use_rk2_for_flow=use_rk2_for_flow,

            freeze_template=freeze_template, use_sobolev_gradient=use_sobolev_gradient,
            smoothing_kernel_width=smoothing_kernel_width,

            initial_control_points=initial_control_points, freeze_control_points=freeze_control_points,
            initial_cp_spacing=initial_cp_spacing,

            initial_momenta=initial_momenta, **kwargs)

        # Deformation.
        self.name = 'SplineRegression'
        self.geodesic = Spline(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(
                deformation_kernel_type, gpu_mode=gpu_mode, kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type, use_rk2_for_flow=use_rk2_for_flow,
            t0=t0, concentration_of_time_points=concentration_of_time_points, geodesic_weight=geodesic_weight)

        # External Forces.
        self.fixed_effects['external_forces'] = torch.zeros(
            concentration_of_time_points, self.number_of_control_points, self.dimension)
        self.freeze_external_forces = freeze_external_forces

        if target_weights is None:
            self.target_weights = torch.ones(concentration_of_time_points)
        else:
            self.target_weights = target_weights

    ####################################################################################################################
    # Encapsulation methods:
    ####################################################################################################################
    # External Forces --------------------------------------------------------------------------------------------------
    def get_external_forces(self):
        return self.fixed_effects['external_forces']

    def set_external_forces(self, force):
        self.fixed_effects['external_forces'] = force

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        out['momenta'] = self.fixed_effects['momenta']
        if not self.freeze_external_forces:
            out['external_forces'] = self.fixed_effects['external_forces']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        self.set_momenta(fixed_effects['momenta'])
        if not self.freeze_external_forces:
            self.set_external_forces(fixed_effects['external_forces'])

    def get_target_weights(self):
        return self.target_weights

    def set_target_weights(self, weights):
        self.target_weights = weights


    ####################################################################################################################
    # Public methods:
    ####################################################################################################################

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, control_points, momenta, external_forces = self._fixed_effects_to_torch_tensors(
            with_grad, device=device)

        # Deform -------------------------------------------------------------------------------------------------------
        attachment, regularity = self.compute_attachment_and_regularity(
            dataset, template_data, template_points, control_points, momenta, external_forces)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad

                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = self.sobolev_kernel.convolve(
                        template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                        gradient['landmark_points'].detach())

            # Control points and momenta.
            if not self.freeze_control_points:
                gradient['control_points'] = control_points.grad
            if not self.freeze_external_forces:
                gradient['external_forces'] = external_forces.grad
            gradient['momenta'] = momenta.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

    ####################################################################################################################
    # Private methods:
    ####################################################################################################################

    def compute_attachment_and_regularity(
            self, dataset, template_data, template_points, control_points, momenta, external_forces):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.set_tmin(min(target_times))
        self.geodesic.set_tmax(max(target_times))
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_control_points_t0(control_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.set_external_force(external_forces)
        self.geodesic.update()

        attachment = 0.
        for j, (time, obj) in enumerate(zip(target_times, target_objects)):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            attachment -= self.target_weights[j] * self.multi_object_attachment.compute_weighted_distance(
                deformed_data, self.template, obj, self.objects_noise_variance)
        regularity = - self.geodesic.get_norm_squared()

        return attachment, regularity

    ####################################################################################################################
    # Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value,
                                                  dtype=self.tensor_scalar_type,
                                                  requires_grad=(not self.freeze_template and with_grad),
                                                  device=device)
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value,
                                                    dtype=self.tensor_scalar_type,
                                                    requires_grad=(not self.freeze_template and with_grad),
                                                    device=device)
                           for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = utilities.move_data(
                control_points, dtype=self.tensor_scalar_type,
                requires_grad=(not self.freeze_control_points and with_grad), device=device)

        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(
            momenta, dtype=self.tensor_scalar_type, requires_grad=with_grad, device=device)

        # External Forces
        external_forces = self.fixed_effects['external_forces']
        external_forces = utilities.move_data(
            external_forces, dtype=self.tensor_scalar_type,
            requires_grad=(not self.freeze_external_forces and with_grad), device=device)

        return template_data, template_points, control_points, momenta, external_forces

    ####################################################################################################################
    # Writing methods:
    ####################################################################################################################
    def _write_model_predictions(self, output_dir, dataset=None,
                                 write_adjoint_parameters=False, compute_residuals=True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, template_points, control_points, momenta, external_forces = self._fixed_effects_to_torch_tensors(
            False)
        target_times = dataset.times[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.tmin = min(target_times)
        self.geodesic.tmax = max(target_times)
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_control_points_t0(control_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.set_external_force(external_forces)
        self.geodesic.update()

        # Write --------------------------------------------------------------------------------------------------------
        # Geodesic flow.
        # self.geodesic.write(self.name, self.objects_name, self.objects_name_extension, self.template, template_data,
        #                     output_dir, write_adjoint_parameters)

        # Model predictions.
        if dataset is not None:
            residuals = []
            for j, time in enumerate(target_times):
                names = []
                for k, (object_name, object_extension) in enumerate(
                        zip(self.objects_name, self.objects_name_extension)):
                    name = self.name + '__Reconstruction__' + object_name + '__tp_' + str(j) + ('__age_%.2f' % time) \
                           + object_extension
                    names.append(name)
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names,
                                    {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

                if compute_residuals:
                    residuals.append(self.multi_object_attachment.compute_distances(
                        deformed_data, self.template, dataset.deformable_objects[0][j]))
                    residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i] for
                                      residuals_i in residuals]
                    write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

    def _write_model_parameters(self, output_dir):
        # Template.
        # template_names = []
        # for k in range(len(self.objects_name)):
        #     aux = self.name + '__EstimatedParameters__Template_' + self.objects_name[k] + '__tp_' \
        #           + str(self.geodesic.backward_exponential.number_of_time_points - 1) \
        #           + ('__age_%.2f' % self.geodesic.t0) + self.objects_name_extension[k]
        #     template_names.append(aux)
        # self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")

        # External Forces
        if not self.freeze_external_forces:
            write_3D_array(
                self.get_external_forces(), output_dir, self.name + "__EstimatedParameters__ExternalForces.txt")
