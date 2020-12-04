import logging
import torch
import numpy as np
import vtk

from core import default
from core.models.deterministic_atlas import DeterministicAtlas
from in_out.array_readers_and_writers import read_3D_array, write_2D_list
from support import utilities


logger = logging.getLogger(__name__)


class VolumeConstrainedShooting(DeterministicAtlas):

    def __init__(self, template_specifications,
                 number_of_subjects=1,
                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot,
                 use_rk2_for_flow=default.use_rk2_for_flow,
                 use_rk4_for_shoot=False,
                 use_svf=False, freeze_size_effect=False,

                 initial_control_points=default.initial_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 initial_momenta=default.initial_momenta,
                 gradient_flow=False, regularisation=1.,

                 gpu_mode=default.gpu_mode, **kwargs):

        super(VolumeConstrainedShooting, self).__init__(
            gpu_mode=gpu_mode, dimension=dimension,
            number_of_subjects=number_of_subjects,
            tensor_scalar_type=tensor_scalar_type,
            tensor_integer_type=tensor_integer_type, dense_mode=dense_mode,
            number_of_processes=number_of_processes,
            template_specifications=template_specifications,
            freeze_template=True,

            deformation_kernel_type=deformation_kernel_type,
            deformation_kernel_width=deformation_kernel_width,
            number_of_time_points=number_of_time_points,
            use_svf=use_svf,

            shoot_kernel_type=shoot_kernel_type,
            use_rk2_for_flow=use_rk2_for_flow,
            use_rk2_for_shoot=use_rk2_for_shoot,
            use_rk4_for_shoot=use_rk4_for_shoot,

            initial_control_points=initial_control_points,
            freeze_control_points=True,
            initial_cp_spacing=initial_cp_spacing,

            initial_momenta=initial_momenta)

        self.name = 'VolumeConstrainedShooting'
        # Declare model structure.
        # self.fixed_effects['template_data'] = None
        # self.fixed_effects['control_points'] = None
        # self.fixed_effects['momenta'] = None
        if not gradient_flow:
            control_points = read_3D_array(initial_control_points)
            logger.info('>> Reading %d initial control points from file %s.' % (len(control_points), initial_control_points))
            self.fixed_effects['control_points'] = control_points
            self.number_of_control_points = len(self.fixed_effects['control_points'])

        self.fixed_effects['intercept'] = torch.tensor([1.])
        self.fixed_effects['size_effect'] = torch.tensor([0.])

        self.freeze_template = True
        self.freeze_control_points = True
        self.freeze_momenta = True
        self.freeze_size_effect = freeze_size_effect
        self.gradient_flow = gradient_flow
        self.regularization = regularisation

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(template_specifications['shape']['filename'])
        reader.Update()
        self.polydata = reader.GetOutput()

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def set_intercept(self, inter):
        self.fixed_effects['intercept'] = inter

    def get_intercept(self):
        return self.fixed_effects['intercept']

    def set_size_effect(self, inter):
        self.fixed_effects['size_effect'] = inter

    def get_size_effect(self):
        return self.fixed_effects['size_effect']

    def get_fixed_effects(self):
        out = {'intercept': self.fixed_effects['intercept']}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        if not self.freeze_momenta:
            out['momenta'] = self.fixed_effects['momenta']
        if not self.freeze_size_effect:
            out['size_effect'] = self.fixed_effects['size_effect']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        if not self.freeze_momenta:
            self.set_momenta(fixed_effects['momenta'])
        self.set_intercept(fixed_effects['intercept'])
        if not self.freeze_size_effect:
            self.set_size_effect(fixed_effects['size_effect'])

    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, intercept, size_effect = \
            self._fixed_effects_to_torch_tensors(with_grad, device=device)
        return self.compute_attachment_and_regularity(
            dataset, template_data, template_points, control_points, momenta, intercept, size_effect, with_grad,
            device=device)

    def compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, momenta,
                                          intercept, size_effect, with_grad=False, device='cpu'):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize.
        target_size = dataset['target']
        subject_size = dataset['subject_size']
        attachment = 0.
        regularity = 0.

        # loop for every deformable object
        # deform and update attachment and regularity
        for i, target in enumerate(target_size):
            new_attachment, new_regularity = self.deform_and_compute_attachment_and_regularity(
                self.exponential, template_points, control_points[i], momenta[i],
                self.template, template_data, intercept, size_effect, subject_size, target, device=device)
            attachment += new_attachment
            regularity += new_regularity

        # Compute gradient.
        return self.compute_gradients(attachment, regularity, intercept, size_effect, self.freeze_size_effect,
                                      with_grad)

    def deform_and_compute_attachment_and_regularity(self, exponential, template_points, control_points, momenta,
                                                     template, template_data, intercept, size_effect,
                                                     subject_size, target_size,
                                                     device='cpu'):
        # Deform.
        exponential.set_initial_template_points(template_points)
        scaling = intercept + size_effect * subject_size
        if self.gradient_flow:
            exponential.volume_gradient_flow(end_time=scaling)
        else:
            exponential.set_initial_control_points(control_points)
            exponential.set_initial_momenta(scaling * momenta)
            exponential.move_data_to_(device=device)
            exponential.update()

        # Compute attachment and regularity.
        deformed_points = exponential.get_template_points()
        deformed_data = template.get_deformed_data(deformed_points, template_data)
        deformed_volume = self.volume(tensor=deformed_data['landmark_points'])
        attachment = -((deformed_volume - target_size) / 1000) ** 2 / self.number_of_subjects
        regularity = - self.regularization * (1. - scaling[0]) ** 2

        assert torch.device(
            device) == attachment.device == regularity.device, 'attachment and regularity tensors must be on the same device. ' \
                                                               'device=' + device + \
                                                               ', attachment.device=' + str(attachment.device) + \
                                                               ', regularity.device=' + str(regularity.device)

        return attachment, regularity

    @staticmethod
    def compute_gradients(attachment, regularity, intercept, size_effect, freeze_size_effect,
                          with_grad=False):
        if with_grad:
            total_for_subject = attachment + regularity
            total_for_subject.backward()

            gradient = {'intercept': intercept.grad.detach().cpu().numpy()}
            if not freeze_size_effect:
                gradient['size_effect'] = size_effect.grad.detach().cpu().numpy()

            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

        return res

    def volume(self, tensor):
        polydata = self.polydata
        n = polydata.GetNumberOfCells()
        vol = 0
        for i in range(n):
            cell = polydata.GetCell(i)
            p0 = cell.GetPointId(0)
            p1 = cell.GetPointId(1)
            p2 = cell.GetPointId(2)
            triangle = torch.stack([tensor[p0], tensor[p1], tensor[p2]])
            vol += triangle.det() / 6
        return vol

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                  requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                    requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = utilities.move_data(control_points, device=device, dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_control_points and with_grad))

        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(not self.freeze_momenta and with_grad))

        intercept = self.fixed_effects['intercept']
        intercept = utilities.move_data(intercept, device=device, dtype=self.tensor_scalar_type, requires_grad=True)

        size_effect = self.fixed_effects['size_effect']
        size_effect = utilities.move_data(
            size_effect, device=device, dtype=self.tensor_scalar_type,
            requires_grad=(not self.freeze_size_effect and with_grad))
        return template_data, template_points, control_points, momenta, intercept, size_effect

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):
        device, _ = utilities.get_best_device(self.gpu_mode)

        # Initialize.
        template_data, template_points, control_points, momenta, intercept, size_effect = \
            self._fixed_effects_to_torch_tensors(False, device=device)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset['subject_ids']):
            scaling = intercept + size_effect * dataset['subject_size']
            if self.gradient_flow:
                self.exponential.volume_gradient_flow(end_time=scaling)
            else:
                self.exponential.set_initial_control_points(control_points[i])
                self.exponential.set_initial_momenta(scaling * momenta[i])
                self.exponential.move_data_to_(device=device)
                self.exponential.update()

            # # Writing the whole flow.
            # names = []
            # for k, object_name in enumerate(self.objects_name):
            #     name = self.name + '__flow__' + object_name + '__subject_' + subject_id
            #     names.append(name)
            # self.exponential.write_flow(names, self.objects_name_extension, self.template, template_data, output_dir)

            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append([
                    (self.volume(deformed_data['landmark_points'].detach()) - dataset['target'][i]) / 1000])

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            self.template.write(output_dir, names,
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, output_dir):
        write_2D_list(np.stack([self.get_intercept(), self.get_size_effect()]), output_dir, 'scaling_parameters.txt')
