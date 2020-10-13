import logging
import torch
import vtk

from core import default
from core.models.deterministic_atlas import DeterministicAtlas
from support import utilities


logger = logging.getLogger(__name__)


class VolumeConstrainedShooting(DeterministicAtlas):

    def __init__(self, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot,
                 use_rk2_for_flow=default.use_rk2_for_flow,
                 use_rk4_for_shoot=False,
                 use_svf=False, freeze_size_effect=False,

                 initial_control_points=default.initial_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 initial_momenta=default.initial_momenta,

                 gpu_mode=default.gpu_mode,

                 **kwargs):

        super(VolumeConstrainedShooting, self).__init__(
            self, name='VolumeConstrainedShooting', gpu_mode=gpu_mode, dimension=dimension,
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

            initial_momenta=initial_momenta, **kwargs)

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None
        self.fixed_effects['intercept'] = None
        self.fixed_effects['size_effect'] = None

        self.freeze_template = True
        self.freeze_control_points = True
        self.freeze_momenta = True
        self.freeze_size_effect = freeze_size_effect

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
                self.template, template_data, intercept, size_effect, subject_size[i], target, device=device)

            attachment += new_attachment
            regularity += new_regularity

        # Compute gradient.
        return self.compute_gradients(attachment, regularity, intercept, size_effect, self.freeze_size_effect,
                                      with_grad)

    @classmethod
    def deform_and_compute_attachment_and_regularity(cls, exponential, template_points, control_points, momenta,
                                                     template, template_data, intercept, size_effect,
                                                     subject_size, target_size,
                                                     device='cpu'):
        # Deform.
        exponential.set_initial_template_points(template_points)
        exponential.set_initial_control_points(control_points)
        scaling = intercept + size_effect * subject_size
        exponential.set_initial_momenta(scaling * momenta)
        exponential.move_data_to_(device=device)
        exponential.update()

        # Compute attachment and regularity.
        deformed_points = exponential.get_template_points()
        deformed_data = template.get_deformed_data(deformed_points, template_data)
        deformed_volume = cls.volume(deformed_data)
        attachment = (deformed_volume - target_size) ** 2
        regularity = (1 - scaling) ** 2

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
