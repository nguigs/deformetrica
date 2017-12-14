import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import warnings
import time

from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import *
from pydeformetrica.src.in_out.template_creator import create_template_metadata
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.core.model_tools.deformations.diffeomorphism import Diffeomorphism
from pydeformetrica.src.in_out.utils import *
from torch.autograd import Variable

"""
Basic info printing.

"""

print('')
print('##############################')
print('##### PyDeformetrica 1.0 #####')
print('##############################')
print('')

print('[ shooting function ]')
print('')



"""
Read command line, read xml files, set general settings.

"""

assert len(sys.argv) >= 3, "Usage: " + sys.argv[0] + " <model.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
optimizationParametersXmlPath = sys.argv[2]

xmlParameters = XmlParameters()
xmlParameters.read_all_xmls(modelXmlPath, None, optimizationParametersXmlPath)

Settings().dimension = xmlParameters.dimension




"""
Create the template object
"""

t_list, t_name, t_name_extension, t_noise_variance, t_norm, multi_object_attachment = \
    create_template_metadata(xmlParameters.template_specifications)

print("Object list:", t_list)

template = DeformableMultiObject()
template.object_list = t_list
template.update()

"""
Reading Control points and momenta
"""

if not (os.path.exists(Settings().output_dir)):
    Settings().output_dir


if not xmlParameters.initial_control_points is None:
    control_points = read_2D_array(xmlParameters.initial_control_points)
else:
    raise ArgumentError('Please specify a path to control points to perform a shooting')

if not xmlParameters.initial_momenta is None:
    momenta = read_momenta(xmlParameters.initial_momenta)
else:
    raise ArgumentError('Please specify a path to momenta to perform a shooting')

template_data_numpy = template.get_data()
template_data_torch = Variable(torch.from_numpy(template_data_numpy))

momenta_torch = Variable(torch.from_numpy(momenta))
control_points_torch = Variable(torch.from_numpy(control_points))

diffeo = Diffeomorphism()
diffeo.set_initial_control_points(control_points_torch)
diffeo.set_landmark_points(template_data_numpy)
model.diffeomorphism.kernel = create_kernel(xmlParameters.DeformationKernelType, xmlParameters.DeformationKernelWidth)


for i in range(len(momenta)):
    self.Diffeomorphism.set_initial_momenta(momenta[i])
    self.Diffeomorphism.shoot()
    self.Diffeomorphism.flow()
    deformedPoints = self.Diffeomorphism.get_landmark_points()
    names = [elt + "_"+ str(i) for elt in self.ObjectsName]
    diffeo.write_flow(names, t_name_extension, template)














