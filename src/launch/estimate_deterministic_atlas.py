import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings
from pydeformetrica.src.io.dataset_creator import DatasetCreator

"""
Basic info printing.

"""

print('')
print('##############################')
print('##### PyDeformetrica 1.0 #####')
print('##############################')
print('')

print('[ estimate_deterministic_atlas function ]')
print('')


"""
Read command line, read xml files, create dataset object.

"""

assert len(sys.argv) >= 4, "Usage: " + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml>"
modelXmlPath = sys.argv[1]
datasetXmlPath = sys.argv[2]
optimizationParametersXmlPath = sys.argv[3]

xmlParameters = XmlParameters()
xmlParameters.ReadAllXmls(modelXmlPath, datasetXmlPath, optimizationParametersXmlPath)

datasetCreator = DatasetCreator()
dataset = datasetCreator.CreateDataset(xmlParameters.DatasetFilenames, xmlParameters.VisitAges,
                                       xmlParameters.SubjectIds, xmlParameters.TemplateSpecifications)

"""
Create the model object.

"""

model = DeterministicAtlas()

model.Diffeomorphism.KernelType = xmlParameters.DeformationKernelType
model.Diffeomorphism.KernelWidth = xmlParameters.DeformationKernelWidth
model.Diffeomorphism.NumberOfTimePoints = xmlParameters.NumberOfTimePoints

model.InitializeTemplateAttributes(xmlParameters.TemplateSpecifications)

model.SmoothingKernelWidth = xmlParameters.DeformationKernelWidth * xmlParameters.SmoothingKernelWidthRatio
model.InitialCpSpacing = xmlParameters.InitialCpSpacing
model.NumberOfSubjects = dataset.NumberOfSubjects

model.Update()

"""
Create the estimator object.

"""

estimator = GradientAscent()

estimator.MaxIterations = xmlParameters.MaxIterations
estimator.PrintEveryNIters = xmlParameters.PrintEveryNIters
estimator.SaveEveryNIters = xmlParameters.SaveEveryNIters
estimator.InitialStepSize = xmlParameters.InitialStepSize
estimator.MaxLineSearchIterations = xmlParameters.MaxLineSearchIterations
estimator.LineSearchShrink = xmlParameters.LineSearchShrink
estimator.LineSearchExpand = xmlParameters.LineSearchExpand
estimator.ConvergenceTolerance = xmlParameters.ConvergenceTolerance

estimator.Dataset = dataset
estimator.StatisticalModel = model

"""
Launch.

"""

model.Name = 'DeterministicAtlas'
estimator.Update()