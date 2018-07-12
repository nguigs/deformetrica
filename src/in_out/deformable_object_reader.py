import logging
import warnings

# Image readers
import PIL.Image as pimg
import nibabel as nib
import numpy as np

from core.observations.deformable_objects.image import Image
from core.observations.deformable_objects.landmarks.landmark import Landmark
from core.observations.deformable_objects.landmarks.point_cloud import PointCloud
from core.observations.deformable_objects.landmarks.poly_line import PolyLine
from core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from in_out.image_functions import normalize_image_intensities

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)


class DeformableObjectReader:
    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    connectivity_degrees = {'LINES': 2, 'POLYGONS': 3}

    # Create a PyDeformetrica object from specified filename and object type.
    @staticmethod
    def create_object(object_filename, object_type, tensor_scalar_type, dimension=None):

        if object_type.lower() in ['SurfaceMesh'.lower(), 'PolyLine'.lower(), 'PointCloud'.lower(), 'Landmark'.lower()]:

            if object_type.lower() == 'SurfaceMesh'.lower():
                points, dimension, connectivity = DeformableObjectReader.read_vtk_file(object_filename, dimension, extract_connectivity=True)
                out_object = SurfaceMesh(dimension, tensor_scalar_type)
                out_object.set_points(points)
                out_object.set_connectivity(connectivity)

            elif object_type.lower() == 'PolyLine'.lower():
                points, dimension, connectivity = DeformableObjectReader.read_vtk_file(object_filename, dimension, extract_connectivity=True)
                out_object = PolyLine(dimension, tensor_scalar_type)
                out_object.set_points(points)
                out_object.set_connectivity(connectivity)

            elif object_type.lower() == 'PointCloud'.lower():
                try:
                    points, dimension, connectivity = DeformableObjectReader.read_vtk_file(object_filename, dimension, extract_connectivity=True)
                    out_object = PointCloud(dimension, tensor_scalar_type)
                    out_object.set_points(points)
                    out_object.set_connectivity(connectivity)
                except KeyError:
                    points, dimension = DeformableObjectReader.read_vtk_file(object_filename, dimension, extract_connectivity=False)
                    out_object = PointCloud(dimension, tensor_scalar_type)
                    out_object.set_points(points)

            elif object_type.lower() == 'Landmark'.lower():
                try:
                    points, dimension, connectivity = DeformableObjectReader.read_vtk_file(object_filename, dimension, extract_connectivity=True)
                    out_object = Landmark(dimension, tensor_scalar_type)
                    out_object.set_points(points)
                    out_object.set_connectivity(connectivity)
                except KeyError:
                    points, dimension = DeformableObjectReader.read_vtk_file(object_filename, dimension, extract_connectivity=False)
                    out_object = Landmark(dimension, tensor_scalar_type)
                    out_object.set_points(points)
            else:
                raise TypeError('Object type ' + object_type + ' was not recognized.')

            out_object.update()

        elif object_type.lower() == 'Image'.lower():
            if object_filename.find(".png") > 0:
                img_data = np.array(pimg.open(object_filename))
                dimension = len(img_data.shape)
                img_affine = np.eye(dimension + 1)
                if len(img_data.shape) > 2:
                    msg = 'Multi-channel images are not managed (yet). Defaulting to the first channel.'
                    warnings.warn(msg)
                    img_data = img_data[:, :, 0]

            elif object_filename.find(".npy") > 0:
                img_data = np.load(object_filename)
                dimension = len(img_data.shape)
                img_affine = np.eye(dimension + 1)

            elif object_filename.find(".nii") > 0:
                img = nib.load(object_filename)
                img_data = img.get_data()
                dimension = len(img_data.shape)
                img_affine = img.affine
                assert len(img_data.shape) == 3, "Multi-channel images not available (yet!)."

            else:
                raise TypeError('Unknown image extension for file: %s' % object_filename)

            # Rescaling between 0. and 1.
            img_data, img_data_dtype = normalize_image_intensities(img_data)
            out_object = Image(dimension, tensor_scalar_type)
            out_object.set_intensities(img_data)
            out_object.set_affine(img_affine)
            out_object.intensities_dtype = img_data_dtype

            dimension_image = len(img_data.shape)
            if dimension_image != dimension:
                logger.warning('I am reading a {}d image but the dimension is set to {}'
                               .format(dimension_image, dimension))

            out_object.update()

        else:
            raise RuntimeError('Unknown object type: ' + object_type)

        return out_object

    @staticmethod
    def read_vtk_file(filename, dimension=None, extract_connectivity=False):
        """
        Routine to read  vtk files
        Probably needs new case management
        """

        with open(filename, 'r') as f:
            content = f.readlines()
        fifth_line = content[4].strip().split(' ')

        assert fifth_line[0] == 'POINTS'
        assert fifth_line[2] == 'float'

        nb_points = int(fifth_line[1])
        points = []
        line_start_connectivity = None

        if dimension is None:
            # Try to determine dimension from VTK file: check last element in first 2 points to see if filled with 0.00000, if so 2D else 3D
            if float(content[5].strip().split(' ')[2]) == 0. and float(content[6].strip().split(' ')[2]) == 0.:
                dimension = 2
            else:
                dimension = 3

            if dimension is None:
                raise RuntimeError('Could not automatically determine data dimension. Please manually specify value.')

        assert isinstance(dimension, int)
        logger.debug('Using dimension ' + str(dimension) + ' for file ' + filename)

        # Reading the points:
        for i in range(5, len(content)):
            line = content[i].strip().split(' ')
            # Saving the position of the start for the connectivity
            if line == ['']:
                continue
            elif line[0] in ['LINES', 'POLYGONS']:
                line_start_connectivity = i
                connectivity_type, nb_faces, nb_vertices_in_faces = line[0], int(line[1]), int(line[2])
                break
            else:
                points_for_line = np.array(line, dtype=float).reshape(int(len(line)/3), 3)[:, :dimension]
                for p in points_for_line:
                    points.append(p)
        points = np.array(points)
        assert len(points) == nb_points, 'Something went wrong during the vtk reading'

        # Reading the connectivity, if needed.
        if extract_connectivity:
            if line_start_connectivity is None:
                raise KeyError('Could not read the connectivity for the given vtk file')

            connectivity = []

            for i in range(line_start_connectivity + 1, line_start_connectivity + 1 + nb_faces):
                line = content[i].strip().split(' ')
                number_vertices_in_line = int(line[0])

                if connectivity_type == 'POLYGONS':
                    assert number_vertices_in_line == 3, 'Invalid connectivity: ' \
                                                         'deformetrica only handles triangles for now.'
                    connectivity.append([int(elt) for elt in line[1:]])

                elif connectivity_type == 'LINES':
                    assert number_vertices_in_line >= 2, 'Should not happen.'
                    for i in range(1, number_vertices_in_line):
                        connectivity.append([int(line[i]), int(line[i+1])])

            connectivity = np.array(connectivity)

            # Some sanity checks:
            if connectivity_type == 'POLYGONS':
                assert len(connectivity) == nb_faces, 'Found an unexpected number of faces.'
                assert len(connectivity) * 4 == nb_vertices_in_faces

            return points, dimension, connectivity

        return points, dimension
