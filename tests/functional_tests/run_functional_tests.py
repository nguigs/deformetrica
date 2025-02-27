#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import unittest

from functional_tests.data.atlas.brain_structures.run import AtlasBrainStructures
from functional_tests.data.atlas.digits.run import AtlasDigits
from functional_tests.data.atlas.skulls.run import AtlasSkulls
from functional_tests.data.parallel_transport.alien.run import ParallelTransportAlien
from functional_tests.data.parallel_transport.snowman.run import ParallelTransportSnowman
from functional_tests.data.registration.points.run import RegistrationPoints
from functional_tests.data.registration.tetris.run import RegistrationTetris
from functional_tests.data.regression.cross.run import RegressionCross
from functional_tests.data.regression.skulls.run import RegressionSkulls
from functional_tests.data.regression.surprise.run import RegressionSurprise
from functional_tests.data.shooting.grid.run import ShootingGrid
from functional_tests.data.shooting.snowman.run import ShootingSnowman
from functional_tests.data.principal_geodesic_analysis.digits.run import PrincipalGeodesicAnalysisDigits
from functional_tests.data.longitudinal_atlas.starmen.run import LongitudinalAtlasStarmen
from functional_tests.data.longitudinal_atlas.hippocampi.run import LongitudinalAtlasHippocampi
from functional_tests.data.longitudinal_atlas.digits.run import LongitudinalAtlasDigits

TEST_MODULES = [AtlasSkulls, AtlasBrainStructures, AtlasDigits,
                RegressionSkulls, RegressionSurprise, RegressionCross,
                RegistrationPoints, RegistrationTetris,
                ParallelTransportSnowman, ParallelTransportAlien,
                ShootingGrid, ShootingSnowman,
                PrincipalGeodesicAnalysisDigits,
                LongitudinalAtlasStarmen, LongitudinalAtlasHippocampi, LongitudinalAtlasDigits
                ]

# TEST_MODULES = [LongitudinalAtlasStarmen]


def setup_conda_env():
    path_to_environment_file = os.path.normpath(
        os.path.join(os.path.abspath(__file__), '../../../environment.yml'))
    cmd = 'hostname && ' \
          'if [ -f ~/.profile ]; then . ~/.profile; fi &&' \
          'conda env create -f %s' % path_to_environment_file
    os.system(cmd)


def main():
    setup_conda_env()
    success = True

    for t in TEST_MODULES:
        res = unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))
        success = success and res.wasSuccessful()

    if not success:
        sys.exit('Test failure !')


if __name__ == '__main__':
    main()
