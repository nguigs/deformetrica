import os.path
import sys
import numpy as np
import warnings
import torch
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.support.utilities.general_settings import Settings
import matplotlib.pyplot as plt
from torch.autograd import Variable
"""
Generic geodesic. It wraps a manifold (e.g. OneDimensionManifold) and uses 
its exponential attributes to make manipulations more convenient (e.g. backward and forward) 
The handling is radically different between exponential with closed form and the ones without.

The velocity is the initial criterion here. The translation to momenta is done, if needed, in the exponential objects.
"""

class GenericGeodesic:
    def __init__(self, exponential_factory):
        self.t0 = None
        self.tmax = None
        self.tmin = None
        self.concentration_of_time_points = 10

        self.position_t0 = None
        self.velocity_t0 = None

        self.forward_exponential = exponential_factory.create()
        self.backward_exponential = exponential_factory.create()

        self.manifold_type = exponential_factory.manifold_type

        self.is_modified = True
        self._times = None
        self._geodesic_trajectory = None

    def set_t0(self, t0):
        self.t0 = t0
        self.is_modified = True

    def set_tmin(self, tmin):
        self.tmin = tmin
        self.is_modified = True

    def set_tmax(self, tmax):
        self.tmax = tmax
        self.is_modified = True

    def set_position_t0(self, position_t0):
        self.position_t0 = position_t0
        self.is_modified = True

    def set_velocity_t0(self, velocity_t0):
        self.velocity_t0 = velocity_t0
        self.is_modified = True

    def set_concentration_of_time_points(self, ctp):
        self.concentration_of_time_points = ctp
        self.is_modified = True

    def get_geodesic_point(self, time):
        if self.forward_exponential.has_closed_form:
            return self.forward_exponential.closed_form(self.position_t0, self.velocity_t0, time - self.t0)

        else:

            time_np = time.data.numpy()[0]

            assert self.tmin <= time_np <= self.tmax
            if self.is_modified:
                msg = "Asking for geodesic point but the geodesic was modified and not updated"
                warnings.warn(msg)

            times = self._get_times()

            # Deal with the special case of a geodesic reduced to a single point.
            if len(times) == 1:
                print('>> The geodesic seems to be reduced to a single point.')
                return self.position_t0

            # Standard case.
            if time_np <= self.t0:
                if self.backward_exponential.number_of_time_points <= 2:
                    j = 1
                else:
                    dt = (self.t0 - self.tmin) / (self.backward_exponential.number_of_time_points - 1)
                    j = int((time_np-self.tmin)/dt) + 1
                    assert times[j - 1] <= time_np
                    assert times[j] >= time_np
            else:
                if self.forward_exponential.number_of_time_points <= 2:
                    j = len(times) - 1
                else:
                    dt = (self.tmax - self.t0) / (self.forward_exponential.number_of_time_points - 1)
                    j = min(len(times)-1,
                            int((time_np - self.t0) / dt) + self.backward_exponential.number_of_time_points)

                    assert times[j - 1] <= time_np
                    assert times[j] >= time_np

            # print(times[j-1], time_np, times[j], self.tmin, self.tmax, j, len(times))

            weight_left = (times[j] - time) / (times[j] - times[j - 1])
            weight_right = (time - times[j - 1]) / (times[j] - times[j - 1])
            geodesic_t = self._get_geodesic_trajectory()
            geodesic_point = weight_left * geodesic_t[j - 1] + weight_right * geodesic_t[j]
            return geodesic_point

    def update(self):
        assert self.t0 >= self.tmin, "tmin should be smaller than t0"
        assert self.t0 <= self.tmax, "tmax should be larger than t0"

        if not self.forward_exponential.has_closed_form:
            # Backward exponential -----------------------------------------------------------------------------------------
            delta_t = self.t0 - self.tmin
            self.backward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
            if self.is_modified:
                self.backward_exponential.set_initial_position(self.position_t0)
                self.backward_exponential.set_initial_velocity(- self.velocity_t0 * delta_t)
            if self.backward_exponential.number_of_time_points > 1:
                self.backward_exponential.update()
            else:
                self.backward_exponential.update_norm_squared()

            # Forward exponential ------------------------------------------------------------------------------------------
            delta_t = self.tmax - self.t0
            self.forward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
            if self.is_modified:
                self.forward_exponential.set_initial_position(self.position_t0)
                self.forward_exponential.set_initial_velocity(self.velocity_t0 * delta_t)
            if self.forward_exponential.number_of_time_points > 1:
                self.forward_exponential.update()
            else:
                self.forward_exponential.update_norm_squared()

        self._update_geodesic_trajectory()
        self._update_times()
        self.is_modified = False

    def _update_times(self):
        if not self.forward_exponential.has_closed_form:
            times_backward = [self.t0]
            if self.backward_exponential.number_of_time_points > 1:
                times_backward = np.linspace(
                    self.t0, self.tmin, num=self.backward_exponential.number_of_time_points).tolist()

            times_forward = [self.t0]
            if self.forward_exponential.number_of_time_points > 1:
                times_forward = np.linspace(
                    self.t0, self.tmax, num=self.forward_exponential.number_of_time_points).tolist()

            self._times = times_backward[::-1] + times_forward[1:]

        else:
            delta_t = self.tmax - self.tmin
            self._times = np.linspace(self.tmin, self.tmax, delta_t * self.concentration_of_time_points)

    def _get_times(self):
        if self.is_modified:
            msg = "Asking for geodesic times but the geodesic was modified and not updated"
            warnings.warn(msg)

        return self._times

    def _update_geodesic_trajectory(self):
        if not self.forward_exponential.has_closed_form:
            backward_geodesic_t = [self.backward_exponential.get_initial_position()]
            if self.backward_exponential.number_of_time_points > 1:
                backward_geodesic_t = self.backward_exponential.position_t

            forward_geodesic_t = [self.forward_exponential.get_initial_position()]
            if self.forward_exponential.number_of_time_points > 1:
                forward_geodesic_t = self.forward_exponential.position_t

            self._geodesic_trajectory = backward_geodesic_t[::-1] + forward_geodesic_t[1:]

    def _get_geodesic_trajectory(self):
        if self.is_modified and not self.forward_exponential.has_closed_form:
            msg = "Trying to get geodesic trajectory in non updated geodesic."
            warnings.warn(msg)

        if self.forward_exponential.has_closed_form:
            trajectory = [self.get_geodesic_point(time)
                          for time in self._get_times()]
            return trajectory

        return self._geodesic_trajectory

    def set_parameters(self, extra_parameters):
        """
        Setting extra parameters for the exponentials
        e.g. parameters for the metric
        """
        self.forward_exponential.set_parameters(extra_parameters)
        self.backward_exponential.set_parameters(extra_parameters)
        self.is_modified = True

    def save_metric_plot(self):
        """
        Plot the metric (if it's 1D)
        """
        times = np.linspace(-0.4, 1.2, 300)
        times_torch = Variable(torch.from_numpy(times)).type(torch.DoubleTensor)
        metric_values = [self.forward_exponential.inverse_metric(t).data.numpy()[0] for t in times_torch]
        # square_root_metric_values = [np.sqrt(elt) for elt in metric_values]
        plt.plot(times, metric_values)
        plt.ylim(0., 1.)
        plt.savefig(os.path.join(Settings().output_dir, "inverse_metric_profile.pdf"))
        plt.clf()

    def save_geodesic_plot(self, name=None):
        """
        Plot a geodesic (if it's 1D)
        """
        times = self._get_times()
        geodesic_values = [elt.data.numpy()[0] for elt in self._get_geodesic_trajectory()]
        plt.plot(times, geodesic_values)
        plt.savefig(os.path.join(Settings().output_dir, "reference_geodesic.pdf"))
        plt.clf()
        # We also save a txt file with trajectory.
        XY = np.stack((times, geodesic_values))
        if name is not None:
            np.savetxt(os.path.join(Settings().output_dir, name+"_reference_geodesic_trajectory.txt"), XY)
        else:
            np.savetxt(os.path.join(Settings().output_dir, "reference_geodesic_trajectory.txt"), XY)

    def parallel_transport(self, vector_to_transport_t0, with_tangential_component=True):
        """
        :param vector_to_transport_t0: the vector to parallel transport, given at t0 and carried at position_t0
        :returns: the full trajectory of the parallel transport, from tmin to tmax
        """

        if self.shoot_is_modified:
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)

        if self.backward_exponential.number_of_time_points > 1:
            backward_transport = self.backward_exponential.parallel_transport(vector_to_transport_t0,
                                                                              with_tangential_component)
        else:
            backward_transport = [vector_to_transport_t0]

        if self.forward_exponential.number_of_time_points > 1:
            forward_transport = self.forward_exponential.parallel_transport(vector_to_transport_t0,
                                                                            with_tangential_component)
        else:
            forward_transport = []

        return backward_transport[::-1] + forward_transport[1:]

    def _write(self):
        print("Write method not implemented for the generic geodesic !")