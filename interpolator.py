import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

class Interpolator(object):
    def __init__(self, elevclass_bounds, topo, field, mean_deviations=None):
        """Create an interpolator object with N elevation classes.

        The constructor does all of the work of determining the gradients.

        elevclass_bounds(n+1): bounds of each elevation class
        topo(n): mean topographic height of each elevation class
        field(n): mean field value in each elevation class
        mean_deviations(n): mean(abs(topo(j) - topo(k))) for all CISM cells j
          falling in this CLM cell in class k. This should actually be an
          area-weighted mean. In principle, we need mean deviations both for the
          upper half of the elevation class and the lower half. However, for
          two-way coupling, these upper and lower mean deviations are guaranteed
          to be the same (because the mean topographic height of the CLM cell in
          class k is exactly the mean topographic height of the underlying CISM
          cells). For one-way coupling, conservation is not important, and I
          think we'll get better results from the interpolation if we assume
          that the upper and lower mean deviations are the same.
        """

        self._nelev = len(topo)
        assert len(field) == self._nelev
        assert len(elevclass_bounds) == self._nelev + 1

        self._elevclass_bounds = np.array(elevclass_bounds)
        self._topo = np.array(topo)
        self._field = np.array(field)

        if mean_deviations is None:
            mean_deviations = self.guess_mean_deviations(
                self._elevclass_bounds, self._topo)
        else:
            assert len(mean_deviations) == self._nelev
        self._mean_deviations = np.array(mean_deviations)

        self._field_at_mean_topo = np.zeros(self._nelev)
        self._gradients = np.zeros(self._nelev - 1)

        self._compute_gradients()

        self._glint_gradients = np.zeros(self._nelev - 1)
        self._set_glint_gradients()

    @classmethod
    def from_file(cls, filename, multiplier=1):
        """Create an Interpolator object by reading a file

        Field values are multiplied by multiplier

        File should be formatted as:
        nelev (int)
        elevclass_bounds (list of floats; length nelev+1)
        topo (list of floats; length nelev)
        field (list of floats; length nelev)

        For example:
        3
        0. 10. 20. 30.
        5. 15. 25.
        -3. 7. 15.

        mean_deviations are guessed automatically
        """

        with open(filename) as f:
            nelev = int(f.readline())
            elevclass_bounds = [float(x) for x in f.readline().split()]
            topo = [float(x) for x in f.readline().split()]
            field = [float(x) * multiplier for x in f.readline().split()]

        return cls(elevclass_bounds, topo, field)

    @staticmethod
    def guess_mean_deviations(elevclass_bounds, topo):
        nelev = len(topo)
        assert(len(elevclass_bounds) == nelev + 1)

        mean_deviations = []
        for ec in range(nelev):
            # Look for which is smaller: the distance from the mean to the upper
            # bound or the distance from the mean to the lower bound. Whichever
            # is smaller, assume that the mean deviation is half that distance.
            dev_lower = topo[ec] - elevclass_bounds[ec]
            dev_upper = elevclass_bounds[ec + 1] - topo[ec]
            dev_min = min(dev_lower, dev_upper)
            mean_deviations.append(dev_min/2)

        return mean_deviations


    def _compute_gradients(self):
        """Compute gradients and store them for later retrieval"""

        # Compute the gradients as well as field values at each mean topo value
        # by solving the matrix equation A*x = b
        #
        # where:
        #
        # x = [field_at_mean_topo(0), field_at_mean_topo(1), ..., field_at_mean_topo(nelev), gradient(0), gradient(1), ..., gradient(nelev-1)]
        #
        # b = [field(0), field(1), ..., field(nelev), 0, 0, ..., 0]
        #
        # and A is an (2*nelev-1) x (2*nelev-1) matrix described in more detail below

        b = np.zeros(2*self._nelev - 1)
        b[:self._nelev] = self._field

        A = np.zeros([2*self._nelev - 1, 2*self._nelev - 1])

        # Set coefficients for conservation equations in each elevation class
        #
        # Note that, for n elevation classes, we have (n-1) gradients: gradient
        # 0 connects classes 0 and 1, gradient 1 connects classes 1 and 2, etc.
        # We assume gradients are 0 in the lower half of the first elevation
        # class and the upper half of the last elevation class.
        #
        # In general, the equation looks like:
        #   (field_at_mean_topo(i) - mean_deviations(i)*gradient(i-1) + field_at_mean_topo(i) + mean_deviations(i)*gradient(i))/2 = field(i)
        #   => field_at_mean_topo(i) - mean_deviations(i)/2 * gradient(i-1) + mean_deviations(i)/2 * gradient(i) = field(i)
        #
        # For the lowest elevation class, we assume a gradient of 0 in the lower
        # half of the elevation class, so the equation looks like:
        #   field_at_mean_topo(0) + mean_deviations(0)/2 * gradient(0) = field(0)
        #
        # For the highest elevation class, we assume a gradient of 0 in the
        # upper half of the elevation class, so the equation looks like (keeping
        # in mind the 0-based indexing in python):
        #  field_at_mean_topo(nelev-1) - mean_deviations(nelev-1)/2 * gradient(nelev-2) = field(nelev-1)
        for ec in range(self._nelev):
            # Coefficient on field_at_mean_topo(ec)
            A[ec,ec] = 1.
            if (ec > 0):
                # Coefficient on gradient(ec-1)
                A[ec,self._nelev+ec-1] = -self._mean_deviations[ec]/2
            if (ec < self._nelev-1):
                # Coefficient on gradient(ec)
                A[ec,self._nelev+ec] = self._mean_deviations[ec]/2

        # Determine widths between mean topographic height and the elevation
        # class boundaries. For each elevation class i, dl(i) gives the distance
        # from the mean topographic height to the lower bound, and du(i) gives
        # the distance to the upper bound.
        dl = self._topo - self._elevclass_bounds[:-1]
        du = self._elevclass_bounds[1:] - self._topo

        # Set coefficients for equations that ensure continuity at the
        # interfaces between elevation classes
        #
        # For gradient i, the equation looks like:
        #  field_at_mean_topo(i) + du(i) * gradient(i) = field_at_mean_topo(i+1) - dl(i+1) * gradient(i)
        #  => field_at_mean_topo(i) - field_at_mean_topo(i+1) + (du(i) + dl(i+1)) * gradient(i) = 0
        for gradient_num in range(self._nelev-1):
            row = self._nelev + gradient_num
            # Coefficient on field_at_mean_topo(gradient_num)
            A[row,gradient_num] = 1.
            # Coefficient on field_at_mean_topo(gradient_num+1)
            A[row,gradient_num+1] = -1.
            # Coefficient on gradient(gradient_num)
            A[row,row] = du[gradient_num] + dl[gradient_num+1]

        # Solve the matrix problem
        x = np.linalg.solve(A, b)
        self._field_at_mean_topo = x[:self._nelev]
        self._gradients = x[self._nelev:]

        # Sanity check: make sure that the gradients are truly the slope between
        # adjacent points
        for gradient_num in range(self._nelev-1):
            np.testing.assert_approx_equal(self._gradients[gradient_num],
                (self._field_at_mean_topo[gradient_num+1] -
                self._field_at_mean_topo[gradient_num]) /
                (self._topo[gradient_num+1] - self._topo[gradient_num]))


    def _set_glint_gradients(self):
        for ec in range(self._nelev - 1):
            self._glint_gradients[ec] = (self._field[ec+1] - self._field[ec]) / \
              (self._topo[ec+1] - self._topo[ec])

    def __str__(self):
        field_at_mean_topo_str = "Values at mean topo: " + str(self._field_at_mean_topo)
        gradient_str = "Gradients: " + str(self._gradients)
        return field_at_mean_topo_str + "\n" + gradient_str

    def get_mean(self, ec):
        """Get the mean smb in elevation class ec"""

        if (ec == 0):
            mean_lower = self._mean_value_lower(
                ec, self._field_at_mean_topo[ec], 0)
        else:
            mean_lower = self._mean_value_lower(
                ec, self._field_at_mean_topo[ec], self._gradients[ec-1])

        if (ec == self._nelev-1):
            mean_upper = self._mean_value_upper(
                ec, self._field_at_mean_topo[ec], 0)
        else:
            mean_upper = self._mean_value_upper(
                ec, self._field_at_mean_topo[ec], self._gradients[ec])

        return (mean_lower + mean_upper)/2

    def get_value(self, ec, topo):
        """Get the value in the given elevation class at the given topographic
        height"""

        return self._get_value_generic(ec, topo, self._gradients,
                                       self._field_at_mean_topo)

    def get_glint_value(self, ec, topo):
        """Get glint's value in the given elevation class at the given
        topgraphic height"""

        return self._get_value_generic(ec, topo, self._glint_gradients,
                                       self._field)

    def _get_value_generic(self, ec, topo, gradients, field):

        assert ec >= 0
        assert ec < self._nelev

        if topo < self._topo[ec]:
            # In lower half
            if (ec == 0):
                gradient = 0
            else:
                gradient = gradients[ec-1]
        else:
            # In upper half
            if (ec == self._nelev-1):
                gradient = 0
            else:
                gradient = gradients[ec]

        return field[ec] + gradient * (topo - self._topo[ec])

    def get_glint_mean(self, ec):
        """Get the mean smb in elevation class ec if we used glint-style
        interpolation"""

        if (ec == 0):
            mean_lower = self._mean_value_lower(
                ec, self._field[ec], 0)
        else:
            mean_lower = self._mean_value_lower(
                ec, self._field[ec], self._glint_gradients[ec-1])

        if (ec == self._nelev-1):
            mean_upper = self._mean_value_upper(
                ec, self._field[ec], 0)
        else:
            mean_upper = self._mean_value_upper(
                ec, self._field[ec], self._glint_gradients[ec])

        return (mean_lower + mean_upper)/2

    def _mean_value_upper(self, ec, field, gradient):
        """Return the mean value for the upper side of an elevation class"""
        return field + self._mean_deviations[ec]*gradient

    def _mean_value_lower(self, ec, field, gradient):
        """Return the mean value for the lower side of an elevation class"""
        return field - self._mean_deviations[ec]*gradient


    def draw_figure(self, output_filename):
        """Draw a figure of this gradient info, and save it to
        output_filename.

        Each figure shows:

        - original SMB in each elevation class (red circles and dashed black lines)

        - this new mean-preserving interpolation (blue lines and blue circles)

        - glint-style interpolation (red lines)

        - mean CISM SMB in each elevation class with each approach (blue & red numbers) 
        """

        field_min = min(self._field)
        field_max = max(self._field)

        # Draw points
        plt.plot(self._topo, self._field, 'ro')
        plt.plot(self._topo, self._field_at_mean_topo, 'bo')

        # Draw dashed lines at the clm-computed values
        for ec in range(self._nelev):
            plt.plot([self._elevclass_bounds[ec], self._elevclass_bounds[ec+1]],
                     [self._field[ec], self._field[ec]], 'k--')

        # Limit upper bound of top elevation class
        upper_bound = min(self._elevclass_bounds[self._nelev],
                          self._topo[self._nelev-1] +
                          (self._topo[self._nelev-1] - self._elevclass_bounds[self._nelev-1]))

        # Draw lines between points
        label_glint = 'glint'
        label_mp = 'mean-preserving'
        for ec in range(self._nelev - 1):
            plt.plot([self._topo[ec], self._topo[ec+1]],
                     [self._field_at_mean_topo[ec],
                      self._field_at_mean_topo[ec+1]],
                     'b', label=label_mp)
            plt.plot([self._topo[ec], self._topo[ec+1]],
                     [self._field[ec], self._field[ec+1]],
                     'r', label=label_glint)
            # Make sure each label only appears once
            label_glint = None
            label_mp = None

        # Draw gradient in lower half of lowest EC
        plt.plot([self._elevclass_bounds[0], self._topo[0]],
                 [self._field[0], self._field[0]], 'r')
        plt.plot([self._elevclass_bounds[0], self._topo[0]],
                 [self._field_at_mean_topo[0], self._field_at_mean_topo[0]], 'b')

        # Draw gradient in upper half of highest EC
        plt.plot([self._topo[-1], upper_bound],
                 [self._field[-1], self._field[-1]], 'r')
        plt.plot([self._topo[-1], upper_bound],
                 [self._field_at_mean_topo[-1], self._field_at_mean_topo[-1]], 'b')

        # Limit x axis
        plt.xlim([self._elevclass_bounds[0], upper_bound])

        # Set y axes ourselves, rather than letting them be dynamic, for easier
        # comparison between figures
        y_range = field_max - field_min
        y_max = field_max + 0.2 * y_range
        y_min = field_min - 0.2 * y_range
        y_mean = (y_min+y_max)/2
        plt.ylim([y_min, y_max])

        # Plot elevation class bounds - vertical lines
        # (don't draw upper bound of last EC)
        for ec_bound in self._elevclass_bounds[:-1]:
            plt.plot([ec_bound, ec_bound], [y_min, y_max], 'k')

        # Write some statistics
        # For lower elevation classes, write them above the point; for upper
        # classes, below the point
        for ec in range(self._nelev):
            if (self._field[ec] < y_mean):
                yloc = self._field[ec] + y_range/5
            else:
                yloc = self._field[ec] - y_range/5
            num_format = "{:6.2f}"

            # I have done some spot-checks to confirm that this mean agrees with
            # self._field[ec]; if we implement this for real, we should add some
            # unit tests and/or inline tests to confirm that
            plt.text(self._elevclass_bounds[ec], yloc,
                     num_format.format(self.get_mean(ec)),
                     fontsize=10, color='b')

            plt.text(self._elevclass_bounds[ec], yloc-y_range/20,
                     num_format.format(self.get_glint_mean(ec)),
                     fontsize=10, color='r')

        pylab.legend(loc='best', fontsize='x-small')
        pylab.savefig(output_filename)
        plt.close()

        
