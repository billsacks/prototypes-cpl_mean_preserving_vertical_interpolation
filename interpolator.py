import numpy as np

class interpolator(object):
    def __init__(self, elevclass_bounds, topo, field, mean_deviations):
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
        assert len(mean_deviations) == self._nelev
        assert len(elevclass_bounds) == self._nelev + 1

        self._elevclass_bounds = np.array(elevclass_bounds)
        self._topo = np.array(topo)
        self._field = np.array(field)
        self._mean_deviations = np.array(mean_deviations)

        self._field_at_mean_topo = np.zeros(self._nelev)
        self._gradients = np.zeros(self._nelev - 1)

        self._compute_gradients()

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

    def __str__(self):
        field_at_mean_topo_str = "Values at mean topo: " + str(self._field_at_mean_topo)
        gradient_str = "Gradients: " + str(self._gradients)
        return field_at_mean_topo_str + "\n" + gradient_str


