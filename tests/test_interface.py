# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================


import pytest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_equal
from simulai import sim


# ============================================================================
# TESTS
# ============================================================================


@patch.object(sim.BasePlant, 'process_simulation', return_v=bool)
def test_process_simulation(mock_method2):
    """Test that the connection() function returns a boolean type value.

    Use the mock of the simulation software.
    """
    sim.BasePlant.process_simulation()
    mock_method2.assert_called_with()