# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Plant simulation with autonomous decision system."""

# ============================================================================
# IMPORTS
# ============================================================================

from abc import ABCMeta, abstractmethod
import attr
import numpy as np
from interface import CommunicationInterface


# ============================================================================
# PLANTS
# ============================================================================


@attr.s
class Plant(metaclass=ABCMeta):
    """Metaclass to generate various simulated manufacturing plants.

    Parameters
    ----------
    method: str
        Name of the chosen AI method.
    """

    method = attr.ib()

    def __attrs_post_init__(self):
        """Attrs initialization."""
        self.method.register(self)

    def connection(self):
        """Connect function."""
        file_name = self.get_file_name_plant()
        self.connect = CommunicationInterface(file_name)
        return self.connect.connection()

    @abstractmethod
    def get_file_name_plant(self):
        """Name of the given plant file."""
        pass

    @abstractmethod
    def process_simulation(self):
        """Simulate in Tecnomatix."""
        pass

    @abstractmethod
    def update(self, data):
        """Update.

        Parameters
        ----------
        data: int
            Simulation data.
        """
        pass


@attr.s
class BasePlant(Plant):
    """A particularly adaptable plant.

    Parameters
    ----------
    method: str
        Name of the chosen AI method.
    v_i: list
        List of chosen input variables.
    v_o: list
        List of chosen output variables.
    filename: str
        Tecnomatix Plant Simulation complete file name (.spp)
    modelname: str
        Model frame name of the file, Default value="Model".
    """

    v_i = attr.ib()
    v_o = attr.ib()
    filename = attr.ib()
    modelname = attr.ib(default="Model")

    @v_i.validator
    def _validate_v_i(self, attribute, value):
        """Input value validator.

        Parameters
        ----------
        value: list
            User-selected value.
        """
        if not isinstance(value, list):
            raise TypeError("v_i: Argument must be a list.")

    @v_o.validator
    def _validate_v_o(self, attribute, value):
        """Output value validator.

        Parameters
        ----------
        value: list
            User-selected value.
        """
        if not isinstance(value, list):
            raise TypeError("v_o: Argument must be a list.")

    @filename.validator
    def _validate_filename(self, attribute, value):
        """File validator.

        Parameters
        ----------
        value: str
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("File Name: Argument must be a string.")

    @modelname.validator
    def _validate_modelname(self, attribute, value):
        """Model validator.

        Parameters
        ----------
        value: str
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Model Name: Argument must be a string.")

    def get_file_name_plant(self):
        """Get the name of the plant file.

        Return
        ------
        filename: str
            Name of the file.
        """
        return self.filename

    def update(self, data):
        """Update.

        Parameters
        ----------
        data: int
            Simulation data.

        Return
        -------
        r:float
            Reward value.
        """
        for idx, x in enumerate(self.v_i):
            self.connect.setvalue(x.path, data[idx])

        self.connect.startsimulation(".Models.{}".format(self.modelname))

        r = 0
        for idx, x in enumerate(self.v_o):
            a_idx = np.zeros(x.num_rows)
            for h in range(1, x.num_rows + 1):
                a_idx[h - 1] = self.connect.getvalue(
                    x.path + str([x.column, h])
                )
            b_idx = np.sum(a_idx)
            r += b_idx / len(self.v_o)

        self.connect.resetsimulation(".Models.{}".format(self.modelname))
        return r

    def process_simulation(self):
        """Process simulation."""
        if self.connection():
            self.connect.setvisible(True)
            self.method.process()


# ============================================================================
# METHODS
# ============================================================================

@attr.s
class AutonomousDecisionSystem(metaclass=ABCMeta):
    """Autonomous decision system class."""

    method = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Attrs initialization."""
        self.method = ""

    def register(self, who):
        """Subscribe registration.

        Parameters
        ----------
        who:str
            Node to subscribe.
        """
        self.subscriber = who

    @abstractmethod
    def process(self):
        """Process."""
        pass
