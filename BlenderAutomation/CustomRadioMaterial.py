"""
CustomRadioMaterial class written by Sionna to help with the integration of our own radio materials in the future
Modified slightly by myself
"""

import sionna.rt
import mitsuba as mi
import drjit as dr
from typing import Tuple
from sionna.rt import RadioMaterialBase

class CustomRadioMaterial(RadioMaterialBase):

    # The __init__ method builds the radio material from:
    # - A unique `name` to identify the material instance in the scene
    # - The gain parameter `g`
    # - An optional `color` for displaying the material in the previewer and renderer
    # Providing these 3 parameters to __init__ is how an instance of this radio material
    # is built programmatically.
    #
    # When loading a scene from an XML file, Mitsuba provides to __init__
    # only an `mi.Properties` object containing all the properties of the material
    # read from the XML scene file. Therefore, when a `props` object is provided,
    # the other parameters are ignored and should not be given.
    def __init__(self,
                 name : str | None = None,
                 g : float | mi.Float | None = None,
                 color : Tuple[float, float, float] | None = None,
                 props : mi.Properties | None = None):

        # If `props` is `None`, then one is built from the
        # other parameters
        if props is None:
            props = mi.Properties("custom-radio-material")
            # Name of the radio material
            props.set_id(name)
            props["g"] = g
            if color is not None:
                props["color"] = mi.ScalarColor3f(color)

        # Read the gain from `props`
        g = 0.0
        if props.has_property("g"):
            g = props["g"]
            props.remove_property('g')
        self._g = mi.Float(g)

        # The other parameters (`name`, `color`) are given to the
        # base class to complete the initialization of the material
        super().__init__(props)

    def sample(
        self,
        ctx : mi.BSDFContext,
        si : mi.SurfaceInteraction3f,
        sample1 : mi.Float,
        sample2 : mi.Point2f,
        active : bool | mi.Bool = True
    ) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:

        # Read the incident direction of propagation in the local coordinate
        # system
        ki_prime = si.wi

        # Build the 3x3 change-of-basis matrix from the local basis to the world
        # basis.
        # `si.sh_frame` stores the three vectors that define the local interaction basis
        # in the world coordinate system.
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        # Direction of propagation of the reflected field in the local coordinate system
        kr_prime = mi.reflect(-ki_prime)

        # Compute the Jones matrix in the implicit world coordinate system
        # The `jones_matrix_to_world_implicit()` builds the Jones matrix with the
        # structure we need.
        sqrt_g = mi.Complex2f(dr.sqrt(self._g), 0.)
        jones_mat = sionna.rt.utils.jones_matrix_to_world_implicit(c1=sqrt_g,
                                                   c2=sqrt_g,
                                                   to_world=to_world,
                                                   k_in_local=ki_prime,
                                                   k_out_local=kr_prime)

        ## We now only need to prepare the outputs

        # Cast the Jones matrix to a `mi.Spectrum` to meet the requirements of
        # the BSDF interface of Mitsuba
        jones_mat = mi.Spectrum(jones_mat)

        # Instantiate and set the BSDFSample object
        bs = mi.BSDFSample3f()
        # Specifies the type of interaction that was computed
        bs.sampled_component = sionna.rt.constants.InteractionType.SPECULAR
        # Direction of the scattered wave in the world frame
        bs.wo = to_world@kr_prime
        # The next field of `bs` stores the probability that the sampled
        # interaction type and direction of scattering are sampled conditioned
        # on the given direction of incidence.
        # As only one event and direction of scattering are possible with this model,
        # this probability is set to 1.
        bs.pdf = mi.Float(1.)
        # Not used but required to be set
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.eta = 1.0

        return bs, jones_mat

    def eval(
        self,
        ctx : mi.BSDFContext,
        si : mi.SurfaceInteraction3f,
        wo : mi.Vector3f,
        active : bool | mi.Bool = True
    ) -> mi.Spectrum:

        # Read the incident direction of propagation in the local coordinate
        # system
        ki_prime = si.wi

        # Build the 3x3 change-of-basis matrix from the local basis to the world
        # basis.
        # `si.sh_frame` stores the three vectors that define the local interaction basis
        # in the world coordinate system.
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        # Direction of propagation of the reflected field in the local coordinate system
        kr_prime = mi.reflect(-ki_prime)

        # Compute the Jones matrix in the implicit world coordinate system
        # The `jones_matrix_to_world_implicit()` builds the Jones matrix with the
        # structure we need.
        sqrt_g = mi.Complex2f(dr.sqrt(self._g), 0.)
        jones_mat = sionna.rt.utils.jones_matrix_to_world_implicit(c1=sqrt_g,
                                                   c2=sqrt_g,
                                                   to_world=to_world,
                                                   k_in_local=ki_prime,
                                                   k_out_local=kr_prime)

        # This model only scatters energy in the direction of the specular reflection.
        # Any other direction provided by the user `wo` should therefore lead to no energy.
        is_valid = sionna.rt.utils.isclose(dr.dot(kr_prime, wo), mi.Float(1.))
        jones_mat = dr.select(is_valid, jones_mat, 0.)

        # Cast the Jones matrix to a `mi.Spectrum` to meet the requirements of
        # the BSDF interface of Mitsuba
        jones_mat = mi.Spectrum(jones_mat)

        return jones_mat

    def pdf(
        self,
        ctx : mi.BSDFContext,
        si : mi.SurfaceInteraction3f,
        wo : mi.Vector3f,
        active : bool | mi.Bool = True
    ) -> mi.Float:

        # Read the incident direction of propagation in the local coordinate
        # system
        ki_prime = si.wi

        # Direction of propagation of the reflected field in the local coordinate system
        kr_prime = mi.reflect(-ki_prime)

        # As only one event and direction of scattering are possible with this model,
        # the probability is set to 1 for this direction and 0 for any other.
        is_valid = sionna.rt.utils.isclose(dr.dot(kr_prime, wo), mi.Float(1.))
        return dr.select(is_valid, mi.Float(1.), mi.Float(0.))

    def traverse(self, callback : mi.TraversalCallback):
        # Registers the `g` parameter as a differentiable
        # parameter of the scene
        callback.put_parameter('g', self._g,
                               mi.ParamFlags.Differentiable)

    def to_string(self) -> str:
        # Returns a humanly readable description of the material
        s = f"CustomRadioMaterial["\
            f"g={self._g}"\
            f"]"
        return s

    # We add a getter and setter to access `g`
    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, v):
        self._g = mi.Float(v)