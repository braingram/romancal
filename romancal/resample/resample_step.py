import logging
import os
from copy import deepcopy

import asdf
import numpy as np
from roman_datamodels import datamodels
from stcal.alignment import util

from ..datamodels import ModelLibrary
from ..stpipe import RomanStep
from . import resample

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = ["ResampleStep"]


class ResampleStep(RomanStep):
    """
    Resample input data onto a regular grid using the drizzle algorithm.

    .. note::
        When supplied via ``output_wcs``, a custom WCS overrides other custom
        WCS parameters such as ``output_shape`` (now computed from by
        ``output_wcs.bounding_box``), ``crpix``

    Parameters
    -----------
    input : str, `roman_datamodels.datamodels.DataModel`, or `~romancal.datamodels.ModelLibrary`
        If a string is provided, it should correspond to either a single ASDF filename
        or an association filename. Alternatively, a single DataModel instance can be
        provided instead of an ASDF filename. Multiple files can be processed via
        either an association file or wrapped by a
        `~romancal.datamodels.ModelLibrary`.

    Returns
    -------
    : `roman_datamodels.datamodels.MosaicModel`
        A mosaic datamodel with the final output frame.
    """  # noqa: E501

    class_alias = "resample"

    spec = """
        pixfrac = float(default=1.0)
        kernel = string(default='square')
        fillval = string(default='INDEF' )
        weight_type = option('ivm', 'exptime', None, default='ivm')
        output_shape = int_list(min=2, max=2, default=None)  # [x, y] order
        crpix = float_list(min=2, max=2, default=None)
        crval = float_list(min=2, max=2, default=None)
        rotation = float(default=None)
        pixel_scale_ratio = float(default=1.0) # Ratio of input to output pixel scale
        pixel_scale = float(default=None) # Absolute pixel scale in arcsec
        output_wcs = string(default='')  # Custom output WCS.
        good_bits = string(default='~DO_NOT_USE+NON_SCIENCE')  # The good bits to use for building the resampling mask.
    """  # noqa: E501

    reference_file_types = []

    def process(self, input):
        if isinstance(input, datamodels.DataModel):
            input_models = ModelLibrary([input])
            # set output filename from meta.filename found in the first datamodel
            output = input.meta.filename
        elif isinstance(input, str):
            # either a single asdf filename or an association filename
            try:
                # association filename
                input_models = ModelLibrary(input)
            except Exception:
                # single ASDF filename
                input_models = ModelLibrary([input])
            output = input_models.asn["products"][0]["name"]
        elif isinstance(input, ModelLibrary):
            input_models = input
            # set output filename using the common prefix of all datamodels
            output = f"{os.path.commonprefix([x['expname'] for x in input_models.asn['products'][0]['members']])}.asdf"
            if len(output) == 0:
                # set default filename if no common prefix can be determined
                output = "resample_output.asdf"
        else:
            raise TypeError(
                "Input must be an ASN filename, a ModelLibrary, "
                "a single ASDF filename, or a single Roman DataModel."
            )

        # Check that input models are 2D images
        with input_models:
            example_model = input_models.borrow(0)
            data_shape = example_model.data.shape
            input_models.shelve(example_model, 0, modify=False)
            if len(data_shape) != 2:
                # resample can only handle 2D images, not 3D cubes, etc
                raise RuntimeError(f"Input {input_models[0]} is not a 2D image.")

        # Issue a warning about the use of exptime weighting
        if self.weight_type == "exptime":
            self.log.warning("Use of EXPTIME weighting will result in incorrect")
            self.log.warning("propagated errors in the resampled product")

        # Custom output WCS parameters.
        self.output_shape = self._check_list_pars(
            self.output_shape, "output_shape", min_vals=[1, 1]
        )
        self.output_wcs = self._load_custom_wcs(self.output_wcs, self.output_shape)
        self.crpix = self._check_list_pars(self.crpix, "crpix")
        self.crval = self._check_list_pars(self.crval, "crval")

        # Call the resampling routine
        resamp = resample.ResampleData(
            input_models,
            output=output,
            pixfrac=self.pixfrac,
            kernel=self.kernel,
            fillval=self.fillval,
            weight_type=self.weight_type,
            good_bits=self.good_bits,
            pixel_scale_ratio=self.pixel_scale_ratio,
            pixel_scale=self.pixel_scale,
            output_wcs=self.output_wcs,
            output_shape=self.output_shape,
            crpix=self.crpix,
            crval=self.crval,
            rotation=self.rotation,
        )
        result = resamp.resample_many_to_one(input_models)

        with result:
            for i, model in enumerate(result):
                self._final_updates(model, input_models)
                result.shelve(model, i)
            if len(result) == 1:
                model = result.borrow(0)
                result.shelve(model, 0, modify=False)
                return model

        return result

    def _final_updates(self, model, input_models):
        model.meta.cal_step["resample"] = "COMPLETE"
        model.meta.wcsinfo.s_region = util.compute_s_region_imaging(
            model.meta.wcs, model.data.shape
        )

        # if pixel_scale exists, it will override pixel_scale_ratio.
        # calculate the actual value of pixel_scale_ratio based on pixel_scale
        # because source_catalog uses this value from the header.
        model.meta.resample.pixel_scale_ratio = (
            self.pixel_scale / np.sqrt(model.meta.photometry.pixelarea_arcsecsq)
            if self.pixel_scale
            else self.pixel_scale_ratio
        )
        model.meta.resample.pixfrac = self.pixfrac
        self.update_phot_keywords(model)
        model.meta.resample["good_bits"] = self.good_bits

    @staticmethod
    def _check_list_pars(vals, name, min_vals=None):
        """
        Check if a specific keyword parameter is properly formatted.

        Parameters
        ----------
        vals : list or tuple
            A list or tuple containing a pair of values currently assigned to the
            keyword parameter `name`. Both values must be either `None` or not `None`.
        name : str
            The name of the keyword parameter.
        min_vals : list or tuple, optional
            A list or tuple containing a pair of minimum values to be assigned
            to `name`, by default None.

        Returns
        -------
        None or list
            If either `vals` is set to `None` (or both of its elements), the
            returned result will be `None`. Otherwise, the returned result will be
            a list containing the current values assigned to `name`.

        Raises
        ------
        ValueError
            This error will be raised if any of the following conditions are met:
            - the number of elements of `vals` is not 2;
            - the currently assigned values of `vals` are smaller than the
            minimum value provided;
            - one element is `None` and the other is not `None`.
        """
        if vals is None:
            return None
        if len(vals) != 2:
            raise ValueError(f"List '{name}' must have exactly two elements.")
        n = sum(x is None for x in vals)
        if n == 2:
            return None
        elif n == 0:
            if min_vals and sum(x >= y for x, y in zip(vals, min_vals)) != 2:
                raise ValueError(
                    f"'{name}' values must be larger or equal to {list(min_vals)}"
                )
            return list(vals)
        else:
            raise ValueError(f"Both '{name}' values must be either None or not None.")

    @staticmethod
    def _load_custom_wcs(asdf_wcs_file, output_shape):
        if not asdf_wcs_file:
            return None

        with asdf.open(asdf_wcs_file) as af:
            wcs = deepcopy(af.tree["wcs"])

        if output_shape is not None:
            wcs.array_shape = output_shape[::-1]
        elif wcs.pixel_shape is not None:
            wcs.array_shape = wcs.pixel_shape[::-1]
        elif wcs.bounding_box is not None:
            wcs.array_shape = tuple(
                int(axs[1] - axs[0] + 0.5)
                for axs in wcs.bounding_box.bounding_box(order="C")
            )
        elif wcs.array_shape is None:
            raise ValueError(
                "Step argument 'output_shape' is required when custom WCS "
                "does not have neither of 'array_shape', 'pixel_shape', or "
                "'bounding_box' attributes set."
            )

        return wcs

    def update_phot_keywords(self, model):
        """Update pixel scale keywords"""
        if model.meta.photometry.pixelarea_steradians is not None:
            model.meta.photometry.pixelarea_steradians *= (
                model.meta.resample.pixel_scale_ratio**2
            )
        if model.meta.photometry.pixelarea_arcsecsq is not None:
            model.meta.photometry.pixelarea_arcsecsq *= (
                model.meta.resample.pixel_scale_ratio**2
            )
