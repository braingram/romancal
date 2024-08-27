import copy
import logging

import numpy as np
from astropy import units as u
from roman_datamodels import datamodels, maker_utils

from ..datamodels import ModelLibrary
from . import meta_blender, resample_utils, resampler

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = ["OutputTooLargeError", "ResampleData"]


class OutputTooLargeError(MemoryError):
    """Raised when the output is too large for in-memory instantiation"""


class ResampleData:
    """
    This is the controlling routine for the resampling process.

    Notes
    -----
    This routine performs the following operations:

      1. Extracts parameter settings from input model, such as pixfrac,
         weight type, exposure time (if relevant), and kernel, and merges
         them with any user-provided values.
      2. Creates output WCS based on input images and define mapping function
         between all input arrays and the output array. Alternatively, a custom,
         user-provided WCS object can be used instead.
      3. Updates output data model with output arrays from drizzle, including
         a record of metadata from all input models.
    """

    def __init__(
        self,
        input_models,
        output=None,
        pixfrac=1.0,
        kernel="square",
        fillval="INDEF",
        weight_type="ivm",
        good_bits="0",
        pixel_scale_ratio=1.0,
        pixel_scale=None,
        output_wcs=None,
        output_shape=None,
        crpix=None,
        crval=None,
        rotation=None,
    ):
        """
        Parameters
        ----------
        input_models : ~romancal.datamodels.ModelLibrary
            A `~romancal.datamodels.ModelLibrary` object containing the data
            to be processed.

        output : str
            filename for output

            .. note::
                ``output_shape`` is in the ``x, y`` order.
        """
        if (input_models is None) or (len(input_models) == 0):
            raise ValueError(
                "No input has been provided. Input must be a non-empty ModelLibrary"
            )

        input_models = input_models
        self.output_filename = output
        self.pixel_scale_ratio = pixel_scale_ratio
        self.pixfrac = pixfrac
        self.kernel = kernel
        self.fillval = fillval
        self.weight_type = weight_type
        self.good_bits = good_bits

        log.info(f"Driz parameter kernel: {self.kernel}")
        log.info(f"Driz parameter pixfrac: {self.pixfrac}")
        log.info(f"Driz parameter fillval: {self.fillval}")
        log.info(f"Driz parameter weight_type: {self.weight_type}")

        if pixel_scale is not None:
            log.info(f"Output pixel scale: {pixel_scale} arcsec.")
            pixel_scale /= 3600.0
        else:
            log.info(f"Output pixel scale ratio: {pixel_scale_ratio}")

        # build the output WCS object
        if output_wcs:
            # use the provided WCS object
            self.output_wcs = output_wcs
            if output_shape is not None:
                self.output_wcs.array_shape = output_shape[::-1]
        else:
            with input_models:
                models = list(input_models)
                # determine output WCS based on all inputs, including a reference WCS
                self.output_wcs = resample_utils.make_output_wcs(
                    models,
                    pscale_ratio=self.pixel_scale_ratio,
                    pscale=pixel_scale,
                    rotation=rotation,
                    shape=None if output_shape is None else output_shape[::-1],
                    crpix=crpix,
                    crval=crval,
                )
                for i, m in enumerate(models):
                    input_models.shelve(m, i, modify=False)

        log.debug(f"Output mosaic size: {self.output_wcs.array_shape}")

    def resample_many_to_one(self, input_models):
        """Resample and coadd many inputs to a single output.
        Used for level 3 resampling
        """
        # this requires:
        # -- computed --
        # - self.blank_output (FIXME to remove this)
        #
        # -- from args or computed --
        # - self.output_wcs
        #
        # -- from args --
        # - self.output_filename
        # - self.input_models
        # - self.pixfrac
        # - self.kernel
        # - self.fillval
        # - self.weight_type
        # - self.good_bits
        #
        # - self.resample_variance_array (function call)
        #   - self.drizzle_arrays (function call)
        #   - (other stuff from above)
        # - self.resample_exposure_time (function call)
        #   - (same as resample_variance_array)
        # - self.update_exposure_times (function call)
        # don't make a context array as the dtype does not match and
        # it won't be used below
        output_model = maker_utils.mk_datamodel(
            datamodels.MosaicModel,
            shape=tuple(self.output_wcs.array_shape),
            context=np.zeros((0, 0, 0), dtype=np.uint32),
        )
        output_model.meta.wcs = copy.deepcopy(self.output_wcs)
        blender = meta_blender.MetaBlender(output_model)

        output_model.meta.filename = self.output_filename

        # copy over asn information
        if (asn_pool := input_models.asn.get("asn_pool", None)) is not None:
            output_model.meta.asn.pool_name = asn_pool
        if (asn_table_name := input_models.asn.get("table_name", None)) is not None:
            output_model.meta.asn.table_name = asn_table_name

        # pre-allocate the context array
        nplanes = int(np.ceil(len(input_models) / 32))
        data_shape = output_model.data.shape
        outcon = np.zeros((nplanes, data_shape[0], data_shape[1]), dtype=np.int32)

        resamplers = {}

        # Initialize the output with the wcs
        resamplers["data"] = resampler.Resampler(
            output_model.data.value,
            output_model.weight,
            outcon,
            self.output_wcs,
            pixfrac=self.pixfrac,
            kernel=self.kernel,
            fillval=self.fillval,
        )

        # Initialize the variance arrays
        # We can just use a single outwht for
        # all the non-data arrays (as it's not used)
        out_var_wht = np.zeros_like(output_model.var_rnoise, dtype="f4")

        var_sums = {}
        for var_name in ("var_rnoise", "var_poisson", "var_flat"):
            # TODO use a single consistent data_shape?
            var_sums[var_name] = np.full(
                getattr(output_model, var_name).shape, np.nan, dtype="f4"
            )
            getattr(output_model, var_name).value[:] = np.nan
            resamplers[var_name] = resampler.Resampler(
                getattr(output_model, var_name).value,
                out_var_wht,
                None,
                self.output_wcs,
                pixfrac=self.pixfrac,
                kernel=self.kernel,
                fillval=np.nan,
            )

        # Initialize the exposure time result
        exptime_tot = np.zeros(output_model.data.shape, dtype="f4")
        resamplers["exptime"] = resampler.Resampler(
            np.zeros(output_model.data.shape, dtype="f4"),
            out_var_wht,
            None,
            self.output_wcs,
            pixfrac=1.0,
            kernel=self.kernel,
            fillval=0,
        )

        log.info("Resampling science data")
        with input_models:
            for i, img in enumerate(input_models):
                data_wht = resample_utils.build_driz_weight(
                    img,
                    weight_type=self.weight_type,
                    good_bits=self.good_bits,
                )
                if (
                    hasattr(img.meta, "background")
                    and img.meta.background.subtracted is False
                    and img.meta.background.level is not None
                ):
                    data = img.data - img.meta.background.level
                else:
                    data = img.data

                # compute pixmap to use for all operations...
                pixmap = resample_utils.calc_gwcs_pixmap(
                    img.meta.wcs, self.output_wcs, data.shape
                )

                resamplers["data"].add_image(
                    data.value,
                    img.meta.wcs,
                    data_wht,
                    pixmap=pixmap,
                )
                del data, data_wht

                var_wht = resample_utils.build_driz_weight(
                    img,
                    weight_type=None,
                    good_bits=self.good_bits,
                )

                # I think we can use resamplers here as long as we
                # reset the outsci array after each use (to the fillval)

                # dispatch other arrays to resample: (all with weight_type=None)
                for var_name in ("var_rnoise", "var_poisson", "var_flat"):
                    var_resampler = resamplers[var_name]
                    var_resampler.add_image(
                        getattr(img, var_name).value,
                        img.meta.wcs,
                        var_wht,
                        pixmap=pixmap,
                    )
                    mask = var_resampler.outsci > 0
                    var_sum = var_sums[var_name]
                    # TODO improve this to avoid the extra copies
                    var_sum[mask] = np.nansum(
                        [
                            var_sum[mask],
                            np.reciprocal(var_resampler.outsci[mask]),
                        ],
                        axis=0,
                    )
                    # np.add(var_sum, resampler.outsci, out=var_sum, where=resampler.outsci > 0)

                    # reset resampler for next image
                    var_resampler.outsci[:] = np.nan
                    del var_resampler

                # exposure time
                # TODO pre-allocate...
                exp_data = np.full(
                    img.data.shape,
                    img.meta.exposure.effective_exposure_time,
                    dtype="f4",
                )
                resamplers["exptime"].add_image(
                    exp_data,
                    img.meta.wcs,
                    var_wht,
                    pixmap=pixmap,
                )
                exptime_tot += resamplers["exptime"].outsci
                resamplers["exptime"].outsci[:] = 0
                del exp_data, var_wht

                blender.blend(img)

                input_models.shelve(img, i, modify=False)
                del img

        # TODO: fix RAD to expect a context image datatype of int32
        # TODO do I need to copy back data and weight
        output_model.context = resamplers["data"].outcon.astype(
            np.uint32
        )  # FIXME this cast copies
        del resamplers

        # record the actual filenames (the expname from the association)
        # for each file used to generate the output_model
        output_model.meta.resample["members"] = [
            m["expname"] for m in input_models.asn["products"][0]["members"]
        ]

        # finalize variance calculations
        for var_name in ("var_rnoise", "var_poisson", "var_flat"):
            getattr(output_model, var_name).value[:] = np.reciprocal(
                var_sums[var_name], out=var_sums[var_name]
            )
        del var_sums

        # TODO: fix unit here
        # TODO avoid the extra copies
        output_model.err = u.Quantity(
            np.sqrt(
                np.nansum(
                    [
                        output_model.var_rnoise,
                        output_model.var_poisson,
                        output_model.var_flat,
                    ],
                    axis=0,
                )
            ),
            unit=output_model.err.unit,
        )

        self.update_exposure_times(input_models, output_model, exptime_tot)
        del exptime_tot

        output_model.meta.resample.weight_type = self.weight_type
        output_model.meta.resample.pointings = len(input_models.group_names)
        blender.finalize()

        return ModelLibrary([output_model])

    def update_exposure_times(self, input_models, output_model, exptime_tot):
        """Update exposure time metadata (in-place)."""
        m = exptime_tot > 0
        total_exposure_time = np.mean(exptime_tot[m]) if np.any(m) else 0
        max_exposure_time = np.max(exptime_tot)
        log.info(
            f"Mean, max exposure times: {total_exposure_time:.1f}, "
            f"{max_exposure_time:.1f}"
        )
        exposure_times = {"start": [], "end": []}
        with input_models:
            for group_id, indices in input_models.group_indices.items():
                index = indices[0]
                model = input_models.borrow(index)
                exposure_times["start"].append(model.meta.exposure.start_time)
                exposure_times["end"].append(model.meta.exposure.end_time)
                input_models.shelve(model, index, modify=False)

        # Update some basic exposure time values based on output_model
        output_model.meta.basic.mean_exposure_time = total_exposure_time
        output_model.meta.basic.time_first_mjd = min(exposure_times["start"]).mjd
        output_model.meta.basic.time_last_mjd = max(exposure_times["end"]).mjd
        output_model.meta.basic.max_exposure_time = max_exposure_time
        output_model.meta.resample.product_exposure_time = max_exposure_time
