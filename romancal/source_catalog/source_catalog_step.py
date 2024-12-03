"""
Module for the source catalog step.
"""

import numpy as np
from astropy.table import Table
from roman_datamodels import datamodels, maker_utils
from roman_datamodels.datamodels import ImageModel, MosaicModel
from roman_datamodels.maker_utils import mk_datamodel

from romancal.source_catalog.background import RomanBackground
from romancal.source_catalog.detection import convolve_data, make_segmentation_image
from romancal.source_catalog.reference_data import ReferenceData
from romancal.source_catalog.source_catalog import RomanSourceCatalog
from romancal.stpipe import RomanStep

__all__ = ["SourceCatalogStep"]


class SourceCatalogStep(RomanStep):
    """
    Create a catalog of sources including photometry and basic shape
    measurements.

    Parameters
    -----------
    input : str, `ImageModel`, or `MosaicModel`
        Path to an ASDF file, or an `ImageModel` or `MosaicModel`.
    """

    class_alias = "source_catalog"
    reference_file_types = []

    spec = """
        bkg_boxsize = integer(default=1000)   # background mesh box size in pixels
        kernel_fwhm = float(default=2.0)      # Gaussian kernel FWHM in pixels
        snr_threshold = float(default=3.0)    # per-pixel SNR threshold above the bkg
        npixels = integer(default=25)         # min number of pixels in source
        deblend = boolean(default=False)      # deblend sources?
        aperture_ee1 = integer(default=30)    # aperture encircled energy 1
        aperture_ee2 = integer(default=50)    # aperture encircled energy 2
        aperture_ee3 = integer(default=70)    # aperture encircled energy 3
        ci1_star_threshold = float(default=2.0)  # CI 1 star threshold
        ci2_star_threshold = float(default=1.8)  # CI 2 star threshold
        suffix = string(default='cat')        # Default suffix for output files
        fit_psf = boolean(default=True)      # fit source PSFs for accurate astrometry?
    """

    def process(self, step_input):
        if isinstance(step_input, datamodels.DataModel):
            input_model = step_input
        else:
            input_model = datamodels.open(step_input)

        if not isinstance(input_model, (ImageModel, MosaicModel)):
            raise ValueError("The input model must be an ImageModel or MosaicModel.")

        # Copy the data and error arrays to avoid modifying the input
        # model. We use mk_datamodel to copy *only* the data and err
        # arrays. The metadata and dq and weight arrays are not copied
        # because they are not modified in this step. The other model
        # arrays (e.g., var_rnoise) are not currently used by this step.
        if isinstance(input_model, ImageModel):
            model = mk_datamodel(
                ImageModel,
                meta=input_model.meta,
                shape=(0, 0),
                data=input_model.data.copy(),
                err=input_model.err.copy(),
                dq=input_model.dq,
            )
        elif isinstance(input_model, MosaicModel):
            model = mk_datamodel(
                MosaicModel,
                meta=input_model.meta,
                shape=(0, 0),
                data=input_model.data.copy(),
                err=input_model.err.copy(),
                weight=input_model.weight,
            )

        if isinstance(model, ImageModel):
            cat_model = datamodels.ImageSourceCatalogModel
        else:
            cat_model = datamodels.MosaicSourceCatalogModel
        source_catalog_model = maker_utils.mk_datamodel(cat_model)

        for key in source_catalog_model.meta.keys():
            value = (
                model.meta.instrument[key]
                if key == "optical_element"
                else model.meta[key]
            )
            source_catalog_model.meta[key] = value
        aperture_ee = (self.aperture_ee1, self.aperture_ee2, self.aperture_ee3)
        refdata = ReferenceData(model, aperture_ee)
        aperture_params = refdata.aperture_params

        mask = np.isnan(model.data)
        coverage_mask = np.isnan(model.err)
        bkg = RomanBackground(
            model.data,
            box_size=self.bkg_boxsize,
            mask=mask,
            coverage_mask=coverage_mask,
        )
        model.data -= bkg.background

        convolved_data = convolve_data(
            model.data, kernel_fwhm=self.kernel_fwhm, mask=coverage_mask
        )

        segment_img = make_segmentation_image(
            convolved_data,
            snr_threshold=self.snr_threshold,
            npixels=self.npixels,
            bkg_rms=bkg.background_rms,
            deblend=self.deblend,
            mask=coverage_mask,
        )

        if segment_img is None:  # no sources found
            source_catalog_model.source_catalog = Table()
        else:
            ci_star_thresholds = (
                self.ci1_star_threshold,
                self.ci2_star_threshold,
            )
            catobj = RomanSourceCatalog(
                model,
                segment_img,
                convolved_data,
                aperture_params,
                ci_star_thresholds,
                self.kernel_fwhm,
                self.fit_psf,
            )

            # put the resulting catalog in the model
            source_catalog_model.source_catalog = catobj.catalog

        # always save the segmentation image and source catalog
        self.save_base_results(segment_img, source_catalog_model)

        # Return the source catalog object or the input model. If the
        # input model is an ImageModel, the metadata is updated with the
        # source catalog filename.
        if getattr(self, "return_updated_model", False):
            # define the catalog filename; self.save_model will
            # determine whether to use a fully qualified path
            output_catalog_name = self.make_output_path(
                basepath=model.meta.filename, suffix="cat"
            )

            # set the suffix to something else to prevent the step from
            # overwriting the source catalog file with a datamodel
            self.suffix = "sourcecatalog"

            if isinstance(input_model, ImageModel):
                update_metadata(input_model, output_catalog_name)

            result = input_model
        else:
            result = source_catalog_model

        return result

    def save_base_results(self, segment_img, source_catalog_model):
        # save the segmentation map and source catalog
        output_filename = (
            self.output_file
            if self.output_file is not None
            else source_catalog_model.meta.filename
        )

        if isinstance(source_catalog_model, datamodels.ImageSourceCatalogModel):
            seg_model = datamodels.SegmentationMapModel
        else:
            seg_model = datamodels.MosaicSegmentationMapModel

        segmentation_model = maker_utils.mk_datamodel(seg_model)
        for key in segmentation_model.meta.keys():
            segmentation_model.meta[key] = source_catalog_model.meta[key]

        if segment_img is not None:
            segmentation_model.data = segment_img.data.astype(np.uint32)
            self.save_model(
                segmentation_model,
                output_file=output_filename,
                suffix="segm",
                force=True,
            )

        # save the source catalog
        self.save_model(
            source_catalog_model,
            output_file=output_filename,
            suffix="cat",
            force=True,
        )


def update_metadata(model, output_catalog_name):
    # update datamodel to point to the source catalog file destination
    model.meta["source_catalog"] = maker_utils.mk_source_catalog(
        tweakreg_catalog_name=output_catalog_name
    )
    model.meta.cal_step.source_catalog = "COMPLETE"
