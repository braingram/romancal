.. _outlier-detection-imaging:

Outlier Detection Algorithm
===========================

This module serves as the interface for applying ``outlier_detection`` to direct
image observations. The code implements the basic outlier detection algorithm used
with JWST data, but adapted to Roman.

Specifically, this routine performs the following operations:

#. Extract parameter settings from input model and merge them with any user-provided
   values.

   * See :ref:`outlier detection arguments <outlier_detection_step_args>` for
     the full list of parameters.

#. By default, resample all input images.

   * The resampling step starts by computing an output WCS that is large enough
     to encompass all the input images.
   * All 18 detectors from the *same exposure* will get resampled onto this output
     WCS to create a mosaic of all the chips for that exposure.  This product
     is referred to as a "grouped mosaic" since it groups all the chips from
     the same exposure into a single image.
   * Each dither position will result in a separate grouped mosaic, so only
     a single exposure ever contributes to each pixel in these mosaics.
   * The ``fillval`` parameter specifies what value to use in the output
     resampled image for any pixel which has no valid contribution from any
     input exposure.
   * The resampling can be controlled with the ``pixfrac``, ``kernel`` and
     ``weight_type`` parameters.
   * The ``pixfrac`` indicates the fraction by
     which input pixels are "shrunk" before being drizzled onto the
     output image grid, given as a real number between 0 and 1. This specifies
     the size of the footprint, or "dropsize", of a pixel in units of the input
     pixel size.
   * The ``kernel`` specifies the form of the kernel function used to distribute flux onto
     the separate output images.
   * The ``weight_type`` indicates the type of weighting image to apply with the bad pixel mask.
     Available options are ``ivm`` (default) for computing and using an inverse-variance map
     and ``exptime`` for weighting by the exposure time.
   * The ``good_bits`` parameter specifies what DQ values from the input exposure
     should be used when resampling to create the output mosaic.  Any pixel with a
     DQ value not included in this value (or list of values) will be ignored when
     resampling.
   * Resampled images will be written out to disk with suffix `_outlier_coadd` by default.
   * **If resampling is turned off** through the use of the ``resample_data`` parameter,
     a copy of the unrectified input images (as a ModelLibrary)
     will be used for subsequent processing.

#. Create a median image from all grouped observation mosaics.

   * The median image is created by combining all grouped mosaic images or
     non-resampled input data pixel-by-pixel.
   * The ``maskpt`` parameter sets the percentage of the weight image values to
     use, and any pixel with a weight below this value gets flagged as "bad" and
     ignored when resampled.

#. By default, the median image is blotted back (inverse of resampling) to
   match each original input image.

   * **If resampling is turned off**, the median image is compared directly to
     each input image.

#. Perform statistical comparison between blotted image and original image to identify outliers.

   * This comparison uses the original input images, the blotted
     median image, and the derivative of the blotted image to
     create a cosmic ray mask for each input image.
   * The derivative of the blotted image gets created using the blotted
     median image to compute the absolute value of the difference between each pixel and
     its four surrounding neighbors with the largest value being the recorded derivative.
   * These derivative images are used to flag cosmic rays
     and other blemishes, such as moving object trails. Where the difference is larger
     than can be explained by noise statistics, the flattening effect of taking the
     median, or an error in the shift (the latter two effects are estimated using
     the image derivative), the suspect pixel is masked.
   * The ``backg`` parameter specifies a user-provided value to be used as the
     background estimate.  This gets added to the background-subtracted
     blotted image to attempt to match the original background levels of the
     original input mosaic so that cosmic-rays (bad pixels) from the input
     mosaic can be identified more easily as outliers compared to the blotted
     mosaic.
   * Cosmic rays are flagged using the following rule:

     .. math:: | image\_input - image\_blotted | > scale*image\_deriv + SNR*noise

   * The ``scale`` is defined as the multiplicative factor applied to the
     derivative which is used to determine if the difference between the data
     image and the blotted image is large enough to require masking.
   * The ``noise`` is calculated using a combination of the detector read
     noise and the poisson noise of the blotted median image plus the sky background.
   * The user must specify two cut-off signal-to-noise values using the
     ``snr`` parameter for determining whether a pixel should be masked:
     the first for detecting the primary cosmic ray, and the second for masking
     lower-level bad pixels adjacent to those found in the first pass. Since
     cosmic rays often extend across several pixels, the adjacent pixels make
     use of a slightly lower SNR threshold.

#. Update input data model DQ arrays with mask of detected outliers.

Memory Model for Outlier Detection Algorithm
---------------------------------------------
The outlier detection algorithm can end up using massive amounts of memory
depending on the number of inputs, the size of each input, and the size of the
final output product.  Specifically,

#. The input :py:class:`~romancal.datamodels.ModelLibrary` all input exposures would
   have been kept open in memory to make processing more efficient.

#. The initial resample step creates an output product for EACH input that is the
   same size as the final
   output product, which for imaging modes can span all chips in the detector while
   also accounting for all dithers.  For some Level 3 products, each resampled image can
   be on the order of 2Gb or more.

#. The median combination step then needs to have all pixels at the same position on
   the sky in memory in order to perform the median computation.  The simplest implementation
   for this step requires keeping all resampled outputs fully in memory at the same time.

Many Level 3 products only include a modest number of input exposures that can be
processed using less than 32Gb of memory at a time.  However, there are a number of
ways this memory limit can be exceeded.  This has been addressed by implementing an
overall memory model for the outlier detection that includes options to minimize the
memory usage at the expense of file I/O.  The control over this memory model happens
with the use of the ``in_memory`` parameter.  The full impact of this parameter
during processing includes:

#. The ``on_disk`` parameter gets set to `True`
   when opening the input :py:class:`~romancal.datamodels.library.ModelLibrary`
   object. This causes modified models to be written to temporary files.

#. Computing the median image uses temporary files. Each resampled group
   is split into sections (1 per "row") and each section is appended to a different
   temporary file. After resampling all groups, each temporary file is read and a
   median is computed for all sections in that file (yielding a median for that
   section across all resampled groups). Finally, these median sections are
   combined into a final median image.

These changes result in a minimum amount of memory usage during processing at the obvious
expense of reading and writing the products from disk.
