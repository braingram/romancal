import pytest
import roman_datamodels as rdm

from romancal.stpipe import RomanStep

from .regtestdata import compare_asdf

# mark all tests in this module
pytestmark = [pytest.mark.bigdata]


@pytest.fixture(scope="module")
def run_image_flat(rtdata_module):
    rtdata = rtdata_module

    rtdata.get_data("WFI/image/r0000101001001001001_0001_wfi01_assignwcs.asdf")
    output = "r0000101001001001001_0001_wfi01_flat.asdf"
    rtdata.output = output
    rtdata.get_truth(f"truth/WFI/image/{output}")
    RomanStep.from_cmdline(["romancal.step.FlatFieldStep", rtdata.input])
    return rtdata


@pytest.fixture(scope="module")
def run_image_flat_new_time(rtdata_module):
    rtdata = rtdata_module

    rtdata.get_data(
        "WFI/image/r0000101001001001001_0001_wfi01_changetime_assignwcs.asdf"
    )
    output = "r0000101001001001001_0001_wfi01_changetime_flat.asdf"
    rtdata.output = output
    rtdata.get_truth(f"truth/WFI/image/{output}")
    RomanStep.from_cmdline(["romancal.step.FlatFieldStep", rtdata.input])
    return rtdata


@pytest.fixture(scope="module")
def run_grism_flat(rtdata_module):
    rtdata = rtdata_module

    rtdata.get_data("WFI/grism/r0000201001001001001_0001_wfi01_assignwcs.asdf")
    output = "r0000201001001001001_0001_wfi01_flat.asdf"
    rtdata.output = output
    rtdata.get_truth(f"truth/WFI/grism/{output}")
    RomanStep.from_cmdline(["romancal.step.FlatFieldStep", rtdata.input])
    return rtdata


@pytest.fixture(scope="module")
def image_output_model(run_image_flat):
    with rdm.open(run_image_flat.output) as model:
        yield model


@pytest.fixture(scope="module")
def grism_output_model(run_grism_flat):
    with rdm.open(run_grism_flat.output) as model:
        yield model


@pytest.fixture(scope="module")
def new_time_output_model(run_image_flat_new_time):
    with rdm.open(run_image_flat_new_time.output) as model:
        yield model


@pytest.mark.parametrize(
    "run_name", ["run_image_flat", "run_image_flat_new_time", "run_grism_flat"]
)
def test_output_matches_truth(run_name, ignore_asdf_paths, request):
    rtdata = request.getfixturevalue(run_name)
    diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    assert diff.identical, diff.report()


def test_image_ref_file(image_output_model):
    assert "roman_wfi_flat" in image_output_model.meta.ref_file.flat


def test_grism_ref_file(grism_output_model):
    assert "N/A" in grism_output_model.meta.ref_file.flat


def test_grism_skipped(grism_output_model):
    assert grism_output_model.meta.cal_step.flat_field == "SKIPPED"


@pytest.mark.soctests
def test_flat_field_crds_match_image_step(image_output_model, new_time_output_model):
    assert image_output_model.meta.ref_file.flat != new_time_output_model


# TODO check that ref file has useafter < exposure.start_time for image and new_time

# @pytest.mark.bigdata
# @pytest.mark.soctests
# def old_test_flat_field_crds_match_image_step(rtdata, ignore_asdf_paths):
#     """DMS79 Test: Testing that different datetimes pull different
#     flat files and successfully make level 2 output"""
#
#     # First file
#     input_l2_file = "r0000101001001001001_0001_wfi01_assignwcs.asdf"
#     rtdata.get_data(f"WFI/image/{input_l2_file}")
#     rtdata.input = input_l2_file
#
#     # Test CRDS
#     step = FlatFieldStep()
#     model = rdm.open(rtdata.input)
#     step.log.info(
#         "DMS79 MSG: Testing retrieval of best ref file, Success is flat file with"
#         " correct use after date"
#     )
#
#     step.log.info(f"DMS79 MSG: First data file: {rtdata.input.rsplit('/', 1)[1]}")
#     step.log.info(f"DMS79 MSG: Observation date: {model.meta.exposure.start_time}")
#
#     ref_file_path = step.get_reference_file(model, "flat")
#     step.log.info(
#         f"DMS79 MSG: CRDS matched flat file: {ref_file_path.rsplit('/', 1)[1]}"
#     )
#     flat = rdm.open(ref_file_path)
#     step.log.info(f"DMS79 MSG: flat file UseAfter date: {flat.meta.useafter}")
#     step.log.info(
#         "DMS79 MSG: UseAfter date before observation date? :"
#         f" {(flat.meta.useafter < model.meta.exposure.start_time)}"
#     )
#
#     # Test FlatFieldStep
#     output = "r0000101001001001001_0001_wfi01_flat.asdf"
#     rtdata.output = output
#     args = ["romancal.step.FlatFieldStep", rtdata.input]
#     step.log.info(
#         "DMS79 MSG: Running flat fielding step. The first ERROR is"
#         "expected, due to extra CRDS parameters not having been "
#         "implemented yet."
#     )
#     RomanStep.from_cmdline(args)
#     rtdata.get_truth(f"truth/WFI/image/{output}")
#
#     diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
#     step.log.info(
#         f"DMS79 MSG: Was proper flat fielded Level 2 data produced? : {diff.identical}"
#     )
#     assert diff.identical, diff.report()
#
#     # This test requires a second file, in order to meet the DMS79 requirement.
#     # The test will show that two files with different observation dates match
#     #  to separate flat files in CRDS.
#
#     # Second file
#     input_file = "r0000101001001001001_0001_wfi01_changetime_assignwcs.asdf"
#     rtdata.get_data(f"WFI/image/{input_file}")
#     rtdata.input = input_file
#
#     # Test CRDS
#     step = FlatFieldStep()
#     model = rdm.open(rtdata.input)
#
#     step.log.info(f"DMS79 MSG: Second data file: {rtdata.input.rsplit('/', 1)[1]}")
#     step.log.info(f"DMS79 MSG: Observation date: {model.meta.exposure.start_time}")
#
#     ref_file_path_b = step.get_reference_file(model, "flat")
#     step.log.info(
#         f"DMS79 MSG: CRDS matched flat file: {ref_file_path_b.rsplit('/', 1)[1]}"
#     )
#     flat = rdm.open(ref_file_path_b)
#     step.log.info(f"DMS79 MSG: flat file UseAfter date: {flat.meta.useafter}")
#     step.log.info(
#         "DMS79 MSG: UseAfter date before observation date? :"
#         f" {(flat.meta.useafter < model.meta.exposure.start_time)}"
#     )
#
#     # Test FlatFieldStep
#     output = "r0000101001001001001_0001_wfi01_changetime_flat.asdf"
#     rtdata.output = output
#     args = ["romancal.step.FlatFieldStep", rtdata.input]
#     step.log.info(
#         "DMS79 MSG: Running flat fielding step. The first ERROR is"
#         "expected, due to extra CRDS parameters not having been "
#         "implemented yet."
#     )
#     RomanStep.from_cmdline(args)
#     rtdata.get_truth(f"truth/WFI/image/{output}")
#     diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
#     step.log.info(
#         f"DMS79 MSG: Was proper flat fielded Level 2 data produced? : {diff.identical}"
#     )
#     assert diff.identical, diff.report()
#
#     # Test differing flat matches
#     step.log.info(
#         "DMS79 MSG REQUIRED TEST: Are the two data files "
#         "matched to different flat files? : "
#         f"{('/'.join(ref_file_path.rsplit('/', 3)[1:]))} != "
#         f"{('/'.join(ref_file_path_b.rsplit('/', 3)[1:]))}"
#     )
#     assert "/".join(ref_file_path.rsplit("/", 1)[1:]) != "/".join(
#         ref_file_path_b.rsplit("/", 1)[1:]
#     )
