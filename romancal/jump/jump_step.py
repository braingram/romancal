"""
Detect jumps in a science image
"""

import roman_datamodels as rdm
import numpy as np
import time
from ..stpipe import RomanStep
from .. roman_datamodels import dqflags
from stcal.jump.jump import detect_jumps

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


__all__ = ["JumpStep"]


class JumpStep(RomanStep):
    """
    JumpStep: Performs CR/jump detection. The 2-point difference method is
    applied.
    """

    spec = """
        rejection_threshold = float(default=4.0,min=0) # CR sigma rej thresh
        three_group_rejection_threshold = float(default=6.0,min=0) # CR sigma rej thresh
        four_group_rejection_threshold = float(default=5.0,min=0) # CR sigma rej thresh
        maximum_cores = option('none', 'quarter', 'half', 'all', default='none') # max number of processes to create
        flag_4_neighbors = boolean(default=True) # flag the four perpendicular neighbors of each CR
        max_jump_to_flag_neighbors = float(default=1000) # maximum jump sigma that will trigger neighbor flagging
        min_jump_to_flag_neighbors = float(default=10) # minimum jump sigma that will trigger neighbor flagging
    """

    reference_file_types = ['gain', 'readnoise']

    def process(self, input):

        # Open input as a Roman DataModel (single integration; 3D arrays)
        with rdm.RampModel(input) as input_model:

            # Extract the needed info from the Roman Data Model
            meta = input_model.meta
            r_data = input_model.data
            r_gdq = input_model.groupdq
            r_pdq = input_model.pixeldq
            r_err = input_model.err

            frames_per_group = meta.exposure.nframes

            # Modify the arrays for input into the 'common' jump (4D)
            data = np.broadcast_to(r_data, (1,) + r_data.shape)
            gdq = np.broadcast_to(r_gdq, (1,) + r_gdq.shape)
            pdq = np.broadcast_to(r_pdq, (1,) + r_pdq.shape)
            err = np.broadcast_to(r_err, (1,) + r_err.shape)

            tstart = time.time()

            # Check for an input model with NGROUPS<=2
            ngroups = data.shape[1]

            if ngroups <= 2:
                self.log.warning('Cannot apply jump detection as NGROUPS<=2;')
                self.log.warning('Jump step will be skipped')

                result = input_model.copy()

                result.meta.cal_step.jump = 'SKIPPED'
                return result

            # Retrieve the parameter values
            rej_thresh = self.rejection_threshold
            three_grp_rej_thresh = self.three_group_rejection_threshold
            four_grp_rej_thresh = self.four_group_rejection_threshold
            max_cores = self.maximum_cores
            max_jump_to_flag_neighbors = self.max_jump_to_flag_neighbors
            min_jump_to_flag_neighbors = self.min_jump_to_flag_neighbors
            flag_4_neighbors = self.flag_4_neighbors

            self.log.info('CR rejection threshold = %g sigma', rej_thresh)
            if self.maximum_cores != 'none':
                self.log.info('Maximum cores to use = %s', max_cores)

            # Get the gain and readnoise reference files
            gain_filename = self.get_reference_file(input_model, 'gain')
            self.log.info('Using GAIN reference file: %s', gain_filename)
            gain_model = rdm.GainModel(gain_filename)
            gain_2d = gain_model.data

            readnoise_filename = self.get_reference_file(input_model, 'readnoise')
            self.log.info('Using READNOISE reference file: %s',
                          readnoise_filename)
            readnoise_model = rdm.ReadnoiseModel(readnoise_filename)
            readnoise_2d = readnoise_model.data

            dqflags_d = {}  # Dict of DQ flags
            dqflags_d = {
                "GOOD": dqflags.group["GOOD"],
                "DO_NOT_USE": dqflags.group["DO_NOT_USE"],
                "SATURATED":  dqflags.group["SATURATED"],
                "JUMP_DET":  dqflags.group["JUMP_DET"]
            }

            gdq, pdq = detect_jumps(frames_per_group, data, gdq, pdq, err,
                                    gain_2d, readnoise_2d, rej_thresh,
                                    three_grp_rej_thresh, four_grp_rej_thresh,
                                    max_jump_to_flag_neighbors,
                                    min_jump_to_flag_neighbors,
                                    flag_4_neighbors,
                                    dqflags_d)

            gain_model.close()
            readnoise_model.close()
            tstop = time.time()
            self.log.info('The execution time in seconds: %f', tstop - tstart)

        result.meta.cal_step.jump = 'COMPLETE'

        return result