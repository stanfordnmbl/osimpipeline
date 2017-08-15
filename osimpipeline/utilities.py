
import os

class working_directory():
    """Use this to temporarily run code with some directory as a working
    directory and to then return to the original working directory::

        with working_directory('<dir>'):
            pass
    """
    def __init__(self, path):
        self.path = path
        self.original_working_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.path)
    def __exit__(self, *exc_info):
        os.chdir(self.original_working_dir)

class GaitLandmarks(object):
    def __init__(self,
            primary_leg=None,
            cycle_start=None,
            cycle_end=None,
            left_strike=None,
            left_toeoff=None,
            right_strike=None,
            right_toeoff=None):
        self.primary_leg  = primary_leg
        self.cycle_start  = cycle_start
        self.cycle_end    = cycle_end
        self.left_strike  = left_strike
        self.left_toeoff  = left_toeoff
        self.right_strike = right_strike
        self.right_toeoff = right_toeoff

    def cycle_duration(self):
        return self.cycle_end - self.cycle_start