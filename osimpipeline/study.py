import os
import yaml

from perimysium.dataman import GaitLandmarks
from numpy import loadtxt

import vital_tasks

class Cycle(object):
    """A subject may walk for multiple gait cycles in a given trial,
    particularly on a treadmill."""
    def __init__(self, trial, num, gait_landmarks, metadata=None):
        self.trial = trial
        self.condition = trial.condition
        self.subject = trial.subject
        self.study = trial.study
        self.num = num
        self.name = 'cycle%02i' % num
        self.metadata = metadata
        self.id = '_'.join([trial.id, self.name])
        self.gl = gait_landmarks
        self.start = gait_landmarks.cycle_start
        self.end = gait_landmarks.cycle_end

        self.tasks = list()

    def add_task(self, cls, *args, **kwargs):
        """Add a TrialTask for this cycle.
            
           TrialTasks for cycles can only be created
           by Trial objects. 
        """
        if 'cycle' not in kwargs:
            kwargs['cycle'] = self

        task = cls(self.trial, *args, **kwargs)
        self.tasks.append(task)
        return task

class Trial(object):
    """Trials may have multiple cycles. If a condition has only one
    trial, you can omit the unnecessary trial
    directory using `omit_trial_dir`.

    Examples
    --------

    ```
    new_trial = cond.add_trial(1)

    ```

    ```
    new_trial = cond.add_trial(1, start_time=0.0, end_time=0.5)

    ```

    ```
    new_trial = cond.add_trial(1, right_strikes=[45.900, 47.000, 48.133]
                                  right_toeoffs=[46.642, 47.758],  
                                  left_strikes=[46.450, 47.567],
                                  left_toeoffs=[46.075, 47.192],
                                  omit_trial_dir=True)
    ```
    """
    def __init__(self, condition, num, metadata=None,
            omit_trial_dir=False, primary_leg=False,
            model_to_adjust_fpath=None,
            start_time=None, end_time=None,
            right_strikes=None, right_toeoffs=None,
            left_strikes=None, left_toeoffs=None,
            ):
        self.condition = condition
        self.subject = condition.subject
        self.study = self.subject.study
        self.num = num
        self.name = 'trial%02i' % num 
        self.metadata = metadata
        self.cycles = list()
        self.rel_path = (condition.rel_path if omit_trial_dir else
                os.path.join(condition.rel_path, self.name))
        self.results_exp_path = os.path.join(self.study.config['results_path'],
                'experiments', self.rel_path)

        def list_condition_names():
            """Iterate through all conditions under which this trial sits."""
            cond_names = list()
            cur_cond = condition
            while cur_cond != None:
                cond_names.append(cur_cond.name)
                cur_cond = condition.parent_condition
            return cond_names
        self.id = '_'.join([self.subject.name] + list_condition_names() +
                [] if omit_trial_dir else [self.name])

        self.expdata_path = os.path.join(
                self.study.config['results_path'], 'experiments',
                self.rel_path, 'expdata')
        self.marker_trajectories_fpath = os.path.join(
                self.expdata_path, 'marker_trajectories.trc')
        self.ground_reaction_fpath = os.path.join(
                self.expdata_path, 'ground_reaction_orig.mot')
        # Model used by RRA to create adjusted model. By default, this is set 
        # to the scaled model, but it can be set to a different model if the 
        # scaled model must be modifed (adding a backpack, etc.)
        self.model_to_adjust_fpath = (self.subject.scaled_model_fpath if 
            not model_to_adjust_fpath else model_to_adjust_fpath)
        self.tasks = list()

        # One type of time information input supported for a given trial
        if (start_time or end_time) and (right_strikes or left_strikes):
            raise Exception("Please specify either simulation time window "
                "information (start_time, end_time) or gait cycle events "
                "(right_strikes, etc.) both specified, but not both.")
 
        # Determine gait cycle division and labeling
        if not right_strikes and not left_strikes:
            if not start_time:
                start_time = self.get_mocap_start_time()
            if not end_time:
                end_time = self.get_mocap_end_time()

            heel_strikes = [start_time, end_time]

        elif ((right_strikes and not left_strikes) or 
             (len(right_strikes) == len(left_strikes)+1)):
            heel_strikes=right_strikes
            primary_leg='right'

        elif ((left_strikes and not right_strikes) or 
             (len(left_strikes) == len(right_strikes)+1)):
            heel_strikes=left_strikes
            primary_leg='left'

        else:
            raise Exception("Invalid gait landmarks specified: ensure "
                "specified heel strikes and toeoffs are consistent "
                "with an integer number of gait cycles.")

        # Divide trial based on provided heel strikes and create individaul
        # cycle objects. This also supports cases where the notion of a cycle
        # may not exist (e.g. overground trials where only start and end times
        # given).
        for icycle in range(len(heel_strikes) - 1):
            start = heel_strikes[icycle]
            end = heel_strikes[icycle + 1]
            gait_landmarks = GaitLandmarks(
                    cycle_start=start,
                    cycle_end=end,
                    )
            if primary_leg:
                gait_landmarks.primary_leg = primary_leg
            if left_strikes:
                gait_landmarks.left_strike = left_strikes[icycle]
            if left_toeoffs:
                gait_landmarks.left_toeoff = left_toeoffs[icycle]
            if right_strikes:
                gait_landmarks.right_strike = right_strikes[icycle]
            if right_toeoffs:
                gait_landmarks.right_toeoff = right_toeoffs[icycle]

            self._add_cycle(icycle+1, gait_landmarks)

    def get_mocap_start_time(self):
        mocap_data = loadtxt(self.marker_trajectories_fpath, skiprows=6)
        # First row, second column.
        return mocap_data[0][1]

    def get_mocap_end_time(self):
        mocap_data = loadtxt(self.marker_trajectories_fpath, skiprows=6)
        # Last row, second column.
        return mocap_data[-1][1]

    def _add_cycle(self, *args, **kwargs):
        cycle = Cycle(self, *args, **kwargs)
        assert not self.contains_cycle(cycle.num)
        self.cycles.append(cycle)
        return cycle

    def get_cycle(self, num):
        for cycle in self.cycles:
            if cycle.num == num:
                return cycle
        return None

    def contains_cycle(self, num):
        return (self.get_cycle(num) != None)

    def add_task(self, cls, *args, **kwargs):
        """Add a TrialTask for this trial.
        """
        task = cls(self, *args, **kwargs)
        self.tasks.append(task)
        return task

    def add_task_cycles(self, cls, *args, **kwargs):
        """Add a TrialTask for each cycle in this trial.
        """
        orig_args = args
        setup_tasks = None
        if 'setup_tasks' in kwargs:
            setup_tasks = kwargs['setup_tasks']
            kwargs.pop('setup_tasks', None)

        tasks = list()
        for i, cycle in enumerate(self.cycles):

            if setup_tasks:
                args = orig_args + (setup_tasks[i],)

            task = cycle.add_task(cls, *args, **kwargs)
            tasks.append(task)

        return tasks

class Condition(object):
    """There can be multiple tiers of conditions; conditions can be nested
    within conditions."""
    def __init__(self, subject, parent_condition, name, metadata=None):
        """Users do not call this constructor; instead, use
        `Subject.add_condition()` or `Condition.add_condition()`. The first two
        arguments are provided internally by `add_condition()`.
        """
        self.subject = subject
        self.study = subject.study
        self.parent_condition = parent_condition
        # If this condition is within another condition, then that condition is
        # the parent to this object. Otherwise, the subject is the parent.
        self.parent = parent_condition if parent_condition != None else subject
        self.name = name
        self.metadata = metadata
        self.rel_path = os.path.join(self.parent.rel_path, name)
        self.results_exp_path = os.path.join(self.study.config['results_path'],
            'experiments', self.rel_path)
        self.conditions = list()
        self.trials = list()
    def add_condition(self, *args, **kwargs):
        """Example: `new_cond = cond.add_condition('walk0')`"""
        cond = Condition(self.subject, self, *args, **kwargs)
        assert not self.contains_condition(cond.name)
        self.conditions.append(cond)
        return cond
    def get_condition(self, name):
        for cond in self.conditions:
            if cond.name == name:
                return cond
        return None
    def contains_condition(self, name):
        return (self.get_condition(name) != None)

    def add_trial(self, *args, **kwargs):
        trial = Trial(self, *args, **kwargs)
        assert not self.contains_trial(trial.num)
        self.trials.append(trial)
        return trial
       
    def get_trial(self, num):
        for t in self.trials:
            if t.num == num:
                return t
        return None
    def contains_trial(self, num):
        return (self.get_trial(num) != None)
        
class Subject(object):
    def __init__(self, study, num, mass, metadata=None):
        self.study = study
        self.num = num
        self.name = 'subject%02i' % num
        self.mass = mass
        self.metadata = metadata
        # Relative path to the subject folder; can be used for the source
        # directory or the results directory.
        self.rel_path = self.name
        self.results_exp_path = os.path.join(self.study.config['results_path'],
            'experiments', self.rel_path)
        self.scaled_model_fpath = os.path.join(self.results_exp_path, 
            '%s.osim' % self.name)
        self.residual_actuators_fpath = os.path.join(self.results_exp_path, 
            '%s_residual_actuators.xml' % self.name)
        self.conditions = list()
        self.tasks = list()
    def add_condition(self, *args, **kwargs):
        """Example: `cond.add_condition('loaded')`"""
        cond = Condition(self, None, *args, **kwargs)
        assert not self.contains_condition(cond.name)
        self.conditions.append(cond)
        return cond
    def get_condition(self, name):
        for cond in self.conditions:
            if cond.name == name:
                return cond
        return None
    def contains_condition(self, name):
        return (self.get_condition(name) != None)
    def add_task(self, cls, *args, **kwargs):
        """Add a SubjectTask for this subject.
        Example: `subject.add_task(TaskCopyMotionCaptureData, fcn)`
        """
        task = cls(self, *args, **kwargs)
        self.tasks.append(task)
        return task

class Study(object):
    """
    
    Configuration file
    ------------------
    We expect that the current directory contains a `config.yaml` file with
    the following fields:
    
      - motion_capture_data_path
      - results_path

    An example `config.yaml` may look like the following:

    ```
    motion_capture_data_path: /home/fred/data
    results_path: /home/fred/results
    ```

    The paths do not need to be absolute; they could also be relative. It is
    expected that the `config.yaml` file is not committed to any source code
    repository, since different users might choose different values for these
    settings.
    """
    def __init__(self, name, generic_model_fpath, rra_actuators_fpath=None,
        cmc_actuators_fpath=None):
        self.name = name
        self.source_generic_model_fpath = generic_model_fpath
        self.source_rra_actuators_fpath = rra_actuators_fpath
        self.source_cmc_actuators_fpath = cmc_actuators_fpath
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.load(f)
        except Exception as e:
            raise Exception(e.message +
                    "\nMake sure there is a config.yaml next to dodo.py")
        # The copy in the results directory.
        self.generic_model_fpath = os.path.join(self.config['results_path'],
                'generic_model.osim')
                #os.path.basename(generic_model_fpath))
        self.rra_actuators_fpath = os.path.join(self.config['results_path'],
                'rra_actuators.xml')
        self.cmc_actuators_fpath = os.path.join(self.config['results_path'],
                'cmc_actuators.xml')

        self.subjects = list() 
        self.tasks = list()

        #self.add_task(vital_tasks.TaskCopyGenericModelToResults)

    def add_subject(self, *args, **kwargs):
        subj = Subject(self, *args, **kwargs)
        assert not self.contains_subject(subj.num)
        self.subjects.append(subj)
        return subj
    def get_subject(self, num):
        for subj in self.subjects:
            if subj.num == num:
                return subj
        return None
    def contains_subject(self, num):
        return (self.get_subject(num) != None)

    def add_task(self, cls, *args, **kwargs):
        """Add a StudyTask for the study.
        """
        task = cls(self, *args, **kwargs)
        self.tasks.append(task)
        return task
