import os

import yaml

from perimysium.dataman import GaitLandmarks

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
        self.cycle_start = gait_landmarks.cycle_start
        self.cycle_end = gait_landmarks.cycle_end

        self.tasks = list()

    def add_task(self, cls, trial, *args, **kwargs):
        """Add a TrialTask for this cycle.
            
           TrialTasks for cycles can only be created
           by Trial objects. 
        """
        kwargs['cycle']=self
        task = cls(trial, *args, **kwargs)
        self.tasks.append(task)
        return task

class Trial(object):
    def __init__(self, condition, num, metadata=None,
            omit_trial_dir=False,
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
                self.expdata_path, 'ground_reaction.mot')

        self.tasks = list()
        # TODO self.add_task(vital_tasks.TaskGRFGaitLandmarks)

    def add_task(self, cls, *args, **kwargs):
        """Add a TrialTask for this trial.
        """
        task = cls(self, *args, **kwargs)
        self.tasks.append(task)
        return task

    def add_task_cycles(self, cls, *args, **kwargs):
        """Add a CycleTask for each cycle in this trial.
        """
        for cycle in self.cycles:
            cycle.add_task(cls, self, *args, **kwargs)

class OvergroundTrial(Trial):
    """Overground trials have just one cycle."""
    def __init__(self, condition, num, metadata=None,omit_trial_dir=False):
        super(OvergroundTrial, self).__init__(condition, num,
                metadata=metadata,omit_trial_dir=omit_trial_dir)
        self.type = 'overground'
                
class TreadmillTrial(Trial):
    """Treadmill trials may have multiple cycles. If a condition has only one
    trial and it's a treadmill trial, you can omit the unnecessary trial
    directory using `omit_trial_dir`.
    
    Example
    -------
    ```
    trial = condition.add_treadmill_trial(1, omit_trial_dir=True)
    ```

    The provided strike times are used to automatically create cycles.
    """
    def __init__(self, condition, num, 
            right_strikes=None, right_toeoffs=None,
            left_strikes=None, left_toeoffs=None,
            metadata=None,
            omit_trial_dir=False,
            ):
        super(TreadmillTrial, self).__init__(condition, num, metadata=metadata,
                omit_trial_dir=omit_trial_dir)
        self.type = 'treadmill'

        # Loop through provided times and create Cycles.
        # TODOs: this code may not be sufficiently generic to go here.
        #        error if no cycle times given
        if right_strikes:
            for icycle in range(len(right_strikes) - 1):
                cycle_num = icycle + 1
                start = right_strikes[icycle]
                end = right_strikes[icycle + 1]
                gait_landmarks = GaitLandmarks(
                        primary_leg='right',
                        cycle_start=start,
                        cycle_end=end,
                        left_strike=left_strikes[icycle],
                        left_toeoff=left_toeoffs[icycle],
                        right_strike=start,
                        right_toeoff=right_toeoffs[icycle],
                        )
                self._add_cycle(cycle_num, gait_landmarks)
        else:
            raise Exception('TreadmillTrial: please provide gait landmarks '
                            'for at least one gait cycles')

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
    def add_overground_trial(self, *args, **kwargs):
        """
        Examples
        --------
        
        ```
        new_trial = cond.add_overground_trial(1)
        ```
        """
        trial = OvergroundTrial(self, *args, **kwargs)
        assert not self.contains_trial(trial.num)
        self.trials.append(trial)
        return trial
    def add_treadmill_trial(self, *args, **kwargs):
        """
        Examples
        --------
        
        ```
        new_trial = cond.add_treadmill_trial(1, start_time, end_time)
        ```
        """
        if 'use_type' in kwargs:
            treadmill_trial_type = kwargs.pop('use_type')
        else:
            treadmill_trial_type = TreadmillTrial
        trial = treadmill_trial_type(self, *args, **kwargs)
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
        self.model_name = self.name
        self.scaled_model_fpath = os.path.join(self.results_exp_path, '%s.osim' % self.model_name)
        # Model used by RRA to create adjusted model. By default, this is set 
        # to the scaled model, but it can be set to a different model if the 
        # scaled model must be modifed (adding a backpack, etc.)
        self.model_to_adjust_fpath = self.scaled_model_fpath
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
    def __init__(self, name, generic_model_fpath):
        self.name = name
        self.source_generic_model_fpath = generic_model_fpath
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
