import os

import yaml

class Trial(object):
    def __init__(self, condition, num, metadata=None):
        self.condition = condition
        self.subject = condition.subject
        self.study = self.subject.study
        self.num = num
        self.name = 'trial%02i' % num
        self.rel_path = os.path.join(condition.rel_path, self.name)
        def list_condition_names():
            """Iterate through all conditions under which this trial sits."""
            cond_names = list()
            cur_cond = condition
            while cur_cond != None:
                cond_names.append(cur_cond.name)
                cur_cond = condition.parent_condition
            return cond_names
        self.id = '_'.join([self.subject.name] + list_condition_names() +
                [self.name])

class OvergroundTrial(Trial):
    """Overground trials have their own expdata folder within the trial
    folder."""
    def __init__(self, condition, num, metadata=None):
        super(OvergroundTrial, self).__init__(condition, num,
                metadata=metadata)
        self.marker_trajectories_fpath = os.path.join(
                self.study.config['results_path'], 'experiments',
                self.rel_path, 'expdata', 'marker_trajectories.trc')
                
class TreadmillTrial(Trial):
    """Treadmill trials have a single expdata folder for all trials, within the
    parent conditions folder."""
    def __init__(self, condition, num, start_time, end_time, metadata=None):
        super(TreadmillTrial, self).__init__(condition, num, metadata=metadata)
        self.marker_trajectories_fpath = os.path.join(
                self.study.config['results_path'], 'experiments',
                condition.rel_path, 'expdata', 'marker_trajectories.trc')

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
        trial = TreadmillTrial(self, *args, **kwargs)
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
        self.generic_model_fpath = generic_model_fpath
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.load(f)
        except Exception as e:
            raise Exception(e.message +
                    "\nMake sure there is a config.yaml next to dodo.py")

        self.subjects = list() 
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
