import yaml

class Trial(object):
    def __init__(self, condition, num, metadata=None):
        self.condition = condition
        self.subject = condition.subject
        self.num = num
        self.name = 'trial%02i' % num
        def list_condition_names():
            cond_names = list()
            cur_cond = condition
            while cur_cond != None:
                cond_names.append(cur_cond.name)
                cur_cond = condition.parent_condition
            return cond_names
        self.id = '_'.join([self.subject.name] + list_condition_names() +
                [self.name])

class TreadmillTrial(Trial):
    def __init__(self, condition, num, start_time, end_time, metadata=None):
        super(TreadmillTrial, self).__init__(condition, num, metadata=metadata)
        # TODO

class Condition(object):
    """There can be multiple tiers of conditions."""
    def __init__(self, subject, parent_condition, name, metadata=None):
        """Users do not call this constructor; instead, use
        `Subject.add_condition()` or `Condition.add_condition()`. The first two
        arguments are provided internally by `add_condition()`.
        """
        self.subject = subject
        self.parent_condition = parent_condition
        self.name = name
        self.metadata = metadata
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
    def add_treadmill_trial(self, *args, **kwargs):
        """Example: `new_trial = cond.add_trial(1, start_time, end_time)` """
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
        self.tasks.append(cls(self, *args, **kwargs))

class Study(object):
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
