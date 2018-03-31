import os
import yaml

from numpy import loadtxt

import vital_tasks
import utilities 

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
    Passing no arguments creates a single trial object that begins and ends
    with the times specified in the mocap data.
    ```
    new_trial = cond.add_trial(1)

    ```

    Use the 'interval' dictionary argument to pick a specific start and/or end
    time. The beginning or end time in the mocap file will be chosen if the
    start or end time is left out of the argument, respectively.
    ```
    interval = dict()
    interval['start_time'] = 0.0
    interval['end_time'] = 0.5
    new_trial = cond.add_trial(1, interval=interval)

    ```

    Use the 'gait_events' dictionary argument to provide the events
    corresponding to individual gait cycles within a trial. A TrialTask will
    be created for each individual cycle. Either the 'right_strikes' or
    'left_strikes' entry is required to create gait cycles. If both are
    provided, then the primary leg must have one more heel strike than the
    opposite leg for consistency and automatic primary leg detection. If the
    desired gait cycles in the trial are not consecutive, then the
    'stride_times' entry is also required. The opposite foot heel strikes and
    toeoffs from both feet can be included as optional dictionary entries.

    ```
    gait_events = dict()
    gait_events['right_strikes'] = [24.575, 30.650, 32.683, 33.683]
    gait_events['right_toeoffs'] = [25.242, 31.317, 33.342]
    gait_events['left_toeoffs']  = [24.742, 30.808, 32.842]
    gait_events['stride_times'] = [1.015, 1.021, 1.096]
    new_trial = cond.add_trial(1, gait_events=gait_events,
                                  omit_trial_dir=True)

    ```
    """
    def __init__(self, condition, num, metadata=None,
            omit_trial_dir=False, primary_leg=None,
            model_to_adjust_fpath=None,
            interval=None, gait_events=None,
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
        self.primary_leg = primary_leg
        self.right_strikes = None
        self.right_toeoffs = None
        self.left_strikes = None
        self.left_toeoffs = None
        self.stride_times = None
        self.start_time = None
        self.end_time = None

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
        # Model used by RRA to create adjusted model. By default, this is set 
        # to the scaled model, but it can be set to a different model if the 
        # scaled model must be modifed (adding a backpack, etc.)
        self.model_to_adjust_fpath = (self.subject.scaled_model_fpath if 
            not model_to_adjust_fpath else model_to_adjust_fpath)
        self.tasks = list()

        # One type of time information input supported for a given trial
        if interval:
            self.start_time = interval.get('start_time')
            self.end_time = interval.get('end_time')

            if not self.start_time:
                self.start_time = self.get_mocap_start_time()
            if not self.end_time:
                self.end_time = self.get_mocap_end_time()

            self.heel_strikes = [self.start_time, self.end_time]

        elif gait_events:
            self.right_strikes = gait_events.get('right_strikes')
            self.right_toeoffs = gait_events.get('right_toeoffs')
            self.left_strikes = gait_events.get('left_strikes')
            self.left_toeoffs = gait_events.get('left_toeoffs')
            self.stride_times = gait_events.get('stride_times')

            # Determine gait cycle division and labeling
            if not self.right_strikes and not self.left_strikes:
                self.heel_strikes = [self.start_time, self.end_time]

            elif ((self.right_strikes and not self.left_strikes) or 
                 (len(self.right_strikes) == len(self.left_strikes)+1)):
                self.heel_strikes = self.right_strikes
                self.primary_leg = 'right'

            elif ((self.left_strikes and not self.right_strikes) or 
                 (len(self.left_strikes) == len(self.right_strikes)+1)):
                self.heel_strikes = self.left_strikes
                self.primary_leg='left'

            else:
                raise Exception("Invalid gait landmarks specified: ensure "
                    "specified heel strikes and toeoffs are consistent "
                    "with an integer number of gait cycles.")

        elif not interval and not gait_events:
            self.start_time = self.get_mocap_start_time()
            self.end_time = self.get_mocap_end_time()
            self.heel_strikes = [self.start_time, self.end_time]

        elif interval and gait_events:
            raise Exception("Please specify either simulation time "
                "interval or gait cycle events, but not both.")

        # Divide trial based on provided heel strikes and create individaul
        # cycle objects. This also supports cases where the notion of a cycle
        # may not exist (e.g. overground trials where only start and end times
        # given).
        for icycle in range(len(self.heel_strikes) - 1):
            start = self.heel_strikes[icycle]
            if self.stride_times:
                end = start + self.stride_times[icycle]
            else:
                end = self.heel_strikes[icycle + 1]
            gait_landmarks = utilities.GaitLandmarks(
                    cycle_start=start,
                    cycle_end=end,
                    )
            if self.primary_leg:
                gait_landmarks.primary_leg = self.primary_leg
            if self.left_strikes:
                gait_landmarks.left_strike = self.left_strikes[icycle]
            if self.left_toeoffs:
                gait_landmarks.left_toeoff = self.left_toeoffs[icycle]
            if self.right_strikes:
                gait_landmarks.right_strike = self.right_strikes[icycle]
            if self.right_toeoffs:
                gait_landmarks.right_toeoff = self.right_toeoffs[icycle]

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

            # Put setup task at front of argument list
            if setup_tasks:
                args = (setup_tasks[i],) + orig_args 

            task = cycle.add_task(cls, *args, **kwargs)

            if (("Inverse Kinematics" in task.doc) or 
                ("Inverse Dynamics" in task.doc)):
                raise Exception("TrialTask creation for individual cycles not "
                    " currently supported for the Inverse Kinematics and "
                    " Inverse Dynamics tools")

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
        self.cond_args = dict()
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
    def __init__(self, name, generic_model_fpath, reserve_actuators_fpath,
        rra_actuators_fpath=None, cmc_actuators_fpath=None):
        self.name = name
        self.source_generic_model_fpath = generic_model_fpath
        self.source_reserve_actuators_fpath = reserve_actuators_fpath
        self.source_rra_actuators_fpath = rra_actuators_fpath
        self.source_cmc_actuators_fpath = cmc_actuators_fpath
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.load(f)
        except Exception as e:
            raise Exception(e.message +
                    "\nMake sure there is a config.yaml next to dodo.py")
            
        if not 'results_path' in self.config:
            self.config['results_path'] = '../results'
        if not 'analysis_path' in self.config:
            self.config['analysis_path'] = '../analysis'

        # The copy in the results directory.
        self.generic_model_fpath = os.path.join(self.config['results_path'],
                'generic_model.osim')
                #os.path.basename(generic_model_fpath))
        self.reserve_actuators_fpath = os.path.join(
            self.config['results_path'], 'reserve_actuators.xml')
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
