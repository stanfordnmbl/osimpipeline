"""
These abstract base classes define a 'task' for python-doit. 
"""

import os

from doit.action import CmdAction

class Task(object):
    """The derived class must set `self.name` somewhere within its constructor.
    The derived class must also have its own class variable `REGISTRY = []`.
    """
    REGISTRY = []
    def __init__(self):
        self.uptodate = []
        self.file_dep = []
        self.actions = []
        self.targets = []
        self.task_dep = []
        self.doc = None 
        self.title = None
        #self.clean = []
        # Add this specific instance to the class variable REGISTRY.
        self.REGISTRY.append(self)

    def add_action(self, file_dep, target, member_function):
        """A convenient way to add an action to your task: file dependencies,
        targets, and the action's member function are grouped together. Call
        this within a derived class. The `file_dep` and `target` arguments are
        passed down to the `member_function`. So the signature of
        `member_function` should look like the following:

        ```
        def member_function(self, file_dep, target):
        ```
        
        Make sure the `target` option is not named `targets`; python-doit tries
        to be smart about actions with a `targets` parameter, and overrides the
        behavior we want here.
        

        The arguments `file_dep` and `target` should be lists or dicts.

        """
        if type(file_dep) == list:
            self.file_dep += file_dep
        else:
            self.file_dep += file_dep.values()
        if type(target) == list:
            self.targets += target
        else:
            self.targets += target.values()
        self.actions.append((member_function, [file_dep, target]))

    def copy_file(self, file_dep, target):
        import shutil
        to_dir = os.path.basename(target[0])
        if not os.path.exists(to_dir): os.makedirs(to_dir)
        shutil.copyfile(file_dep[0], target[0])

    @classmethod
    def create_doit_tasks(cls):
        """Create a specific task for each registered instance of this
        class.
        """
        # Python-doit invokes this function for any object in the `doit`
        # namespace that has it; this is how python-doit registers the tasks.
        for instance in cls.REGISTRY:
            yield {'basename': instance.name,
                    'file_dep': instance.file_dep,
                    'actions': instance.actions,
                    'targets': instance.targets,
                    'uptodate': instance.uptodate,
                    'task_dep': instance.task_dep,
                    'title': instance.title,
                    'doc': instance.doc,
                    #'clean': instance.clean,
                    }

class StudyTask(Task):
    def __init__(self, study):
        super(StudyTask, self).__init__()
        self.study = study

class SubjectTask(StudyTask):
    def __init__(self, subject):
        super(SubjectTask, self).__init__(subject.study)
        self.subject = subject

class TrialTask(SubjectTask):
    def __init__(self, trial):
        super(TrialTask, self).__init__(trial.subject)
        self.trial = trial

class ToolTask(TrialTask):
    def __init__(self, trial, tool_folder, exec_name=None, cmd=None, env=None):
        super(ToolTask, self).__init__(trial)
        self.name = '%s_%s' % (trial.id, tool_folder)
        self.path = os.path.join(trial.results_exp_path, tool_folder)

        self.file_dep = [
                '%s/setup.xml' % self.path
                ]

        if exec_name == None:
            exec_name = tool_folder
        if cmd == None:
            cmd_action = CmdAction('%s/bin/%s -S setup.xml' %
                    (self.study.config['opensim_home'], exec_name),
                    cwd=os.path.abspath(self.path),
                    env=env)
        else:
            cmd_action = CmdAction(cmd, cwd=os.path.abspath(self.path),
                    env=env)
        self.actions = [
                cmd_action,
                ]


