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

    def add_action(self, file_dep, target, member_function, *args_to_member):
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

        You can use `args_to_member` to pass additional arguments to the
        `member_function`.
        
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
        args = [file_dep, target]
        if len(args_to_member):
            args += args_to_member
        self.actions.append((member_function, args))

    def copy_file(self, file_dep, target):
        """This can be used as the action for derived classes that want to copy
        a file from one place to another (e.g., from the source to the results
        directory).
        """
        import shutil
        to_dir = os.path.split(target[0])[0]
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

class CycleTask(TrialTask):
    def __init__(self, cycle):
        super(CycleTask, self).__init__(cycle.trial)
        self.cycle = cycle

class ToolTrialTask(TrialTask):
    def __init__(self, trial, tool_folder, exec_name=None, cmd=None, env=None):
        super(ToolTrialTask, self).__init__(trial)
        self.name = '%s_%s' % (trial.id, tool_folder)
        self.path = os.path.join(trial.results_exp_path, tool_folder)

        self.file_dep = [
                '%s/setup.xml' % self.path
                ]

        #if env == None:
        #    from sys import platform
        #    if platform == 'win32':
        #        env = {'PATH':
        #                os.path.join(self.study.config['opensim_home'], 'bin')}
        #    elif platform == 'darwin': # mac
        #        env = {'DYLD_LIBRARY_PATH':
        #                os.path.join(self.study.config['opensim_home'], 'lib')}
        #    else:
        #        env = {'LD_LIBRARY_PATH': 
        #                os.path.join(self.study.config['opensim_home'], 'lib')}

        if exec_name == None:
            exec_name = tool_folder
        if cmd == None:
            cmd_action = CmdAction('"' + os.path.join(
                self.study.config['opensim_home'], 'bin', exec_name)
                + '" -S setup.xml',
                cwd=os.path.abspath(self.path),
                env=env)
        else:
            cmd_action = CmdAction(cmd, cwd=os.path.abspath(self.path),
                    env=env)
        self.actions = [
                cmd_action,
                ]

class ToolCycleTask(CycleTask):
    def __init__(self, cycle, tool_folder, exec_name=None, cmd=None, env=None):
        super(ToolCycleTask, self).__init__(cycle)
        self.name = '%s_%s' % (cycle.id, tool_folder)
        self.path = os.path.join(cycle.results_exp_path, tool_folder)

        self.file_dep = [
                '%s/setup.xml' % self.path
                ]

        #if env == None:
        #    from sys import platform
        #    if platform == 'win32':
        #        env = {'PATH':
        #                os.path.join(self.study.config['opensim_home'], 'bin')}
        #    elif platform == 'darwin': # mac
        #        env = {'DYLD_LIBRARY_PATH':
        #                os.path.join(self.study.config['opensim_home'], 'lib')}
        #    else:
        #        env = {'LD_LIBRARY_PATH': 
        #                os.path.join(self.study.config['opensim_home'], 'lib')}

        if exec_name == None:
            exec_name = tool_folder
        if cmd == None:
            cmd_action = CmdAction('"' + os.path.join(
                self.study.config['opensim_home'],'bin',exec_name)
                + '" -S setup.xml',
                cwd=os.path.abspath(self.path),
                env=env)
        else:
            cmd_action = CmdAction(cmd, cwd=os.path.abspath(self.path),
                    env=env)
        self.actions = [
                cmd_action,
                ]
