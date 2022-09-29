"""
These abstract base classes define a 'task' for python-doit. 
"""

import os
from numpy import loadtxt
from doit.action import CmdAction
import pylab as pl

# Import postprocessing subroutines
from postprocessing import plot_lower_limb_kinematics

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

    def add_action(self, file_dep, target, member_function, *args_to_member,
        **kwargs_to_member):
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
        args = [file_dep, target]
        if len(args_to_member):
            args += args_to_member
        
        if bool(kwargs_to_member):
            self.actions.append((member_function, args, kwargs_to_member))
        else:
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

class SetupTask(TrialTask):
    def __init__(self, tool, trial, cycle=None):
        super(SetupTask, self).__init__(trial)
        self.tool = tool
        self.trial = trial
        self.tool_path = os.path.join(trial.results_exp_path, self.tool)
        self.doit_path = self.study.config['doit_path']
        self.cycle = cycle

        # Generate setup file based on whether or not a specific cycle has been
        # specified
        if self.cycle:
            self.tricycle = self.cycle
            self.create_setup_deps = (True if self.cycle.name=='cycle01' else
                False)
            self.name = '%s_%s_setup_%s' % (trial.id, self.tool, 
                self.cycle.name)
            self.add_cycle_dir()
            self.path = os.path.join(self.tool_path, self.cycle.name)
            self.init_time=self.cycle.start
            self.final_time=self.cycle.end
        else:
            self.tricycle = trial
            self.create_setup_deps = True
            self.name = '%s_%s_setup' % (trial.id, self.tool)
            self.add_tool_dir()
            self.path = self.tool_path
            first_cycle = trial.cycles[0]
            last_cycle = trial.cycles[-1]
            self.init_time = first_cycle.start
            self.final_time = last_cycle.end

        self.source_path = os.path.join(trial.rel_path, self.tool)
        self.source_extloads_fpath = os.path.join(self.source_path,
            'external_loads.xml')
        self.results_extloads_fpath = os.path.join(self.path,
                os.path.basename(self.source_extloads_fpath))
        self.source_tasks_fpath = os.path.join(self.source_path,
            'tasks.xml')
        self.results_tasks_fpath = os.path.join(self.path, 
                os.path.basename(self.source_tasks_fpath))
        self.adjusted_model = '%s_adjusted.osim' % self.subject.name    
        self.adjusted_model_fpath = os.path.join(self.path, 
            self.adjusted_model)
        self.results_setup_fpath = os.path.join(self.path, 'setup.xml')  

    def create_setup_action(self): 
        self.add_action(
                    ['templates/%s/setup.xml' % self.tool],
                    [self.results_setup_fpath],
                    self.fill_setup_template,  
                    init_time=self.init_time,
                    final_time=self.final_time,      
                    )

    def create_external_loads_action(self, rel_kinematics_fpath):
        self.add_source_dir()
        if (not os.path.exists(self.source_extloads_fpath) and 
            self.create_setup_deps):
            # The user does not yet have a external_loads.xml in place; fill 
            # out the template.
            self.add_action(
                ['templates/%s/external_loads.xml' % self.tool],
                [self.source_extloads_fpath],
                self.fill_external_loads_template,
                rel_kinematics_fpath)
            self.actions.append((self.copy_file,
                [[self.source_extloads_fpath], [self.results_extloads_fpath]]))
        else:
            # We have already filled out the template external loads file,
            # and the user might have made changes to it.
            self.add_action(
                    [self.source_extloads_fpath],
                    [self.results_extloads_fpath],
                    self.copy_file) 

    def create_tasks_action(self):
        self.add_source_dir()
        if (not os.path.exists(self.source_tasks_fpath) and
            self.create_setup_deps):
            # The user does not yet have a tasks.xml in place; fill out the
            # template.
            self.add_action(
                    ['templates/%s/tasks.xml' % self.tool],
                    [self.source_tasks_fpath],
                    self.fill_tasks_template)
            self.actions.append((self.copy_file,
                [[self.source_tasks_fpath], [self.results_tasks_fpath]]))
        else:
            # We have already filled out the template tasks file,
            # and the user might have made changes to it.
            self.add_action(
                    [self.source_tasks_fpath],
                    [self.results_tasks_fpath],
                    self.copy_file)

    def fill_setup_template(self):
        raise NotImplementedError()

    def fill_external_loads_template(self, file_dep, target,
        rel_kinematics_fpath):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.trial.id)
            content = content.replace('@KINEMATICS_FILE@', 
                rel_kinematics_fpath)

        with open(target[0], 'w') as f:
            f.write(content)

    def fill_tasks_template(self, file_dep, target):
        # Don't overwrite existing tasks file
        if not os.path.exists(target[0]):
            with open(file_dep[0]) as ft:
                content = ft.read()
                content = content.replace('@STUDYNAME@', self.study.name)
                content = content.replace('@NAME@', self.trial.id)
    
            with open(target[0], 'w') as f:
                f.write(content)

    def add_tool_dir(self):
        if not os.path.exists(self.tool_path): os.makedirs(self.tool_path)

    def add_cycle_dir(self):
        self.add_tool_dir()
        cycle_path = os.path.join(self.tool_path, self.cycle.name)
        if not os.path.exists(cycle_path): os.makedirs(cycle_path)

    def add_source_dir(self):
        if not os.path.exists(self.source_path): os.makedirs(self.source_path)

class ToolTask(TrialTask):
    def __init__(self, setup_task, trial, cycle=None,
                 exec_name=None, env=None, opensim=True):
        super(ToolTask, self).__init__(trial)
        self.exec_name = exec_name
        self.env = env
        self.path = setup_task.path

        if cycle:
            self.name = '%s_%s_%s' % (trial.id, setup_task.tool, cycle.name)
        else:
            self.name = '%s_%s' % (trial.id, setup_task.tool)

        if opensim:
            if self.exec_name == None:
                self.exec_name = setup_task.tool

            self.add_action(['%s/setup.xml' % self.path], [],
                self.execute_tool)

    def execute_tool(self, file_dep, target):
        import subprocess
        exec_path = os.path.join(self.study.config['opensim_home'],
                        'bin', self.exec_name) 
        print(exec_path)
        print(file_dep[0])
        p = subprocess.Popen('%s -S %s' % (exec_path, file_dep[0]),
            cwd=self.path, env=self.env)
        p.wait()
        if p.returncode != 0:
            raise Exception('Non-zero exit status: code %s.' % p.returncode)

class PostTask(TrialTask):
    def __init__(self, setup_task, trial, cycle=None):
        super(PostTask, self).__init__(trial)
        self.methods = list()
        self.path = setup_task.path
        self.gl = None if not cycle else cycle.gl

        if cycle:
            self.name = '%s_%s_post_%s' % (trial.id, setup_task.tool, 
                cycle.name)
        else:
            self.name = '%s_%s_post' % (trial.id, setup_task.tool)





