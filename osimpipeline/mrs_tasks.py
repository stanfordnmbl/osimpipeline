import os
import numpy as np
import shutil

import task
import utilities as util
import postprocessing as pp

class TaskMRSDeGrooteSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, **kwargs):
        super(TaskMRSDeGrooteSetup, self).__init__('mrs', trial, **kwargs)
        self.doc = "Create a setup file for the DeGroote Muscle Redundancy Solver tool."
        self.kinematics_file = os.path.join(self.trial.results_exp_path, 'ik',
                '%s_%s_ik_solution.mot' % (self.study.name, self.trial.id))
        self.rel_kinematics_file = os.path.relpath(self.kinematics_file,
                self.path)
        self.kinetics_file = os.path.join(self.trial.results_exp_path,
                'id', 'results', '%s_%s_id_solution.sto' % (self.study.name,
                    self.trial.id))
        self.rel_kinetics_file = os.path.relpath(self.kinetics_file,
                self.path)
        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        self.results_post_fpath = os.path.join(self.path, 'postprocess.m')
        self.results_output_fpath = os.path.join(self.path, '%s_%s_mrs.mat' % (
            self.study.name, self.tricycle.id)) 

        self.file_dep += [
            self.kinematics_file,
            self.kinetics_file
        ]

        # Fill out setup.m template and write to results directory
        self.create_setup_action()

        # Fill out postprocess.m template and write to results directory
        self.add_action(
                ['templates/mrs/postprocess.m'],
                [self.results_post_fpath],
                self.fill_postprocess_template)

    def create_setup_action(self): 
        self.add_action(
                    ['templates/%s/setup.m' % self.tool],
                    [self.results_setup_fpath],
                    self.fill_setup_template,  
                    init_time=self.init_time,
                    final_time=self.final_time,      
                    )

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            # TODO should this be an RRA-adjusted model?
            content = content.replace('@MODEL@', os.path.relpath(
                self.subject.scaled_model_fpath, self.path))
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            # TODO provide slop on either side? start before the cycle_start?
            # end after the cycle_end?
            content = content.replace('@INIT_TIME@',
                    '%.5f' % init_time)
            content = content.replace('@FINAL_TIME@', 
                    '%.5f' % final_time)
            content = content.replace('@IK_SOLUTION@',
                    self.rel_kinematics_file)
            content = content.replace('@ID_SOLUTION@',
                    self.rel_kinetics_file)
            content = content.replace('@SIDE@',
                    self.trial.primary_leg[0])

        with open(target[0], 'w') as f:
            f.write(content)

    def fill_postprocess_template(self, file_dep, target):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            content = content.replace('@MODEL@', os.path.relpath(
                self.subject.scaled_model_fpath, self.path))

        with open(target[0], 'w') as f:
            f.write(content)

class TaskMRSDeGroote(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, mrs_setup_task, **kwargs):
        super(TaskMRSDeGroote, self).__init__(mrs_setup_task, trial, 
            opensim=False, **kwargs)
        self.doc = 'Run DeGroote Muscle Redundancy Solver in MATLAB.'
        self.results_setup_fpath  = mrs_setup_task.results_setup_fpath
        self.results_output_fpath = mrs_setup_task.results_output_fpath

        self.file_dep += [
                self.results_setup_fpath,
                self.subject.scaled_model_fpath,
                mrs_setup_task.kinematics_file,
                mrs_setup_task.kinetics_file,
                ]

        self.actions += [
                self.run_muscle_redundancy_solver,
                self.delete_muscle_analysis_results,
                ]

        self.targets += [
                self.results_output_fpath,
                ]

    def run_muscle_redundancy_solver(self):
        with util.working_directory(self.path):
            # On Mac, CmdAction was causing MATLAB ipopt with GPOPS output to
            # not display properly.

            status = os.system('matlab %s -logfile matlab_log.txt -wait -r "try, '
                    "run('%s'); disp('SUCCESS'); "
                    'catch ME; disp(getReport(ME)); exit(2), end, exit(0);"\n'
                    % ('-automation' if os.name == 'nt' else '',
                        self.results_setup_fpath)
                    )
            if status != 0:
                raise Exception('Non-zero exit status.')

            # Wait until output mat file exists to finish the action
            while True:
                time.sleep(3.0)

                mat_exists = os.path.isfile(self.results_output_fpath)
                if mat_exists:
                    break

    def delete_muscle_analysis_results(self):
        if os.path.exists(os.path.join(self.path, 'results')):
            import shutil
            shutil.rmtree(os.path.join(self.path, 'results'))

class TaskMRSDeGrootePost(task.PostTask):
    REGISTRY = []
    def __init__(self, trial, mrs_setup_task, **kwargs):
        super(TaskMRSDeGrootePost, self).__init__(mrs_setup_task, trial, 
            **kwargs)
        self.doc = 'Postprocess DeGroote Muscle Redundancy Solver in MATLAB.'
        self.id = mrs_setup_task.tricycle.id
        self.results_output_fpath = mrs_setup_task.results_output_fpath

        # Plot muscle excitations, activations, and reserve activations
        self.add_action([self.results_output_fpath],
                        [os.path.join(self.path, 'muscle_activity.pdf'),
                        os.path.join(self.path, 'reserve_activity.pdf')],
                        self.plot_activations
                        )

        # Plot joint moment breakdown
        self.add_action([self.results_output_fpath],
                        [os.path.join(self.path, 'joint_moment_breakdown.pdf'),
                        os.path.join(self.path, '%s_%s_mrs_moments.csv' %
                        (self.study.name, mrs_setup_task.tricycle.id))],
                        self.plot_joint_moment_breakdown
                        )

    def plot_activations(self, file_dep, target):

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        exc = util.hdf2pandas(file_dep[0], 'MExcitation', labels=muscle_names)
        act = util.hdf2pandas(file_dep[0], 'MActivation', labels=muscle_names)
        dof_names = util.hdf2list(file_dep[0], 'DatStore/DOFNames', 
            type=str)
        reserves = util.hdf2pandas(file_dep[0],'RActivation', labels=dof_names)

        # Create plots
        pp.plot_muscle_activity(target[0], exc=exc, act=act)
        pp.plot_reserve_activity(target[1], reserves)

    def plot_joint_moment_breakdown(self, file_dep, target):

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames', type=str)
        num_dofs = len(dof_names)
        num_muscles = len(muscle_names)
        joint_moments_exp = util.hdf2numpy(file_dep[0], 'DatStore/T_exp')
        tendon_forces = util.hdf2numpy(file_dep[0], 'TForce')
        exp_time = util.hdf2numpy(file_dep[0], 'DatStore/time').transpose()[0]
        time = util.hdf2numpy(file_dep[0], 'Time').transpose()[0]
        moment_arms_exp = util.hdf2numpy(file_dep[0], 'DatStore/dM').transpose()

        # Clip large tendon forces at final time
        from warnings import warn
        for imusc in range(len(muscle_names)):
            tendon_force = tendon_forces[:,imusc]
            if (tendon_force[-1] > 10*tendon_force[-2]):
                tendon_force[-1] = tendon_force[-2]
                tendon_forces[:,imusc] = tendon_force
                warn('WARNING: large %s tendon force at final time. '
                    'Clipping...' % muscle_names[imusc])

        # Interpolate to match solution time
        from scipy.interpolate import interp1d
        ma_shape = (len(time), moment_arms_exp.shape[1], 
            moment_arms_exp.shape[2])
        moment_arms = np.empty(ma_shape)
        for i in range(moment_arms_exp.shape[2]):
            func_moment_arms_interp = interp1d(exp_time, 
                moment_arms_exp[:,:,i].squeeze(), axis=0)
            moment_arms[:,:,i] = func_moment_arms_interp(time)

        func_joint_moments_interp = interp1d(exp_time, joint_moments_exp,
            axis=0)
        joint_moments = func_joint_moments_interp(time)

        # Generate plots
        pp.plot_joint_moment_breakdown(time, joint_moments, tendon_forces,
            moment_arms, dof_names, muscle_names, target[0], target[1],
            mass=self.subject.mass)

class TaskMRSDeGrooteMod(task.ToolTask):
    def __init__(self, trial, mrs_setup_task, mod_name, description,
        mrsflags, **kwargs):
        """
        Parameters
        ----------
        mrsflags:
            A function that takes a Cycle and returns a list of flags
            (formatted like "study='ISB2017/Collins2017'), or a list of
            flags.
        adds_parameters: bool
            Does the modified MRS optimization problem include
            constant parameters as variables?
        """
        self.mod_name = mod_name
        self.tool = 'mrsmod_%s' % self.mod_name
        mrs_setup_task.tool = self.tool

        super(TaskMRSDeGrooteMod, self).__init__(mrs_setup_task, trial,
            opensim=False, **kwargs)
        self.mrs_setup_task = mrs_setup_task
        self.description = description
        self.mrsflags = mrsflags
        self.doc = 'Run a modified DeGroote Muscle Redundancy Solver in MATLAB.'
        self.basemrs_path = mrs_setup_task.path
        self.tricycle = mrs_setup_task.tricycle
        self.path = os.path.join(self.study.config['results_path'],
            'mrsmod_%s' % self.mod_name, trial.rel_path, 'mrs',
            mrs_setup_task.cycle.name if mrs_setup_task.cycle else '')
        self.setup_template_fpath = 'templates/mrs/setup.m'
        self.setup_fpath = os.path.join(self.path, 'setup.m')
        self.kinematics_fpath = mrs_setup_task.kinematics_file
        self.kinetics_fpath = mrs_setup_task.kinetics_file
        self.results_output_fpath = os.path.join(self.path,
                    '%s_%s_mrs.mat' % 
                    (self.study.name, mrs_setup_task.tricycle.id))

        self.file_dep += [
                self.setup_template_fpath,
                self.subject.scaled_model_fpath,
                self.kinematics_fpath,
                self.kinetics_fpath,
                ]

        self.actions += [
                self.make_path,
                ]

        self.add_action(
                    [],
                    [],
                    self.fill_setup_template,  
                    init_time=mrs_setup_task.init_time,
                    final_time=mrs_setup_task.final_time,      
                    )

        self.actions += [
                self.run_muscle_redundancy_solver,
                self.delete_muscle_analysis_results,
                ]

        self.targets += [
                self.setup_fpath,
                self.results_output_fpath,
                ]

    def make_path(self):
        if not os.path.exists(self.path): os.makedirs(self.path)

    def fill_setup_template(self, file_dep, target, 
                            init_time=None, final_time=None):
        with open(self.setup_template_fpath) as ft:
            content = ft.read()

            if type(self.mrsflags) is list:
                list_of_flags = self.mrsflags 
            else:
             list_of_flags = self.mrsflags(self.cycle)

            # Insert flags for the mod.
            flagstr = ''
            for flag in list_of_flags:
                flagstr += 'Misc.%s;\n' % flag

            content = content.replace('Misc = struct();',
                    'Misc = struct();\n' +
                    flagstr + '\n' +
                    # In case the description has multiple lines, add comment
                    # symbol in front of every line.
                    '% ' + self.description.replace('\n', '\n% ') + '\n')

            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            # TODO should this be an RRA-adjusted model?
            content = content.replace('@MODEL@', os.path.relpath(
                self.subject.scaled_model_fpath, self.path))
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            # TODO provide slop on either side? start before the cycle_start?
            # end after the cycle_end?
            content = content.replace('@INIT_TIME@',
                    '%.5f' % init_time)
            content = content.replace('@FINAL_TIME@', 
                    '%.5f' % final_time)

            content = content.replace('@IK_SOLUTION@',
                    os.path.relpath(self.kinematics_fpath, self.path))
            content = content.replace('@ID_SOLUTION@',
                    os.path.relpath(self.kinetics_fpath, self.path))
            content = content.replace('@SIDE@',
                    self.trial.primary_leg[0])

        with open(self.setup_fpath, 'w') as f:
            f.write(content)

    def run_muscle_redundancy_solver(self):
        with util.working_directory(self.path):

            status = os.system('matlab '
                '%s -logfile matlab_log.txt -wait -r "try, '
                "run('%s'); disp('SUCCESS'); "
                'catch ME; disp(getReport(ME)); exit(2), end, exit(0);"\n'
                % ('-automation' if os.name == 'nt' else '',
                    self.setup_fpath)
                )

            if status != 0:
                raise Exception('Non-zero exit status.')

             # Wait until output mat file exists to finish the action
            while True:
                time.sleep(3.0)

                mat_exists = os.path.isfile(self.results_output_fpath)
                if mat_exists:
                    break

    def delete_muscle_analysis_results(self):
        if os.path.exists(os.path.join(self.path, 'results')):
            import shutil
            shutil.rmtree(os.path.join(self.path, 'results'))

class TaskMRSDeGrooteModPost(task.PostTask):
    REGISTRY = []
    def __init__(self, trial, mrsmod_task, **kwargs):
        super(TaskMRSDeGrooteModPost, self).__init__(mrsmod_task, trial, 
            **kwargs)
        self.mrs_setup_task = mrsmod_task.mrs_setup_task
        self.doc = 'Postprocess modified DeGroote Muscle Redundancy Solver problem in MATLAB.'
        self.mrsmod_task = mrsmod_task
        self.path = self.mrsmod_task.path
        self.id = self.mrs_setup_task.tricycle.id
        self.results_output_fpath = mrsmod_task.results_output_fpath

        # Plot muscle excitations, activations, and reserve activations
        self.add_action([self.results_output_fpath],
                        [os.path.join(self.path, 'muscle_activity.pdf'),
                        os.path.join(self.path, 'reserve_activity.pdf')],
                        self.plot_activations
                        )

        # Plot joint moment breakdown
        self.add_action([self.results_output_fpath],
                        [os.path.join(self.path, 'joint_moment_breakdown.pdf'),
                        os.path.join(self.path, '%s_%s_mrs_moments.csv' %
                        (self.study.name, self.id))],
                        self.plot_joint_moment_breakdown
                        )

    def plot_activations(self, file_dep, target):

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        exc = util.hdf2pandas(file_dep[0], 'MExcitation', labels=muscle_names)
        act = util.hdf2pandas(file_dep[0], 'MActivation', labels=muscle_names)
        dof_names = util.hdf2list(file_dep[0], 'DatStore/DOFNames', 
            type=str)
        reserves = util.hdf2pandas(file_dep[0],'RActivation', labels=dof_names)

        # Create plots
        pp.plot_muscle_activity(target[0], exc=exc, act=act)
        pp.plot_reserve_activity(target[1], reserves)

    def plot_joint_moment_breakdown(self, file_dep, target):

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames', type=str)
        num_dofs = len(dof_names)
        num_muscles = len(muscle_names)
        joint_moments_exp = util.hdf2numpy(file_dep[0], 'DatStore/T_exp')
        tendon_forces = util.hdf2numpy(file_dep[0], 'TForce')
        exp_time = util.hdf2numpy(file_dep[0], 'DatStore/time').transpose()[0]
        time = util.hdf2numpy(file_dep[0], 'Time').transpose()[0]
        moment_arms_exp = util.hdf2numpy(file_dep[0], 'DatStore/dM').transpose()
        
        # Clip large tendon forces at final time
        from warnings import warn
        for imusc in range(len(muscle_names)):
            tendon_force = tendon_forces[:,imusc]
            if (tendon_force[-1] > 5*tendon_force[-2]):
                tendon_force[-1] = tendon_force[-2]
                tendon_forces[:,imusc] = tendon_force
                warn('WARNING: large %s tendon force at final time. '
                    'Clipping...' % muscle_names[imusc])

        # Interpolate to match solution time
        from scipy.interpolate import interp1d
        ma_shape = (len(time), moment_arms_exp.shape[1], 
            moment_arms_exp.shape[2])
        moment_arms = np.empty(ma_shape)
        for i in range(moment_arms_exp.shape[2]):
            func_moment_arms_interp = interp1d(exp_time, 
                moment_arms_exp[:,:,i].squeeze(), axis=0)
            moment_arms[:,:,i] = func_moment_arms_interp(time)

        func_joint_moments_interp = interp1d(exp_time, joint_moments_exp,
            axis=0)
        joint_moments = func_joint_moments_interp(time)

        # Generate plots
        pp.plot_joint_moment_breakdown(time, joint_moments, tendon_forces,
            moment_arms, dof_names, muscle_names, target[0], target[1],
            mass=self.subject.mass)