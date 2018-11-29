import os
import time
import numpy as np
import shutil

import task
import utilities as util
import postprocessing as pp


import pdb


class TaskMRSDeGrooteSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, cost='Default', use_filtered_id_results=False,
                **kwargs):
        # assigns the cost that will be used as the one passed in the method call
        self.cost = cost
        self.costdir = ''
        # modifies the name of the task and output directory based on the cost used
        if not (self.cost == 'Default'):
            self.costdir = cost
            # self.name += '_%s' % self.cost

        super(TaskMRSDeGrooteSetup, self).__init__('mrs', trial, 
            pathext=self.costdir, **kwargs)
        self.doc = "Create a setup file for the DeGroote Muscle Redundancy Solver tool."
        # specifies the kinematics file location and name
        self.kinematics_file = os.path.join(self.trial.results_exp_path, 'ik',
                '%s_%s_ik_solution.mot' % (self.study.name, self.trial.id))
        # provides a path to the first locaiton from the specified one
        # in this case it gives the path from the path of the task, to the path of the kinematics_file
        self.rel_kinematics_file = os.path.relpath(self.kinematics_file,
                self.path)
        # string manipulation to grab the filtered or non filtered ID result
        id_suffix = '_filtered' if use_filtered_id_results else ''
        # specifies the actual ID result file
        self.kinetics_file = os.path.join(self.trial.results_exp_path,
                'id', 'results', '%s_%s_id_solution%s.sto' % (self.study.name,
                    self.trial.id, id_suffix))
        # gives the path to the kinetics file from the task directory
        self.rel_kinetics_file = os.path.relpath(self.kinetics_file,
                self.path)
        # creates a results setup matlab file
        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        # creates a results postprocess file
        self.results_post_fpath = os.path.join(self.path, 'postprocess.m')
        # creates the output workspace from matlab that will be used
        self.results_output_fpath = os.path.join(self.path, '%s_%s_mrs.mat' % (
            self.study.name, self.tricycle.id)) 
        #!!! what the heck is tricycle??

        # sets up some file dependencies -> not super sure what these do.!! 
        self.file_dep += [
            self.kinematics_file,
            self.kinetics_file
        ]

        # Fill out setup.m template and write to results directory
        self.create_setup_action()

        #!!! why are these commented out? seems like they are just creating the postprocess files to be used
        # Fill out postprocess.m template and write to results directory
        # self.add_action(
        #         ['templates/mrs/postprocess.m'],
        #         [self.results_post_fpath],
        #         self.fill_postprocess_template)


        # would like an explanation of what is actually happening here!!!
    def create_setup_action(self): 
        self.add_action(
                    ['templates/%s/setup.m' % self.tool],
                    [self.results_setup_fpath],
                    self.fill_setup_template,  
                    init_time=self.init_time,
                    final_time=self.final_time,      
                    )

    # this function is what is going into the template setup file for the mrs tasks
    # and filling it with all of the actual study data
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
            content = content.replace('@COST@', self.cost)

        # writes all of the content changes to the file
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
        # this is a tool task subclass, so here it is defining the superclasses properties
        super(TaskMRSDeGroote, self).__init__(mrs_setup_task, trial, 
            opensim=False, **kwargs)
        self.doc = 'Run DeGroote Muscle Redundancy Solver in MATLAB.'
        # copies the setup path over from the setup task
        self.results_setup_fpath  = mrs_setup_task.results_setup_fpath
        # copies over the output path files
        self.results_output_fpath = mrs_setup_task.results_output_fpath
        # based on the cost type, it names the task accordingly
        if not (mrs_setup_task.cost == 'Default'):
            self.name += '_%s' % mrs_setup_task.cost

        # not sure what is actually happening here - allows the task to access these files?!!!
        self.file_dep += [
                self.results_setup_fpath,
                self.subject.scaled_model_fpath,
                mrs_setup_task.kinematics_file,
                mrs_setup_task.kinetics_file,
                ]

        # adds the actions to run the class methods below
        self.actions += [
                self.run_muscle_redundancy_solver,
                self.delete_muscle_analysis_results,
                ]

        # sets the target files as the output folder for doit pipeline
        self.targets += [
                self.results_output_fpath,
                ]

    # class method -> called when added to the actions list in the base class
    # !!! going to want a walkthrough of how this interfaces with Matlab
    def run_muscle_redundancy_solver(self):
        # sets the working directory as the path of the task
        with util.working_directory(self.path):
            # On Mac, CmdAction was causing MATLAB ipopt with GPOPS output to
            # not display properly.

            # not sure
            # sets up the matlab_log.txt file -> outputs of the matlab commands
            status = os.system('matlab %s -logfile matlab_log.txt -wait -r "try, '
                    "run('%s'); disp('SUCCESS'); "
                    'catch ME; disp(getReport(ME)); exit(2), end, exit(0);"\n'
                    % ('' if os.name == 'nt' else '',
                        self.results_setup_fpath)
                    )
            # exception catch for failed matlab commands
            if status != 0:
                # print 'Non-zero exist status. Continuing....'
                raise Exception('Non-zero exit status.')

            # Wait until output mat file exists to finish the action
            while True:
                time.sleep(3.0)
                mat_exists = os.path.isfile(self.results_output_fpath)
                if mat_exists:
                    break

    # class method that gets called each time the MRS tasks are called
    # removes the previous results of any MRS tasks that were run for the specific trail or cycle 
    def delete_muscle_analysis_results(self):
        if os.path.exists(os.path.join(self.path, 'results')):
            import shutil
            shutil.rmtree(os.path.join(self.path, 'results'))


# post task for the MRS tasks
# this is a subclass of task
class TaskMRSDeGrootePost(task.PostTask):
    REGISTRY = []
    def __init__(self, trial, mrs_setup_task, **kwargs):
        # initializes the base post task structure
        super(TaskMRSDeGrootePost, self).__init__(mrs_setup_task, trial, 
            **kwargs)
        self.doc = 'Postprocess DeGroote Muscle Redundancy Solver in MATLAB.'
        #!!! what is this?
        self.id = mrs_setup_task.tricycle.id
        # sets the output path for all of the results
        self.results_output_fpath = mrs_setup_task.results_output_fpath
        # print "\nhere we go:"
        # print self.results_output_fpath
        

        # edits the name of the task based on the cost name
        if not (mrs_setup_task.cost == 'Default'):
            self.name += '_%s' % mrs_setup_task.cost

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
    # class method that is called to plot the muscle excitations and activations and reserve activations
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

    # class method called to plot the joint moments
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
        # iterate through the muscles
        for imusc in range(len(muscle_names)):
            # assign the actual tendon force for that muscle
            tendon_force = tendon_forces[:,imusc]
            # cutt off the ends of the force if they are uncharacteristically large
            # note: only does this at the end (not the beginning)
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
    REGISTRY = []
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
        # print "debug"
        # pdb.set_trace()
        self.mod_name = mod_name
        # print self.mod_name
        self.tool = 'mrsmod_%s' % self.mod_name
        # print self.tool
        mrs_setup_task.tool = self.tool 

        super(TaskMRSDeGrooteMod, self).__init__(mrs_setup_task, trial,
            opensim=False, **kwargs)
        self.cost = mrs_setup_task.cost
        # print self.cost
        self.costdir = ''
        # print self.costdir
        if not (self.cost == 'Default'):
            self.name += "_%s" % self.cost
            self.costdir = self.cost
        # print self.name
        # print self.costdir
        self.mrs_setup_task = mrs_setup_task
        # print self.mrs_setup_task
        self.description = description
        # print self.description
        self.mrsflags = mrsflags
        # print self.mrsflags
        self.doc = 'Run a modified DeGroote Muscle Redundancy Solver in MATLAB.'
        # print self.doc
        self.basemrs_path = mrs_setup_task.path
        # print self.basemrs_path
        self.tricycle = mrs_setup_task.tricycle
        # print self.tricycle
        


        self.path = os.path.join(self.study.config['results_path'],
            'mrsmod_%s' % self.mod_name, trial.rel_path, 'mrs',
            mrs_setup_task.cycle.name if mrs_setup_task.cycle else '', 
            self.costdir)
        # print self.path
        self.setup_template_fpath = 'templates/mrs/setup.m'
        # print self.setup_template_fpath
        self.setup_fpath = os.path.join(self.path, 'setup.m')
        # print self.setup_fpath
        self.kinematics_fpath = mrs_setup_task.kinematics_file
        # print self.kinematics_fpath
        self.kinetics_fpath = mrs_setup_task.kinetics_file
        # print self.kinetics_fpath
        self.results_output_fpath = os.path.join(self.path,
                    '%s_%s_mrs.mat' % 
                    (self.study.name, mrs_setup_task.tricycle.id))
        # print self.results_output_fpath
        self.cost = mrs_setup_task.cost
        # print self.cost
        
        # print "\nat end of initial"
        # print self.path




        self.file_dep += [
                self.setup_template_fpath,
                self.subject.scaled_model_fpath,
                self.kinematics_fpath,
                self.kinetics_fpath,
                ]
        
        # print "\n here \n"

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
        
        # print "\nin make_path"
        # print self.path

        if not os.path.exists(self.path): os.makedirs(self.path)

    def fill_setup_template(self, file_dep, target, 
                            init_time=None, final_time=None):
        
        # print "\nin fill_setup_template"
        # print self.setup_fpath
        # print self.results_output_fpath
        
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

            if 'cycle' in self.tricycle.name:
                flagstr += 'Misc.cycle=%s;\n' % self.tricycle.num

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
            content = content.replace('@COST@', self.cost)

        with open(self.setup_fpath, 'w') as f:
            f.write(content)

    def run_muscle_redundancy_solver(self):
        with util.working_directory(self.path):
            
            # print "\nin run_muscle_redundancy_solver"
            # print self.setup_fpath

            status = os.system('matlab %s -logfile matlab_log.txt -wait -r "try, '
                "run('%s'); disp('SUCCESS'); "
                'catch ME; disp(getReport(ME)); exit(2), end, exit(0);"\n'
                % ('' if os.name == 'nt' else '',
                    self.setup_fpath)
                )

            if status != 0:
                # print 'Non-zero exist status. Continuing....'
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

        if not (self.mrs_setup_task.cost == 'Default'):
            self.name += '_%s' % self.mrs_setup_task.cost

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
        # Load mat file fields
        
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