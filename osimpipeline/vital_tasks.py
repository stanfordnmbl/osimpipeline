
import os
import task
from doit.action import CmdAction
from numpy import loadtxt
import shutil
from perimysium.postprocessing import plot_rra_gait_info, plot_cmc_gait_info, \
        plot_so_gait_info, GaitScrutinyReport, plot_pgc, verify_kinematics, \
        plot_lower_limb_kinematics, plot_joint_torque_contributions, \
        plot_lower_limb_kinetics, \
        plot_marker_error, \
        plot_marker_error_general

class TaskCopyGenericModelToResults(task.StudyTask):
    REGISTRY = []
    def __init__(self, study):
        super(TaskCopyGenericModelToResults, self).__init__(study)
        self.name = '%s_copy_generic_model' % study.name
        self.doc = 'Copy generic model to the results directory.'
        self.add_action(
                [study.source_generic_model_fpath],
                [study.generic_model_fpath],
                self.copy_file)

class TaskCopyMotionCaptureData(task.StudyTask):
    """This a very generic task for copying motion capture data (marker
    trajectories, ground reaction, electromyography) and putting it in
    place for creating simulations.

    You may want to create your own custom task(s) that is tailored to the
    organization of your experimental data.

    The other tasks expect an `expdata` folder in the condition folder (for
    treadmill trials) that contains `marker_trajectories.trc` and
    `ground_reaction.mot`.
    
    Task name: `<study.name>_copy_data`
    """
    REGISTRY = [] # TODO Find a way to make this unnecessary.
    def __init__(self, study, regex_replacements):
        """Do not use this constructor directly; use `study.add_task()`.

        Parameters
        ----------
        study : 
            This argument is provided internally by `study.add_task()`.
        regex_replacements : list of tuples
            Each tuple should have two elements: (a) the pattern to match with
            the path (relative to the motion capture data path) of any file
            within the `motion_capture_data_path`, and (b) the replacement that
            provides the path to where the file should be copied (relative to
            the `results_path`).  The list contains as many of these tuples as
            you'd like. The regular expression replacements are performed with
            Python's `re.sub()`.

        Examples
        --------
        ```
        study.add_task(TaskCopyMotionCaptureData, [
                ('subject01/Data/Walk_100 02.trc',
                    'subject01/walk1/expdata/marker_trajectories.trc')])
        ```
        """
        super(TaskCopyMotionCaptureData, self).__init__(study)
        self.name = '_'.join([study.name, 'copy_data'])
        self.doc = 'Copy and organize motion capture data.'
        self.regex_replacements = regex_replacements
        self.register_files()

        self.add_action(self.registry.keys(), self.registry.values(),
                self.copy_files)

    def register_files(self):
        # Keys are source file paths (file_dep), values are destination paths
        # (targets).
        self.registry = dict()
        mocap_dir = self.study.config['motion_capture_data_path']
        results_dir = self.study.config['results_path']
        # Use regular expressions to copy/rename files.
        import re

        # Check if each file in the mocap_dir matches any of the regular
        # expressions given to this task.
        for dirpath, dirnames, filenames in os.walk(mocap_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                # Form path relative to the mocap directory.
                fpath_rel_to_mocap = os.path.relpath(fpath, mocap_dir)
                for pattern, replacement in self.regex_replacements:
                    match = re.search(pattern, fpath_rel_to_mocap)
                    if match != None:
                        # Found at least one match.
                        destination = os.path.join(results_dir, re.sub(pattern,
                            replacement, fpath_rel_to_mocap))
                        self.registry[fpath] = destination

    def copy_files(self, file_dep, target):
        for source, destination in self.registry.items():
            fname = os.path.split(source)[1]
            to_dir = os.path.split(destination)[0]
            if not os.path.exists(to_dir): os.makedirs(to_dir)

            if os.path.exists(destination):
                overwriting = '(overwriting)'
            else:
                overwriting = ''

            print('%-30s -> %s %s' % (fname, to_dir, overwriting))

            import shutil
            shutil.copyfile(source, destination)

class TaskScaleSetup(task.SubjectTask):
    """Create a setup file for the OpenSim Scale tool. You must place a
    template model markerset located at
    `templates/scale/prescale_markerset.xml`. This task creates a copy of this
    file for each subject, since you may need to tweak the markerset for
    individual subjects. The fields @STUDYNAME@ and @SUBJECTNAME@ in the
    template will be replaced by the correct values. You can find an example
    template in osimpipeline's templates directory.
    
    Task name: `subject<num>_scale_setup`
    """
    REGISTRY = []
    def __init__(self, subject, init_time, final_time,
            mocap_trial, edit_setup_function, addtl_file_dep=[]):
        """
        Parameters
        ----------
        init_time : float
            The initial time from the motion capture trial at which to start
            averaging the marker frames.
        final_time : float
            The final time from the motion capture trial at which to stop
            averaging the marker frames.
        mocap_trial : study.Trial
            The Trial whose marker trajectories to use for both the model
            scaling and marker placer steps. This is ususally a static trial
            but it does not need to be.
        addtl_file_dep : list of str
            Any other files that should be added as file dependencies, changes
            to which would make this task out of date. Usually, you might
            specify the dodo file and whichever file contains the
            `edit_setup_function`.

        """
        super(TaskScaleSetup, self).__init__(subject)
        self.subj_mass = self.subject.mass
        self.init_time = init_time
        self.final_time = final_time
        self.mocap_trial = mocap_trial
        self.name = '%s_scale_setup' % (self.subject.name)
        self.doc = "Create a setup file for OpenSim's Scale Tool."
        self.edit_setup_function = edit_setup_function
        self.results_scale_path = os.path.join(
                self.study.config['results_path'], 'experiments',
                self.subject.rel_path, 'scale')
        self.output_model_fpath = os.path.join(
                self.study.config['results_path'], 'experiments',
                self.subject.rel_path, '%s.osim' % self.subject.name)
        self.output_motion_fpath = os.path.join(self.results_scale_path,
                '%s_%s_ik_solution.mot' % (self.study.name, self.subject.name))
        self.output_markerset_fpath = os.path.join(self.results_scale_path,
                '%s_%s_markerset.xml' % (self.study.name, self.subject.name))

        # setup.xml
        # ---------
        self.setup_fpath = os.path.join(self.results_scale_path, 'setup.xml')
        self.add_action(
                {'marker_traj':
                    self.mocap_trial.marker_trajectories_fpath,
                 'generic_model': self.study.generic_model_fpath,
                    },
                {'setup': self.setup_fpath
                    },
                self.create_scale_setup)

        # MarkerSet for the Scale Tool.
        # -----------------------------
        self.source_scale_path = os.path.join(self.subject.rel_path, 'scale')
        self.prescale_template_fpath = 'templates/scale/prescale_markerset.xml'
        self.prescale_markerset_fname = '%s_prescale_markerset.xml' % (
                self.subject.name)
        self.source_prescale_markerset_fpath = os.path.join(
                self.source_scale_path, self.prescale_markerset_fname)
        self.results_prescale_markerset_fpath = os.path.join(
                self.results_scale_path, self.prescale_markerset_fname)
        if not os.path.exists(self.source_prescale_markerset_fpath):
            # The user does not yet have a markerset in place; fill out the
            # template.
            self.add_action(
                    {'template': self.prescale_template_fpath},
                    {'subjspecific': self.source_prescale_markerset_fpath},
                    self.fill_prescale_markerset_template)
            self.actions.append((self.copy_file,
                [[self.source_prescale_markerset_fpath],
                    [self.results_prescale_markerset_fpath]]))
        else:
            # We have already filled out the template prescale markerset,
            # and the user might have made changes to it.
            self.file_dep.append(self.source_prescale_markerset_fpath)
            self.add_action(
                    [self.source_prescale_markerset_fpath],
                    [self.results_prescale_markerset_fpath],
                    self.copy_file)

        self.file_dep += addtl_file_dep

    def fill_prescale_markerset_template(self, file_dep, target):
        if not os.path.exists(target['subjspecific']):
            ft = open(file_dep['template'])
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@SUBJECTNAME@', self.subject.name)
            ft.close()
            if not os.path.exists(self.source_scale_path):
                os.makedirs(self.source_scale_path)
            f = open(target['subjspecific'], 'w')
            f.write(content)
            f.close()

    def create_scale_setup(self, file_dep, target):
        import opensim as osm
        from perimysium import modeling as pmm

        # EDIT THESE FIELDS IN PARTICULAR.
        # --------------------------------
        time_range = osm.ArrayDouble()
        time_range.append(self.init_time)
        time_range.append(self.final_time)

        tool = osm.ScaleTool()
        tool.setName('%s_%s' % (self.study.name, self.subject.name))
        tool.setSubjectMass(self.subject.mass)

        # GenericModelMaker
        # =================
        gmm = tool.getGenericModelMaker()
        gmm.setModelFileName(os.path.relpath(file_dep['generic_model'],
            self.results_scale_path))
        gmm.setMarkerSetFileName(os.path.relpath(self.prescale_markerset_fname))

        # ModelScaler
        # ===========
        scaler = tool.getModelScaler()
        scaler.setPreserveMassDist(True)
        marker_traj_rel_fpath = os.path.relpath(file_dep['marker_traj'],
            self.results_scale_path)
        scaler.setMarkerFileName(marker_traj_rel_fpath)

        scale_order_str = osm.ArrayStr()
        scale_order_str.append('manualScale')
        scale_order_str.append('measurements')
        scaler.setScalingOrder(scale_order_str)

        scaler.setTimeRange(time_range)

        mset = scaler.getMeasurementSet()

        # Manual scalings
        # ---------------
        sset = scaler.getScaleSet()

        # MarkerPlacer
        # ============
        placer = tool.getMarkerPlacer()
        placer.setStaticPoseFileName(marker_traj_rel_fpath)
        placer.setTimeRange(time_range)
        placer.setOutputModelFileName(os.path.relpath(
            self.output_model_fpath, self.results_scale_path))
        placer.setOutputMotionFileName(os.path.relpath(
            self.output_motion_fpath, self.results_scale_path))
        placer.setOutputMarkerFileName(os.path.relpath(
            self.output_markerset_fpath, self.results_scale_path))
        ikts = pmm.IKTaskSet(placer.getIKTaskSet())

        self.edit_setup_function(pmm, mset, sset, ikts)

        # Validate Scales
        # ===============
        model = osm.Model(file_dep['generic_model'])
        bset = model.getBodySet()
        for iscale in range(sset.getSize()):
            segment_name = sset.get(iscale).getSegmentName()
            if not bset.contains(segment_name):
                raise Exception("You specified a Scale for "
                        "body %s but it's not in the model." % segment_name)

        if not os.path.exists(self.results_scale_path):
            os.makedirs(self.results_scale_path)
        tool.printToXML(target['setup'])
            

class TaskScale(task.SubjectTask):
    REGISTRY = []
    residual_actuators_template = 'templates/residual_actuators.xml'
    def __init__(self, subject, scale_setup_task, 
            ignore_nonexistant_data=False,
            ):
        super(TaskScale, self).__init__(subject)
        self.name = '%s_scale' % (self.subject.name)
        self.doc = "Run OpenSim's Scale Tool."
        self.ignore_nonexistant_data = ignore_nonexistant_data

        # file_dep
        # --------
        setup_fname = 'setup.xml'
        self.setup_fpath = scale_setup_task.setup_fpath
        self.generic_model_fpath = self.study.generic_model_fpath
        self.marker_trajectories_fpath = \
                scale_setup_task.mocap_trial.marker_trajectories_fpath
        self.prescale_markerset_fpath = \
                scale_setup_task.results_prescale_markerset_fpath
        self.file_dep = [
                self.setup_fpath,
                self.generic_model_fpath,
                self.prescale_markerset_fpath,
                self.marker_trajectories_fpath,
                self.residual_actuators_template,
                ]

        # actions
        # -------
        self.actions += [
                self.check_tasks,
                CmdAction(
                    '"' + os.path.join(self.study.config['opensim_home'],
                        'bin','scale') + '" -S %s' % (setup_fname),
                    cwd=scale_setup_task.results_scale_path),
                self.create_residual_actuators,
                ]

        # targets
        # -------
        self.output_model_fpath = scale_setup_task.output_model_fpath
        self.residual_actuators_fpath = os.path.join(
                self.study.config['results_path'], 'experiments',
                self.subject.rel_path, '%s_residual_actuators.xml' %
                self.subject.name)
        self.targets += [
                self.output_model_fpath,
                scale_setup_task.output_motion_fpath,
                scale_setup_task.output_markerset_fpath,
                self.residual_actuators_fpath,
                ]

    def check_tasks(self):
        """Lists tasks that are <apply>'d for markers that either
        don't exist in the model or are not in the TRC file.

        Also lists tasks for which there is data, but that are either not in
        the model or do not have an IK task.

        """
        from opensim import ScaleTool, Model, MarkerSet
        from perimysium.dataman import TRCFile

        scale = ScaleTool(self.setup_fpath)
        tasks = scale.getMarkerPlacer().getIKTaskSet()
        trc = TRCFile(self.marker_trajectories_fpath)
        trc_names = trc.marker_names
        model = Model(self.generic_model_fpath)
        markerset = MarkerSet(self.prescale_markerset_fpath)

        # Markers with IK tasks but without data.
        # ---------------------------------------
        markers_without_data = []
        for i in range(tasks.getSize()):
            task = tasks.get(i)
            name = task.getName()
            applied = task.getApply()

            if applied:
                if (not name in trc_names) or (not markerset.contains(name)):
                    if task.getConcreteClassName() != 'IKCoordinateTask':
                        markers_without_data.append(name)

        if markers_without_data != [] and not self.ignore_nonexistant_data:
            raise Exception('There are IK tasks for the following markers, '
                    'yet data does not exist for them: {}'.format(
                        markers_without_data))
        del name

        # Markers for which there is data but they're not specified elsewhere.
        # --------------------------------------------------------------------
        unused_markers = []
        for name in trc.marker_names:
            if (not markerset.contains(name)) or (not tasks.contains(name)):
                unused_markers.append(name)
        if unused_markers != []:
            raise Exception("You have data for the following markers, but "
                    "you are not using them in Scale's IK: {}".format(
                        unused_markers))

        # No data for these markers in the model or prescale markerset.
        # -------------------------------------------------------------
        excess_model_markers = []
        for im in range(markerset.getSize()):
            name = markerset.get(im).getName()
            if not tasks.contains(name):
                excess_model_markers.append(name)
        if excess_model_markers != []:
            raise Exception("The following model markers do not have tasks or "
                    "experimental data: {}".format(excess_model_markers))

    def create_residual_actuators(self):
        ft = open(self.residual_actuators_template)
        content = ft.read()
        content = content.replace('@STUDYNAME@', self.study.name)
        content = content.replace('@SUBJECTNAME@', self.subject.name)

        def com_in_pelvis():
            import opensim
            m = opensim.Model(self.output_model_fpath)
            init_state = m.initSystem()
            com_in_ground = m.calcMassCenterPosition(init_state)
            com_in_pelvis = opensim.Vec3()
            simbody_engine = m.getSimbodyEngine()
            simbody_engine.transformPosition(init_state,
                    m.getBodySet().get('ground'), com_in_ground,
                    m.getBodySet().get('pelvis'), com_in_pelvis)
            com_xmeas = str(com_in_pelvis.get(0))
            com_ymeas = str(com_in_pelvis.get(1))
            com_zmeas = str(com_in_pelvis.get(2))
            return com_xmeas, com_ymeas, com_zmeas

        com_xmeas, com_ymeas, com_zmeas = com_in_pelvis()

        content = content.replace('@SYSTEM_COM_GLOBAL_X_MEAS@', com_xmeas)
        content = content.replace('@SYSTEM_COM_GLOBAL_Y_MEAS@', com_ymeas)
        content = content.replace('@SYSTEM_COM_GLOBAL_Z_MEAS@', com_zmeas)

        ft.close()

        f = open(self.residual_actuators_fpath, 'w')
        f.write(content)
        f.close()


class TaskGRFGaitLandmarks(task.TrialTask):
    # TODO not actually a trial task if for treadmill...
    REGISTRY = []
    def __init__(self, trial,
                right_grfy_column_name='ground_force_r_vy',
                left_grfy_column_name='ground_force_l_vy',
                **kwargs):
        super(TaskGRFGaitLandmarks, self).__init__(trial)
        self.name = '%s_gait_landmarks' % trial.id
        self.doc = 'Plot vertical ground reaction force.'
        self.right_grfy_column_name = right_grfy_column_name
        self.left_grfy_column_name = left_grfy_column_name
        self.kwargs = kwargs
        self.add_action(
                [trial.ground_reaction_fpath],
                [os.path.join(trial.expdata_path, '..', '%s.pdf' % self.name)],
                self.save_gait_landmarks_fig)

    def save_gait_landmarks_fig(self, file_dep, target):
        from perimysium.postprocessing import gait_landmarks_from_grf
        import pylab as pl
        gait_landmarks_from_grf(file_dep[0],
                right_grfy_column_name=self.right_grfy_column_name,
                left_grfy_column_name=self.left_grfy_column_name,
                do_plot=True,
                **self.kwargs)
        pl.gcf().savefig(target[0])


class TaskIKSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial):
        super(TaskIKSetup, self).__init__('ik', trial)
        self.doc = 'Create a setup file for Inverse Kinematics.'

        # Fill out tasks.xml template and copy over to results directory
        self.copy_over_tasks()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.trial.id)
            content = content.replace('@MODEL@', 
                self.subject.scaled_model_fpath)
            content = content.replace('@TASKS@', self.results_tasks_fpath)
            content = content.replace('@MARKER_FILE@',
                self.trial.marker_trajectories_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
        
        with open(target[0], 'w') as f:
            f.write(content)

class TaskIK(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task):
        super(TaskIK, self).__init__('ik', trial)
        self.doc = "Run OpenSim's Inverse Kinematics tool."
        self.file_dep += [
                self.subject.scaled_model_fpath,
                ik_setup_task.results_tasks_fpath,
                ]
        self.targets += [
                os.path.join(self.path, '%s_%s_ik_solution.mot' % (
                    self.study.name, trial.id)),
                os.path.join(self.path, 'ik_model_marker_locations.sto'),
                ]

# TODO, abstract RRA setup task?: class TaskRRASetup(task.SetupTask):

class TaskRRAModelSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, 
                 adjust_body='torso',
                 model_to_adjust=None):
        super(TaskRRAModelSetup, self).__init__('rramodel', trial, 
            model_to_adjust=model_to_adjust)
        self.doc = "Create a setup file for the Residual Reduction Algorithm, to create an adjusted model."
        self.rra_act_fpath = os.path.join(self.doit_path, 
            self.study.config['rra_actuators'])
        self.res_act_fpath = os.path.join(self.subject.results_exp_path, 
            '%s_residual_actuators.xml' % self.subject.name)
        self.adjust_body = adjust_body

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.copy_over_external_loads()

        # Fill out tasks.xml template and copy over to results directory
        self.copy_over_tasks()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.trial.id)
            content = content.replace('@MODEL@', 
                                      self.model_to_adjust_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = '%s %s' % (self.rra_act_fpath, 
                                         self.res_act_fpath)
            content = content.replace('@FORCESETFILES@', force_set_files)
            content = content.replace('@ADJUSTCOMBODY@', self.adjust_body)
            # We always compute the mass change, but we just don't always USE
            # the resulting model.
            content = content.replace('@MODELADJUSTED@',
                   self.subject.name  + "_adjusted.osim")
        
        with open(target[0], 'w') as f:
            f.write(content)

# class TaskRRATasksSetup(task.SetupTask):
#     REGISTRY = []
#     def __init__(self, trial):
#         super(TaskRRATasksSetup, self).__init__(trial, 'rratasks')
#         self.doc = "Create a setup file for the Residual Reduction Algorithm tool to adjust kinematics, using IK kinematics."
#         self.rra_actuators = self.study.config['rra_actuators']

#         # Fill out external_loads.xml template and copy over to results 
#         # directory
#         self.copy_over_external_loads()

#         # Fill out tasks.xml template and copy over to results directory
#         self.copy_over_tasks()

#     def fill_setup_template(self, file_dep, target):
#         with open(file_dep[0]) as ft:
#             content = ft.read()
#             content = content.replace('@STUDYNAME@', self.study.name)
#             content = content.replace('@NAME@', self.trial.id)
#             content = content.replace('@MODEL@', 
#                                       self.adjusted_model_fpath)
#             content = content.replace('@INIT_TIME@', '%.4f' % self.init_time)
#             content = content.replace('@FINAL_TIME@', '%.4f' % self.final_time)
            
#             rra_act = os.path.join(self.doit_path, self.rra_actuators)
#             res_act = '../../%s_residual_actuators.xml' % self.subject.name
#             force_set_files = '%s %s' % (rra_act, res_act)
#             content = content.replace('@FORCESETFILES@', force_set_files)

#         with open(target[0], 'w') as f:
#             f.write(content)

class TaskRRAKinSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, adjusted_model=None):
        super(TaskRRAKinSetup, self).__init__('rrakin', trial,  
            adjusted_model=adjusted_model)
        self.doc = "Create a setup file for the Residual Reduction Algorithm tool to adjust kinematics."
        self.rra_act_fpath = os.path.join(self.doit_path, 
            self.study.config['rra_actuators'])
        self.res_act_fpath = os.path.join(self.subject.results_exp_path, 
            '%s_residual_actuators.xml' % self.subject.name)

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.copy_over_external_loads()

        # Fill out tasks.xml template and copy over to results directory
        self.copy_over_tasks()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.trial.id)
            content = content.replace('@MODEL@', 
                                      self.adjusted_model_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = '%s %s' % (self.rra_act_fpath, 
                                         self.res_act_fpath)
            content = content.replace('@FORCESETFILES@', force_set_files)

        with open(target[0], 'w') as f:
            f.write(content)

class TaskRRA(task.ToolTask):
    REGISTRY = []
    def __init__(self, tool_folder, trial):
        super(TaskRRA, self).__init__(tool_folder, trial, exec_name='rra')
        self.doc = "Abstract class for OpenSim's Residual Reduction Algorithm tool."
        self.des_kinematics_fpath = '%s/ik/%s_%s_ik_solution.mot' % (trial.results_exp_path, self.study.name, self.trial.id)
        self.des_kinetics_fpath = \
            '%s/expdata/ground_reaction_orig.mot' % trial.results_exp_path

        # Set file dependencies
        self.file_dep += [self.des_kinematics_fpath,
                          self.des_kinetics_fpath
                         ]

        # Set targets for all RRA outputs
        for rra_output in ['Actuation_force.sto',
                'Actuation_power.sto', 'Actuation_speed.sto',
                'avgResiduals.txt', 'controls.sto', 'controls.xml',
                'Kinematics_dudt.sto', 'Kinematics_q.sto', 'Kinematics_u.sto',
                'pErr.sto', 'states.sto']:
            self.targets += ['%s/results/%s_%s_%s_%s' % (
                self.path, self.study.name, trial.id, tool_folder, rra_output)]

class TaskRRAModel(TaskRRA):
    REGISTRY = []
    def __init__(self, trial, rramodel_setup_task, reenable_probes=False):
        super(TaskRRAModel, self).__init__('rramodel' ,trial)
        self.doc = "Run OpenSim's Residual Reduction Algorithm tool to create an adjusted model."

        # Set file dependencies
        self.file_dep += [rramodel_setup_task.results_setup_fpath,
                          rramodel_setup_task.results_tasks_fpath,
                          rramodel_setup_task.results_extloads_fpath, 
                          rramodel_setup_task.res_act_fpath,
                          ]
        # Set targets
        self.targets += [rramodel_setup_task.adjusted_model_fpath]

        # Deal with the fact that this operation would otherwise overwrite
        # the valuable RRA log.
        cur_log = '%s/out.log' % self.path
        rra_log = '%s/out_rra.log' % self.path

        def copylog():
            if os.path.exists(cur_log): shutil.copyfile(cur_log, rra_log)
        def removelog():
            if os.path.exists(cur_log): os.remove(cur_log)
        def renamelog():
            if os.path.exists(rra_log): os.rename(rra_log, cur_log)
        self.actions += [
                copylog,
                removelog,
                renamelog,
                ]

        if reenable_probes:
            self.actions += [self.reenable_probes]

    def reenable_probes(self, rramodel_setup_task):
        # RRA disables the probes; let's re-enable them.
        import opensim
        m = opensim.Model(rramodel_setup_task.adjusted_model_fpath)
        from perimysium.modeling import enable_probes
        enable_probes(rramodel_setup_task.adjusted_model_fpath)

class TaskRRAKin(TaskRRA):
    REGISTRY = []
    def __init__(self, trial, rrakin_setup_task):
        super(TaskRRAKin, self).__init__('rrakin', trial)
        self.doc = "Run OpenSim's Residual Reduction Algorithm tool to adjust kinematics, using the model from TaskRRAModel."

        # Set file dependencies
        self.file_dep += [
                rrakin_setup_task.adjusted_model_fpath,
                rrakin_setup_task.results_setup_fpath,
                rrakin_setup_task.results_tasks_fpath,
                rrakin_setup_task.results_extloads_fpath,
                ]

class TaskCMCSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, 
                 adjusted_model=None,
                 des_kinematics='rramodel', 
                 control_constraints=None,
                 gait_cycles='concatenated'):
        super(TaskCMCSetup, self).__init__('cmc', trial,  
            adjusted_model=adjusted_model, gait_cycles=gait_cycles)

        self.doc = "Create a setup file for Computed Muscle Control."
        self.cmc_act_fpath = os.path.join(self.doit_path, 
            self.study.config['cmc_actuators'])
        self.res_act_fpath = os.path.join(self.subject.results_exp_path, 
            '%s_residual_actuators.xml' % self.subject.name)
        self.des_kinematics = des_kinematics
        self.control_constraints = control_constraints
        self.gait_cycles=gait_cycles

        # Set desired kinematics path
        if self.des_kinematics=='rrakin':
            self.des_kinematics_fpath = os.path.join(trial.results_exp_path, 
                'rrakin', 'results',
                '%s_%s_rrakin_Kinematics_q.sto' % (self.study.name, trial.id))
        elif self.des_kinematics=='ik':
            self.des_kinematics_fpath = os.path.join(trial.results_exp_path,
                'ik', '%s_%s_ik_solution.mot' % (self.study.name, trial.id))
        else:
            raise Exception("TaskCMCSetup: %s is not a valid kinematics task "
                "source, please choose 'rrakin' or 'ik'." % des_kinematics)
        
        # Set control constraints path (if any)
        if self.control_constraints:
            self.control_constraints_fpath = 'TODO'
        else:
            self.control_constraints_fpath = ''

        # TODO
        # self.coord_act_fpath

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.copy_over_external_loads()

        # Fill out tasks.xml template and copy over to results directory
        self.copy_over_tasks()

    # Override derived action method since different desired kinematics
    # may be specified 
    def fill_external_loads_template(self, file_dep, target):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.trial.id)
            content = content.replace('@DESKINEMATICS@',
                                       self.des_kinematics_fpath)

        with open(target[0], 'w') as f:
            f.write(content)

    def fill_setup_template(self, file_dep, target, 
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.trial.id)
            content = content.replace('@MODEL@', 
                                      self.adjusted_model_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = '%s %s' % (self.cmc_act_fpath, 
                                         self.res_act_fpath)
            content = content.replace('@FORCESETFILES@', force_set_files)
            content = content.replace('@DESKINEMATICS@',
                                       self.des_kinematics_fpath)
            content = content.replace('@CONTROLCONSTRAINTS@',
                                       self.control_constraints_fpath)
        
        with open(target[0], 'w') as f:
            f.write(content)

class TaskCMC(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, cmc_setup_task, cycle=None):
        super(TaskCMC, self).__init__('cmc', trial, cycle=cycle)
        self.doc = "Run OpenSim's Computed Muscle Control tool."
        self.des_kinetics_fpath = \
            '%s/expdata/ground_reaction_orig.mot' % trial.results_exp_path

        if cycle:
            self.path = os.path.join(trial.results_exp_path, 'cmc', cycle.name)
        else:
            self.path = os.path.join(trial.results_exp_path, 'cmc')

        self.file_dep += [
            cmc_setup_task.adjusted_model_fpath,
            cmc_setup_task.results_extloads_fpath,
            cmc_setup_task.results_tasks_fpath,
            cmc_setup_task.des_kinematics_fpath,
            self.des_kinetics_fpath,
            cmc_setup_task.cmc_act_fpath,
            cmc_setup_task.res_act_fpath,
            ]

        if cmc_setup_task.control_constraints:
            self.file_dep += [self.control_contraints_fpath]

        # Set targets for all CMC outputs
        for cmc_output in ['Actuation_force.sto',
                'Actuation_power.sto', 'Actuation_speed.sto',
                'avgResiduals.txt', 'controls.sto', 'controls.xml',
                'Kinematics_dudt.sto', 'Kinematics_q.sto', 'Kinematics_u.sto',
                'pErr.sto', 'states.sto']:

            self.targets += ['%s/results/%s_%s_%s_%s' % (
                self.path, self.study.name, trial.id, 'cmc', cmc_output)]