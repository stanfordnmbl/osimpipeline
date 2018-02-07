
import os
import time
from doit.action import CmdAction
import shutil
import pylab as pl
import pandas as pd
import numpy as np
import opensim as osm

import task
import utilities as util
import postprocessing as pp


class TaskCopyGenericModelFilesToResults(task.StudyTask):
    REGISTRY = []
    def __init__(self, study):
        super(TaskCopyGenericModelFilesToResults, self).__init__(study)
        self.name = '%s_copy_generic_model_files' % study.name
        self.doc = 'Copy generic model to the results directory.'
        self.add_action(
                [study.source_generic_model_fpath],
                [study.generic_model_fpath],
                self.copy_file)

        self.add_action(
                [study.source_reserve_actuators_fpath],
                [study.reserve_actuators_fpath],
                self.copy_file)

        if study.source_rra_actuators_fpath:
            self.add_action(
                [study.source_rra_actuators_fpath],
                [study.rra_actuators_fpath],
                self.copy_file)

        if study.source_cmc_actuators_fpath:
            self.add_action(
                [study.source_cmc_actuators_fpath],
                [study.cmc_actuators_fpath],
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

        # self.add_action(self.registry.keys(), self.registry.values(),
        #         self.copy_files)
        # May want to copy over files repeatedly during data processing,
        # so get rid of dependencies for now.
        self.actions += [self.copy_files]

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

    def copy_files(self):
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
        ikts = util.IKTaskSet(placer.getIKTaskSet())

        self.edit_setup_function(util, mset, sset, ikts)

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
        scale = osm.ScaleTool(self.setup_fpath)
        tasks = scale.getMarkerPlacer().getIKTaskSet()
        trc = util.TRCFile(self.marker_trajectories_fpath)
        trc_names = trc.marker_names
        model = osm.Model(self.generic_model_fpath)
        markerset = osm.MarkerSet(self.prescale_markerset_fpath)

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
                threshold=5,
                **kwargs):
        super(TaskGRFGaitLandmarks, self).__init__(trial)
        self.name = '%s_gait_landmarks' % trial.id
        self.doc = 'Plot vertical ground reaction force.'
        self.right_grfy_column_name = right_grfy_column_name
        self.left_grfy_column_name = left_grfy_column_name
        self.kwargs = kwargs
        self.threshold = threshold
        self.add_action(
                [trial.ground_reaction_fpath],
                [os.path.join(trial.expdata_path, '..', '%s.pdf' % self.name)],
                self.save_gait_landmarks_fig)

    def save_gait_landmarks_fig(self, file_dep, target):
        util.gait_landmarks_from_grf(file_dep[0],
                right_grfy_column_name=self.right_grfy_column_name,
                left_grfy_column_name=self.left_grfy_column_name,
                threshold=self.threshold,
                do_plot=True,
                **self.kwargs)
        pl.gcf().savefig(target[0])


class TaskIKSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, **kwargs):
        super(TaskIKSetup, self).__init__('ik', trial, **kwargs)
        self.doc = 'Create a setup file for Inverse Kinematics.'
        self.solution_fpath = os.path.join(self.path, 
            '%s_%s_ik_solution.mot' % (self.study.name, self.tricycle.id))
        self.model_markers_fpath = os.path.join(self.path, 
            'ik_model_marker_locations.sto')

        # Fill out tasks.xml template and copy over to results directory
        self.create_tasks_action()

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@', 
                os.path.relpath(self.subject.scaled_model_fpath, self.path))
            content = content.replace('@MARKER_FILE@',
                os.path.relpath(self.trial.marker_trajectories_fpath, 
                    self.path))
            content = content.replace('@TASKS@', os.path.relpath(
                self.results_tasks_fpath, self.path))
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
        
        with open(target[0], 'w') as f:
            f.write(content)


class TaskIK(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task, **kwargs):
        super(TaskIK, self).__init__(ik_setup_task, trial, **kwargs)
        self.doc = "Run OpenSim's Inverse Kinematics tool."
        
        self.file_dep += [
                self.subject.scaled_model_fpath,
                ik_setup_task.results_tasks_fpath,
                ik_setup_task.results_setup_fpath
                ]
        self.targets += [
                ik_setup_task.solution_fpath,
                ik_setup_task.model_markers_fpath
                ]


class TaskIKPost(task.PostTask):
    REGISTRY=[]
    def __init__(self, trial, ik_setup_task, error_markers=None, side=None,
             **kwargs):
        super(TaskIKPost, self).__init__(ik_setup_task, trial, **kwargs)
        self.doc = 'Create plots from the results of Inverse Kinematics.'
        self.joint_angles_plotpath = '%s/joint_angles.pdf' % self.path
        self.marker_errors_plotpath = '%s/marker_error.pdf' % self.path
        self.error_markers = error_markers
        self.side = side

        self.add_action([ik_setup_task.solution_fpath],
                        [self.joint_angles_plotpath],
                        self.joint_angle_plots)

        if self.error_markers:
            self.add_action([self.subject.scaled_model_fpath, 
                            ik_setup_task.model_markers_fpath,
                            self.trial.marker_trajectories_fpath],
                            [self.marker_errors_plotpath],
                            self.marker_error_plots)

    def joint_angle_plots(self, file_dep, target):
        # if os.path.exists(self.fig_fpath):
        #     os.rename(self.fig_fpath,
        #             self.fig_fpath.replace('.pdf', '_backup.pdf'))
        fig = pp.plot_lower_limb_kinematics(file_dep[0], self.gl, 
            side=self.side)
        fig.savefig(target[0])
        pl.close(fig)

    def marker_error_plots(self, file_dep, target):
        # if os.path.exists(self.errorplot_fpath):
        #     os.rename(self.errorplot_fpath,
        #             self.errorplot_fpath.replace('.pdf', '_backup.pdf'))
        pp.plot_marker_error(target[0], self.error_markers, 
            10, self.gl, file_dep[0], file_dep[1], file_dep[2])

    
class TaskIDSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task, **kwargs):
        super(TaskIDSetup, self).__init__('id', trial, **kwargs)
        self.doc = 'Create a setup file for Inverse Dynamics.'
        self.ik_setup_task = ik_setup_task
        self.rel_kinematics_fpath = os.path.relpath(
            ik_setup_task.solution_fpath, self.path)
        self.solution_fpath = os.path.join(
            self.path, 'results','%s_%s_id_solution.sto' % (
            self.study.name, trial.id))

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.create_external_loads_action(self.rel_kinematics_fpath)

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@', 
                os.path.relpath(self.subject.scaled_model_fpath, self.path))
            content = content.replace('@COORDINATES_FILE@',
                self.rel_kinematics_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
        
        with open(target[0], 'w') as f:
            f.write(content)

class TaskID(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, id_setup_task, **kwargs):
        super(TaskID, self).__init__(id_setup_task, trial, **kwargs)
        self.doc = "Run OpenSim's Inverse Dynamics tool."
        self.ik_setup_task = id_setup_task.ik_setup_task
        self.file_dep += [
                self.subject.scaled_model_fpath,
                id_setup_task.results_extloads_fpath,
                id_setup_task.results_setup_fpath,
                self.ik_setup_task.solution_fpath
                ]
        self.targets += [
                id_setup_task.solution_fpath
                ]

class TaskIDPost(task.PostTask):
    REGISTRY = []
    def __init__(self, trial, id_setup_task, **kwargs):
        super(TaskIDPost, self).__init__(id_setup_task, trial, **kwargs)
        self.doc = 'Create plots from the results of Inverse Dynamics.'
        self.trial = trial
        self.ik_setup_task = id_setup_task.ik_setup_task
        self.id_solution_fpath = id_setup_task.solution_fpath
        self.file_dep += [
                self.id_solution_fpath
            ]
        self.actions += [
                self.cycle_joint_torque_plots
            ]

    def cycle_joint_torque_plots(self):

        id_array = util.storage2numpy(self.id_solution_fpath)

        for cycle in self.trial.cycles:
            fname = 'joint_torques_cycle%02d.pdf' % cycle.num
            output_filepath = os.path.join(self.path, fname)

            pp.plot_gait_torques(output_filepath, id_array, 
                self.trial.primary_leg, cycle.start, cycle.end,
                cycle.gl.right_strike, cycle.gl.left_strike, 
                toeoff_time=cycle.gl.right_toeoff)

class TaskSOSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, ik_setup_task, **kwargs):
        super(TaskSOSetup, self).__init__('so', trial, **kwargs)
        self.doc = 'Create a setup file for Static Optimization.'
        self.ik_setup_task = ik_setup_task
        self.kinematics_fpath = ik_setup_task.solution_fpath
        self.rel_kinematics_fpath = os.path.relpath(
            self.kinematics_fpath, self.path)
        self.solution_fpath = os.path.join(
            self.path, 'results','%s_%s_so_StaticOptimization_activation.sto' % (
            self.study.name, self.tricycle.id))

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.create_external_loads_action(self.rel_kinematics_fpath)

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@', 
                os.path.relpath(self.subject.scaled_model_fpath, self.path))
            content = content.replace('@COORDINATES_FILE@',
                self.rel_kinematics_fpath)
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = [
                    os.path.relpath(
                        os.path.join(self.study.config['results_path'],
                            'experiments',
                            self.subject.rel_path, 
                            '%s_residual_actuators.xml' % self.subject.name),
                        self.path),
                    os.path.relpath(
                        self.study.reserve_actuators_fpath, self.path),
                    ]
            force_set_str = ' '.join(force_set_files)
            content = content.replace('@FORCE_SET_FILES@', force_set_str)
        
        with open(target[0], 'w') as f:
            f.write(content)

class TaskSO(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, so_setup_task, **kwargs):
        super(TaskSO, self).__init__(so_setup_task, trial, exec_name='analyze',
            **kwargs)
        self.doc = "Run OpenSim's Static Optimization tool."

        self.file_dep += [
                self.subject.scaled_model_fpath,
                self.study.reserve_actuators_fpath,
                so_setup_task.kinematics_fpath,
                so_setup_task.results_extloads_fpath,
                ]
        self.targets += [
                so_setup_task.solution_fpath,
                ]

class TaskSOPost(task.PostTask):
    REGISTRY = []
    def __init__(self, trial, so_setup_task, **kwargs):
        super(TaskSOPost, self).__init__(so_setup_task, trial, **kwargs)
        self.doc = "Create plots from the results of Static Optimization."
        self.cycle = so_setup_task.cycle

        # Generate muscle activations plots from SO
        self.add_action([so_setup_task.solution_fpath],
                        [os.path.join(self.path, 'activations.pdf')],
                        self.plot_activations)

    def plot_activations(self, file_dep, target):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from copy import copy

        residuals = ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']

        act = util.storage2numpy(file_dep[0])
        names = copy(residuals)
        # To help with plotting left and right actuators on same axes.
        for name in act.dtype.names:
            if name.endswith('_r'):
                names.append(name[:-2])
        n_subplots = len(names)
        n_cols = 6
        n_rows = n_subplots / n_cols + 1

        fig = plt.figure(figsize=(4.5 * n_cols, 4 * n_rows))
        grid = gridspec.GridSpec(n_rows, n_cols)
        i_row = 0
        i_col = 0

        for name in names:
            ax = plt.Subplot(fig, grid[i_row, i_col])
            fig.add_subplot(ax)

            if name in residuals:
                pp.plot_pgc(act['time'], act[name], self.cycle.gl,
                        side=self.cycle.gl.primary_leg,
                        axes=ax)
            else:
                for s in ['left', 'right']:
                    pp.plot_pgc(act['time'], act['%s_%s' % (name, s[0])],
                            self.cycle.gl, side=s, axes=ax)

            ax.set_title(name)
            ax.set_xlim(0, 100)

            if not (name in residuals or name.startswith('reserve_')):
                # This is a muscle; set the ylims to [0, 1].
                ax.set_ylim(0, 1)

            if i_col < (n_cols - 1):
                i_col += 1
            else:
                i_col = 0
                i_row += 1
        plt.tight_layout()
        fig.savefig(target[0])
        plt.close(fig)

class TaskRRAModelSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, adjust_body='torso', **kwargs):
        super(TaskRRAModelSetup, self).__init__('rramodel', trial, **kwargs)
        self.doc = "Create a setup file for the Residual Reduction Algorithm, to create an adjusted model."
        self.adjust_body = adjust_body

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.create_external_loads_action()

        # Fill out tasks.xml template and copy over to results directory
        self.create_tasks_action()

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@', 
                os.path.relpath(self.trial.model_to_adjust_fpath, self.path))
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = '%s %s' % (
                os.path.relpath(self.study.rra_actuators_fpath, self.path), 
                os.path.relpath(self.subject.residual_actuators_fpath, 
                    self.path))
            content = content.replace('@FORCESETFILES@', force_set_files)
            content = content.replace('@ADJUSTCOMBODY@', self.adjust_body)
            # We always compute the mass change, but we just don't always USE
            # the resulting model.
            content = content.replace('@MODELADJUSTED@',
                   self.adjusted_model)
        
        with open(target[0], 'w') as f:
            f.write(content)

class TaskRRAKinSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, **kwargs):
        super(TaskRRAKinSetup, self).__init__('rrakin', trial, **kwargs)
        self.doc = "Create a setup file for the Residual Reduction Algorithm tool to adjust kinematics."

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.create_external_loads_action()

        # Fill out tasks.xml template and copy over to results directory
        self.create_tasks_action()

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@', 
                os.join.relpath(self.adjusted_model_fpath, self.path))
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = '%s %s' % (
                os.path.relpath(self.study.rra_actuators_fpath, self.path), 
                os.path.relpath(self.subject.residual_actuators_fpath, 
                    self.path))
            content = content.replace('@FORCESETFILES@', force_set_files)

        with open(target[0], 'w') as f:
            f.write(content)

class TaskRRA(task.ToolTask):
    REGISTRY = []
    def __init__(self, setup_task, trial, **kwargs):
        kwargs['exec_name'] = 'rra'
        super(TaskRRA, self).__init__(setup_task, trial, **kwargs)
        self.doc = "Abstract class for OpenSim's RRA tool."
        self.des_kinematics_fpath = '%s/ik/%s_%s_ik_solution.mot' % (
            trial.results_exp_path, self.study.name, setup_task.tricycle.id)
        self.des_kinetics_fpath = \
            '%s/expdata/ground_reaction_orig.mot' % trial.results_exp_path

        # Set file dependencies
        self.file_dep += [
                self.des_kinematics_fpath,
                self.des_kinetics_fpath,
                self.subject.residual_actuators_fpath,
                self.study.rra_actuators_fpath
                ]

        # Set targets for all RRA outputs
        for rra_output in ['Actuation_force.sto',
                'Actuation_power.sto', 'Actuation_speed.sto',
                'avgResiduals.txt', 'controls.sto', 'controls.xml',
                'Kinematics_dudt.sto', 'Kinematics_q.sto', 'Kinematics_u.sto',
                'pErr.sto', 'states.sto']:
            self.targets += ['%s/results/%s_%s_%s_%s' % (
                self.path, self.study.name, setup_task.tricycle.id,
                setup_task.tool, rra_output)]

class TaskRRAModel(TaskRRA):
    REGISTRY = []
    def __init__(self, trial, rramodel_setup_task, reenable_probes=False, 
            **kwargs):
        super(TaskRRAModel, self).__init__(rramodel_setup_task, trial, 
            **kwargs)
        self.doc = "Run OpenSim's RRA tool to create an adjusted model."

        # Set common file dependencies
        self.file_dep += [
            trial.model_to_adjust_fpath,
            rramodel_setup_task.results_setup_fpath,
            rramodel_setup_task.results_tasks_fpath,
            rramodel_setup_task.results_extloads_fpath, 
            ]

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
        util.enable_probes(rramodel_setup_task.adjusted_model_fpath)

class TaskRRAKin(TaskRRA):
    REGISTRY = []
    def __init__(self, trial, rrakin_setup_task, **kwargs):
        super(TaskRRAKin, self).__init__(rrakin_setup_task, trial, **kwargs)
        self.doc = "Run OpenSim's RRA tool to adjust model kinematics"
            
        # Set file dependencies
        self.file_dep += [
                rrakin_setup_task.adjusted_model_fpath,
                rrakin_setup_task.results_setup_fpath,
                rrakin_setup_task.results_tasks_fpath,
                rrakin_setup_task.results_extloads_fpath
                ]

class TaskCMCSetup(task.SetupTask):
    REGISTRY = []
    def __init__(self, trial, des_kinematics='rrakin', 
                 control_constraints=None, **kwargs):
        super(TaskCMCSetup, self).__init__('cmc', trial, **kwargs)
        self.doc = "Create a setup file for Computed Muscle Control."
        self.des_kinematics = des_kinematics
        self.control_constraints = control_constraints

        # Set desired kinematics path
        if self.des_kinematics=='rrakin':
            self.des_kinematics_fpath = os.path.join(trial.results_exp_path, 
                'rrakin', 'results',
                '%s_%s_rrakin_Kinematics_q.sto' % (self.study.name, 
                    self.tricycle.id))
        elif self.des_kinematics=='ik':
            self.des_kinematics_fpath = os.path.join(trial.results_exp_path,
                'ik', '%s_%s_ik_solution.mot' % (self.study.name, 
                    self.tricycle.id))
        else:
            raise Exception("TaskCMCSetup: %s is not a valid kinematics task "
                "source, please choose 'rrakin' or 'ik'." % des_kinematics)
        
        # Set control constraints path (if any)
        if self.control_constraints:
            raise Exception("TaskCMCSetup: control constraints have yet to be "
                "implemented.")
        else:
            self.control_constraints_fpath = ''

        # TODO
        # self.coord_act_fpath

        # Fill out external_loads.xml template and copy over to results 
        # directory
        self.create_external_loads_action()

        # Fill out tasks.xml template and copy over to results directory
        self.create_tasks_action()

        # Fill out setup.xml template and write to results directory
        self.create_setup_action()

    # Override derived action method since different desired kinematics
    # may be specified 
    def fill_external_loads_template(self, file_dep, target):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@DESKINEMATICS@', 
                os.join.relpath(self.des_kinematics_fpath, self.path))

        with open(target[0], 'w') as f:
            f.write(content)

    def fill_setup_template(self, file_dep, target, 
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@MODEL@', 
                os.path.relpath(self.adjusted_model_fpath, self.path))
            content = content.replace('@INIT_TIME@', '%.4f' % init_time)
            content = content.replace('@FINAL_TIME@', '%.4f' % final_time)
            force_set_files = '%s %s' % (
                os.path.relpath(self.study.cmc_actuators_fpath, self.path), 
                os.path.relpath(self.subject.residual_actuators_fpath, 
                    self.path))
            content = content.replace('@FORCESETFILES@', force_set_files)
            content = content.replace('@DESKINEMATICS@',
                os.path.relpath(self.des_kinematics_fpath, self.path))
            content = content.replace('@CONTROLCONSTRAINTS@', 
                os.path.relpath(self.control_constraints_fpath, self.path))
        
        with open(target[0], 'w') as f:
            f.write(content)

class TaskCMC(task.ToolTask):
    REGISTRY = []
    def __init__(self, trial, cmc_setup_task, **kwargs):
        super(TaskCMC, self).__init__(cmc_setup_task, trial, **kwargs)
        self.doc = "Run OpenSim's Computed Muscle Control tool."
        self.des_kinetics_fpath = \
            '%s/expdata/ground_reaction_orig.mot' % trial.results_exp_path

        self.file_dep += [
            cmc_setup_task.adjusted_model_fpath,
            cmc_setup_task.results_setup_fpath,
            cmc_setup_task.results_extloads_fpath,
            cmc_setup_task.results_tasks_fpath,
            cmc_setup_task.des_kinematics_fpath,
            self.des_kinetics_fpath,
            self.subject.residual_actuators_fpath,
            self.study.cmc_actuators_fpath
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
                self.path, self.study.name, cmc_setup_task.tricycle.id, 
                'cmc', cmc_output)]