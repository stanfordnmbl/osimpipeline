
import os
import task

class TaskCopyMotionCaptureData(task.SubjectTask):
    """This a very generic task for copying motion capture data (marker
    trajectories, ground reaction, electromyography) and putting it in
    place for creating simulations.

    You may want to create your own custom task(s) that is tailored to the
    organization of your experimental data.

    The other tasks expect an `expdata` folder in the condition folder (for
    treadmill trials) that contains `marker_trajectories.trc` and
    `ground_reaction.mot`.
    
    Task name: `subject<num>_copy_data`
    """
    # TODO Turn into a StudyTask, since there is very little about this task
    # that is subject-specific.
    REGISTRY = [] # TODO Find a way to make this unnecessary.
    def __init__(self, subject, regex_replacements):
        """Do not use this constructor directly; use `study.Subject.add_task()`.

        Parameters
        ----------
        subject : study.Subject
            This argument is provided internally by `study.Subject.add_task()`.
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
        subject.add_task(TaskCopyMotionCaptureData, [
                ('subject01/Data/Walk_100 02.trc',
                    'subject01/walk1/expdata/marker_trajectories.trc')])
        ```
        """
        super(TaskCopyMotionCaptureData, self).__init__(subject)
        self.name = '_'.join([subject.name, 'copy_data'])
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
    template will be replaced by the correct values.
    
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

        self.add_action(
                {'marker_traj':
                    self.mocap_trial.marker_trajectories_fpath,
                 'generic_model': self.study.generic_model_fpath,
                    },
                {'setup': os.path.join(self.results_scale_path, 'setup.xml')
                    },
                self.create_scale_setup)

        # MarkerSet for the Scale Tool.
        self.source_scale_path = os.path.join(self.subject.rel_path, 'scale')
        self.prescale_template_fpath = 'templates/scale/prescale_markerset.xml'
        self.prescale_fpath = os.path.join(self.source_scale_path, 
                '%s_prescale_markerset.xml' % self.subject.name)
        if not os.path.exists(self.prescale_fpath):
            # The user does not yet have a markerset in place; fill out the
            # template.
            self.add_action(
                    {'template': self.prescale_template_fpath},
                    {'subjspecific': self.prescale_fpath},
                    self.fill_prescale_markerset_template)
        else:
            # We have already filled out the template prescale markerset,
            # and the user might have made changes to it.
            self.file_dep.append(self.prescale_fpath)

        # TODO self.actions += [self.prescale_markerset_template]

        #self.prescale_rel_fpath = ('subject%02i_prescale_markerset.xml' %
        #        self.subj)
        #self.prescale_fpath = join(self.subj_str, self.prescale_rel_fpath)

        #if not os.path.exists(self.prescale_fpath):
        #    self.file_dep += [self.prescale_template_fpath]
        #    self.targets += [self.prescale_fpath]

        #else:
        #    self.file_dep += [self.prescale_fpath]

        self.file_dep += addtl_file_dep

    def fill_prescale_markerset_template(self, file_dep, target):
        # TODO
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
        gmm.setMarkerSetFileName(os.path.relpath(self.prescale_fpath,
            self.results_scale_path))

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
        placer.setOutputModelFileName('%s.osim' % (self.subject.name))
        placer.setOutputMotionFileName(
                '%s_%s_staticpose.mot' % (self.study.name, self.subject.name))
        placer.setOutputMarkerFileName(
                '%s_%s_markerset.xml' % (self.study.name, self.subject.name))
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
            


