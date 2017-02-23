
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
    
    Task name: `subject<num>_copy_data`"""
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
        # Use regular expressions to copy/rename files.
        import re

        # Check if each file in the mocap_dir matches any of the regular
        # expressions given to this task.
        for dirpath, dirnames, filenames in os.walk(mocap_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                fpath_rel_to_mocap = os.path.relpath(fpath, mocap_dir)
                for pattern, replacement in self.regex_replacements:
                    match = re.search(pattern, fpath_rel_to_mocap)
                    if match != None:
                        # Found at least one match.
                        destination = re.sub(pattern, replacement,
                                fpath_rel_to_mocap)
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

            


