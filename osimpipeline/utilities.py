
import os
import sys
#import opensim as osim
import numpy as np
import pandas as pd
import pylab as pl
import warnings
import h5py
import opensim as osim

class working_directory():
    """Use this to temporarily run code with some directory as a working
    directory and to then return to the original working directory::

        with working_directory('<dir>'):
            pass
    """
    def __init__(self, path):
        self.path = path
        self.original_working_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.path)
    def __exit__(self, *exc_info):
        os.chdir(self.original_working_dir)

def toarray(simtk_vector):
    array = np.empty(simtk_vector.size())
    for i in range(simtk_vector.size()):
        array[i] = simtk_vector[i]
    return array

def publication_spines(axes):
    axes.spines['right'].set_visible(False)
    axes.yaxis.set_ticks_position('left')
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.tick_params(axis='both', direction='in')

def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.

    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.

    Returns
    -------
    data : np.ndarry (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.

    Examples
    --------
    Columns from the storage file can be obtained as follows:

        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']

    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

def ndarray2storage(ndarray, storage_fpath, name=None, in_degrees=False):
    """Saves an ndarray, with named dtypes, to an OpenSim Storage file.
    Parameters
    ----------
    ndarray : numpy.ndarray
    storage_fpath : str
    in_degrees : bool, optional
    name : str
        Name of Storage object.
    """
    n_rows = ndarray.shape[0]
    n_cols = len(ndarray.dtype.names)

    f = open(storage_fpath, 'w')
    f.write('%s\n' % (name if name else storage_fpath,))
    f.write('version=1\n')
    f.write('nRows=%i\n' % n_rows)
    f.write('nColumns=%i\n' % n_cols)
    f.write('inDegrees=%s\n' % ('yes' if in_degrees else 'no',))
    f.write('endheader\n')
    for line_num, col in enumerate(ndarray.dtype.names):
        if line_num != 0:
            f.write('\t')
        f.write('%s' % col)
    f.write('\n')

    for i_row in range(n_rows):
        for line_num, col in enumerate(ndarray.dtype.names):
            if line_num != 0:
                f.write('\t')
            f.write('%f' % ndarray[col][i_row])
        f.write('\n')

    f.close()

def filter_critically_damped(data, sampling_rate, lowpass_cutoff_frequency,
        order=4):
    """See Robertson, 2003. This code is transcribed from some MATLAB code that
    Amy Silder gave me. This implementation is slightly different from that
    appearing in Robertson, 2003. We only allow lowpass filtering.
    Parameters
    ----------
    data : array_like
        The signal to filter.
    sampling_rate : float
    lowpass_cutoff_frequency : float
        In Hertz (not normalized).
    order : int, optional
        Number of filter passes.
    Returns
    -------
    data : array_like
        Filtered data.
    """
    # 3 dB cutoff correction.
    Clp = (2.0 ** (1.0 / (2.0 * order)) - 1.0) ** (-0.5)

    # Corrected cutoff frequency.
    flp = Clp * lowpass_cutoff_frequency / sampling_rate

    # Warp cutoff frequency from analog to digital domain.
    wolp = np.tan(np.pi * flp)

    # Filter coefficients, K1 and K2.
    # lowpass: a0 = A0, a1 = A1, a2 = A2, b1 = B2, b2 = B2
    K1lp = 2.0 * wolp
    K2lp = wolp ** 2

    # Filter coefficients.
    a0lp = K2lp / (1.0 + K1lp + K2lp)
    a1lp = 2.0 * a0lp
    a2lp = a0lp
    b1lp = 2.0 * a0lp  * (1.0 / K2lp - 1.0)
    b2lp = 1.0 - (a0lp + a1lp + a2lp + b1lp)

    num_rows = len(data)
    temp_filtered = np.zeros(num_rows)
    # For order = 4, we go forward, backward, forward, backward.
    for n_pass in range(order):
        for i in range(2, num_rows):
            temp_filtered[i] = (a0lp * data[i] +
                    a1lp * data[i - 1] +
                    a2lp * data[i - 2] +
                    b1lp * temp_filtered[i - 1] +
                    b2lp * temp_filtered[i - 2])
        # Perform the filter backwards.
        data = np.flipud(temp_filtered)
        temp_filtered = np.zeros(num_rows)

    return data

def plot_joint_moment_breakdown(model, moco_traj,
                                coord_paths, muscle_paths=None,
                                coordact_paths=[]):
    model.initSystem()

    num_coords = len(coord_paths)

    if not muscle_paths:
        muscle_paths = list()
        for muscle in model.getMuscleList():
            muscle_paths.append(muscle.getAbsolutePathString())
    num_muscles = len(muscle_paths)

    num_coordact = len(coordact_paths)


    net_joint_moments = None
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        id_tool = osim.InverseDynamicsTool()
        modelID = osim.Model(model)
        id_tool.setModel(modelID)
        table = moco_traj.exportToStatesTable()
        labels = list(table.getColumnLabels())
        import re
        for ilabel in range(len(labels)):
            labels[ilabel] = labels[ilabel].replace('/value', '')
            labels[ilabel] = re.sub('/jointset/(.*?)/', '', labels[ilabel])
        table.setColumnLabels(labels)

        storage = osim.Storage()
        stolabels = osim.ArrayStr()
        stolabels.append('time')
        for label in labels:
            stolabels.append(label)
        storage.setColumnLabels(stolabels)

        times = table.getIndependentColumn()
        for i_time in np.arange(table.getNumRows()):
            rowVector = osim.RowVector(table.getRowAtIndex(int(i_time)))
            storage.append(times[i_time], rowVector.transpose())
        
        id_tool.setCoordinateValues(storage)
        # id_tool.setExternalLoadsFileName(extloads_fpath)
        excludedForces = osim.ArrayStr()
        excludedForces.append('ACTUATORS')
        id_tool.setExcludedForces(excludedForces)
        id_result = 'joint_moment_breakdown_residuals.sto'
        id_tool.setResultsDir(tmpdirname)
        id_tool.setOutputGenForceFileName(id_result)
        # TODO: Remove muscles from the model?
        id_tool.run()

        net_joint_moments = osim.TimeSeriesTable(
            os.path.join(tmpdirname, id_result))

    time = moco_traj.getTimeMat()

    states_traj = moco_traj.exportToStatesTrajectory(model)

    # TODO for models without activation dynamics, we must prescribeControlsToModel().

    fig = pl.figure(figsize=(8.5, 11))
    tendon_forces = np.empty((len(time), num_muscles))
    for imusc, muscle_path in enumerate(muscle_paths):
        muscle = model.getComponent(muscle_path)
        for itime in range(len(time)):
            state = states_traj.get(itime)
            model.realizeDynamics(state)
            tendon_forces[itime, imusc] = muscle.getTendonForce(state)


    coordact_moments = np.empty((len(time), num_coordact))
    for ica, coordact_paths in enumerate(coordact_paths):
        coordact = model.getComponent(coordact_paths)
        for itime in range(len(time)):
            state = states_traj.get(itime)
            model.realizeDynamics(state)
            coordact_moments[itime, ica] = coordact.getActuation(state)


    for icoord, coord_path in enumerate(coord_paths):
        coord = model.getComponent(coord_path)

        label = os.path.split(coord_path)[-1] + '_moment'
        net_moment = toarray(net_joint_moments.getDependentColumn(label))

        moment_arms = np.empty((len(time), num_muscles))
        for imusc, muscle_path in enumerate(muscle_paths):
            muscle = model.getComponent(muscle_path)
            for itime in range(len(time)):
                state = states_traj.get(itime)
                moment_arms[itime, imusc] = \
                    muscle.computeMomentArm(state, coord)

        ax = fig.add_subplot(num_coords, 1, icoord + 1)
        net_integ = np.trapz(np.abs(net_moment), x=time)
        sum_actuators_shown = np.zeros_like(time)
        for imusc, muscle_path in enumerate(muscle_paths):
            if np.any(moment_arms[:, imusc]) > 0.00001:
                this_moment = tendon_forces[:, imusc] * moment_arms[:, imusc]
                mom_integ = np.trapz(np.abs(this_moment), time)
                if mom_integ > 0.01 * net_integ:
                    ax.plot(time, this_moment, label=muscle_path)

                    sum_actuators_shown += this_moment

        for ica, coordact_path in enumerate(coordact_paths):
            this_moment = coordact_moments[:, ica]
            ax.plot(time, this_moment, label=coordact_path)
            sum_actuators_shown += this_moment

        ax.plot(time, sum_actuators_shown,
                label='sum actuators shown', color='gray', linewidth=2)

        ax.plot(time, net_moment,
                label='net', color='black', linewidth=2)

        ax.set_title(coord_path)
        ax.set_ylabel('moment (N-m)')
        ax.legend(frameon=False, bbox_to_anchor=(1, 1),
                  loc='upper left', ncol=2)
        ax.tick_params(axis='both')
    ax.set_xlabel('time (% gait cycle)')

    fig.tight_layout()
    return fig

class GaitLandmarks(object):
    def __init__(self,
            primary_leg=None,
            cycle_start=None,
            cycle_end=None,
            left_strike=None,
            left_toeoff=None,
            right_strike=None,
            right_toeoff=None):
        self.primary_leg  = primary_leg
        self.cycle_start  = cycle_start
        self.cycle_end    = cycle_end
        self.left_strike  = left_strike
        self.left_toeoff  = left_toeoff
        self.right_strike = right_strike
        self.right_toeoff = right_toeoff

    def cycle_duration(self):
        return self.cycle_end - self.cycle_start

def enable_probes(model_fpath):
    """Ensures that all probes are enabled (isDisabled is false). Writes over
    the given model file.

    Parameters
    ----------
    model_fpath : str
        Path to a model (.OSIM) file.

    """
    model = osim.Model(model_fpath)
    n_probes = model.getProbeSet().getSize()
    for i_probe in range(n_probes):
        model.updProbeSet().get(i_probe).setDisabled(False)
    printobj(model, model_fpath)

def gait_landmarks_from_grf(mot_file,
        right_grfy_column_name='ground_force_vy',
        left_grfy_column_name='1_ground_force_vy',
        threshold=1e-5,
        do_plot=False,
        min_time=None,
        max_time=None,
        plot_width=6,
        show_legend=True,
        ):
    """
    Obtain gait landmarks (right and left foot strike & toe-off) from ground
    reaction force (GRF) time series data.

    Parameters
    ----------
    mot_file : str
        Name of *.mot (OpenSim Storage) file containing GRF data.
    right_grfy_column_name : str, optional
        Name of column in `mot_file` containing the y (vertical) component of
        GRF data for the right leg.
    left_grfy_column_name : str, optional
        Same as above, but for the left leg.
    threshold : float, optional
        Below this value, the force is considered to be zero (and the
        corresponding foot is not touching the ground).
    do_plot : bool, optional (default: False)
        Create plots of the detected gait landmarks on top of the vertical
        ground reaction forces.
    min_time : float, optional
        If set, only consider times greater than `min_time`.
    max_time : float, optional
        If set, only consider times greater than `max_time`.
    plot_width : float, optional
        If plotting, this is the width of the plotting window in inches.
    show_legend : bool, optional
        If plotting, show a legend.
    Returns
    -------
    right_foot_strikes : np.array
        All times at which right_grfy is non-zero and it was 0 at the preceding
        time index.
    left_foot_strikes : np.array
        Same as above, but for the left foot.
    right_toe_offs : np.array
        All times at which left_grfy is 0 and it was non-zero at the preceding
        time index.
    left_toe_offs : np.array
        Same as above, but for the left foot.

    """
    data = storage2numpy(mot_file)

    time = data['time']
    right_grfy = data[right_grfy_column_name]
    left_grfy = data[left_grfy_column_name]

    def nearest_index(time, index_time):
        return np.argmin(np.abs(time-index_time))

    # Time range to consider.
    if max_time == None: max_idx = len(time)
    else: max_idx = nearest_index(time, max_time)

    if min_time == None: min_idx = 1
    else: min_idx = max(1, nearest_index(time, min_time))

    index_range = range(min_idx, max_idx)

    # Helper functions
    # ----------------
    def zero(number):
        return abs(number) < threshold

    def birth_times(ordinate):
        births = list()
        for i in index_range:
            # 'Skip' first value because we're going to peak back at previous
            # index.
            if zero(ordinate[i - 1]) and (not zero(ordinate[i])):
                births.append(time[i])
        return np.array(births)

    def death_times(ordinate):
        deaths = list()
        for i in index_range:
            if (not zero(ordinate[i - 1])) and zero(ordinate[i]):
                deaths.append(time[i])
        return np.array(deaths)

    right_foot_strikes = birth_times(right_grfy)
    left_foot_strikes = birth_times(left_grfy)
    right_toe_offs = death_times(right_grfy)
    left_toe_offs = death_times(left_grfy)

    if do_plot:

        pl.figure(figsize=(plot_width, 6))
        ones = np.array([1, 1])

        def myplot(index, label, ordinate, foot_strikes, toe_offs):
            ax = pl.subplot(2, 1, index)
            pl.plot(time[min_idx:max_idx], ordinate[min_idx:max_idx], 'k')
            pl.ylabel('vertical ground reaction force (N)')
            pl.title('%s (%i foot strikes, %i toe-offs)' % (
                label, len(foot_strikes), len(toe_offs)))

            for i, strike in enumerate(foot_strikes):
                if i == 0: kwargs = {'label': 'foot strikes'}
                else: kwargs = dict()
                pl.plot(strike * ones, ax.get_ylim(), 'r', **kwargs)
                pl.text(strike, .03 * ax.get_ylim()[1], ' %.3f' % round(strike,
                    3))

            for i, off in enumerate(toe_offs):
                if i == 0: kwargs = {'label': 'toe-offs'}
                else: kwargs = dict()
                pl.plot(off * ones, ax.get_ylim(), 'b', **kwargs)
                pl.text(off, .03 * ax.get_ylim()[1], ' %.3f' % round(off, 3))

        # We'll place the legend on the plot with less strikes.
        n_left = len(left_toe_offs) + len(left_foot_strikes)
        n_right = len(right_toe_offs) + len(right_foot_strikes)

        myplot(1, 'left foot', left_grfy, left_foot_strikes, left_toe_offs)

        if show_legend and n_left <= n_right:
            pl.legend(loc='best')

        myplot(2, 'right foot', right_grfy, right_foot_strikes, right_toe_offs)

        if show_legend and n_left > n_right:
            pl.legend(loc='best')

        pl.xlabel('time (s)')

    return right_foot_strikes, left_foot_strikes, right_toe_offs, left_toe_offs


def hdf2pandas(filename, fieldname, type=float, labels=None, index=None):
    """A function to extract data from HDF5 files into a useable format for
    scripting, in this case a Pandas data structure. This function may be used
    to import MATLAB files (.mat) provided that they are saved as version 7.3,
    to ensure HDF compatibility.

    Parameters
    ----------
    filename : str
        Filename including full (or relative) path to HDF file.
    fieldname : str
        Path to desired field in HDF file. Traversing layers of file is done by
        using forward slashes, e.g. 'structLevel1/structLevel2/desiredData'.
    type : (optional)
        Specify data type in HDF file. Only used for data extraction in the 
        special case for strings.
    labels : list<string>
        List of strings containing labels to be set as the index in the case 
        of a 1D HDF to pandas.Series conversion, or set as the column labels
        in a 2D HDF to pandas.DataFrame conversion.
    """
    f = h5py.File(filename)
    refs = f[fieldname]

    if len(refs.shape)==3:
        if type is str:
            # TODO
            NotImplementedError("Conversion from HDF to Panel for type string "
                " not supported.")
        else:
            # Get transpose by flipping indices in list comprehension
            data = [[refs[i,j,:] for j in range(refs.shape[1])] for i in range(
                refs.shape[0])]

        return pd.Panel(data)
    else:
        if type is str:
            data = [f[ref].value.tobytes()[::2].decode() for ref in refs[:,0]]
        else:
            data = [refs[i,:] for i in range(refs.shape[0])]

        # Transpose 2D list
        data = zip(*data)
        if len(refs.shape)==1:
            if index is None:
                index = labels
            series_of_tuples = pd.Series(data, index=index)
            return series_of_tuples.apply(pd.Series)
        elif len(refs.shape)==2:
            return pd.DataFrame(data, columns=labels, index=index)


def hdf2numpy(filename, fieldname, type=float):
    """A function to extract data from HDF5 files into a useable format for
    scripting, in this case a NumPy array. This function may be used to import
    MATLAB files (.mat) provided that they are saved as version 7.3, to ensure
    HDF compatibility.

    Parameters
    ----------
    filename : str
        Filename including full (or relative) path to HDF file.
    fieldname : str
        Path to desired field in HDF file. Traversing layers of file is done by
        using forward slashes, e.g. 'structLevel1/structLevel2/desiredData'.
    type : (optional)
        Specify data type in HDF file. Only used for data extraction in the 
        special case for strings.
    """
    f = h5py.File(filename)
    refs = f[fieldname]

    if len(refs.shape)==3:
        if type is str:
            # TODO
            NotImplementedError("Conversion from HDF to Panel for type string "
                " not supported.")
        else:
            # Get transpose by flipping indices in list comprehension
            data = [[refs[i,j,:] for j in range(refs.shape[1])] for i in range(
                refs.shape[0])]

    else:
        if type is str:
            data = [f[ref].value.tobytes()[::2].decode() for ref in refs[:,0]]
        else:
            data = [refs[i,:] for i in range(refs.shape[0])]

        # Transpose 2D list
        data = zip(*data)

    return np.array(data)


def hdf2list(filename, fieldname, type=float):
    """A function to extract data from HDF5 files into a useable format for
    scripting, in this case a Python list. This function may be used to import
    MATLAB files (.mat) provided that they are saved as version 7.3, to ensure
    HDF compatibility.

    Parameters
    ----------
    filename : str
        Filename including full (or relative) path to HDF file.
    fieldname : str
        Path to desired field in HDF file. Traversing layers of file is done by
        using forward slashes, e.g. 'structLevel1/structLevel2/desiredData'.
    type : (optional)
        Specify data type in HDF file. Only used for data extraction in the 
        special case for strings.
    """
    f = h5py.File(filename)
    refs = f[fieldname]

    if type is str:
        data = [f[ref].value.tobytes()[::2].decode() for ref in refs[:,0]]
    else:
        data = [f[ref].value for ref in refs[:,0]]

    return data


class TRCFile(object):
    """A plain-text file format for storing motion capture marker trajectories.
    TRC stands for Track Row Column.

    The metadata for the file is stored in attributes of this object.

    See
    http://simtk-confluence.stanford.edu:8080/display/OpenSim/Marker+(.trc)+Files
    for more information.

    """
    def __init__(self, fpath=None, **kwargs):
            #path=None,
            #data_rate=None,
            #camera_rate=None,
            #num_frames=None,
            #num_markers=None,
            #units=None,
            #orig_data_rate=None,
            #orig_data_start_frame=None,
            #orig_num_frames=None,
            #marker_names=None,
            #time=None,
            #):
        """
        Parameters
        ----------
        fpath : str
            Valid file path to a TRC (.trc) file.

        """
        self.marker_names = []
        if fpath != None:
            self.read_from_file(fpath)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def read_from_file(self, fpath):
        # Read the header lines / metadata.
        # ---------------------------------
        # Split by any whitespace.
        # TODO may cause issues with paths that have spaces in them.
        f = open(fpath)
        # These are lists of each entry on the first few lines.
        first_line = f.readline().split()
        # Skip the 2nd line.
        f.readline()
        third_line = f.readline().split()
        fourth_line = f.readline().split()
        f.close()

        # First line.
        if len(first_line) > 3:
            self.path = first_line[3]
        else:
            self.path = ''

        # Third line.
        self.data_rate = float(third_line[0])
        self.camera_rate = float(third_line[1])
        self.num_frames = int(third_line[2])
        self.num_markers = int(third_line[3])
        self.units = third_line[4]
        self.orig_data_rate = float(third_line[5])
        self.orig_data_start_frame = int(third_line[6])
        self.orig_num_frames = int(third_line[7])

        # Marker names.
        # The first and second column names are 'Frame#' and 'Time'.
        self.marker_names = fourth_line[2:]

        len_marker_names = len(self.marker_names)
        if len_marker_names != self.num_markers:
            warnings.warn('Header entry NumMarkers, %i, does not '
                    'match actual number of markers, %i. Changing '
                    'NumMarkers to match actual number.' % (
                        self.num_markers, len_marker_names))
            self.num_markers = len_marker_names

        # Load the actual data.
        # ---------------------
        col_names = ['frame_num', 'time']
        # This naming convention comes from OpenSim's Inverse Kinematics tool,
        # when it writes model marker locations.
        for mark in self.marker_names:
            col_names += [mark + '_tx', mark + '_ty', mark + '_tz']
        dtype = {'names': col_names,
                'formats': ['int'] + ['float64'] * (3 * self.num_markers + 1)}
        self.data = np.loadtxt(fpath, delimiter='\t', skiprows=6, dtype=dtype)
        self.time = self.data['time']

        # Check the number of rows.
        n_rows = self.time.shape[0]
        if n_rows != self.num_frames:
            warnings.warn('%s: Header entry NumFrames, %i, does not '
                    'match actual number of frames, %i, Changing '
                    'NumFrames to match actual number.' % (fpath,
                        self.num_frames, n_rows))
            self.num_frames = n_rows

    def __getitem__(self, key):
        """See `marker()`.

        """
        return self.marker(key)

    def marker(self, name):
        """The trajectory of marker `name`, given as a `self.num_frames` x 3
        array. The order of the columns is x, y, z.

        """
        this_dat = np.empty((self.num_frames, 3))
        this_dat[:, 0] = self.data[name + '_tx']
        this_dat[:, 1] = self.data[name + '_ty']
        this_dat[:, 2] = self.data[name + '_tz']
        return this_dat

    def add_marker(self, name, x, y, z):
        """Add a marker, with name `name` to the TRCFile.

        Parameters
        ----------
        name : str
            Name of the marker; e.g., 'R.Hip'.
        x, y, z: array_like
            Coordinates of the marker trajectory. All 3 must have the same
            length.

        """
        if (len(x) != self.num_frames or len(y) != self.num_frames or len(z) !=
                self.num_frames):
            raise Exception('Length of data (%i, %i, %i) is not '
                    'NumFrames (%i).', len(x), len(y), len(z), self.num_frames)
        self.marker_names += [name]
        self.num_markers += 1
        if not hasattr(self, 'data'):
            self.data = np.array(x, dtype=[('%s_tx' % name, 'float64')])
            self.data = append_fields(self.data,
                    ['%s_t%s' % (name, s) for s in 'yz'],
                    [y, z], usemask=False)
        else:
            self.data = append_fields(self.data,
                    ['%s_t%s' % (name, s) for s in 'xyz'],
                    [x, y, z], usemask=False)

    def marker_at(self, name, time):
        x = np.interp(time, self.time, self.data[name + '_tx'])
        y = np.interp(time, self.time, self.data[name + '_ty'])
        z = np.interp(time, self.time, self.data[name + '_tz'])
        return [x, y, z]

    def marker_exists(self, name):
        """
        Returns
        -------
        exists : bool
            Is the marker in the TRCFile?

        """
        return name in self.marker_names

    def write(self, fpath):
        """Write this TRCFile object to a TRC file.

        Parameters
        ----------
        fpath : str
            Valid file path to which this TRCFile is saved.

        """
        f = open(fpath, 'w')

        # Line 1.
        f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.path.split(fpath)[0])

        # Line 2.
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')

        # Line 3.
        f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            self.data_rate, self.camera_rate, self.num_frames,
            self.num_markers, self.units, self.orig_data_rate,
            self.orig_data_start_frame, self.orig_num_frames))

        # Line 4.
        f.write('Frame#\tTime\t')
        for imark in range(self.num_markers):
            f.write('%s\t\t\t' % self.marker_names[imark])
        f.write('\n')

        # Line 5.
        f.write('\t\t')
        for imark in np.arange(self.num_markers) + 1:
            f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
        f.write('\n')

        # Line 6.
        f.write('\n')

        # Data.
        for iframe in range(self.num_frames):
            f.write('%i' % (iframe + 1))
            f.write('\t%.5f' % self.time[iframe])
            for mark in self.marker_names:
                idxs = [mark + '_tx', mark + '_ty', mark + '_tz']
                f.write('\t%.3f\t%.3f\t%.3f' % tuple(
                    self.data[coln][iframe] for coln in idxs))
            f.write('\n')

        f.close()

    def add_noise(self, noise_width):
        """ add random noise to each component of the marker trajectory
            The noise mean will be zero, with the noise_width being the
            standard deviation.

            noise_width : int
        """
        for imarker in range(self.num_markers):
            components = ['_tx', '_ty', '_tz']
            for iComponent in range(3):
                # generate noise
                noise = np.random.normal(0, noise_width, self.num_frames)
                # add noise to each component of marker data.
                self.data[self.marker_names[imarker] + components[iComponent]] += noise

class Scale:
    """Wrapper of org.opensim.modeling.Scale, that adds a convenience
    constructor.
    """
    def __init__(self, body_name, x, y, z, scale_set=None):
        """
        Parameters
        ----------
        body_name : str
            org.opensim.modeling.Scale.setSegmentName(body_name)
        x, y, z : float
            org.opensim.modeling.Scale.setScaleFactors([x, y, z])
        scaleset : org.opensim.modeling.ScaleSet, optional
            A ScaleSet to adopt-and-append the org.opensim.modeling.Scale to.
        """
        self.scale = osim.Scale()
        self.scale.setName(body_name)
        self.scale.setSegmentName(body_name)
        self.scale.setScaleFactors(osim.Vec3(x, y, z))
        if scale_set:
            scale_set.cloneAndAppend(self.scale)
                
class Measurement:
    """Wrapper of org.opensim.modeling.Measurement with convenience methods.

    """

    def __init__(self, name, measurement_set=None):
        """
        Parameters
        ----------
        name : str
            org.opensim.modeling.Measurement.setName(name)
        measurement_set : org.opensim.modeling.MeasurementSet, optional
            A MeasurementSet to adopt-and-append the
            org.opensim.modeling.Measurement to.

        """
        self.measurement = osim.Measurement()
        self.measurement.setName(name)
        if measurement_set:
            measurement_set.adoptAndAppend(self.measurement)

    def add_bodyscale(self, name, axes='XYZ'):
        """Adds a BodyScale to the Measurement.

        Parameters
        ----------
        name : str
            org.opensim.modeling.BodyScale.setName(name)
        axes : str, optional
            e.g., 'X', 'XY', ...
            org.opensim.modeling.BodyScale.setAxisNames().
            Default is isometric.

        """
        bs = osim.BodyScale()
        bs.setName(name)
        axis_names = osim.ArrayStr()
        for ax in axes:
            axis_names.append(ax)
        bs.setAxisNames(axis_names)
        self.measurement.getBodyScaleSet().cloneAndAppend(bs)

    def add_bodyscale_bilateral(self, name, *args, **kwargs):
        """Adds a BodyScale to both sides of a model. If `name` is 'calf', then
        the same BodyScale is added to the two bodies 'calf_l' and 'calf_r'.

        Parameters
        ----------
        name : str
            Shared prefix of the body.
        axes : list of str's
            See `add_bodyscale`.
        """
        self.add_bodyscale('%s_l' % name, *args, **kwargs)
        self.add_bodyscale('%s_r' % name, *args, **kwargs)

    def add_markerpair(self, marker0, marker1):
        """Adds a MarkerPair to the Measurement's MarkerPairSet.

        Parameters
        ----------
        marker0 : str
            Name of the first marker in the pair.
        marker1 : str
            Name of the second marker in the pair.

        """
        mp = osim.MarkerPair()
        mp.setMarkerName(0, marker0)
        mp.setMarkerName(1, marker1)
        self.measurement.getMarkerPairSet().cloneAndAppend(mp)

    def add_markerpair_bilateral(self, marker0, marker1):
        """Adds two MarkerPair's to the Measurement's MarkerPairSet; assuming
        the name convention: if `marker0` is 'Heel', and `marker1` is 'Toe',
        then we add the following marker pairs: 'RHeel' and 'RToe', and 'LHeel'
        and 'LToe'.

        """
        self.add_markerpair('L%s' % marker0, 'L%s' % marker1)
        self.add_markerpair('R%s' % marker0, 'R%s' % marker1)


class IKTaskSet:
    """Wrapper of org.opensim.modeling.IKTaskSet with convenience methods.

    """
    def __init__(self, iktaskset=None):
        """Creates an org.opensim.modeling.IKTaskSet, or just uses the one
        provided, if provided.

        """
        if iktaskset:
            self.iktaskset = iktaskset
        else:
            self.iktaskset = osim.IKTaskSet()

    def add_ikmarkertask(self, name, do_apply, weight):
        """Creates an IKMarkerTask and appends it to the IKTaskSet.

        Parameters
        ----------
        name : str
            org.opensim.modeling.IKMarkerTask.setName(name)
        do_apply : bool
            org.opensim.modeling.IKMarkerTask.setApply(do_apply)
        weight : float
            org.opensim.modeling.IKMarkerTask.setWeight(weight)

        """
        ikt = osim.IKMarkerTask()
        ikt.setName(name)
        ikt.setApply(do_apply)
        ikt.setWeight(weight)
        self.iktaskset.cloneAndAppend(ikt)

    def add_ikmarkertask_bilateral(self, name, do_apply, weight):
        """Adds two IKMarkerTask's to the IKTaskSet.

        Parameters
        ----------
        name : str
            If 'name' is 'Elbow', then two tasks for markers 'LElbow' and
            'MElbow' will be added.
        do_apply, weight :
            See `add_ikmarkertask`.


        """
        self.add_ikmarkertask('L%s' % name, do_apply, weight)
        self.add_ikmarkertask('R%s' % name, do_apply, weight)

    def add_ikcoordinatetask(self, name, do_apply, manual_value, weight):
        """Creates an IKCoordinateTask (using a manual value) and appends it to
        the IKTaskSet.

        name : str
            org.opensim.modeling.IKMarkerTask.setName(name)
        do_apply : bool
            org.opensim.modeling.IKMarkerTask.setApply(do_apply)
        manual_value : float
            The desired value for this coordinate.
        weight : float
            org.opensim.modeling.IKMarkerTask.setWeight(weight)

        """
        ikt = osim.IKCoordinateTask()
        ikt.setName(name)
        ikt.setApply(do_apply)
        ikt.setValueType(ikt.ManualValue)
        ikt.setValue(manual_value)
        ikt.setWeight(weight)
        self.iktaskset.cloneAndAppend(ikt)

    def add_ikcoordinatetask_bilateral(self, name, do_apply, manual_value,
            weight):
        """Adds two IKCoordinateTask's to the IKTaskSet.

        Parameters
        ----------
        name : str
            if 'name' is 'hip_flexion', then two tasks for coordinates
            'hip_flexion_r' and 'hip_flexion_l' will be added.
        do_apply, manual_value, weight :
            See `add_ikcoordinatetask`.

        """
        self.add_ikcoordinatetask('%s_l' % name, do_apply, manual_value,
                weight)
        self.add_ikcoordinatetask('%s_r' % name, do_apply, manual_value,
                weight)