
import os
import sys
import opensim as osm
import numpy as np
import pandas as pd
import pylab as pl
import warnings
import h5py

import platform

def running_in_jython():
    return platform.system() == 'Java'

if running_in_jython():
    import org.opensim.modeling as osm
else:
    import opensim as osm

def printobj(obj, fname):
    if running_in_jython():
        exec('obj.print(fname)')
    else:
        obj.printToXML(fname)

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
    model = osm.Model(model_fpath)
    n_probes = model.getProbeSet().getSize()
    for i_probe in range(n_probes):
        model.updProbeSet().get(i_probe).setDisabled(False)
    printobj(model, model_fpath)

def get_sum_total_muscle_volume(model_fpath, specific_tension):
    """Returns the sum total of all the masses of the model muscles.
    Parameters
    ----------
    model_fpath : str
        Path to model (.OSIM) file for the model to get muscle volumes.
    specific_tension : float
        Estimate of the specific tension of all model muscles.

    """
    m = osm.Model(model_fpath)
    Vtotal = 0.0
    for i_m in range(m.getMuscles().getSize()):
        musc = m.getMuscles().get(i_m)
        Fmax = musc.getMaxIsometricForce() # N
        Lopt = musc.getOptimalFiberLength() # m

        PCSA = Fmax / specific_tension
        Vm = Lopt * PCSA

        Vtotal += Vm

    return Vtotal

def strengthen_muscles(model_fpath, new_model_fpath, scale_factor):
    """Scales all muscles' maximum isometric force by `scale_factor`.
    Parameters
    ----------
    model_fpath : str
        Path to model (.OSIM) file for the model to strengthen.
    new_model_fpath : str
        Path to which to save the strengthened model.
    scale_factor : float
        All muscle optimal forces are scaled by this number.
    """
    m = osm.Model(model_fpath)
    for i_m in range(m.getMuscles().getSize()):
        m.updMuscles().get(i_m).setMaxIsometricForce(
                m.getMuscles().get(i_m).getMaxIsometricForce() * scale_factor)
    printobj(m, new_model_fpath)

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
        self.scale = osm.Scale()
        self.scale.setName(body_name)
        self.scale.setSegmentName(body_name)
        self.scale.setScaleFactors(osm.Vec3(x, y, z))
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
        self.measurement = osm.Measurement()
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
        bs = osm.BodyScale()
        bs.setName(name)
        axis_names = osm.ArrayStr()
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
        mp = osm.MarkerPair()
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
            self.iktaskset = osm.IKTaskSet()

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
        ikt = osm.IKMarkerTask()
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
        ikt = osm.IKCoordinateTask()
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

### Metabolic probes ###
########################

twitch_ratios_2392 = {
            'glut_med1': 0.55, 'glut_med2': 0.55, 'glut_med3': 0.55,
            'glut_min1': 0.55, 'glut_min2': 0.55, 'glut_min3': 0.55,
            'semimem': 0.4925, 'semiten': 0.425,
            'bifemlh': 0.5425, 'bifemsh': 0.529,
            'add_mag1': 0.552, 'add_mag2': 0.552, 'add_mag3': 0.552,
            'add_mag4': 0.552, # Chris' change to Apoorva's model.
            'glut_max1': 0.55, 'glut_max2': 0.55, 'glut_max3': 0.55,
            'iliacus': 0.5, 'psoas': 0.5, 'rect_fem': 0.3865,
            'vas_med': 0.503, 'vas_int': 0.543, 'vas_lat': 0.455,
            'med_gas': 0.566, 'lat_gas': 0.507, 'soleus': 0.803,
            'tib_post': 0.6, 'flex_dig': 0.6, 'flex_hal': 0.6, 'tib_ant': 0.7,
            'per_brev': 0.6, 'per_long': 0.6, 'per_tert': 0.75,
            'ext_dig': 0.75, 'ext_hal': 0.75,
            'ercspn': 0.6, 'intobl': 0.56, 'extobl': 0.58,
            'sar': -1, 'add_long': -1, 'add_brev': -1,
            'tfl': -1, 'pect': -1, 'grac': -1,
            'quad_fem': -1, 'gem': -1, 'peri': -1}

twitch_ratios_1018 = {
            'hamstrings': 0.49, 'bifemsh': 0.53, 'glut_max': 0.55,
            'iliopsoas': 0.50, 'rect_fem': 0.39, 'vasti': 0.50,
            'gastroc': 0.54, 'soleus': 0.80,
            'tib_ant': 0.70}

# For the muscles that are divided in OpenSim across multiple paths,
# divide the published mass evenly between them.
# Volumes are in cm^3.
_H2014glut_med_volume = 323.2
_H2014glut_min_volume = 104.5
_H2014add_mag_volume = 559.8
_H2014glut_max_volume = 849.0
_H2014small_ext_rotators = 16.1
_H2014extensors_volume = 102.3
_H2014peroneals_volume = 130.8
_Handsfield2014_muscle_volumes = {
        'glut_med1_': _H2014glut_med_volume / 3.0,
        'glut_med2_': _H2014glut_med_volume / 3.0,
        'glut_med3_': _H2014glut_med_volume / 3.0,
        'glut_min1_': _H2014glut_min_volume / 3.0,
        'glut_min2_': _H2014glut_min_volume / 3.0,
        'glut_min3_': _H2014glut_min_volume / 3.0,
        'semimem_': 245.4,
        'semiten_': 186.0,
        'bifemlh_': 206.5,
        'bifemsh_': 100.1,
        'sar_': 163.7,
        'add_long_': 162.1,
        'add_brev_': 104.0,
        'add_mag1_': _H2014add_mag_volume / 3.0,
        'add_mag2_': _H2014add_mag_volume / 3.0,
        'add_mag3_': _H2014add_mag_volume / 3.0,
        'tfl_': 64.9,
        'pect_': 66.3,
        'grac_': 104.0,
        'glut_max1_': _H2014glut_max_volume / 3.0,
        'glut_max2_': _H2014glut_max_volume / 3.0,
        'glut_max3_': _H2014glut_max_volume / 3.0,
        'iliacus_': 176.8,
        'psoas_': 274.8,
        'quad_fem_': 32.4,
        'gem_': _H2014small_ext_rotators,
        'peri_': 42.8, # piriformis.
        'rect_fem_': 269.0,
        'vas_med_': 423.6,
        'vas_int_': 270.5,
        'vas_lat_': 830.9,
        'med_gas_': 257.4,
        'lat_gas_': 150.0,
        'soleus_': 438.2,
        'tib_post_': 104.8,
        'flex_dig_': 30.0,
        'flex_hal_': 78.8,
        'tib_ant_': 135.2,
        'per_brev_': _H2014peroneals_volume / 2.0,
        'per_long_': _H2014peroneals_volume / 2.0,
        'per_tert_': _H2014extensors_volume / 3.0,
        'ext_dig_': _H2014extensors_volume / 3.0,
        'ext_hal_': _H2014extensors_volume / 3.0,
        #'ercspn_': ,
        #'intobl_': ,
        #'extobl_': ,
        }

# In kg.
Handsfield2014_muscle_masses = dict()
muscle_density = 0.001056 # kg / cm^3
for key, val in _Handsfield2014_muscle_volumes.items():
    Handsfield2014_muscle_masses[key] = muscle_density * val

"""
_W2009glut_max_mass = 547.2
_W2009glut_med_mass = 273.5
_Ward2009_muscle_volumes_grams = {
        'glut_med1_': _W2009glut_med_mass / 3.0,
        'glut_med2_': _W2009glut_med_mass / 3.0,
        'glut_med3_': _W2009glut_med_mass / 3.0,
        'glut_min1_': _W2009glut_,
        'glut_min2_':,
        'glut_min3_':,
        'semimem_':,
        'semiten_':,
        'bifemlh_':,
        'bifemsh_':,
        'sar_':,
        'add_long_':,
        'add_brev_':,
        'add_mag1_':,
        'add_mag2_':,
        'add_mag3_':,
        'tfl_':,
        'pect_':,
        'grac_':,
        'glut_max1_':,
        'glut_max2_':,
        'glut_max3_':,
        'iliacus_': 113.7,
        'psoas_': 97.7,
        'quad_fem_':,
        'gem_':,
        'peri_':, # piriformis.
        'rect_fem_':,
        'vas_med_':,
        'vas_int_':,
        'vas_lat_':,
        'med_gas_':,
        'lat_gas_':,
        'soleus_':,
        'tib_post_':,
        'flex_dig_':,
        'flex_hal_':,
        'tib_ant_':,
        'per_brev_':,
        'per_long_':,
        'per_tert_':,
        'ext_dig_':,
        'ext_hal_':,
        #'ercspn_': ,
        #'intobl_': ,
        #'extobl_': ,
        }
Ward2009_muscle_volumes = dict()
for key, val in _Ward2009_muscle_volumes_grams.items():
    Ward2009_muscle_volumes[key] = 0.001 * val
"""

def add_metabolics_probes(model, twitch_ratio_set='gait2392',
        activationMaintenanceRateOn=True,
        shorteningRateOn=True,
        basalRateOn=False,
        mechanicalWorkRateOn=True,
        muscle_masses=None,
        muscle_effort_scaling_factor=None,
        exclude=[],
        specific_tension=None
        ):
    """Adds Umberger2010MuscleMetabolicsProbes to an OpenSim model. Adds a
    probe for each muscle, as well as a whole-body probe that returns
    cumulative energy expended across all muscles in the model. When possible,
    we use published twitch ratios for the probes. For muscles for which we do
    not have data, we use a twitch ratio of 0.5. This method doesn't return
    anything; the model given to the method is simply modified.
    Parameters
    ----------
    model : org.opensim.modeling.Model
        An OpenSim Model that has muscles.
    twitch_ratio_set : float or str ('gait2392' or 'gait1018')
        The experimental data set to use for the model, depending on the model
        we are adding probes to. If a float, use that constant value for all
        muscles.
    activationMaintenanceRateOn : bool, optional
    shorteningRateOn : bool, optional
    basalRateOn : bool, optional
    mechanicalWorkRateOn : bool, optional
    muscle_masses : str, or dict; optional
        * str: 'Handsfield2014' or 'Ward2009' to use lower body masses from the
          respective paper. NOTE: this set of muscle masses does NOT contain
          ercspn, intobl, or extobl masses. Consider excluding those muscles.
        * dict: For muscles in this dict, use the given value as the muscle's
          mass. If the muscle is not specified in this dict, compute the
          muscle's mass from the model's muscle properties.
    muscle_effort_scaling_factor : float, optional (default: 1.0)
    exclude : list of str's, optional
        List of muscle names to exclude.
    specific_tension : float, optional
        Set the specific tension to use for each muscle (N/m^2). We set this
        even for muscles for which you supply muscle masses, but OpenSim will
        ignore its value in those cases.
    """
    # Twitch ratios
    # -------------
    if type(twitch_ratio_set) == float:
        twitchRatios = twitch_ratio_set
    elif twitch_ratio_set == 'gait2392':
        twitchRatios = twitch_ratios_2392
    elif twitch_ratio_set == 'gait1018':
        twitchRatios = twitch_ratios_1018
    else:
        raise Exception("Invalid value for `twitch_ratio_set`.")

    # Muscle masses
    # -------------
    if muscle_masses:
        if muscle_masses == 'Handsfield2014':
            muscle_masses = Handsfield2014_muscle_masses
        else:
            raise Exception("Unexpected muscle_masses {}.".format(
                muscle_masses))
    else:
        muscle_masses = dict()

    # The mass of each muscle will be calculated using data from the model:
    #   muscleMass = (maxIsometricForce / sigma) * rho * optimalFiberLength
    # where sigma = 0.25e6 is the specific tension of mammalian muscle (in
    # Pascals) and rho = 1059.7 is the density of mammalian muscle (in kg/m^3).

    # The slow-twitch ratio used for muscles that either do not appear in the
    # file, or appear but whose proportion of slow-twitch fibers is unknown.
    defaultTwitchRatio = 0.5

    # Whole-body probe
    # ----------------
    # Define a whole-body probe that will report the total metabolic energy
    # consumption over the simulation.
    wholeBodyProbe = osm.Umberger2010MuscleMetabolicsProbe(
        activationMaintenanceRateOn,
        shorteningRateOn,
        basalRateOn,
        mechanicalWorkRateOn)
    wholeBodyProbe.setOperation("value")
    wholeBodyProbe.set_report_total_metabolics_only(False);
    if muscle_effort_scaling_factor:
        wholeBodyProbe.set_muscle_effort_scaling_factor(
                muscle_effort_scaling_factor)

    # Add the probe to the model and provide a name.
    model.addProbe(wholeBodyProbe)
    wholeBodyProbe.setName("metabolic_power")

    # Loop through all muscles, adding parameters for each into the whole-body
    # probe.
    for iMuscle in range(model.getMuscles().getSize()):
        thisMuscle = model.getMuscles().get(iMuscle)

        if not (thisMuscle.getName() in exclude):

            # Get the slow-twitch ratio from the data we read earlier. Start
            # with the default value.
            slowTwitchRatio = defaultTwitchRatio
    
            # Set the slow-twitch ratio to the physiological value, if it is
            # known.
            if type(twitch_ratio_set) == float:
                slowTwitchRatio = twitchRatios
            else:
                for key, val in twitchRatios.items():
                    if thisMuscle.getName().startswith(key) and val != -1:
                        slowTwitchRatio = val
    
            # Add this muscle to the whole-body probe. The arguments are muscle
            # name, slow-twitch ratio, and muscle mass. Note that the muscle
            # mass is ignored unless we set useProvidedMass to True.
            wholeBodyProbe.addMuscle(thisMuscle.getName(),
                                     slowTwitchRatio)

            if specific_tension != None:
                wholeBodyProbe.setSpecificTension(thisMuscle.getName(),
                        specific_tension)
    
            # If we are given a muscle mass, use it in the probe.
            for key, val in muscle_masses.items():
                if thisMuscle.getName().startswith(key):
                    wholeBodyProbe.useProvidedMass(thisMuscle.getName(),
                            val)