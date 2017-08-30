"""Contains methods for postprocessing simulations, or computing derived
quantities, such as sum of squared activations, from simulation (e.g., CMC)
results.

"""
import collections
import copy
import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import tables
from scipy.signal import butter, filtfilt
from scipy.stats import nanmean, nanstd

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

pi = 3.14159265359

def savefigtolog(figname, *args, **kwargs):
    pl.savefig(os.path.join(os.environ['LOGFIGS'], figname), *args, **kwargs)

def nearest_index(array, val):
    return np.abs(array - val).argmin()

def plot_lower_limb_kinematics(kinematics_q_fpath, gl=None,
        kinematics_q_compare_fpath=None, compare_name=None):
    """Plots pelvis tilt, pelvis list, pelvis rotation, hip adduction, hip
    flexion, knee angle, and ankle angle for both limbs.

    Parameters
    ----------
    kinematics_q_fpath : str
        Path to a Kinematics_q.sto file.
    gl : dataman.GaitLandmarks, optional
        If provided, the plots are for a single gait cycle.
    kinematics_q_compare_fpath : str, optional
        Want to compare kinematics to another set of kinematics? Provide the
        kinematics_q file here.
    compare_name : str, optional
        The name to use in the legend for the above comparison kinematics. Must
        be provided if kinematics_q_compare_fpath is provided.

    Returns
    -------
    fig : pylab.figure

    """
    fig = pl.figure(figsize=(7, 10))
    dims = (4, 2)

    sto = storage2numpy(kinematics_q_fpath)
    if kinematics_q_compare_fpath:
        sto2 = storage2numpy(kinematics_q_compare_fpath)
        pl.suptitle('transparent lines: %s' % compare_name)

    def common():
        pl.minorticks_on()
        pl.grid(b=True, which='major', axis='y', color='gray', linestyle='--')
        pl.grid(b=True, which='minor', axis='y', color='gray', linestyle=':')
        if gl != None:
            pl.xlim(0, 100)

    def plot(time, y, label, side, *args, **kwargs):
        if gl != None:
            plot_pgc(time, y, gl, side=side, plot_toeoff=True, label=label,
                    *args, **kwargs)

        else:
            pl.plot(time, y, label=label, *args, **kwargs)

    def plot_coord(coord, side='right', *args, **kwargs):
        if kinematics_q_compare_fpath:
            plot(sto2['time'], sto2[coord], None, side, alpha=0.5,
                    *args, **kwargs)
        plot(sto['time'], sto[coord], side, side,
                *args, **kwargs)
    def plot_one(loc, coord, ylim):
        ax = pl.subplot2grid(dims, loc)
        plot_coord(coord, color='blue')
        pl.ylim(ylim)
        pl.axhline(0, color='gray', zorder=0)
        pl.title(coord)
        common()
    colors = {'left': 'blue', 'right': 'red'}
    def plot_both_sides(loc, coord_pre, ylim):
        ax = pl.subplot2grid(dims, loc)
        for side in ['left', 'right']:
            coord = '%s_%s' % (coord_pre, side[0])
            plot_coord(coord, side, color=colors[side])
        pl.legend(frameon=False)
        pl.ylim(ylim)
        pl.axhline(0, color='gray', zorder=0)
        pl.title(coord_pre)
        common()

    plot_one((0, 0), 'pelvis_tilt', [-20, 10])
    plot_one((1, 0), 'pelvis_list', [-15, 15])
    plot_one((2, 0), 'pelvis_rotation', [-10, 10])
    plot_both_sides((3, 0), 'hip_rotation', [-20, 20])
    pl.xlabel('time (% gait cycle)')

    plot_both_sides((0, 1), 'hip_adduction', [-15, 15])
    plot_both_sides((1, 1), 'hip_flexion', [-30, 50])
    plot_both_sides((2, 1), 'knee_angle', [-10, 90])
    plot_both_sides((3, 1), 'ankle_angle', [-40, 25])
    pl.xlabel('time (% gait cycle)')

    pl.tight_layout() #fig) #, rect=[0, 0, 1, 0.95])
    return fig

def marker_error(model_filepath, model_markers_filepath, 
    marker_trc_filepath):
    """Creates an ndarray containing time histories of marker errors between
    experimental marker trajectories and model marker trajectories from IK.

    Parameters
    ----------
    model_filepath : opensim.Model
        Model used to generate the joint-space kinematics. Must contain
        opensim.Marker's.
    model_markers_filepath : str
        A Storage file containing model marker trajectories.
    marker_trc_filepath : str
        The path to a TRCFile containing experimental marker trajectories.
    indegrees: optional, bool
        True if the states are in degrees instead of radians. Causes all state
        variables to be multiplied by pi/180.

    Returns
    -------
    marker_err : numpy.ndarray
        Each entry in the ndarray contains the error for a marker. We include
        only the markers that are present in the model and the TRCFile.

    """
    marker_names = list()
    m = osm.Model(model_filepath)
    trc = TRCFile(marker_trc_filepath)
    for i in range(m.getMarkerSet().getSize()):
        model_marker_name = m.getMarkerSet().get(i).getName()
        if trc.marker_exists(model_marker_name):
            # Model marker has corresponding experimental data.
            marker_names.append(model_marker_name)

    marker_err = dict()
    marker_err['time'] = []
    for mname in marker_names:
        marker_err[mname] = []

    model_markers = storage2numpy(model_markers_filepath)
    time = model_markers['time']

    for mname in marker_names:
 
        if not (mname + '_tx') in model_markers.dtype.names:
            print ("WARNING: " + mname + " model marker locations not found. "
                "An IK task may not have been specified for this marker. "
                "Skipping...")
        else:
            x_model_loc = model_markers[mname + '_tx']  
            y_model_loc = model_markers[mname + '_ty']
            z_model_loc = model_markers[mname + '_tz']

            for i in range(len(time)):

                exp_loc = trc.marker_at(mname, time[i])
                exp_loc = np.array(exp_loc) * 0.001
                
                distance = np.sqrt(
                            (x_model_loc[i] - exp_loc[0])**2 +
                            (y_model_loc[i] - exp_loc[1])**2 +
                            (z_model_loc[i] - exp_loc[2])**2
                            )
                marker_err[mname].append(distance)
    
    for t in model_markers['time']:
        marker_err['time'].append(t) 

    return marker_err


def marker_error_from_kinematics(model_filepath, states_storage, 
    marker_trc_filepath, indegrees=False):
    """Creates an ndarray containing time histories of marker errors between
    experimental marker trajectories and joint-space kinematics (from RRA
    or CMC).

    Parameters
    ----------
    model_filepath : opensim.Model
        Model used to generate the joint-space kinematics. Must contain
        opensim.Marker's.
    states_storage : str
        A Storage file containing joint space kinematics.
    marker_trc_filepath : str
        The path to a TRCFile containing experimental marker trajectories.
    indegrees: optional, bool
        True if the states are in degrees instead of radians. Causes all state
        variables to be multiplied by pi/180.

    Returns
    -------
    marker_err : numpy.ndarray
        Each entry in the ndarray contains the error for a marker. We include
        only the markers that are present in the model and the TRCFile.

    """
    marker_names = list()
    m = osm.Model(model_filepath)
    trc = TRCFile(marker_trc_filepath)
    for i in range(m.getMarkerSet().getSize()):
        model_marker_name = m.getMarkerSet().get(i).getName()
        if trc.marker_exists(model_marker_name):
            # Model marker has corresponding experimental data.
            marker_names.append(model_marker_name)

    engine = m.getSimbodyEngine()

    data = dict()
    for mname in marker_names:
        data[mname] = []

    def marker_error_for_frame(model, state, data, marker_names, trc):
        for mname in marker_names:
            marker = model.getMarkerSet().get(mname)
            modelMarkerPosInGround = osm.Vec3()
            engine.transformPosition(state,
                    marker.getBody(), marker.getOffset(),
                    model.getGroundBody(), modelMarkerPosInGround)
            expMarkerPosInGround = trc.marker_at(mname, state.getTime())

            a = modelMarkerPosInGround
            b = np.array(expMarkerPosInGround) * 0.001
            distance = np.sqrt(
                    (a.get(0) - b[0])**2 +
                    (a.get(1) - b[1])**2 +
                    (a.get(2) - b[2])**2
                    )
            data[mname].append(distance)

    time, _ = analysis(m, states_storage,
            lambda m, s: marker_error_for_frame(m, s, data, marker_names, trc),
            indegrees=indegrees
            )

    n_times = len(data[marker_names[0]])
    marker_err = np.empty(n_times, dtype={'names': ['time'] + marker_names,
        'formats': (len(marker_names) + 1) * ['f4']})

    marker_err['time'] = time
    for mname in marker_names:
        marker_err[mname] = data[mname]

    return marker_err


def plot_marker_error_general(output_filepath, marker_names, ymax, gl,
        data, mult=100):

    def xlim(times):
        if gl != None:
            pl.xlim(0, 100)
            pl.xlabel('time (% gait cycle)')
        else:
            pl.xlim(times[0], times[-1])
            pl.xlabel('time (seconds)')

    def plot(time, y, side='right', *args, **kwargs):
        if gl != None:
            if side.lower() == 'r': side = 'right'
            elif side.lower() == 'l': side = 'left'
            plot_pgc(time, y, gl, side=side, plot_toeoff=True, *args, **kwargs)

        else:
            pl.plot(time, y, *args, **kwargs)

    fig = pl.figure(figsize=(12, 4 * np.ceil(len(marker_names) * 0.5)))
    for imark, marker_name in enumerate(marker_names):
        pl.subplot(np.ceil(len(marker_names) * 0.5), 2, imark + 1)
        if (marker_name[0] == '.' or marker_name[0] == '_' 
            or marker_name[0]=='*'):

            if marker_name[0]=='*':
                marker_name = marker_name[1:]

            for side in ['R', 'L']:
                name = '%s%s' % (side, marker_name)
                plot(data['time'], mult * data[name], side, label=name)
                
        else:
            plot(np.array(data['time']), mult * np.array(data[marker_name]),
                label=marker_name)
        pl.legend(frameon=False, loc='best')
        pl.ylim(ymin=0, ymax=ymax)
        xlim(data['time'])
        pl.ylabel('marker error (cm)')
        pl.axhline(1, c='gray', ls='--')
        pl.axhline(2, c='gray', ls='--')
        pl.axhline(3, c='gray', ls='--')
        pl.axhline(4, c='gray', ls='--')
        pl.axhline(6, c='gray', ls='--')

    pl.tight_layout()
    fig.savefig(output_filepath)


def plot_marker_error(output_filepath, marker_names, ymax, gl, *args, 
    **kwargs):
    data = marker_error(*args, **kwargs)
    plot_marker_error_general(output_filepath, marker_names, ymax, gl, data)


def plot_marker_error_from_kinematics(output_filepath, marker_names, ymax, gl,
    *args, **kwargs):
    data = marker_error_from_kinematics(*args, **kwargs)
    plot_marker_error_general(output_filepath, marker_names, ymax, gl, data)


def plot_pgc(time, data, gl, side='left', axes=None, plot_toeoff=False, *args,
        **kwargs):

    pgc, ys = data_by_pgc(time, data, gl, side=side)

    if axes:
        ax = axes
    else:
        ax = pl
    ax.plot(pgc, ys, *args, **kwargs)

    if plot_toeoff:
        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = 'gray'
        if 'label' in kwargs: kwargs.pop('label')
        plot_toeoff_pgc(gl, side, ax, *args, zorder=0, **kwargs)


def data_by_pgc(time, data, gl, side='left'):

    if side == 'left':
        strike = gl.left_strike
    elif side == 'right':
        strike = gl.right_strike
    else:
        raise Exception("side '%s' not recognized." % side)

    cycle_duration = (gl.cycle_end - gl.cycle_start)

    if strike < gl.cycle_start:
        strike += cycle_duration
    if strike > gl.cycle_end:
        strike -= cycle_duration                   

    ts, ys = shift_data_to_cycle(gl.cycle_start,
            gl.cycle_end, strike, time, data)

    pgc = percent_duration(ts, 0, cycle_duration)

    if np.any(pgc > 100.0):
        print('Percent gait cycle greater than 100: %f' % np.max(pgc))
        # TODO DEBUG
        # TODO import traceback
        # TODO traceback.print_stack()
    if np.any(pgc < 0):
        print('Percent gait cycle less than 0: %f' % np.min(pgc))
    if np.any(pgc > 100.01) or np.any(pgc < 0.0):
        raise Exception('Percent gait cycle out of range.')

    return pgc, ys


def shift_data_to_cycle(
        arbitrary_cycle_start_time, arbitrary_cycle_end_time,
        new_cycle_start_time, time, ordinate, cut_off=True):
    """
    Takes data (ordinate) that is (1) a function of time and (2) cyclic, and
    returns data that can be plotted so that the data starts at the desired
    part of the cycle.

    Used to shift data to the desired part of a gait cycle, for plotting
    purposes.  Data may be recorded from an arbitrary part
    of the gait cycle, but we might desire to plot the data starting at a
    particular part of the gait cycle (e.g., right foot strike).
    Another example use case is that one might have data for both right and
    left limbs, but wish to plot them together, and thus must shift data for
    one of the limbs by 50% of the gait cycle.

    This method also cuts the data so that your data covers at most a full gait
    cycle but not more.

    The first three parameters below not need exactly match times in the `time`
    array.

    This method can also be used just to truncate data, by setting
    `new_cycle_start_time` to be the same as `arbitrary_cycle_start_time`.

    Parameters
    ----------
    arbitrary_cycle_start_time : float
        Choose a complete cycle/period from the original data that you want to
        use in the resulting data. What is the initial time in this period?
    arbitrary_cycle_end_time : float
        See above; what is the final time in this period?
    new_cycle_start_time : float
        The time at which the shifted data should start. Note that the initial
        time in the shifted time array will regardlessly be 0.0, not
        new_cycle_start_time.
    time : np.array
        An array of times that must correspond with ordinate values (see next),
        and must contain arbitrary_cycle_start_time and
        arbitrary_cycle_end_time.
    ordinate : np.array
        The cyclic function of time, values corresponding to the times given.
    cut_off : bool, optional
        Sometimes, there's a discontinuity in the data that prevents obtaining
        a smooth curve if the data wraps around. In order prevent
        misrepresenting the data in plots, etc., an np.nan is placed in the
        appropriate place in the data.

    Returns
    -------
    shifted_time : np.array
        Same size as time parameter above, but its initial value is 0 and its
        final value is the duration of the cycle (arbitrary_cycle_end_time -
        arbitrary_cycle_start_time).
    shifted_ordinate : np.array
        Same ordinate values as before, but they are shifted so that the first
        value is ordinate[{index of arbitrary_cycle_start_time}] and the last
        value is ordinate[{index of arbitrary_cycle_start_time} - 1].

    Examples
    --------
    Observe that we do not require a constant interval for the time:

        >>> ordinate = np.array([2, 1., 2., 3., 4., 5., 6.])
        >>> time = np.array([0.5, 1.0, 1.2, 1.35, 1.4, 1.5, 1.8])
        >>> arbitrary_cycle_start_time = 1.0
        >>> arbitrary_cycle_end_time = 1.5
        >>> new_cycle_start_time = 1.35
        >>> shifted_time, shifted_ordinate = shift_data_to_cycle(
                ...     arbitrary_cycle_start_time, arbitrary_cycle_end_time,
                ...     new_cycle_start_time,
                ...     time, ordinate)
        >>> shifted_time
        array([ 0.  ,  0.05,  0.15,  0.3 ,  0.5 ])
        >>> shifted_ordinate
        array([3., 4., nan, 1., 2.])

    In order to ensure the entire duration of the cycle is kept the same,
    the time interval between the original times "1.5" and "1.0" is 0.1, which
    is the time gap between the original times "1.2" and "1.3"; the time
    between 1.2 and 1.3 is lost, and so we retain it in the place where we
    introduce a new gap (between "1.5" and "1.0"). NOTE that we only ensure the
    entire duration of the cycle is kept the same IF the available data covers
    the entire time interval [arbitrary_cycle_start_time,
    arbitrary_cycle_end_time].

    """
    # TODO gaps in time can only be after or before the time interval of the
    # available data.

    if new_cycle_start_time > arbitrary_cycle_end_time:
        raise Exception('(`new_cycle_start_time` = %f) > (`arbitrary_cycle_end'
                '_time` = %f), but we require that `new_cycle_start_time <= '
                '`arbitrary_cycle_end_time`.' % (new_cycle_start_time,
                    arbitrary_cycle_end_time))
    if new_cycle_start_time < arbitrary_cycle_start_time:
        raise Exception('(`new_cycle_start_time` = %f) < (`arbitrary_cycle'
                '_start_time` = %f), but we require that `new_cycle_start_'
                'time >= `arbitrary_cycle_start_time`.' % (new_cycle_start_time,
                    arbitrary_cycle_start_time))


    # We're going to modify the data.
    time = copy.deepcopy(time)
    ordinate = copy.deepcopy(ordinate)

    duration = arbitrary_cycle_end_time - arbitrary_cycle_end_time

    old_start_index = nearest_index(time, arbitrary_cycle_start_time)
    old_end_index = nearest_index(time, arbitrary_cycle_end_time)

    new_start_index = nearest_index(time, new_cycle_start_time)

    # So that the result matches exactly with the user's desired times.
    if new_cycle_start_time > time[0] and new_cycle_start_time < time[-1]:
        time[new_start_index] = new_cycle_start_time
        ordinate[new_start_index] = np.interp(new_cycle_start_time, time,
                ordinate)

    data_exists_before_arbitrary_start = old_start_index != 0
    if data_exists_before_arbitrary_start:
        #or (old_start_index == 0 and
        #    time[old_start_index] > arbitrary_cycle_start_time):
        # There's data before the arbitrary start.
        # Then we can interpolate to get what the ordinate SHOULD be exactly at
        # the arbitrary start.
        time[old_start_index] = arbitrary_cycle_start_time
        ordinate[old_start_index] = np.interp(arbitrary_cycle_start_time, time,
                ordinate)
        gap_before_avail_data = 0.0
    else:
        if not new_cycle_start_time < time[old_start_index]:
            gap_before_avail_data = (time[old_start_index] -
                    arbitrary_cycle_start_time)
        else:
            gap_before_avail_data = 0.0
    data_exists_after_arbitrary_end = time[-1] > arbitrary_cycle_end_time
    # TODO previous: old_end_index != (len(time) - 1)
    if data_exists_after_arbitrary_end:
        #or (old_end_index == (len(time) - 1)
        #and time[old_end_index] < arbitrary_cycle_end_time):
        time[old_end_index] = arbitrary_cycle_end_time
        ordinate[old_end_index] = np.interp(arbitrary_cycle_end_time, time,
                ordinate)
        gap_after_avail_data = 0
    else:
        gap_after_avail_data = arbitrary_cycle_end_time - time[old_end_index]

    # If the new cycle time sits outside of the available data, our job is much
    # easier; just add or subtract a constant from the given time.
    if new_cycle_start_time > time[-1]:
        time_at_end = arbitrary_cycle_end_time - new_cycle_start_time
        missing_time_at_beginning = \
                max(0, time[0] - arbitrary_cycle_start_time)
        move_forward = time_at_end + missing_time_at_beginning
        shift_to_zero = time[old_start_index:] - time[old_start_index]
        shifted_time = shift_to_zero + move_forward
        shifted_ordinate = ordinate[old_start_index:]
    elif new_cycle_start_time < time[0]:
        move_forward = time[0] - new_cycle_start_time
        shift_to_zero = time[:old_end_index + 1] - time[old_start_index]
        shifted_time = shift_to_zero + move_forward
        shifted_ordinate = ordinate[:old_end_index + 1]
    else:
        # We actually must cut up the data and move it around.

        # Interval of time in
        # [arbitrary_cycle_start_time, arbitrary_cycle_end_time] that is 'lost' in
        # doing the shifting.
        if new_cycle_start_time < time[old_start_index]:
            lost_time_gap = 0.0
        else:
            lost_time_gap = time[new_start_index] - time[new_start_index - 1]

        # Starts at 0.0.
        if new_cycle_start_time < time[0]:
            addin = gap_before_avail_data
        else:
            addin = 0
        first_portion_of_new_time = (time[new_start_index:old_end_index+1] -
                new_cycle_start_time + addin)

        # Second portion: (1) shift to 0, then move to the right of first portion.
        second_portion_to_zero = \
                time[old_start_index:new_start_index] - arbitrary_cycle_start_time
        second_portion_of_new_time = (second_portion_to_zero +
                first_portion_of_new_time[-1] + lost_time_gap +
                gap_after_avail_data)

        shifted_time = np.concatenate(
                (first_portion_of_new_time, second_portion_of_new_time))

        # Apply cut-off:
        if cut_off:
            ordinate[old_end_index] = np.nan

        # Shift the ordinate.
        shifted_ordinate = np.concatenate(
                (ordinate[new_start_index:old_end_index+1],
                    ordinate[old_start_index:new_start_index]))

    return shifted_time, shifted_ordinate


def percent_duration_single(time, start, end):
    """Converts a single time value to a percent duration (e.g., percent gait
    cycle) value. The difference between this method and `percent_duration` is
    that this works on a single float, rather than on an array.

    Parameters
    ----------
    time : float
        The time value to convert, with units of time (e.g., seconds).
    start : float
        The start of the duration (0 %), in the same units of time.
    end : float
        The end of the duration (100 %), in the same units of time.

    """
    return (time - start) / (end - start) * 100.0


def percent_duration(time, start=None, end=None):
    """Converts a time array to percent duration (e.g., percent gait cycle).

    Parameters
    ----------
    time : np.array
        The time data to convert, with units of time (e.g., seconds).
    start : float, optional
        Start time of the duration. If not provided, we use time[0].
    end : float, optional
        End time of the duration. If not provided, we use time[-1].

    Returns
    -------
    percent_duration : np.array
        Varies from 0 to 100.

    """
    if start == None: start = time[0]
    if end == None: end = time[-1]
    return (time - start) / (end - start) * 100.0


def toeoff_pgc(gl, side):
    toeoff = getattr(gl, side + '_toeoff')
    strike = getattr(gl, side + '_strike')
    cycle_duration = (gl.cycle_end - gl.cycle_start)
    while toeoff < strike:
        toeoff += cycle_duration
    while toeoff > strike + cycle_duration:
        toeoff -= cycle_duration
    return percent_duration_single(toeoff,
            strike,
            strike + cycle_duration)


def plot_toeoff_pgc(gl, side, axes=None, *args, **kwargs):
    if axes:
        ax = axes
    else:
        ax = pl
    ax.axvline(toeoff_pgc(gl, side), *args, **kwargs)


## Data Management ##
## =============== ##

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

## Modeling ##
## ======== ##

def analysis(model, storage, fcn, times=None, indegrees=False):
    """This basically does the same thing as an OpenSim analysis. Compute the
    result of `fcn` for each time in the states_sto, using the model's state,
    and return the resulting array.

    Parameters
    ----------
    model : str or opensim.Model
        If str, a valid path to an OpenSim model file (.osim).
    storage : str or opensim.Storage
        If str, a valid path to a states Storage file.
    fcn : function
        This function must have a signature like:

            qty = fcn(model, state)

        where model is an opensim.Model, and state is a
        simtk.State. Note that you can grab the time via state.getTime().
    times : array_like of float's
        Times at which to evaluate `fcn`.
    indegrees: optional, bool
        True if the states are in degrees instead of radians. Causes all state
        variables to be multiplied by pi/180.

    Returns
    -------
    times : list of float's
        The times corresponding to the evaluations of `fcn`.
    qty : list of float's
        This is the result of `qty` at all the times in the states Storage. It
        has the same length as a column in `storage`.

    """
    if type(model) == str:
        model = osm.Model(model)
    if type(storage) == str:
        storage = osm.Storage(storage)

    state = model.initSystem()

    sto_times = osm.ArrayDouble()
    storage.getTimeColumn(sto_times)

    if times == None:
        times = sto_times.getSize() * [0]
        for i in range(sto_times.getSize()):
            times[i] = sto_times.getitem(i)

    qty = len(times) * [0]
    for i, t in enumerate(times):
        this_state = set_model_state_from_storage(model, storage, t,
                state=state, indegrees=indegrees)
        qty[i] = fcn(model, this_state)

    return times, qty

def set_model_state_from_storage(model, storage, time, state=None,
        indegrees=False):
    """Set the state of the model from a state described in a states Storage
    (.STO) file, at the specified time in the Storage file. Note that the model
    is not modified in any way; we just use the model to set the state.

    The storage should have beeng generated with a model that has the same
    exact states.

    Parameters
    ----------
    model : str or opensim.Model
        If str, a valid path to an OpenSim model file (.osim).
    storage : str or opensim.Storage
        If str, a valid path to a states Storage file.
    time : float
        A time, within the range of the times in the storage file, at which we
        should extract the state from the storage file.
    state : simtk.State
        If you don't want us to call `initSystem()` on the model, then give us
        a state!
    indegrees: optional, bool
        True if the states are in degrees instead of radians. Causes all state
        variables to be multiplied by pi/180.

    Returns
    -------
    state : simtk.State
        A state object that represents the state given in `storage` at time
        `time`.

    """
    if type(model) == str:
        model = osm.Model(model)
    if type(storage) == str:
        storage = osm.Storage(storage)

    if state == None:
        state = model.initState()

    state_names = storage.getColumnLabels()
    
    n_states = state_names.getSize()

    # Interpolate the data to obtain the state (placed into sto_state) at the
    # specified time. Grab all the states (n_states). I'm assuming that these
    # state values are in the same order as is given by getStateIndex.
    sto_state = osm.ArrayDouble()
    sto_state.setSize(n_states)
    storage.getDataAtTime(time, n_states, sto_state)

    for i in range(state_names.getSize()):
        # I'm not even assuming that these
        # state values are returned in the same order given by state_names.
        if state_names.getitem(i) != 'time':
            sto_idx = storage.getStateIndex(state_names.getitem(i))
            state_value = sto_state.getitem(sto_idx)
            if indegrees:
                state_value *= pi / 180.0
            model.setStateVariable(state, state_names.getitem(i), state_value)

    # TODO Maybe CAN rely on this.
    state.setTime(time)
    
    model.assemble(state)

    return state