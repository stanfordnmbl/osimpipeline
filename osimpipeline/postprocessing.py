"""Contains methods for postprocessing simulations, or computing derived
quantities, such as sum of squared activations, from simulation (e.g., CMC)
results.

"""
import collections
import copy
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from scipy.signal import butter, filtfilt
import scipy.io as sio
from numpy import nanmean, nanstd

import platform

from utilities import storage2numpy, TRCFile

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
        kinematics_q_compare_fpath=None, compare_name=None, side=None):
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
    if side:
        plot_one((3, 0), 'hip_rotation_%s' % side, [-20, 20])
    else:
        plot_both_sides((3, 0), 'hip_rotation', [-20, 20])
    pl.xlabel('time (% gait cycle)')

    if side:
        plot_one((0, 1), 'hip_adduction_%s' % side, [-15, 15])
        plot_one((1, 1), 'hip_flexion_%s' % side, [-30, 50])
        plot_one((2, 1), 'knee_angle_%s' % side, [-10, 90])
        plot_one((3, 1), 'ankle_angle_%s' % side, [-40, 25])
    else:
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
        pl.subplot(int(np.ceil(len(marker_names) * 0.5)), 2, int(imark + 1))
        if (marker_name[0] == '.' or marker_name[0] == '_' 
            or marker_name[0]=='*'):

            if marker_name[0]=='*':
                marker_name = marker_name[1:]

            for side in ['R', 'L']:
                name = '%s%s' % (side, marker_name)
                plot(data['time'], mult * np.array(data[name]), side, 
                    label=name)
                
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
    pl.close(fig)


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


def plot_force_plate_data(mot_file):
    """Plots all force componenets, center of pressure components, and moment
    components, for both legs.

    Parameters
    ----------
    mot_file : str
        Name of *.mot (OpenSim Storage) file containing force plate data.

    """
    data = storage2numpy(mot_file)
    time = data['time']

    pl.figure(figsize=(5 * 2, 4 * 3))

    for i, prefix in enumerate([ '1_', '']):
        pl.subplot2grid((3, 2), (0, i))
        for comp in ['x', 'y', 'z']:
            pl.plot(time, 1.0e-3 * data['%sground_force_v%s' % (prefix, comp)],
                    label=comp)
        if i == 0:
            pl.title('left foot')
            pl.ylabel('force components (kN)')
        if i == 1:
            pl.title('right foot')
            pl.legend(loc='upper left', bbox_to_anchor=(1, 1))

    for i, prefix in enumerate([ '1_', '']):
        pl.subplot2grid((3, 2), (1, i))
        for comp in ['x', 'y', 'z']:
            pl.plot(time, data['%sground_force_p%s' % (prefix, comp)],
                    label=comp)

        if i == 0: pl.ylabel('center of pressure components (m)')

    for i, prefix in enumerate([ '1_', '']):
        pl.subplot2grid((3, 2), (2, i))
        for comp in ['x', 'y', 'z']:
            pl.plot(time, data['%sground_torque_%s' % (prefix, comp)],
                    label=comp)

        if i == 0: pl.ylabel('torque components (N-m)')
        pl.xlabel('time (s)')


def plot_gait_torques(output_filepath, actu, primary_leg, cycle_start,
        cycle_end, primary_footstrike, opposite_footstrike,
        toeoff_time=None):
    """Plots hip, knee, and ankle torques, as a function of percent
    gait cycle, for one gait cycle. Gait cycle starts with a footstrike (start
    of stance). Torques are plotted for both legs; the data for the
    'opposite' leg is properly shifted so that we plot it starting from a
    footstrike as well.

    We assume torques are in units of N-m.

    Knee torque is negated: In OpenSim, a flexion torque is represented by a
    negative value for the torque. In literature, flexion is shown
    as positive.

    Parameters
    ----------
    actu : pytables.Group or dict
        Actuation_force group from a simulation containing joint torques
        (probably an RRA output). If dict, must have fields 'time',
        'hip_flexion_r', 'knee_angle_r' (extension), 'ankle_angle_r'
        (dorsiflexion), 'hip_flexion_l', 'knee_angle_l' (extension),
        'ankle_angle_l' (dorsiflexion).
    primary_leg : str; 'right' or 'left'
        The first and second foot strikes are for which leg?
    cycle_start : float
        Time, in seconds, at which the gait cycle starts.
    cycle_end : float
        Time, in seconds, at which the gait cycle ends.
    primary_footstrike : float
        Time, in seconds, at which the primary leg foot-strikes.
    opposite_footstrike : float
        In between the other two footstrikes, the opposite foot also strikes
        the ground. What's the time at which this happens? This is used to
        shift the data for the opposite leg so that it lines up with the
        `primary` leg's data.
    toeoff_time : bool, optional
        Draw a vertical line on the plot to indicate when the primary foot
        toe-off occurs by providing the time at which this occurs.

    """
    # TODO compare to experimental data.
    # TODO compare to another simulation.
    if primary_leg == 'right':
        opposite_leg = 'left'
    elif primary_leg == 'left':
        opposite_leg = 'right'
    else:
        raise Exception("'primary_leg' is '%s'; it must "
                "be 'left' or 'right'." % primary_leg)

    def plot_for_a_leg(coordinate_name, leg, new_start, color='k', mult=1.0):
        if type(actu) == np.ndarray:
            raw_time = actu['time']
            raw_y = actu[coordinate_name + '_%s_moment' % leg[0]] # TODO uhoh
        else:
            raw_time = actu.cols.time[:]
            raw_y = getattr(actu.cols, '%s_%s' % (coordinate_name, leg[0]))
        time, angle = shift_data_to_cycle( cycle_start, cycle_end, new_start,
                raw_time, raw_y)
        pl.plot(percent_duration(time), mult * angle, color=color,
                label=leg)

    def plot_primary_leg(coordinate_name, **kwargs):
        plot_for_a_leg(coordinate_name, primary_leg, primary_footstrike,
                **kwargs)

    def plot_opposite_leg(coordinate_name, **kwargs):
        plot_for_a_leg(coordinate_name, opposite_leg, opposite_footstrike,
                color=(0.5, 0.5, 0.5), **kwargs)

    def plot_coordinate(index, name, negate=False, label=None):
        ax = pl.subplot(3, 1, index)
        if negate:
            plot_primary_leg(name, mult=-1.0)
            plot_opposite_leg(name, mult=-1.0)
        else:
            plot_primary_leg(name)
            plot_opposite_leg(name)
        # TODO this next line isn't so great of an idea:
        if label == None:
            label = name.replace('_', ' ')
        pl.ylabel('%s (N-m)' % label)
        pl.legend()

        if toeoff_time != None:
            duration = cycle_end - cycle_start
            # 'pgc' is percent gait cycle
            if toeoff_time > primary_footstrike:
                toeoff_pgc = percent_duration_single(toeoff_time,
                        primary_footstrike, duration +
                        primary_footstrike)
            else:
                chunk1 = cycle_end - primary_footstrike
                chunk2 = toeoff_time - cycle_start
                toeoff_pgc = (chunk1 + chunk2) / duration * 100.0
            pl.plot(toeoff_pgc * np.array([1, 1]), ax.get_ylim(),
                    c=(0.5, 0.5, 0.5))

        pl.xticks([0.0, 25.0, 50.0, 75.0, 100.0])
        pl.minorticks_on()
        pl.grid(b=True, which='major', color='gray', linestyle='--')
        pl.grid(b=True, which='minor', color='gray', linestyle='--')

    fig = pl.figure(figsize=(4, 12))
    plot_coordinate(1, 'hip_flexion', label='hip flexion moment')
    plot_coordinate(2, 'knee_angle', negate=True, label='knee flexion moment')
    plot_coordinate(3, 'ankle_angle', label='ankle dorsiflexion moment')
    pl.xlabel('percent gait cycle')

    pl.tight_layout()
    fig.savefig(output_filepath)
    pl.close(fig)

def plot_muscle_activity(filepath, exc=None, act=None):

    """Plots provided muscle activity and saves to a pdf file.

    Parameters
    ----------
    filepath: string
        Path name of pdf file to print to.
    exc: pandas DataFrame
        DataFrame containing muscle excitation and muscle name information.
    act: pandas DataFrame
        DataFrame containing muscle activation and muscle name information.
    """

    if (exc is None) and (act is None):
        raise Exception("Please provide either excitation or "
            "activation information")
    elif act is None:
        N = len(exc.columns)
    else: 
        N = len(act.columns)

    # Create plots
    num_rows = 5
    num_cols = np.ceil(float(N) / num_rows)
    fig = pl.figure(figsize=(11, 8.5))
    for i in range(N):
        pl.subplot(num_rows, num_cols, i + 1)
        if not (exc is None):
            pl.plot(exc.index, exc[exc.columns[i]], label='e')
        if not (act is None):
            pl.plot(act.index, act[act.columns[i]], label='a')
        pl.ylim(0, 1)
        if i == 1:
            pl.legend(frameon=False, fontsize=8)
        if not (exc is None):
            pl.title(exc.columns[i], fontsize=8)
        else:
            pl.title(act.columns[i], fontsize=8)
        pl.autoscale(enable=True, axis='x', tight=True)
        pl.xticks([])
        pl.yticks([])
    pl.tight_layout()
    pl.savefig(filepath)
    pl.close(fig)

def plot_reserve_activity(filepath, reserves):
    
    """Plots provided reservec acutator activations and saves to a pdf file.

    Parameters
    ----------
    filepath: string
        Path name of pdf file to print to.
    reserves: pandas DataFrame
        DataFrame containing reserve activity and name information.
    """

    fig = pl.figure()
    NR = len(reserves.columns)
    for i in range(NR):
        pl.subplot(NR, 1, i + 1)
        pl.plot(reserves.index, reserves[reserves.columns[i]])
        pl.ylim(-1, 1)
        pl.title(reserves.columns[i], fontsize=8)
        pl.axhline(0)
        pl.autoscale(enable=True, axis='x', tight=True)
    pl.tight_layout()
    pl.savefig(filepath)
    pl.close(fig)

def plot_joint_moment_breakdown(time, joint_moments, tendon_forces, 
    moment_arms, dof_names, muscle_names, pdf_path, csv_path, ext_moments=None,
     ext_names=None, ext_colors=None, mass=None):

    """Plots net joint moments, individual muscle moments, and, if included,
    any external moments in the system. Prints a pdf file with plots and a csv
    file containing data.

    N:     # of time points
    Ndof:  # of degrees-of-freedom
    Nmusc: # of muscles 

    Parameters
    ----------
    time (N,): Numpy array
        Vector of time points
    joint_moments (N, Ndof): Numpy array
        Array of experimental net joint moments.
    tendon_forces (N, Nmusc): Numpy array
        Array of computed tendon forces.
    moment_arms (N, Ndof, Nmusc): Numpy array
        Array of muscle moment arms, corresponding to each joint.
    dof_names (Ndof,): list
        List of strings containing degree-of-freedom names.
    muscle_names (Nmusc,): list
        List of strings containing muscle names.
    pdf_path: string
        Path and filename for PDF of final plots.
    csv_path: string
        Path and filename for CSV of moment breakdown data
    mass: kg
        (Optional). Subject mass to normalize moments by.
    ext_moments:
        (Optional). External moments applied to each degree-of-freedom (i.e.
        such as an exoskeleton device torque).

    """

    num_dofs = len(dof_names)
    num_muscles = len(muscle_names)

    # For writing moments to a file.
    dof_array = list()
    actuator_array = list()
    moments_array = list()
    all_moments = list()

    import seaborn.apionly as sns
    palette = sns.color_palette('muted')
    num_colors = 6
    sns.set_palette(palette)

    pgc = 100.0 * (time - time[0]) / (time[-1] - time[0])
    pgc_csv = np.linspace(0, 100, 400)
    fig = pl.figure(figsize=(8.5, 11))
    for idof in range(num_dofs):
        dof_name = dof_names[idof]
        ax = fig.add_subplot(num_dofs, 2, 2 * idof + 1)
        net_integ = np.trapz(np.abs(joint_moments[:, idof]),
                x=time)
        sum_actuators_shown = np.zeros_like(time)
        icolor = 0
        for imusc in range(num_muscles):
            muscle_name = muscle_names[imusc]
            if np.any(moment_arms[:, idof, imusc]) > 0.00001:
                this_moment = \
                        tendon_forces[:, imusc] * moment_arms[:, idof, imusc];
                mom_integ = np.trapz(np.abs(this_moment), time)
                if mom_integ > 0.05 * net_integ:
                    if np.floor(icolor / num_colors) == 0:
                        ls = '-'
                    elif np.floor(icolor / num_colors) == 1:
                        ls = '--'
                    else:
                        ls = '-.'
                    ax.plot(pgc, this_moment, label=muscle_name,
                            linestyle=ls)
                    dof_array.append(dof_name)
                    actuator_array.append(muscle_name)
                    all_moments.append(
                            np.interp(pgc_csv, pgc, this_moment))

                    sum_actuators_shown += this_moment
                    icolor += 1

        if ext_moments:
            num_ext = len(ext_moments)
            for iext in range(num_ext):

                ext_moment = ext_moments[iext]
                if ext_names:
                    ext_name = ext_names[iext]
                if ext_colors:
                    ext_color = ext_colors[iext]

                pgc_mod = 100.0 * (
                    (ext_moment.index - ext_moment.index[0])*1.0 /
                    (ext_moment.index[-1] - ext_moment.index[0])*1.0)
 
                ext_col = -1
                for colname in ext_moment.columns:
                    if colname in dof_name:
                        ext_col = colname
                        break
                if ext_col == -1:
                    raise Exception('Could not find exo torque for DOF %s.' %
                            dof_name)
                if np.sum(np.abs(ext_moment[ext_col])) != 0:
                    sum_actuators_shown += ext_moment[ext_col]
                    ax.plot(pgc_mod, ext_moment[ext_col], 
                        color= ext_color if ext_color else 'blue',
                        label= ext_name if ext_name else 'external',
                        ls='dashed',
                        linewidth=1.5)
                    dof_array.append(dof_name)
                    actuator_array.append(ext_name if ext_name else 'external')
                    all_moments.append(
                            np.interp(pgc_csv, pgc_mod, ext_moment[ext_col]))

        ax.plot(pgc, sum_actuators_shown,
                label='sum actuators shown', color='gray', linewidth=2)

        ax.plot(pgc, joint_moments[:, idof], label='net',
                color='black', linewidth=2)
        dof_array.append(dof_name)
        actuator_array.append('net')
        all_moments.append(
            np.interp(pgc_csv, pgc, joint_moments[:, idof]))

        ax.set_title(dof_name, fontsize=8)
        ax.set_ylabel('moment (N-m)', fontsize=8)
        ax.legend(frameon=False, bbox_to_anchor=(1, 1),
                loc='upper left', ncol=2, fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
    ax.set_xlabel('time (% gait cycle)', fontsize=8)

    fig.tight_layout()
    fig.savefig(pdf_path)
    pl.close(fig)

    multiindex_arrays = [dof_array, actuator_array]
    columns = pd.MultiIndex.from_arrays(multiindex_arrays,
            names=['DOF', 'actuator'])

    # Normalize by subject mass.
    if mass:
        all_moments_array = (np.array(all_moments).transpose() / mass)
        moments_df = pd.DataFrame(all_moments_array, columns=columns,
            index=pgc_csv)
        with file(csv_path, 'w') as f:
            f.write('# all columns are moments normalized by subject '
                'mass (N-m/kg).\n')
            moments_df.to_csv(f)

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

