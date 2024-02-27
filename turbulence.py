"""
Turbulence Project Data Analysis 
As part of Xin Wen's Ph.D. thesis
Authors: Xin Wen, Michael Fitzsimmons
version 1.0 04.18.2022
"""

from re import L
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import spe2py as spe
import spe_loader as sl
import getopt
import scipy.io as sio
import scipy.optimize as opt
from scipy import ndimage, misc
from datetime import datetime
from datetime import timedelta 
from dateutil import parser
import photutils.detection
import copy
import ipympl
import gc # release memroy
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
# For correlator
import itertools as it
from IPython.display import display, HTML
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Read in spe file and remove background Extract SpeNet and timestamps into numpy arrays
# version 1.2 
# Return xdim_raw and xdim_back both as xdim because they are the same.
# Instead of outputing BackAvg (256x256 array), output background (object)
def read_spe(BackFile, InputFile):

    DefaultDir=os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")

    if BackFile == None:
        usage()
        sys.exit()
    else:
        print(DefaultDir+BackFile)
        background = sl.load_from_files([DefaultDir+BackFile])
        nframes_back = background.nframes
        xdim_back = background.xdim[0]
        ydim_back = background.ydim[0]
        print("Number of frames in background = %s"%(nframes_back))
        print("Dimensions of a frame are %s by %s"%(xdim_back,ydim_back))
        back_avg = np.sum(background.data,axis=0,dtype=np.float32)/nframes_back	
    # the format is frames by x by y, we make no assumption about size of data matrix.

    if InputFile == None:
        usage()
        sys.exit()
    else:
        print(DefaultDir+InputFile)
        try:
            spe_raw = sl.load_from_files([DefaultDir+InputFile])
        except IOError:
            sys.exit('File not found: %s'%(DefaultDir+InputFile))        
        nframes_raw = spe_raw.nframes
        xdim_raw = spe_raw.xdim[0]
        ydim_raw = spe_raw.ydim[0]
        print("Number of frames in raw data = %s"%(nframes_raw))
        print("Dimensions of a frame are %s by %s"%(xdim_raw,ydim_raw))
    #    spe_net = np.zeros((nframes_raw,xdim_raw,ydim_raw), dtype=np.float32) # will contain raw - background
        
    # The next few lines extract time stamp meta data. start, stop and (false) duration
    # The start of every frame is timestamp
    # Units of time are in microseconds
        Exp_start = spe_raw.footer.SpeFormat.MetaFormat.MetaBlock.TimeStamp[0]['absoluteTime']
        Exp_start_time = parser.isoparse(Exp_start) ### Xin: Convert Exp_start from string to datetime object
        timestamp_array = spe_raw.metadata 
        timestamp_array[:,2] = timestamp_array[:,1] - timestamp_array[:,0]

    # the format is frames by x by y, we make no assumption about size of data matrix.

        spe_net=np.array(spe_raw.data)
        spe_net.resize(nframes_raw,xdim_raw,ydim_raw)
        if xdim_raw != xdim_back or ydim_raw != ydim_back:
            print('Input and Background data files are incompatible sizes.')
            print('%s %s %s %s'%(xdim_raw,ydim_raw,xdim_back,ydim_back))
            sys.exit()
        xdim = xdim_raw
        ydim = ydim_raw
        spe_net = spe_net - back_avg

    absolute_time_array = np.zeros(nframes_raw).astype(datetime) ### Xin: Create a new matrix to store absolute timestamps
    for i in range(nframes_raw):
        absolute_time_array[i] = Exp_start_time + timedelta(microseconds = timestamp_array[i,0]) # Convert timestamp from relative to absolute
        absolute_time_array[i] = absolute_time_array[i].strftime("%Y-%m-%d %H:%M:%S.%f") # Convert timestamp data type from datetime to string

    print("The absolute time of the start of the first frame = %s"%(Exp_start))
    print("The average intensity of one pixel in one frame of background = %s"%(np.sum(back_avg)/xdim_back/ydim_back))
    print("The average intensity of one pixel in one frame of raw data   = %s"%(np.sum(spe_raw.data)/xdim_raw/ydim_raw/nframes_raw))
    print("The average intensity of one pixel in one frame of net data   = %s"%(np.sum(spe_net)/xdim_raw/ydim_raw/nframes_raw))
    print("Total counts in net file = %s"%(np.sum(spe_net)))
    print(np.sum(spe_raw.data))

    # appears to be error with last recorded frame
    spe_net = spe_net[0:nframes_raw-1,:,:]
    timestamp_array = timestamp_array[0:nframes_raw-1,:]
    absolute_time_array = absolute_time_array[0:nframes_raw-1]
    nframes_raw = nframes_raw - 1

    # spe_net_2d = spe_net.reshape(spe_net.shape[0], -1) # Reshape spe_net from 3d to 2d array to save as a text file # This is too time-comsuming.

    if OutputFile == None:
        OFile = sys.stdout
    else:
        # print('Writing result as a numpy binary file: '+DefaultDir+OutputFile+'.npy')
        # np.save(DefaultDir+OutputFile,spe_net) # write the net result as a numpy binary file
        print('Writing background as a text file: '+DefaultDir+OutputFile+'_background_average.txt')
        np.savetxt(DefaultDir+OutputFile+"_background_average.txt",back_avg[0]) # write the background as a text file 
        print('Writing timestamps as a text file: '+DefaultDir+OutputFile+'_timestamps.txt')
        np.savetxt(DefaultDir+OutputFile+"_timestamps.txt",timestamp_array*1.e-6,header='Frame Start\tFrame End\tDuration', comments='', fmt='%.5f\t%.5f\t%.5f') # write the timestamp as a text file
        print('Writing absolute timestamp as a text file: '+DefaultDir+OutputFile+'_absolute_timestamps.txt')
        np.savetxt(DefaultDir+OutputFile+"_absolute_timestamps.txt",absolute_time_array,fmt='%30s') # write the absolute timestamp as a text file
        # print('Writing all as a matlab binary file: '+DefaultDir+OutputFile+'.mat')
        # sio.savemat(DefaultDir+OutputFile+".mat",{"spe_net":spe_net, "background":background.data, "back_avg":back_avg, "timestamp":timestamp_array})	

    # The last frame is suspect
    SpeNet = spe_net
    total_frame_number = len(SpeNet)
    BackAvg = back_avg
    TimeStampArray = timestamp_array
    # AbsoluteTimeArray = absolute_time_array

    return SpeNet, background, BackAvg, TimeStampArray, total_frame_number, xdim, ydim




## This function display the integrated frames over selected slices
## Version 2.0 
# Xin Wen 2022.09.02
def display_frame_integral(SpeNet, beginning_frame, ending_frame, Vmax):
    FrameIntegral = np.sum(SpeNet[beginning_frame:ending_frame],axis=0)
    fig, ax = plt.subplots(figsize = (10,10))
    plt.imshow(np.flip(FrameIntegral,axis=0), origin='lower',extent=([-5,5,-5,5]),interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax))
    plt.xticks((np.linspace(-5,5,11)))
    plt.yticks((np.linspace(-5,5,11)))
    plt.xlabel('Horizaontal position (mm)', fontsize = 16)
    plt.ylabel('Vertical position (mm)', fontsize = 16)
    plt.colorbar(shrink = 0.8)
    plt.title('Integrateion over all frames', fontsize = 18)
    plt.savefig('Integrateion over all frames.jpg',dpi = 300)
    plt.show()




# New function to identify bad pixel and create a bad pixel mask
# version 2.0
# It returns a 256x256 mask of 1 and 0 of bad pixels and a list of the coordinates of the bad pixels, and number of bad pixels
def find_bad_pixel(InputFile, Background_spe_object, Bsig):
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    back_raw = Background_spe_object.data
    back_raw = np.array(back_raw) # Convert list to Numpy array
    nframes_back = Background_spe_object.nframes-1
    xdim_back = Background_spe_object.xdim[0]
    ydim_back = Background_spe_object.ydim[0]
    back_raw.resize(nframes_back,xdim_back,ydim_back) # Resize

    BackRaw = copy.deepcopy(back_raw) # Background raw data (1500,256,256)
    BackRawAvg = np.sum(BackRaw[0:nframes_back],axis=0)/nframes_back
    back_mean_raw = np.mean(BackRawAvg)
    back_stdev_raw = np.std(BackRawAvg)
    print('Back Mean Raw: %.3f\nBack Stdev Raw: %3f'%(back_mean_raw, back_stdev_raw))
    vmin_raw = back_mean_raw - Bsig * back_stdev_raw
    vmax_raw = back_mean_raw + Bsig * back_stdev_raw

    #############################################################################################################
    ## Flatten/Normalize the background
    BackRaw_norm = np.empty(np.shape(BackRaw)) # Array to store the flattened background to calculate mean and standard deviation
    integral = np.trapz(np.transpose(BackRawAvg))
    integral_avg = np.mean(integral)
    for i in range(nframes_back):
        for column in range(xdim_back):
            BackRaw_norm[i][:,column] = integral_avg * BackRaw[i][:,column]/integral[column]
    BackNormAvg = np.sum(BackRaw_norm[0:nframes_back],axis=0)/nframes_back
    plt.figure(figsize = (10,10))
    plt.imshow(np.flip(BackNormAvg,axis=0), origin='lower',extent=([0,256,0,256]),interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=vmin_raw, vmax=vmax_raw))
    plt.colorbar()
    # plt.savefig('Background_norm_avg.jpg', dpi=300)
    plt.show()

    back_mean_norm = np.mean(BackNormAvg)
    back_stdev_norm = np.std(BackNormAvg)
    print('Back Mean Norm: %.3f\nBack Stdev Norm: %3f'%(back_mean_norm, back_stdev_norm))
    bthreshold = back_mean_norm + Bsig * back_stdev_norm
    print('%d sigma threshold: %.3f' % (Bsig,bthreshold))

    ###############################################################################################################
    ## Find pixels that are n-sigma above background
    # if 'BackNsig' not in globals(): # Only need to run this once. If the background already exists, skip the step and go ahead to plot the background
    BackNsig = copy.deepcopy(BackRaw_norm)
    for i in range(BackNsig.shape[0]):
        frame = BackNsig[i]
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                if frame[x][y] < bthreshold:
                    frame[x][y] = 0
    #             else:
    #                 frame[x][y] = 1
        BackNsig[i]=frame

    BackNsigAvg = np.sum(BackNsig[0:nframes_back],axis=0)/nframes_back
    back_mean_nsig = np.mean(BackNsigAvg)
    back_stdev_nsig = np.std(BackNsigAvg)
    print('Back Mean Nsig: %.3f\nBack Stdev Nsig: %3f'%(back_mean_nsig, back_stdev_nsig))
    vmin_nsig = back_mean_nsig - Bsig * back_stdev_nsig
    vmax_nsig = back_mean_nsig + Bsig * back_stdev_nsig

    plt.figure(figsize = (10,10))
    plt.imshow(np.flip(BackNsigAvg,axis=0), origin='lower',extent=([0,256,0,256]),interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=vmax_nsig))
    plt.colorbar()
    plt.show()
    # print(BackNsigIntegral)

    #################################################################################################################
    # Generate the bad pixel mask
    bad_pixel_mask = copy.deepcopy(BackNsigAvg)
    bthreshold = back_mean_nsig + Bsig * back_stdev_nsig
    print('%d sigma threshold: %.3f\n' % (Bsig,bthreshold))
    for x in range(bad_pixel_mask.shape[0]):
        for y in range(bad_pixel_mask.shape[1]):
            if bad_pixel_mask[x][y] <= bthreshold:
                bad_pixel_mask[x][y] = False
            else:
                bad_pixel_mask[x][y] = True
    # bad_pixel_mask[0:2,:] = True
    # bad_pixel_mask[254:256,:] = True
    # bad_pixel_mask[:,0:2] = True
    # bad_pixel_mask[:,254:256] = True

    np.savetxt(OutputFile+'_bad_pixel_mask.txt', bad_pixel_mask, fmt = '%d')
    plt.figure(figsize = (10,10))
    plt.imshow(np.flip(bad_pixel_mask,axis=0), origin='lower',extent=([0,256,0,256]),interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=1))
    plt.colorbar()
    plt.savefig('Background mask.jpg', dpi=300)
    plt.show()

    bad_pixel_list = np.argwhere(bad_pixel_mask == True) # Here the x and y index are actually swapped. We need to swap back.
    bad_pixel_list[:, [1, 0]] = bad_pixel_list[:, [0, 1]]
    np.savetxt(OutputFile+'_bad_pixel_list.txt',bad_pixel_list,fmt = '%d\t%d',  delimiter='\t', header='x\ty\t', comments ='', encoding=None)
    bad_pixel_number = bad_pixel_list.shape[0]
    print('Number of bad pixels is %d'% bad_pixel_number)

    return bad_pixel_mask, bad_pixel_list, bad_pixel_number






## Find peaks using photutils.detection.find_peaks
# version 2.0 
# Allow changing box_size in the photutils.detection.find_peaks function from main code. Default value used in 2nd paper is box_size = 5
# Save original peaks in a array format
def find_peaks(nsig, SpeNet, InputFile, Box_size):
    # Define the threshold to be nsig above the mean of background per our PRL, not the frame.
    # Remember that the mean of the background has already been removed from SpeNet
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    SpeNet_stdev = 15 # This is the standard deviation calculated from frames of lasers on neutrons off
    # nsig = 4
    dthreshold = nsig * SpeNet_stdev
    total_peaks_number = 0
    original_peaks_log = open(OutputFile+'_original_peaks.txt','w')
    peaks_list = list() # Create a list to store all peaks_mat
    total_frame_number = len(SpeNet)
    print ('%d sigma threshold = %s'%(nsig, dthreshold))

    for frame_number in range(total_frame_number): 
        frame_peaks_table = photutils.detection.find_peaks(SpeNet[int(frame_number)],threshold = dthreshold, box_size = Box_size)
        if frame_peaks_table != None:
            number_peaks = np.size(frame_peaks_table)
            total_peaks_number += number_peaks
            peaks_mat = np.empty([number_peaks,4])
            for i in range(number_peaks):
                peaks_mat[i][0] = frame_number
                peaks_mat[i][1] = frame_peaks_table[i][0]
                peaks_mat[i][2] = frame_peaks_table[i][1]
                peaks_mat[i][3] = frame_peaks_table[i][2]
        peaks_list.append(peaks_mat)
    original_peaks = np.vstack(peaks_list)
    print(original_peaks)
    print ("Total number of original peaks: %s"%total_peaks_number)
    np.savetxt(OutputFile+'_original_peaks.txt',original_peaks,fmt = '%d\t%d\t%d\t%.3f',  delimiter='\t', header='Frame\tx\ty\tIntensity', comments ='', encoding=None)

    return peaks_list, original_peaks, total_peaks_number





# ## Remove peaks coinciding with bad pixels
# # version 2.0
def remove_bad_pixels(InputFile, original_peaks, bad_pixel_list, xdim, ydim, DoPlot, Vmax):
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    print('Number of peaks before culling: %d'%original_peaks.shape[0])
    # # original_peaks = original_peaks[original_peaks[:, 0].argsort()]
    # original_peaks_byframe = np.split(original_peaks,np.flatnonzero(np.diff(original_peaks[:,0]))+1) # split the original peaks by frame
    clean_peaks = copy.deepcopy(original_peaks)
    for bad in bad_pixel_list:
        clean_peaks = np.delete(clean_peaks, np.where((clean_peaks[:,1] == bad[0]) & (clean_peaks[:,2] == bad[1]))[0], axis=0)
    print('Number of peaks after culling: %d'%clean_peaks.shape[0])
    np.savetxt(OutputFile+'_clean_peaks.txt',clean_peaks,fmt = '%d\t%d\t%d\t%.3f',  delimiter='\t', header='Frame\tx\ty\tIntensity', comments ='', encoding=None)
    if DoPlot:
        # Plot the remaining peaks
        x = clean_peaks[:,1]
        y = clean_peaks[:,2]

        image = np.zeros((xdim, ydim))
        images = np.zeros((xdim, ydim))
        for i in range(len(clean_peaks)):
            image[int(y[i]), int(x[i])] = 1
            images += image

        plt.figure(figsize = (10,10))
        plt.imshow(np.flip(images,axis = 0), origin='lower',extent=([0,255,0,255]),interpolation='bilinear', cmap=cm.viridis, vmin = 0, vmax = Vmax)
        plt.xlabel('Horizontal Direction')
        plt.ylabel('Vertical Direction')
        plt.colorbar()
        plt.savefig('Clean Peaks.jpg', dpi = 300)
        plt.show()
    
    return clean_peaks




## Frame Binning of SpeNet
def rebin(a, newshape): # This function can have two functions. For 2d array, it change resolution. For 3d array, it integrate frames
    # Single frame binning (Change resolution)
    if len(np.shape(a)) == 2: 
        sh2 = newshape[0],a.shape[0]//newshape[0],newshape[1],a.shape[1]//newshape[1]
        reshaped = a.reshape(sh2).mean(-1).mean(1)
    # Multiple frame binning (Integrate frames)
    elif len(np.shape(a)) == 3: 
        sh3 = newshape[0],a.shape[0]//newshape[0],newshape[1],a.shape[1]//newshape[1], newshape[2],a.shape[2]//newshape[2]
        reshaped = a.reshape(sh3).sum(1).mean(-1).sum(2)
    return reshaped



## Display peaks left after removing bad pixels
# version 1.2
# Xin Wen 2022.09.10
def display_good_peaks(SpeNet, clean_peaks, TimeStampArray, InputFile, axis):
    DefaultDir = os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    # Creat sub diectory 1 to store integrated movie and frames 
    frame_subfolder = str(DefaultDir)+'/peak_frames' 
    if not os.path.exists(frame_subfolder):
        os.makedirs(frame_subfolder)
        print("Created Directory : ", frame_subfolder)
    else:
        print("Directory already existed : ", frame_subfolder)
    os.chdir(frame_subfolder)
    clean_peaks_byframe = np.split(clean_peaks,np.flatnonzero(np.diff(clean_peaks[:,0]))+1)
    for frame_number in range(len(SpeNet)):
        time = TimeStampArray[frame_number,0] * 1e-6
        plt.figure(figsize = (8,8))
        plt.scatter(clean_peaks_byframe[frame_number][:,1]*10/256-5,(256 - clean_peaks_byframe[frame_number][:,2])*10/256-5,s=3,c='k')
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        if axis is True: # Plot with axis
            plt.xticks(np.linspace(-5,5,11), fontsize = 18)
            plt.yticks(np.linspace(-5,5,11), fontsize = 18)
            plt.xlabel('Horizontal position (mm)', fontsize=20)
            plt.ylabel('Vertical postition (mm)', fontsize=20)
            plt.title('Time = %.3f s'% time, fontsize = 20)
            plt.savefig(str(frame_number).zfill(4)+'_Time=%.3f.pdf'% time,  bbox_inches='tight',pad_inches = 0.5) 
            plt.close()
        else: # Plot without axis
            plt.axis('off')
            plt.savefig(str(frame_number).zfill(4)+'_Time=%.3f.jpg'% time, dpi = 80,  bbox_inches='tight',pad_inches = 0.5) 
            plt.close()
    cmd = 'rm '+OutputFile+'_peaks.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -pattern_type glob -i "*Time*.jpg" -vf '+'"pad=ceil(iw/2)*2:ceil(ih/2)*2" '+OutputFile+'_peaks.mp4'
    f = os.system(cmd)
    if f != 0:
        print('Error in ffmpeg command execution: %s = %s'%(cmd,f))
    cmd = 'rm '+OutputFile+'_peaks_4xslow.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -i '+OutputFile+'_peaks.mp4 '+'-filter:v "setpts = 4*PTS" '+OutputFile+'_peaks_4xslow.mp4' # Slow down the movie
    f = os.system(cmd)
    os.chdir(DefaultDir) # Go back to previous working directory




## version 1.1
# Xin Wen Remove redundant conditions 2022.09.06
def display_integrated_frames(InputFile, SpeNet, TimeStampArray, TimeWindow, Vmax):
    DefaultDir=os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    # Creat sub diectory 1 to store integrated movie and frames 
    frame_subfolder = str(DefaultDir)+'/integrated_movie_frames' 
    if not os.path.exists(frame_subfolder):
        os.makedirs(frame_subfolder)
        print("Created Directory : ", frame_subfolder)
    else:
        print("Directory already existed : ", frame_subfolder)
    FrameIntegral = np.sum(SpeNet,axis=0)
    # TimeIntegral = np.sum(np.sum(SpeNet,axis=2),axis=1)
    os.chdir(frame_subfolder)
    
    size = np.shape(SpeNet)[1]
    TMin = np.min(TimeStampArray[:,0]/1e6)
    TMax = np.max(TimeStampArray[:,0]/1e6)
    NLoop = int((TMax-TMin)/TimeWindow)
    VProjection = np.zeros((NLoop+1,size),dtype=np.float32)
    TimeAxis = np.zeros(NLoop+1,dtype=np.float32)
    for i in range(NLoop+1):
        print(i,NLoop,TMin,TMax,TimeWindow)
        FMin = np.min(np.argwhere(TimeStampArray[:,0]/1e6>TMin+i*TimeWindow))
        FMax = np.max(np.argwhere(TimeStampArray[:,0]/1e6<TMin+(i+1)*TimeWindow))
        FMax = np.min([FMax,len(SpeNet)])
        FBar = int((FMax+FMin)/2)
        TBar = TimeStampArray[FBar,0]*1e-6
        TimeAxis[i] = TBar
        FrameIntegral = np.sum(SpeNet[FMin:FMax,:,:],axis=0)
        VProjection[i,:] = np.sum(FrameIntegral, axis=1)
        # vmax = np.mean(FrameIntegral) + 3*np.std(FrameIntegral)
        # vmin = np.max([0,np.mean(FrameIntegral) - 3*np.std(FrameIntegral)])
        # print('Computer image limits: %s %s'%(vmin,vmax))
# Plot image
        plt.figure(figsize = (8,8))
        # plt.figure(figsize = (16,7))
        # plt.subplot(1,2,1)
        plt.xlim((-5,5))
        plt.ylim((-5,5))
        plt.xlabel('Horizontal position (mm)', fontsize=20)
        plt.ylabel('Vertical position (mm)', fontsize=20)
        plt.xticks(np.arange(-5, 6, step = 1), fontsize = 18)
        plt.yticks(np.arange(-5, 6, step = 1), fontsize = 18)
        plt.title('Time = %.2f s'%TBar, fontsize = 20)
        plt.imshow(np.flip(FrameIntegral,axis=0), origin='lower',extent=([-5,5,-5,5]),interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax= Vmax))
        plt.colorbar(shrink = 0.8)
# Plot integrated intensity vs. time
        # plt.subplot(1,2,2)
        # plt.xlabel('Time (s)', fontsize=12)
        # plt.ylabel('Integrated Intensity (counts)', fontsize=12)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        # plt.title(InputFile, fontsize=10)
        # plt.plot(TimeStampArray[:,0]*1e-6,TimeIntegral,'bo',markersize=4)
        # plt.plot(TimeStampArray[FMin:FMax,0]*1e-6,TimeIntegral[FMin:FMax],'ro',markersize=4)
        plt.savefig(str(i).zfill(4)+'xTEMP.jpg', dpi = 80, bbox_inches = 'tight', pad_inches = 0.5) # Saving the net integrated in a time window
        plt.show()
        plt.close()
    cmd = 'rm '+OutputFile+'.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -pattern_type glob -i "*xTEMP.jpg" -vf '+'"pad=ceil(iw/2)*2:ceil(ih/2)*2" '+OutputFile+'.mp4'
    f = os.system(cmd)
    if f != 0:
        print('Error in ffmpeg command execution: %s = %s'%(cmd,f))
    cmd = 'rm '+OutputFile+'_4xslow.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -i '+OutputFile+'.mp4 '+'-filter:v "setpts = 4*PTS" '+OutputFile+'_4xslow.mp4' # Slow down the movie
    f = os.system(cmd)
    os.chdir(DefaultDir) # Go back to previous working directory

    return VProjection





## Version 1.1
# Xin Wen Remove redundant conditions 2022.09.06
# Combine display_net and display_frames into one function because they are the same thing.
def display_net_frames(SpeNet, InputFile, TimeStampArray, Vmax):
    DefaultDir =os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    # Creat sub diectory 2 to store unintegrated movie and frames 
    frame_subfolder2 = str(DefaultDir)+'/net_frames'
    if not os.path.exists(frame_subfolder2):
        os.makedirs(frame_subfolder2)
        print("Created Directory : ", frame_subfolder2)
    else:
        print("Directory already existed : ", frame_subfolder2)
    os.chdir(frame_subfolder2)

    for i in range(len(SpeNet)):
        e_image = SpeNet[i,:,:] # This flip will orient up to up and left to left from camera's view
        plt.figure(figsize = (8,8))
        plt.imshow(np.flip(e_image,axis=0), aspect='equal', origin='lower',interpolation='bilinear', extent=(-5,5,-5,5), cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax)) #cmap=cm.Greys
        ## Circle the peaks above cut-off intensity (birth clouds)
        # birth = birth_peaks_list[iter_frame][:,:]
        # if len(birth) != 0: # Circle the peaks above cut-off intensity (birth clouds)
        #     for p in birth:
        #         plt.scatter(10*p[0]/256-5,10*(255-p[1])/256-5, s=80, facecolors='none', edgecolors='r')
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.xticks(np.linspace(-5,5,11), fontsize = 18)
        plt.yticks(np.linspace(-5,5,11), fontsize = 18)
        plt.xlabel('Horizontal position (mm)', fontsize=20)
        plt.ylabel('Vertical postition (mm)', fontsize=20)
        # plt.colorbar(shrink = 0.8)
        time = TimeStampArray[i,0] * 1e-6
        plt.title('Time = %.3f s'% time, fontsize = 20)
        plt.savefig(str(i).zfill(4)+'_Time=%.3f.jpg'% time, dpi = 80, bbox_inches = 'tight', pad_inches = 0.5)
        plt.close()
    cmd = 'rm '+OutputFile+'_net_frames.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -pattern_type glob -i "*Time*.jpg" '+OutputFile+'_net_frames.mp4'
    f = os.system(cmd)
    if f != 0:
        print('Error in ffmpeg command execution: %s = %s'%(cmd,f))
    cmd = 'rm '+OutputFile+'_net_frames_4xslow.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -i '+OutputFile+'_net_frames.mp4 '+'-filter:v "setpts = 4*PTS" '+OutputFile+'_net_frames_4xslow.mp4' # Slow down the movie
    f = os.system(cmd)
    os.chdir(DefaultDir) # Go back to previous working directory



def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma



def FlushCalculation(xdata_tuple,ACoeff0,ACoeff1,ACoeff2,mP,bP,Sigma,TCoeff0,TCoeff1,TCoeff2,VCoeff1,VCoeff2):
    (t0,v0) = xdata_tuple 
    RLength = np.abs(v0 - mP*t0 - bP)/np.sqrt(1 + mP**2) 
    tP = (mP*v0+t0-mP*bP) / (1 + mP**2)
    A0 = ACoeff0+ACoeff1*tP+ACoeff2*tP**2 # amplitude at the closest approach
    Background = TCoeff0 + TCoeff1*t0 + TCoeff2*t0**2 + VCoeff1*v0 + VCoeff2*v0**2
    g = A0*np.exp(-0.5*(RLength/Sigma)**2) + Background 
    return g.ravel()



## 3D plot of itensity vs vertical position and time
# Version 2.0 Xin Wen 2022.09.06
def threeD_plot(InputFile, SpeNet, VProjection, TimeWindow, TimeStampArray):
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    TMin = np.min(TimeStampArray[:,0]/1e6)
    TMax = np.max(TimeStampArray[:,0]/1e6)
    NLoop = int((TMax-TMin)/TimeWindow)
    TimeAxis = np.zeros(NLoop+1,dtype=np.float32)
    for i in range(NLoop+1):
        FMin = np.min(np.argwhere(TimeStampArray[:,0]/1e6>TMin+i*TimeWindow))
        FMax = np.max(np.argwhere(TimeStampArray[:,0]/1e6<TMin+(i+1)*TimeWindow))
        FMax = np.min([FMax,len(SpeNet)])
        FBar = int((FMax+FMin)/2)
        TBar = TimeStampArray[FBar,0]*1e-6
        TimeAxis[i] = TBar
    #Flip VProjection
    VProjection = np.flip(VProjection,axis=1)
    # Make a 3D plot of VProjection vs. TimeAxis (yaxis) and VDistance (xaxis)
    window = 10
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Vertical position (mm)', fontsize=18)
    ax.set_ylabel('Time (s)', fontsize=18)
    ax.set_zlabel('Signal (counts)', fontsize=18)
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
    for i in range(NLoop):
        VProjectionSmooth = movingaverage(VProjection[i,:],window) # averge over window number in the vertical
        VDistance = np.linspace(-5,5,num=len(VProjectionSmooth))
        ax.scatter(VDistance,TimeAxis[i],VProjectionSmooth,'bo')
    plt.savefig(str(OutputFile)+'_3DPlot.jpg', dpi = 300) # Saving the net integrated in a time window
    plt.show()






def contour_plot(InputFile, SpeNet, binning, TimeStampArray, TimeWindow, Vmax, Levels, xLimit, yLimit):
    DefaultDir=os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    # Creat sub diectory to store contour frames 
    frame_subfolder = str(DefaultDir)+'/contour_frames'
    if not os.path.exists(frame_subfolder):
        os.makedirs(frame_subfolder)
        print("Created Directory : ", frame_subfolder)
    else:
        print("Directory already existed : ", frame_subfolder)
    os.chdir(frame_subfolder)

    size = np.shape(SpeNet)[1]
    newshape = (size//binning, size//binning)
    shape_x = (size,1)
    shape_y = (1,size)
    FrameIntegral = np.sum(SpeNet,axis=0)
    TimeIntegral = np.sum(np.sum(SpeNet,axis=2),axis=1)
    TMin = np.min(TimeStampArray[:,0]/1e6)
    TMax = np.max(TimeStampArray[:,0]/1e6)
    NLoop = int((TMax-TMin)/TimeWindow)
    VProjection = np.zeros((NLoop+1,size),dtype=np.float32)
    TimeAxis = np.zeros(NLoop+1,dtype=np.float32)
    for i in range(NLoop+1):
        FMin = np.min(np.argwhere(TimeStampArray[:,0]/1e6>TMin+i*TimeWindow))
        FMax = np.max(np.argwhere(TimeStampArray[:,0]/1e6<TMin+(i+1)*TimeWindow))
        FMax = np.min([FMax,len(SpeNet)])
        FBar = int((FMax+FMin)/2)
        TBar = TimeStampArray[FBar,0]*1e-6
        TimeAxis[i] = TBar
        FrameIntegral = np.sum(SpeNet[FMin:FMax,:,:],axis=0)
        VProjection[i,:] = np.sum(FrameIntegral, axis=1)
        # vmax = np.mean(FrameIntegral) + 3*np.std(FrameIntegral)
        # vmin = np.max([0,np.mean(FrameIntegral) - 3*np.std(FrameIntegral)])
        # print('Computer image limits: %s %s'%(vmin,vmax))
        plt.figure(figsize=(14,14))
        plt.subplot(2,2,1)
        plt.xlabel('Horizontal position (mm)', fontsize=12)
        plt.ylabel('Vertical postition (mm)', fontsize=12)
        plt.title('Time = %.2f (s)'%TBar)
        plt.imshow(np.flip(FrameIntegral,axis=0), origin='lower',extent=([-5,5,-5,5]),interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax))
        plt.colorbar(shrink = 0.8)
    # Plot contour
        # Levels =np.linspace(34, 50, 1, dtype = np.float32) # For 0.1s integration
        con_x = np.linspace(-5,5,size//binning)
        con_y = np.linspace(-5,5,size//binning)
        plt.contour(con_x, con_y, np.flip(rebin(FrameIntegral,newshape),axis=0),levels=Levels, colors='w', linestyles='-')
    # Plot 1x256
        plt.subplot(2,2,2)
        plt.xlabel('Intensity Integrated along \n the Horizontal Direction', fontsize=12)
        plt.ylabel('y position (mm)', fontsize=12)
    #         plt.xlim((0,200))
        plt.xlim((0,xLimit))
        plt.ylim((-5,5))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.plot(np.flip(rebin(FrameIntegral,shape_x),axis = 0),np.linspace(-5,5,size))
    # Plot 256x1
        plt.subplot(2,2,3)
        plt.xlabel('x position (mm)', fontsize=12)
        plt.ylabel('Intensity Integrated along \n the Vertical Direction', fontsize=12)
        plt.xlim((-5,5))
        plt.ylim((0,yLimit))
        plt.ticklabel_format(axis="y", style="plain", scilimits=(0,0))
        plt.plot(np.linspace(-5,5,size),np.transpose(rebin(FrameIntegral,shape_y)))
    # Plot integrated intensity vs. time
        plt.subplot(2,2,4)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Integrated Signal Intensity', fontsize=12)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #         plt.title(InputFile[6:-8], fontsize=10)
        plt.plot(TimeStampArray[:,0]*1e-6,TimeIntegral,'bo',markersize=4)
        plt.plot(TimeStampArray[FMin:FMax,0]*1e-6,TimeIntegral[FMin:FMax],'ro',markersize=4)
        plt.savefig(str(i).zfill(4)+'xTEMP.jpg', dpi = 300) # Saving the net integrated in a time window
        plt.close()
    cmd = 'rm '+OutputFile+'_contour.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -pattern_type glob -i "*xTEMP.jpg" '+OutputFile+'_contour.mp4'
    f = os.system(cmd)
    if f != 0:
        print('Error in ffmpeg command execution: %s = %s'%(cmd,f))
    cmd = 'rm '+OutputFile+'_contour_4xslow.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -i '+OutputFile+'_contour.mp4 '+'-filter:v "setpts = 4*PTS" '+OutputFile+'_contour_4xslow.mp4' # Slow down the movie
    f = os.system(cmd)
    os.chdir(DefaultDir) # Go back to previous working directory



def integrated_plot(SpeNet, InputFile, TimeStampArray, Vmax):
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    ## Show an image of the signal vs time and (left) vertical and (right) horizontal position
    VProjection = np.sum(SpeNet, axis = 2) # Appears to integrate along the horizontal dimension
    HProjection = np.sum(SpeNet, axis = 1) # Appears to integrate along the horizontal dimension
    TMin = np.min(TimeStampArray[:,0]/1e6)
    TMax = np.max(TimeStampArray[:,0]/1e6)

    # Plot VProjection
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.xlabel('Vertical pixel', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.imshow(np.flip(VProjection,axis=1), origin='lower', aspect='auto', extent=([0,255,TMin,TMax]), interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax))
    plt.colorbar()
    # plt.savefig(str(OutputFile)+'_VProjection.jpg')
    # Plot HProjection
    plt.subplot(1,2,2)
    plt.xlabel('Horizontal pixel', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.imshow(np.flip(HProjection,axis=1), origin='lower', aspect='auto', extent=([0,255,TMin,TMax]), interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax))
    plt.colorbar()
    # plt.savefig(str(OutputFile)+'_HProjection.jpg')
    plt.savefig(str(OutputFile)+'_Projections.jpg')
    plt.show()

    ## Show a smoothed image of the signal vs time and (left) vertical and (right) horizontal position
    VProjectionSmooth = ndimage.uniform_filter(VProjection, size=[32,8])
    HProjectionSmooth = ndimage.uniform_filter(HProjection, size=[32,8])
    # Plot Smoothed VProjection
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.xlabel('Vertical pixel', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.imshow(np.flip(VProjectionSmooth,axis=1), origin='lower', aspect='auto', extent=([0,255,TMin,TMax]), interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax))
    plt.colorbar()
    # Plot Smoothed HProjection
    plt.subplot(1,2,2)
    plt.xlabel('Horizontal pixel', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.imshow(np.flip(HProjectionSmooth,axis=1), origin='lower', aspect='auto', extent=([0,255,TMin,TMax]), interpolation='bilinear', cmap=cm.viridis, norm=colors.Normalize(vmin=0, vmax=Vmax))
    plt.colorbar()
    plt.savefig(str(OutputFile)+'_smoothed_Projections.jpg')
    plt.show()





def clustering(SpeNet, InputFile, BackFile, nsig, TimeStampArray, clean_peaks, n_cluster_min, Bandwidth, quantile,frame_format):
    DefaultDir=os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    # Creat sub diectory 3 to store unintegrated movie and frames 
    frame_subfolder = str(DefaultDir)+'/cluster_frames'
    if not os.path.exists(frame_subfolder):
        os.makedirs(frame_subfolder)
        print("Created Directory : ", frame_subfolder)
    else:
        print("Directory already existed : ", frame_subfolder)

    clean_peaks_byframe = np.split(clean_peaks,np.flatnonzero(np.diff(clean_peaks[:,0]))+1)

    cluster_results = open(OutputFile+'_cluster_results.txt', 'w')
    cluster_results.write(' Fr#  Cl#    N        x    sig_x        y    256-y    sig_y     SigSum\n')
    bandwidth_results = open(OutputFile+'_bandwidth_results_q='+str(quantile)+'_Nmin'+str(n_cluster_min)+'.txt', 'w')
    bandwidth_results.write(' Frame\tbandwidth\t#Peaks\t#Clusters\n')
    # SpeNet_stdev = 15 # This is the standard deviation calculated from frames of lasers on neutrons off
    # dthreshold = nsig * SpeNet_stdev
    background = sl.load_from_files([DefaultDir+BackFile])
    xdim_back = background.xdim[0]
    ydim_back = background.ydim[0]

    for iter_frame in range(len(SpeNet)):
        if np.shape(clean_peaks_byframe[iter_frame])[0] == 0:
            n_peaks = 0
            print('Number of peaks in frame: %d'%n_peaks)

            fig, ax = plt.subplots(figsize=(8, 8))
            plt.clf()
            # 1st plot of data
            # e_image = SpeNet[iter_frame,:,:] # This flip will orient up to up and left to left from camera's view
            # plt.imshow(np.flip(e_image,axis=0), aspect='equal', origin='lower', extent=(-5,5,-5,5), alpha=0.8, cmap=cm.Greys, vmin=0, vmax=dthreshold) #cmap=cm.Greys
    #         plt.xlim([0,xdim_back-1])
    #         plt.ylim([0,ydim_back-1])
            plt.xlabel('Horizontal position (mm)', fontsize=20)
            plt.ylabel('Vertical position (mm)', fontsize=20)
            plt.xlim((-5,5))
            plt.ylim((-5,5))
            plt.xticks(np.arange(-5, 6, step = 1), fontsize = 18)
            plt.yticks(np.arange(-5, 6, step = 1), fontsize = 18)
            ax.add_patch(Rectangle((23*10/256-5, 164*10/256-5), 161*10/256,28*10/256, edgecolor = 'green',facecolor = 'green',alpha = 0.5, fill=True, lw = 1))

            time = TimeStampArray[iter_frame,0] * 1e-6
            plt.title('Time = %.2f s'% time, fontsize=20)

            if frame_format == 'jpg':
                    plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+"_Time=%.3f.jpg"%time, dpi = 80, bbox_inches = 'tight', pad_inches = 0.5)
            elif frame_format == 'pdf':
                    plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+"_Time=%.3f.pdf"%time, bbox_inches = 'tight', pad_inches = 0.5)
            plt.close()

        elif np.shape(clean_peaks_byframe[iter_frame])[0] != 0:
            # Select the x,y coordinates from a specific frame
            X = np.zeros((clean_peaks_byframe[iter_frame].shape[0],2), dtype=np.float32)
            weights = np.zeros((clean_peaks_byframe[iter_frame].shape[0]), dtype=np.float32)
            for peak in range(clean_peaks_byframe[iter_frame].shape[0]):
                X[peak,0] = clean_peaks_byframe[iter_frame][peak][1]
                X[peak,1] = clean_peaks_byframe[iter_frame][peak][2]
                weights[peak] = clean_peaks_byframe[iter_frame][peak][3]
            #    weights[peak] = 1.
            print('Frame = %s'%iter_frame)

            # Compute clustering with MeanShift
            # seems to work remarkably well

            # The following bandwidth can be automatically detected using
            # quantile is float 0 to 1., 0.5 means median of all pairwise distances used
            if Bandwidth == "auto":
                bandwidth = estimate_bandwidth(X, quantile = quantile)
                print('Estimated bandwidth: %.3f'%bandwidth)
                
            else: 
                bandwidth = Bandwidth
                print('Fixed bandwidth: %.1f'% bandwidth)

            if bandwidth == 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                plt.clf()
                # 1st plot of data
                e_image = SpeNet[iter_frame,:,:] # This flip will orient up to up and left to left from camera's view
                # plt.imshow(np.flip(e_image,axis=0), aspect='equal', origin='lower', extent=(-5,5,-5,5), alpha=0.8, cmap=cm.Greys, vmin=0, vmax=dthreshold) #cmap=cm.Greys
        #         plt.xlim([0,xdim_back-1])
        #         plt.ylim([0,ydim_back-1])
                plt.plot(clean_peaks_byframe[iter_frame][:,1]*10/256-5, (256-clean_peaks_byframe[iter_frame][:,2])*10/256-5, 'o', markerfacecolor='grey',
                            markeredgecolor='w', markersize = 5,  alpha = 0.8)
                plt.xlim((-5,5))
                plt.ylim((-5,5))
                plt.xlabel('Horizontal position (mm)', fontsize=20)
                plt.ylabel('Vertical position (mm)', fontsize=20)
                plt.xticks(np.arange(-5, 6, step = 1), fontsize = 18)
                plt.yticks(np.arange(-5, 6, step = 1), fontsize = 18)
                ax.add_patch(Rectangle((23*10/256-5, 164*10/256-5), 161*10/256,28*10/256, edgecolor = 'green',facecolor = 'green',alpha = 0.5, fill=True, lw = 1))
                time = TimeStampArray[iter_frame,0] * 1e-6
                plt.title('Time = %.2f s'% time, fontsize=20)

                if frame_format == 'jpg':
                    plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+"_Time=%.3f.jpg"%time, dpi = 80, bbox_inches = 'tight', pad_inches = 0.5)
                elif frame_format == 'pdf':
                    plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+"_Time=%.3f.pdf"%time, bbox_inches = 'tight', pad_inches = 0.5)
                # plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+'.jpg', dpi = 80,  bbox_inches = 'tight', pad_inches = 0.5)
                plt.close()

            elif bandwidth != 0:
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
                ms.fit(X)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                n_clusters_ = len(labels_unique)
                print("Estimated number of clusters: %d" % n_clusters_)
                # #############################################################################
                # Plot result
                # flip per the camera view
                not_culled = 0
                l_max = 0
                for k in range(n_clusters_):
                    my_members = labels == k
                    l = sum(my_members)
                    if(l > l_max):
                        l_max = l
                    if (l > n_cluster_min):
                        not_culled = not_culled + 1

                fig, ax = plt.subplots(figsize=(8, 8))
                plt.clf()
                # 1st plot of data
                e_image = SpeNet[iter_frame,:,:] # This flip will orient up to up and left to left from camera's view
                # plt.imshow(np.flip(e_image,axis=0), aspect='equal', origin='lower', extent=(-5,5,-5,5), alpha=0.8, cmap=cm.Greys, vmin=0, vmax=dthreshold) #cmap=cm.Greys
                plt.plot(clean_peaks_byframe[iter_frame][:,1]*10/256-5, (256-clean_peaks_byframe[iter_frame][:,2])*10/256-5, 'o', markerfacecolor='grey',
                            markeredgecolor='w', markersize = 6,  alpha = 0.8)
                plt.xlim((-5,5))
                plt.ylim((-5,5))
                plt.xlabel('Horizontal position (mm)', fontsize=20)
                plt.ylabel('Vertical position (mm)', fontsize=20)
                plt.xticks(np.arange(-5, 6, step = 1), fontsize = 18)
                plt.yticks(np.arange(-5, 6, step = 1), fontsize = 18)
                ax.add_patch(Rectangle((23*10/256-5, 164*10/256-5), 161*10/256,28*10/256, edgecolor = 'green',facecolor = 'green',alpha = 0.5, fill=True, lw = 1))

                colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
                for k, col in zip(range(n_clusters_), colors):
                    my_members = labels == k
                    l = sum(my_members)
                    if (l > n_cluster_min):
                        cluster_center = cluster_centers[k]
        #                 plt.plot(X[my_members, 0], ydim_back-1-X[my_members, 1], col + '.') # the 255-x does the flip
                        plt.plot(X[my_members, 0]*10/256-5, (ydim_back-1-X[my_members, 1])*10/256-5, 'o', markerfacecolor=col,
                            markeredgecolor='w', markersize = 7) # the 255-x does the flip
                        plt.plot(cluster_center[0]*10/256-5, (ydim_back-1-cluster_center[1])*10/256-5, 's', markerfacecolor='none',
                            markeredgecolor=col, markersize = 18)# markersize = 3*l_max/n_cluster_min
                        plt.plot(cluster_center[0]*10/256-5, (ydim_back-1-cluster_center[1])*10/256-5, 'x', markerfacecolor=col,
                            markeredgecolor=col, markersize = 14)# markersize = 3*l_max/n_cluster_min

                # hist_x = np.zeros(xdim_back,dtype=np.int32)
                # hist_y = np.zeros(ydim_back,dtype=np.int32)
                # j_X = X[:,0].astype(int)
                # j_Y = X[:,1].astype(int)
        #         for i in range(j_X.size):
        #             hist_x[j_X[i]]=hist_x[j_X[i]]+1
        #             hist_y[ydim_back-1-j_Y[i]]=hist_y[ydim_back-1-j_Y[i]]+1
        # #         plt.hist
        #         plt.step(np.linspace(-5,5,xdim_back),hist_x * 0.5 -5,'b-')
        # #         plt.step(np.arange(xdim_back),20*hist_x,'b-')
        #         plt.step(hist_y * 0.5 -5,np.linspace(-5,5, ydim_back),'r-')

                print (len(np.arange(xdim_back)))
                print (np.arange(xdim_back))
                # print ("hist_x: ", hist_x)
                time = TimeStampArray[iter_frame,0] * 1e-6

                plt.title('Time = %.2f s'% time, fontsize=20)
                if frame_format == 'jpg':
                    plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+"_Time=%.3f.jpg"%time, dpi = 80, bbox_inches = 'tight', pad_inches = 0.5)
                elif frame_format == 'pdf':
                    plt.savefig(str(frame_subfolder)+'/'+str(iter_frame).zfill(4)+"_Time=%.3f.pdf"%time, bbox_inches = 'tight', pad_inches = 0.5)
                plt.close()

                # Calculate standard deviation of cluster
                print(' Fr#  Cl#    N        x    sig_x        y    256-y    sig_y     SigSum')
                centroid_sdev = np.zeros((n_clusters_,2), dtype=np.float32)
                for k in range(n_clusters_):
                    my_members = labels == k
                    l = sum(my_members)
                    if (l > n_cluster_min):
                        SignalSum = np.sum(weights[my_members])
                        centroid_sdev[k, 0] = np.sqrt(np.sum((X[my_members, 0]-cluster_centers[k, 0])**2)/(l - 1))
                        centroid_sdev[k, 1] = np.sqrt(np.sum((X[my_members, 1]-cluster_centers[k, 1])**2)/(l - 1))
                        if centroid_sdev[k, 0] != 0 and centroid_sdev[k, 1] != 0:
                            print('%4d %4d %4d %8.2f %8.2f %8.2f %8.2f %8.2f %10.2f'%(iter_frame,k,l,cluster_centers[k, 0],centroid_sdev[k, 0],cluster_centers[k, 1],ydim_back-1-cluster_centers[k, 1],centroid_sdev[k, 1],SignalSum))  
                            cluster_results.write('%4d %4d %4d %8.2f %8.2f %8.2f %8.2f %8.2f %10.2f\n'%(iter_frame,k,l,cluster_centers[k, 0],centroid_sdev[k, 0],cluster_centers[k, 1],ydim_back-1-cluster_centers[k, 1],centroid_sdev[k, 1],SignalSum)) 
        bandwidth_results.write('%4d\t%10.2f\n'%(iter_frame, bandwidth))
    # Make a movie
    os.chdir(frame_subfolder) # Go into the sub directory and creat a movie 
    cmd = 'rm '+OutputFile+'_ClusterMovie.mp4' # if file exists remove it
    f = os.system(cmd)
    # cmd = 'ffmpeg -pattern_type glob -i "Time=*.jpg" '+OutputFile+'_ClusterMovie.mp4'
    cmd = 'ffmpeg -pattern_type glob -i "*Time*.jpg" -vf '+'"pad=ceil(iw/2)*2:ceil(ih/2)*2" '+OutputFile+'_ClusterMovie.mp4'
    f = os.system(cmd)
    if f != 0:
        print('Error in ffmpeg command execution: %s = %s'%(cmd,f))
    cmd = 'rm '+OutputFile+'_ClusterMovie_4xslow.mp4.mp4' # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -i '+OutputFile+'_ClusterMovie.mp4 '+'-filter:v "setpts = 4*PTS" '+OutputFile+'_ClusterMovie_4xslow.mp4' # Slow down the movie
    f = os.system(cmd)
    os.chdir(DefaultDir) # Go back to previous working directory




## Calculate the percentage of peaks not in cluster by frame
# Version 1.0
# Xin Wen 2022.09.06
def stats_clusters(InputFile, clusters, peaks):
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    clusters_byframe = np.array(np.split(clusters,np.flatnonzero(np.diff(clusters[:,0]))+1),dtype=object)
    peaks_byframe = np.array(np.split(peaks,np.flatnonzero(np.diff(peaks[:,0]))+1),dtype=object)

    percentage_list = np.zeros((999,4))
    for frame in range(999):
        percentage_list[frame,0] = peaks_byframe[frame][0][0]+1 # Frame number
        percentage_list[frame,1] = np.shape(peaks_byframe[frame])[0] # Total number of peaks
    i = 0
    for frame in clusters_byframe:
        frame_number = int(frame[0][0])
        percentage_list[frame_number-1,2] = np.sum(clusters_byframe[i][:,2]) # number of peaks in clusters
        i += 1
    for frame in range(999):
    #     percentage_list[frame_number,3] = np.sum(clusters_byframe[i-1][:,2])/np.shape(peaks_byframe[i-1])[1] # percentage used
        percentage_list[frame,3] = percentage_list[frame][2]/percentage_list[frame][1] # percentage used


    np.savetxt(OutputFile+'_percentage_in_cluster.txt',percentage_list,fmt = '%d\t%d\t%d\t%f',  delimiter='\t', header='frame\ttotal\tin_clusters\tpercentage', comments ='', encoding=None)

    total_percentage = np.sum(percentage_list[:,2])/np.sum(percentage_list[:,1])
    print('Total number of peaks = %d'%np.sum(percentage_list[:,1]))
    print('Number of peaks belong to a cluster = %d'%np.sum(percentage_list[:,2]))
    print('Total number of peaks belong to a cluster / Total number of peaks = %f'%total_percentage)

    plt.figure(figsize = (8,6))
    plt.scatter(percentage_list[:,0], percentage_list[:,3], color = 'k', s = 2)
    plt.xlabel('Frame Number')
    plt.ylabel('Percentage of Peaks in a Cluster')
    plt.title('Percentage of Peaks in a Cluster vs Frame Number')
    plt.grid('on')
    plt.savefig('Percentage of Peaks in a Cluster vs Frame Number.pdf')
    plt.show()
    
    return percentage_list







## Correlator routine
# version 1.0 Mike Fitzsimmons, Xin Wen 2022.09.06
def correlator(InputFile, CorrelatorInput, TimeStampArray, TimeMin, TimeMax, d_limit, DoPlot, frame_format):
    DefaultDir=os.getcwd() + r'/'
    OutputFile = InputFile[InputFile.find('Script'):InputFile.find('-raw')].replace(" ", "_")
    # Creat sub diectory to store correlator frames
    frame_subfolder = str(DefaultDir)+'/correlator_frames'
    if not os.path.exists(frame_subfolder):
        os.makedirs(frame_subfolder)
        print("Created Directory : ", frame_subfolder)
    else:
        print("Directory already existed : ", frame_subfolder)
        
    # Creat sub diectory 8 to store correlator results
    frame_subfolder_results = str(DefaultDir)+'/correlator_frames/correlator_'+OutputFile
    if not os.path.exists(frame_subfolder_results):
        os.makedirs(frame_subfolder_results)
        print("Created Directory : ", frame_subfolder_results)
    else:
        print("Directory already existed : ", frame_subfolder_results)
    os.chdir(DefaultDir)

    data = np.loadtxt(CorrelatorInput, skiprows = 1) # Get the results from clustering routine
    TimeStamps = TimeStampArray[:,0] * 1e-6 # convert to seconds 
    os.chdir(frame_subfolder_results)
    FrameMin = np.min(np.argwhere(TimeStamps >= TimeMin)) # 12s
    FrameMax = np.min(np.argwhere(TimeStamps >= TimeMax)) # 17s
    ResultsFile = open('CorrelatorResults_%d-%ds_dlim=%s.txt'%(TimeMin, round(TimeMax), d_limit),'w')

    if DoPlot:
        # Creat sub diectory 9 to store correlator frames for d_limit
        frame_subfolder_frames = str(DefaultDir)+'/correlator_frames/correlator_frames_dlim%d'%d_limit
        if not os.path.exists(frame_subfolder_frames):
            os.makedirs(frame_subfolder_frames)
            print("Created Directory : ", frame_subfolder_frames)
        else:
            print("Directory already existed : ", frame_subfolder_frames)
            
    for iLoop in range(FrameMin,FrameMax):
        
        iFrame = iLoop
        jFrame = iFrame + 1
        # Get timestamps for the two frames
        TiFrame = TimeStamps[iFrame]
        TjFrame = TimeStamps[jFrame]
        
        NoClusters = True # Assert that a frame will not have clusters
        iChoice = np.argwhere(data[:,0] == iFrame)
        jChoice = np.argwhere(data[:,0] == jFrame)
        if (len(iChoice) + len(jChoice) == 0) or  (len(iChoice) == 0 or len(jChoice) == 0):
            print('A frame does not have clusters.')
            print('Frame %s has %s clusters'%(iFrame,len(iChoice)))
            print('Frame %s has %s clusters'%(jFrame,len(jChoice)))
            NSmallEnough = 0
        else:
            NoClusters = False # Both frames have clusters
            c = list(it.product(iChoice, jChoice))
            Ncombinations = len(c)
            Distance = np.zeros(Ncombinations, dtype = np.float32)
            Choices = np.zeros((Ncombinations,2), dtype = np.int16)
            for i in range(Ncombinations):
                Choices[i,0],Choices[i,1] = (int(c[i][0]),int(c[i][1]))
                Distance[i] = np.sqrt((data[Choices[i][0],3]-data[Choices[i][1],3])**2+(data[Choices[i][0],6]-data[Choices[i][1],6])**2) # Distance between clusters
            sdxSmallEnough = np.flatnonzero(Distance[:] <= d_limit)
            NSmallEnough = len(sdxSmallEnough)  
            ShortestPair = np.zeros((NSmallEnough,3), dtype = np.float32)
            for i in range(NSmallEnough):
                ShortestPair[i,:] = Choices[sdxSmallEnough[i]][0],Choices[sdxSmallEnough[i]][1],Distance[sdxSmallEnough[i]] # load choices into numpy array

        if NSmallEnough == 0: # Did not find any pairs within specified distance
            NoClusters = True
            print('Did not find pairs within %i pixels for frames %i and %i'%(d_limit,iFrame,jFrame))

        # Identify a pair of events in the two frames with distance not exceeding d_limit.  
        if not NoClusters:
            # Find duplicate entries in the source list
            SourceDuplicate = False
            SourceUnique = np.unique(ShortestPair[:,0])
            for i in range(len(SourceUnique)):
                if len(np.nonzero(ShortestPair[:,0] == int(SourceUnique[i]))[0]) > 1:
                    SourceDuplicate = True
            # Remove duplicate sources
            if SourceDuplicate: 
                SourceUnique = np.unique(ShortestPair[:,0])
                for i in range(len(SourceUnique)):
                    sdx = np.argwhere(ShortestPair[:,0]==SourceUnique[i])
                    if len(sdx) > 1: # Choose the shortest separation of the duplicates
                        mask = np.ones(len(ShortestPair[:,0]), dtype=bool)
                        DupMin = np.min(ShortestPair[sdx,2]) # Find the shortest distance
                        ExtraPoints = np.nonzero(ShortestPair[sdx,2] != DupMin) # Find indices of duplicates with greater distances
                        mask[sdx[ExtraPoints[0]]] = False
                        ShortestPair = ShortestPair[mask,:] # Eliminate longest duplicates

            # Find duplicate entries in the source list
            TargetDuplicate = False
            TargetUnique = np.unique(ShortestPair[:,1])
            for i in range(len(TargetUnique)):
                if len(np.nonzero(ShortestPair[:,1] == int(TargetUnique[i]))[0]) > 1:
                    TargetDuplicate = True

            # Remove duplicate targets
            if TargetDuplicate: 
                TargetUnique = np.unique(ShortestPair[:,1])
                for i in range(len(TargetUnique)):
                    sdx = np.argwhere(ShortestPair[:,1]==TargetUnique[i])
                    if len(sdx) > 1: # Choose the shortest separation of the duplicates
                        mask = np.ones(len(ShortestPair[:,0]), dtype=bool)
                        DupMin = np.min(ShortestPair[sdx,2]) # Find the shortest distance
                        ExtraPoints = np.nonzero(ShortestPair[sdx,2] != DupMin) # Find indices of duplicates with greater distances
                        mask[sdx[ExtraPoints[0]]] = False
                        ShortestPair = ShortestPair[mask,:] # Eliminate longest duplicates

            # Calculate x and y sum of displacements, then write to results file.
            NPairs = len(ShortestPair[:,0])
            XSum = 0.
            YSum = 0.
            for i in range(NPairs):
                xRed = data[int(ShortestPair[i,0]),3]
                yRed = data[int(ShortestPair[i,0]),6]
                xBlu = data[int(ShortestPair[i,1]),3]
                yBlu = data[int(ShortestPair[i,1]),6]
                XSum = XSum + (xBlu - xRed) # sum of displacements in the horizontal direction
                YSum = YSum + (yBlu - yRed) # sum of displacements in the vertical direction
                ResultsFile.write('%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%(iFrame,jFrame,TiFrame,TjFrame,XSum,YSum,xRed,yRed,xBlu,yBlu,xBlu-xRed,yBlu-yRed))
            print('%d %d %.2f %.2f %d %d'%(iFrame, jFrame, TiFrame, TjFrame, XSum, YSum))

            if DoPlot:
                os.chdir(frame_subfolder_frames)
                plt.figure(figsize = (8,8))
                plt.xlim(-5,5)
                plt.ylim(-5,5)
                plt.xticks(np.linspace(-5,5,11), fontsize = 18)
                plt.yticks(np.linspace(-5,5,11), fontsize = 18)
                plt.xlabel('Horizontal position (mm)', fontsize=20)
                plt.ylabel('Vertical postition (mm)', fontsize=20)
                # Plot all cluster centroids
                plt.scatter(data[iChoice,3]*10/256-5,data[iChoice,6]*10/256-5, c = 'r', s = 16)
                plt.scatter(data[jChoice,3]*10/256-5,data[jChoice,6]*10/256-5, c = 'b', s = 16)
                # Draw arrows connecting shortest pairs
                NPairs = len(ShortestPair[:,0])
                for i in range(NPairs):
                    xRed = data[int(ShortestPair[i,0]),3]*10/256-5
                    yRed = data[int(ShortestPair[i,0]),6]*10/256-5
                    xBlu = data[int(ShortestPair[i,1]),3]*10/256-5
                    yBlu = data[int(ShortestPair[i,1]),6]*10/256-5
                    plt.arrow(xRed,yRed,xBlu-xRed,yBlu-yRed,head_width=0.1, head_length=0.15, head_starts_at_zero=False, length_includes_head=True, fc='w')
                plt.title('Time of Red: %.2f s  Time of Blue: %.2f s'%(TiFrame, TjFrame), fontsize=20)
                if frame_format == 'jpg':
                    plt.savefig(str(iFrame).zfill(4)+"_Time=%.3f.jpg"%TiFrame, dpi = 80, bbox_inches = 'tight', pad_inches = 0.5)
                elif frame_format == 'pdf':
                    plt.savefig(str(iFrame).zfill(4)+"_Time=%.3f.pdf"%TiFrame, bbox_inches = 'tight', pad_inches = 0.5)
    #             plt.show()
                plt.close()
        if NoClusters:
            if DoPlot:
                os.chdir(frame_subfolder_frames)
                plt.figure(figsize = (8,8))
                plt.xlim(-5,5)
                plt.ylim(-5,5)
                plt.xticks(np.linspace(-5,5,11), fontsize = 18)
                plt.yticks(np.linspace(-5,5,11), fontsize = 18)
                plt.xlabel('Horizontal position (mm)', fontsize=20)
                plt.ylabel('Vertical postition (mm)', fontsize=20)
                # Plot all cluster centroids
                plt.scatter(data[iChoice,3]*10/256-5,data[iChoice,6]*10/256-5, c = 'r', s = 16)
                plt.scatter(data[jChoice,3]*10/256-5,data[jChoice,6]*10/256-5, c = 'b', s = 16)
                plt.title('Time of Red: %.2f s  Time of Blue: %.2f s'%(TiFrame, TjFrame), fontsize=20)
                if frame_format == 'jpg':
                    plt.savefig(str(iFrame).zfill(4)+"_Time=%.3f.jpg"%TiFrame, dpi = 80, bbox_inches = 'tight', pad_inches = 0.5)
                elif frame_format == 'pdf':
                    plt.savefig(str(iFrame).zfill(4)+"_Time=%.3f.pdf"%TiFrame, bbox_inches = 'tight', pad_inches = 0.5)
    #             plt.show()
                plt.close()
  
    cmd = 'rm '+OutputFile+'_CorrelatorMovie_dlim=%d.mp4'%d_limit # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -pattern_type glob -i "*Time*.jpg" -vf "scale='+"'bitand(oh*dar,65534)':'min(720,ih)'"+'" '+OutputFile+'_CorrelatorMovie_dlim=%d.mp4'%d_limit
    f = os.system(cmd)
    if f != 0:
        print('Error in ffmpeg command execution: %s = %s'%(cmd,f))
    cmd = 'rm '+OutputFile+'_CorrelatorMovie_4xslow_dlim=%d.mp4'%d_limit # if file exists remove it
    f = os.system(cmd)
    cmd = 'ffmpeg -i '+OutputFile+'_CorrelatorMovie_dlim=%d.mp4'%d_limit+' -filter:v "setpts = 4*PTS" '+OutputFile+'_CorrelatorMovie_4xslow_dlim=%d.mp4'%d_limit # Slow down the movie
    f = os.system(cmd)

    ResultsFile.close()
    os.chdir(DefaultDir) # Go back to previous working directory
    return frame_subfolder, frame_subfolder_results 
