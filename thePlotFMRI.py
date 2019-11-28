#!/usr/bin/python
"""
ThePlotFMRI allows to create a figure summarizing a functional data set as Jonathan Power introduced it in 2017. The fMRI data set is assumed to be preprocessed with SPM. The functional file and some pre-processed files are mandatory to run this script.
More info on how to pre-process functional files at https://www.fmrwhy.com/2018/07/20/the-plot-spm/.
This script performs the plotting only, not the pre-processing.

Usage:
  thePlotFMRI.py (-f <fmri_file>) (-r <rp_file>) (-g <rc1_file>) (-w <rc2_file>) (-c <rc3_file>) [-o <out_file>] [-p <P>] [-t <title>]
  thePlotFMRI.py -h | --help

Options:
  -h --help            Show this help message
  --version            Show version
  Required inputs
  -f, --fmri           Path to functional data set ("bold_run1.nii.gz")
  -r, --realparams     Path to the realignment parameters file ("rp_bold_run1.txt")
  -g, --rgm            Path to the resliced (to fmri space) GM  segmentation map ("rc1_T1w.nii.gz")
  -w, --rwm            Path to the resliced (to fmri space) WM  segmentation map ("rc2_T1w.nii.gz")
  -c, --rcsf           Path to the resliced (to fmri space) CSF segmentation map ("rc3_T1w.nii.gz")
  Optional outputs
  -o, --out=<file>     Path to the output CarpetPlot graph (.png extension preferred)
  -t, --title=<title>  Title for the figure (ex: subject ID with run ID and experiment name) [default :""]
  -p, --psc=<val>      Sets the min,max range for the carpet plot [-psc,+psc] [default: 2]
"""

import matplotlib
# Matplotlib chooses Xwindows backend by default. Set matplotlib to not use the Xwindows backend.
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
matplotlib.use('Agg') #useful for no display environment, in a remote session for example (ssh)
import numpy as np
import numpy.matlib
import nibabel as nib
import matplotlib.pyplot as plt
from os.path import exists as ope, join as opj, basename as opb
from matplotlib.gridspec import GridSpec
from FigureCreator import CarpetFigureCreator, TISSUE_NAMES
from utils import checkFileExists
from config import PSC_MIN_MAX

DEFAULT_OUT_FILE = 'thePlotFMRI_default.png'

class ThePlot:
    """
    --------------------------------------------------------------------------------------------------------------------
    What this class does :
    This class allows to generate a carpet plot describing a functional dataset given a 4D functional file, resliced (to
    functional data) segmentation files and estimated movement file. Once the user has provided the input files, the
    function plot(outputFile) allows to generate a graph and save it.
    0) Sets the input files
    1) The 4D functional data is reorganised in a 2D array of size NxT where N is the number of voxels that belong
    to one of the following tissue types (GM,WM,CSF) and T is the number of fMRI timepoints (number of acquired volumes)
    2) Detrend functional data
    3) Compute framewise displacement
    --------------------------------------------------------------------------------------------------------------------
    What this class does not :
    The class does not perform pre-processing, to get some help regarding this step, I refer the user to  Stephan Heunis
    web page (info@fmrwhy.com) https://www.fmrwhy.com/2018/07/20/the-plot-spm/, a link to open source Matlab (SPM)
    resources allows you to download and easily run a great script to generate required input files to be able to run
    this code.
    """
    def __init__(self):
        """
        Initializes the class
        """
        self.vols_to_highlight = []
        pass

    def setFunctionalData(self, fn_file):
        """
        Sets the path to nifti (.nii or .nii.gz) functional 4D file. Loads the file with nibabel and sets the data into
        self.f_img
        :param fn_file: string, path to the functional file
        """
        self.functional_fn = checkFileExists(fn_file, 'Functional run (functional_filename.nii)')
        # Load file directly
        f = nib.load(self.functional_fn)
        self.f_hdr = f.get_header()
        self.f_aff = f.get_affine()
        # Sets the raw data directly in class attribute
        self.f_img = f.get_data()
        # (Ni,Nj,Nk) = number of voxels in the (X,Y,Z) dimension
        # Nt = number of fMRI timepoints (=number of volumes)
        self.fn_shape = [self.Ni, self.Nj, self.Nk, self.Nt] = self.f_img.shape
        #Sets the default slice index for generating the plot
        self.setDefaultSliceIndex()

    def setMovementParameterFile(self, rp_file):
        """
        Sets the movement parameters across the functional run (rp_....txt file if SPM was used)
        :param rp_file: string, path to the estimated movement parameters
        """
        self.movementParamFile = checkFileExists(rp_file, 'Estimated movement (rp_filename.txt)')

    def setReslicedGM(self,rgm):
        """
        Sets the resliced gray matter segmentation file (rc1_....nii if SPM was used)
        :param rgm: string, path to the resliced GM segmentation file
        """
        self.rgm = checkFileExists(rgm,'resliced WM segmentation map (rc1_filename.nii)')

    def setReslicedWM(self,rwm):
        """
        Sets the resliced white matter segmentation file (rc2_....nii if SPM was used)
        :param rwm: string, path to the resliced WM segmentation file
        """
        self.rwm = checkFileExists(rwm,'resliced WM segmentation map (rc2_filename.nii)')

    def setReslicedCSF(self,rcsf):
        """
        Sets the resliced cerebrospinal fluid segmentation file (rc3_....nii if SPM was used)
        :param rcsf: string, path to the resliced CSF segmentation file
        """
        self.rcsf = checkFileExists(rcsf,'resliced CSF segmentation map (rc3_filename.nii)')

    def setReslicedSegmentationFiles(self, rgm, rwm, rcsf):
        """
        Other option to set in one call the paths to GM,WM and CSF resliced segmentation files
        :param rgm: string, path to the resliced GM segmentation file
        :param rwm: string, path to the resliced WM segmentation file
        :param rcsf: string, path to the resliced CSF segmentation file
        """
        self.setReslicedGM(rgm)
        self.setReslicedWM(rwm)
        self.setReslicedCSF(rcsf)

    def setAllInputData(self,fn_file, rp_file, rgm, rwm, rcsf):
        """
        Alternative way of calling ThePlot()
        :param fn_file: string, path to the functional file
        :param rp_file: string, path to the estimated movement parameters
        :param rgm_fn: string, path to the resliced GM segmentation file
        :param rwm_fn: string, path to the resliced WM segmentation file
        :param rcsf_fn: string, path to the resliced CSF segmentation file
        """
        self.setFunctionalData(fn_file)
        self.setMovementParameterFile(rp_file)
        self.setReslicedSegmentationFiles(rgm, rwm, rcsf)

    def setVolumesToHighlight(self,volsToHiglight):
        """
        Optional stuff
        :return:
        """
        self.vols_to_highlight = volsToHiglight

    def getFramewiseDisplacement(self):
        """
        0. Load the text file corresponding to the estimated movement parameters
        1. Pre-process movement parameters (de-mean and detrend)
        2. Calculate Framewise Displacement (FD is then set as a class attribute)
        :return vector, computed framewise displacement
        ---------------------------------------------------------------------------------------------------------------
        Python adaptation from Stephan Heunis Matlab scripts.
        More info regarding these Matlab scripts are given in the __init__() function of this class
        ---------------------------------------------------------------------------------------------------------------
        """
        mp = np.loadtxt(self.movementParamFile)
        # 1 Movement parameters pre-processing
        # Remove the mean columns to each element of the matrix
        mp2 = mp - np.mean(mp,0)
        # Create a [1:Nt] vector
        X = np.arange(1,self.Nt+1,1)
        # X2 = X - mean(X);
        X2 = X - np.mean(X)
        # Array of 1 elements in the first column and X2 in the second column
        X3 = np.array([np.ones((self.Nt)), X2]).T
        # X3.b = mp2
        b = np.linalg.lstsq(X3,mp2,rcond=-1)[0]
        # MP_corrected = MP2 - X3(:, 2)*b(2,:);
        mp_corrected = mp2 - np.outer(X3[:,1],b[1,:])
        mp_mm = np.copy(mp_corrected)
        mp_mm[:,3:] = mp_mm[:,3:]*50# 50mm = assumed brain radius; from article
        # 2 Calculate FD
        mp_diff = np.vstack((np.zeros((1,6)),np.diff(mp_mm,axis=0)))
        #Sum row-wise the absolute value of movement parameters
        return np.sum(np.abs(mp_diff),axis=1)

    def createBinarySegments(self,threshold=0.1):
        """
        Transform resliced to functional data SPM probability maps to binary maps. The binary maps are stored under the
        following class attributes : self.gm_bin, self.wm_bin and self.csf_bin
        :param threshold: float, parameter to threshold the probability maps
        ---------------------------------------------------------------------------------------------------------------
        Python adaptation from Stephan Heunis Matlab scripts.
        More info regarding these Matlab scripts are given in the __init__() function of this class
        ---------------------------------------------------------------------------------------------------------------
        """
        #Load nifti files
        img_gm = nib.load(self.rgm)
        img_wm = nib.load(self.rwm)
        img_csf = nib.load(self.rcsf)
        #Load arrays
        gm = img_gm.get_data()
        wm = img_wm.get_data()
        csf = img_csf.get_data()
        # Perform thresholding
        if threshold !=0:
            gm_thresh = np.copy(gm)
            wm_thresh = np.copy(wm)
            csf_thresh = np.copy(csf)
            # 2
            gm_thresh[gm < threshold] = 0
            wm_thresh[wm < threshold] = 0
            csf_thresh[csf < threshold] = 0
            # Make decisions based on data/thresholds
            self.gm_bin = (gm_thresh >= wm_thresh) & (gm_thresh >= csf_thresh) & (gm_thresh != 0)
            self.wm_bin = (wm_thresh > gm_thresh) & (wm_thresh >= csf_thresh) & (wm_thresh != 0)
            self.csf_bin = (csf_thresh > gm_thresh) & (csf_thresh > wm_thresh) & (csf_thresh != 0)
        else:
            self.gm_bin = (gm >= wm) & (gm >= csf) & (gm != 0)
            self.wm_bin = (wm > gm) & (wm >= csf) & (wm != 0)
            self.csf_bin = (csf > gm) & (csf > wm) & (csf != 0)

    def detrendFunctionalData(self):
        """
        Remove linear and polynomial trends from data.
        ---------------------------------------------------------------------------------------------------------------
        Python adaptation from Stephan Heunis Matlab scripts.
        More info regarding these Matlab scripts are given in the __init__() function of this class
        ---------------------------------------------------------------------------------------------------------------
        """
        self.f_2D = np.reshape(self.f_img,(self.Ni * self.Nj * self.Nk, self.Nt), order='F').copy()
        #X_design = [ (1:Nt)' ((1:Nt).^2/(Nt^2))' ((1:Nt).^3/(Nt^3))'];
        x = np.arange(1,self.Nt+1)
        X_design = np.array((x, np.power(x,2)/float(self.Nt**2), np.power(x,3)/float(self.Nt**3))).T
        # X_design = X_design - mean(X_design);
        X_design = X_design - np.mean(X_design,axis=0)
        # X_design = [X_design ones(Nt,1)];
        X_design = np.hstack((X_design, np.ones((self.Nt,1))))
        # betas = X_design\F_2D'
        betas = np.linalg.lstsq(X_design, self.f_2D.T,rcond=-1)[0]
        F_detrended = self.f_2D.T - np.dot(X_design[:,:-1],betas[:-1,:])
        self.F_detrended = F_detrended.T

    def generateMasks(self):
        """
        This function uses the GM,WM,CSF binary segmentations to reshape 4D functional data into a 2D array of size NxT
        where N is the total number of voxels and T is the number of functional volumes.
        ---------------------------------------------------------------------------------------------------------------
        Python adaptation from Stephan Heunis Matlab scripts.
        More info regarding these Matlab scripts are given in the __init__() function of this class
        ---------------------------------------------------------------------------------------------------------------
        """
        #Create the binary segmentation maps from probability tissue maps
        self.createBinarySegments()
        # In Python, reshape(order='F') first and np.where() to get similar behavior as Matlab's find() function
        # see : https://docs.scipy.org/doc/numpy-1.13.0/user/numpy-for-matlab-users.html
        # Generate Brain Mask from the binary segmentations
        self.mask = self.gm_bin | self.wm_bin | self.csf_bin
        #Reshape all the masks
        mask_reshaped = np.reshape(self.mask, (self.Ni * self.Nj * self.Nk), order='F')
        gm_reshaped = np.reshape(self.gm_bin, (self.Ni*self.Nj*self.Nk),order='F')
        wm_reshaped = np.reshape(self.wm_bin, (self.Ni * self.Nj * self.Nk), order='F')
        csf_reshaped = np.reshape(self.csf_bin, (self.Ni * self.Nj * self.Nk), order='F')
        # Get indices where mask==1 (later, allows to access data from specific tissue types)
        self.i_mask = np.where(mask_reshaped)
        self.i_gm = np.where(gm_reshaped)
        self.i_wm = np.where(wm_reshaped)
        self.i_csf = np.where(csf_reshaped)

    def calculatePSC(self):
        """Detrend functional data"""
        self.detrendFunctionalData()
        # F_masked = F_detrended(I_mask,:);
        f_masked = self.F_detrended[self.i_mask[0],:]
        f_mean = np.mean(f_masked,axis=1)
        # F_masked_psc = 100 * (F_masked. / repmat(F_mean, 1, Nt)) - 100;
        """Calculation of the Percent Signal Change (PSC) is performed right here"""
        f_masked_psc = 100 * np.divide(f_masked, numpy.matlib.repmat(f_mean,self.Nt,1).T) - 100
        #Set nan to zero
        f_masked_psc[np.isnan(f_masked_psc)] = 0
        #Create images of the
        f_psc_img = np.zeros((self.Ni, self.Nj, self.Nk, self.Nt))
        self.f_2D_psc = np.reshape(f_psc_img, (self.Ni * self.Nj * self.Nk, self.Nt),order='F')
        self.f_2D_psc[self.i_mask,:] = f_masked_psc
        self.f_psc_img = np.reshape(self.f_2D_psc, (self.Ni, self.Nj, self.Nk, self.Nt),order='F').copy()

    # def getDVARS(self,remove_zerovariance=False,intensity_normalization=1000):
    #     print "DVARS is under test version right now"
    #     # TODO : Work-in-progress, the current version is bugging
    #     from nipype.algorithms.confounds import compute_dvars
    #     self.generateMasks()
    #     f = nib.load(self.functional_fn)
    #     test_filename = '/home/qduche/test_mask_DVARS.nii.gz'
    #     nibImage = nib.Nifti1Image(self.mask,affine=f.get_affine(),header=f.get_header())
    #     nib.save(nibImage,test_filename)
    #     (dvars_stdz, dvars_nstd, dvars_vx_stdz) = compute_dvars(self.functional_fn, test_filename,
    #                                                             remove_zerovariance=remove_zerovariance,
    #                                                             intensity_normalization=intensity_normalization)
    #     plt.plot(dvars_stdz)
    #     plt.show()
    #     pass

    def getCarpetData(self):
        """
        Vertically Stack detrended data into a big array of size NxT.
        The data is ordered ordered as GM,WM and CSF (N_GM + N_WM + N_CSF voxels)
        :return: 2D array, containing the data that is plotted in the carpet plot
        """
        self.generateMasks()
        self.calculatePSC()
        # Get data from tissues
        GM_img = self.f_2D_psc[self.i_gm[0], :]
        WM_img = self.f_2D_psc[self.i_wm[0], :]
        CSF_img = self.f_2D_psc[self.i_csf[0], :]
        # Vertically stack data
        all_img = np.vstack((GM_img, WM_img, CSF_img))
        return all_img

    def get_n_GM(self):
        """
        :return: int, number of GM voxels in the segmentation map (functional space)
        """
        return len(self.i_gm[0])

    def get_n_WM(self):
        """
        :return: int, number of WM voxels in the segmentation map (functional space)
        """
        return len(self.i_wm[0])

    def get_n_CSF(self):
        """
        :return: int, number of CSF voxels in the segmentation map (functional space)
        """
        return len(self.i_csf[0])

    def get_gm_bin(self):
        """
        :return: numpy 3D-array, GM segmentation map (functional space)
        """
        return self.gm_bin

    def get_wm_bin(self):
        """
        :return: numpy 3D-array, WM segmentation map (functional space)
        """
        return self.wm_bin

    def get_csf_bin(self):
        """
        :return: numpy 3D-array, CSF segmentation map (functional space)
        """
        return self.csf_bin

    def saveBinaryMasks(self,outdir,base):
        """
        Saves the computed binary masks in directory
        :param outdir: string, path to the output directory
        :param base: string, prefix for the filenaming
        """
        self.getCarpetData()
        total_bin = self.csf_bin + 2 * self.gm_bin + 3 * self.wm_bin
        arrays = [self.gm_bin,self.wm_bin,self.csf_bin,total_bin]
        tissues = TISSUE_NAMES + ['all']
        for arr,tissue in zip(arrays,tissues):
            img = nib.Nifti1Image(arr,affine=self.f_aff,header=self.f_hdr)
            basename = '_'.join([base,tissue,'binary']) + '.nii.gz'
            bin_fn = opj(outdir,basename)
            if not ope(bin_fn):
                nib.save(img,filename=bin_fn)

    def setDefaultSliceIndex(self):
        self.sliceIndex = self.Nk / 2

    def setSliceIndex(self,si):
        """
        Choose the slice index for showing segmentation maps onto functional data on the left of the plot
        :param si: int, slide index in the Z-axis
        """
        if si>=0 and si< self.Nk:
            self.sliceIndex = si
        else:
            self.setDefaultSliceIndex()
            print "The slice index must be between 0 and {0} while you provided {1}. " \
                  "Default index is chosen (middle z-slice : {2})".format(self.Nk, si,self.sliceIndex)

    def plot(self, output_file='',title='',psc_min_max=PSC_MIN_MAX):
        """
        Generates a figure for the user using the CarpetFigureCreator()
        """
        cfc = CarpetFigureCreator()
        cfc.setFunctionalFileToDisplay(self.functional_fn)
        cfc.setCarpetMatrix(self.getCarpetData())
        cfc.setFramewiseDisplacementVector(self.getFramewiseDisplacement())
        cfc.setSliceIndex(self.sliceIndex)
        cfc.setNVoxelsPerTissue(len(self.i_gm[0]),len(self.i_wm[0]),len(self.i_csf[0]))
        cfc.setBinarySegmentations(self.gm_bin,self.wm_bin,self.csf_bin)
        if title:
            cfc.setTitle(title=title)
        if len(self.vols_to_highlight):
            cfc.setVolumesToHighlight(self.vols_to_highlight)
        cfc.plot(output_file=output_file,psc_min_max=psc_min_max)
        print '\t- SingleRun Plot : ' + opb(output_file) + ' saved'

    def generateTimeseriesPSCSnapViews(self,out_dir,nCols,nRows,title='',basename='psc_img',psc_min_max=PSC_MIN_MAX):
        """
        Create a list (size Nt = N timepoints) of figures. Each figure contains the PSC volume corresponding to the
         timepoint represented as Nk slices in a matrix-shape
        :return:
        """
        lf = []
        for t in range(self.Nt):
            out_graph_vol_fn = opj(out_dir, '_'.join([basename, str(t).zfill(4)]) + '.png')
            if not ope(out_graph_vol_fn):
                # Create a figure
                fig = plt.figure(figsize=(nCols, nRows))
                gs = GridSpec(nRows, nCols)
                gs.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.
                for i in range(nRows):
                    for j in range(nCols):
                        z = i*nCols + j
                        if z<=self.Nt:
                            im = self.f_psc_img[:, :, z, t].T
                            ax = plt.subplot(gs[i, j])
                            ims = ax.imshow(im, vmin=-psc_min_max, vmax=psc_min_max, cmap='seismic')
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.axis('off')
                            # for tissueName, bin_map, tcol in zip(TISSUE_NAMES, [self.gm_bin,self.wm_bin,self.csf_bin], TISSUE_COLORS):
                            #     cmap1 = colors.LinearSegmentedColormap.from_list('my_cmap', [COL_TRANSP, tcol], 256)
                            #     mask = bin_map[:,:,z].T
                            #     ax.imshow(mask, cmap=cmap1, interpolation='none', alpha=.5)
                plt.suptitle(title + ' vol {0}/{1}'.format(t + 1, self.Nt))
                # Try to add colorbar at the bottom
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5]) #left bottom width height
                cbar = fig.colorbar(ims, ticks=[-psc_min_max, 0, psc_min_max],cax=cbar_ax)
                cbar.ax.set_yticklabels(['-{0}%'.format(psc_min_max), '0', '+{0}%'.format(psc_min_max)])  # vertically oriented colorbar
                #cbar.set_label('Signal')
                # Save
                plt.savefig(out_graph_vol_fn)
                lf.append(out_graph_vol_fn)
                plt.close()
        return lf


class ThePlotMultipleRuns:
    """
    This class allows to manipulate functional data from multiple runs/sessions to generate a Carpet Plot Figure.
    The class assumes that all the runs were realigned together with SPM
    """
    def __init__(self):
        self.volsToHighlight = []
        pass
    def setListRuns(self,list_files):
        """
        Sets the path to the 4D functional files
        :param list_files: list, contains paths to each functional run
        """
        self.list_fn_files = list_files
        # Get number of runs
        self.nR = len(self.list_fn_files)

    def setListMovementFiles(self,list_files):
        """
        Sets the path to the movement parameters across the functional runs (rp_....txt file if SPM was used)
        :param list_files: list, contains paths to text files where realignment parameters are stored
        """
        self.list_rp_files = list_files

    def setReslicedGM(self,rgm):
        """
        Sets the resliced gray matter segmentation file (rc1_....nii if SPM was used)
        :param rgm: string, path to the resliced GM segmentation file
        """
        self.rgm = checkFileExists(rgm,'resliced WM segmentation map (rc1_filename.nii)')

    def setReslicedWM(self,rwm):
        """
        Sets the resliced white matter segmentation file (rc2_....nii if SPM was used)
        :param rwm: string, path to the resliced WM segmentation file
        """
        self.rwm = checkFileExists(rwm,'resliced WM segmentation map (rc2_filename.nii)')

    def setReslicedCSF(self,rcsf):
        """
        Sets the resliced cerebrospinal fluid segmentation file (rc3_....nii if SPM was used)
        :param rcsf: string, path to the resliced CSF segmentation file
        """
        self.rcsf = checkFileExists(rcsf,'resliced CSF segmentation map (rc3_filename.nii)')

    def setReslicedSegmentationFiles(self,rgm,rwm,rcsf):
        """
        Other option to set in one call the paths to GM,WM and CSF resliced segmentation files
        :param rgm: string, path to the resliced GM segmentation file
        :param rwm: string, path to the resliced WM segmentation file
        :param rcsf: string, path to the resliced CSF segmentation file
        """
        self.setReslicedGM(rgm)
        self.setReslicedWM(rwm)
        self.setReslicedCSF(rcsf)

    def setVolumesToHighlight(self,volsToHighlight):
        """
        Set the volumes to highlight on the head motion plot.
        :param volsToHighlight: list, list of list of tuples. The list should contain nR (number of runs) sublists, each
         item of this sublist is a tuple defininig starting volume index and ending volume index to highlight
         example : [ [(0,10),(15,25)],          # for run number 1 (highlight volumes 0 to 10 and 15 to 25)
                     [],                        # for run number 2 (no volumes to highlight)
                     [(30,40),(50,60),(70,80)]  # for run number 3
                    ]
        :return:
        """
        self.volsToHighlight = volsToHighlight

    def prepareVolumesToHighlightForPlot(self):
        """
        Rearrange the volumes to highlight as a flat list
        :return:
        """
        self.volsForPlot = []
        for r in range(self.nR):
            if len(self.volsToHighlight[r])>0:
                for (vol1,vol2) in self.volsToHighlight[r]:
                    shift = np.sum(self.Nvols[:r])
                    self.volsForPlot.append((vol1+shift, vol2+shift))

    def checkDimensionsCompatibilities(self):
        """

        :return:
        """
        # Prepare arrays for each dimension (X,Y,Z, time, number of voxels segmented)
        self.Ni = np.zeros((self.nR))
        self.Nj = np.zeros((self.nR))
        self.Nk = np.zeros((self.nR))
        self.Nt = np.zeros((self.nR),dtype=int)
        self.Nvoxels = np.zeros((self.nR), dtype=int)
        self.Nvols = np.zeros((self.nR), dtype=int)
        #List of instances of "The Plot"
        self.tp = []
        for r in range(self.nR):
            #Create instance of the plot for each run
            tp = ThePlot()
            tp.setAllInputData(self.list_fn_files[r],self.list_rp_files[r],self.rgm,self.rwm,self.rcsf)
            self.tp.append(tp)
            #Get dimensions of all data files
            self.Ni[r] = tp.Ni
            self.Nj[r] = tp.Nj
            self.Nk[r] = tp.Nk
            self.Nt[r] = tp.Nt
            #Number of voxels (segmentation)
            self.Nvoxels[r] = tp.getCarpetData().shape[0]
            self.Nvols[r] = tp.getCarpetData().shape[1]

    def setDefaultSliceIndex(self):
        """
        Sets the z-slice index for the plot views
        :return:
        """
        self.sliceIndex = self.Nk[0]/2

    def setSliceIndex(self,si):
        """

        :param si:
        :return:
        """
        if np.all(si<self.Nk):
            self.sliceIndex = si
        else: #otherwise use default slice index
            self.setDefaultSliceIndex()

    def concatenateCarpetMatrices(self):
        """
        Horizontally concatenate carpet matrices calculated for each run
        Sets a numpy 2D-array matrix as attribute
        """
        # Prepare 2D array of size NVoxels rows x 0 columns (matrices will be horizontally stacked)
        self.cp_data = np.zeros((self.Nvoxels[0], 0))
        for tp in self.tp:
            self.cp_data = np.hstack((self.cp_data,tp.getCarpetData()))

    def concatenateFramewiseDisplacements(self):
        """
        Horizontally concatenate framewise displacements estimated for each run
        Sets a numpy 1D-array vector containing estimated FD in each volume as attribute
        """
        self.fd_vec = np.zeros((np.sum(self.Nt)))
        for r,tp in enumerate(self.tp):
            # indices to fill the numpy vector
            i_start = np.sum(self.Nt[:r])
            i_end = i_start + self.Nt[r]
            self.fd_vec[i_start:i_end] = tp.getFramewiseDisplacement()

    def setDataForPlot(self):
        """
        Function that allows to retrieve data from each run and concatenate them into class attributes
        :return:
        """
        # First check that everything is compatible
        self.checkDimensionsCompatibilities()
        self.concatenateCarpetMatrices()
        self.concatenateFramewiseDisplacements()
        #Get Data from the first ThePlot instance(), allows to access binary segmentations and N voxels per tissue
        tp = self.tp[0]
        # Set number of voxels per tissue from the first instance
        self.n_GM, self.n_WM, self.n_CSF = tp.get_n_GM(), tp.get_n_WM(), tp.get_n_CSF()
        # Set tissue segmentation maps
        self.bin_gm, self.wm_bin, self.csf_bin = tp.get_gm_bin(), tp.get_wm_bin(), tp.get_csf_bin()

    def plot(self,output_file='',title='',sliceIndex=''):
        """
        Generates a figure for the user using the CarpetFigureCreator()
        :param output_file: string, path to output file (.png prefered), if empty, a window pops up to view the graph
        :param title: string, Supra Title for the figure, if empty, no supra title
        """
        # Go get the data from each instance of the Plot
        self.setDataForPlot()
        if sliceIndex:
            self.setSliceIndex(sliceIndex)
        else:
            self.setDefaultSliceIndex()
        cpc = CarpetFigureCreator() # Call Carpet Figure Creator to generate the figure
        cpc.setFunctionalFileToDisplay(self.list_fn_files[0]) #Use first functional file to display functional image
        cpc.setCarpetMatrix(self.cp_data) #Set horizontally concatenated carpet matrices from each run
        cpc.setFramewiseDisplacementVector(self.fd_vec) #Horizontally concatenated FD
        cpc.setSliceIndex(self.sliceIndex) #Slice Index used is the middle slice from the first functional run
        cpc.setNVoxelsPerTissue(self.n_GM, self.n_WM, self.n_CSF) # Binary segmentation maps
        cpc.setBinarySegmentations(self.bin_gm, self.wm_bin, self.csf_bin) #N voxels per tissue
        cpc.addRunIndices(self.Nt) # sets vector to display on the graph when a new run has started
        if len(self.volsToHighlight)>0:
            self.prepareVolumesToHighlightForPlot()
            cpc.setVolumesToHighlight(self.volsForPlot)
        cpc.setTitle(title=title) #Set supra title
        cpc.plot(output_file=output_file) #Plot and save the figure
        print '\t- MultiRuns Plot : ' + opb(output_file) + ' saved'


#######################################################################################################################

if __name__ == '__main__':
    # The main function is here
    from docopt import docopt
    args = docopt(__doc__, version='0.1')
    # Get arguments
    fmri_fn = args['<fmri_file>'] # Functional 4D file (acquired fMRI raw data)
    rp_fn = args['<rp_file>']     # Realignment parameters file
    rgm_fn = args['<rc1_file>']   # Resliced GM segmentation file
    rwm_fn = args['<rc2_file>']   # Resliced WM segmentation file
    rcsf_fn = args['<rc3_file>']  # Resliced CSF segmentation file

    print args

    if args['--out']:
        out_file = args['--out'] # Output file (graph) saved as an image
    else:
        out_file = DEFAULT_OUT_FILE
        print "[thePlotFMRI.py] Argument -o was not defined, the default output file '{0}' is being used".format(out_file)

    if not ope(out_file):
        # From here, everything should be automated
        # --------------------------------------------------
        # Create Instance of the class 'The Plot'
        tp = ThePlot()
        # Set the data
        tp.setFunctionalData(fmri_fn)
        tp.setMovementParameterFile(rp_fn)
        tp.setReslicedGM(rgm_fn)
        tp.setReslicedWM(rwm_fn)
        tp.setReslicedCSF(rcsf_fn)
        # Generates the figure
        tp.plot(out_file,title=args['--title'],psc_min_max=float(args['--psc']))
    else:
        print "[thePlotFMRI.py] output file already exists : {0}".format(out_file)

    # Example : setting the inputs in several ways
    # ----------------------------------------------------------------------------------------------
    # fmri_fn = 'path/to/your/4d_functional_file.nii'  # Functional 4D file (acquired fMRI raw data)
    # rp_fn = 'path/to/your/rp_....txt'                # Realignment parameters file
    # rgm_fn = 'path/to/your/rc1_....nii'              # Resliced GM segmentation file
    # rwm_fn = 'path/to/your/rc2_....nii'              # Resliced WM segmentation file
    # rcsf_fn = 'path/to/your/rc3_....nii'             # Resliced CSF segmentation file
    # out_file = 'path/to/output/graph/filename.png'   # Output file (graph) saved as an image
    # --------------------------------------------------
    # Set the data, version 1
    # --------------------------------------------------
    # tp.setFunctionalData(fmri_fn)
    # tp.setMovementParameterFile(rp_fn)
    # tp.setReslicedGM(rgm_fn)
    # tp.setReslicedWM(rwm_fn)
    # tp.setReslicedCSF(rcsf_fn)
    # --------------------------------------------------
    # Set the data, version 2
    # --------------------------------------------------
    # tp.setFunctionalData(fn_file)
    # tp.setMovementParameterFile(rp_fn)
    # tp.setReslicedFiles(rgm_fn, rwm_fn, rcsf_fn)
    # --------------------------------------------------
    # Set the data, version 3
    # --------------------------------------------------
    # tp.setAllInputData(fn_file, rp_fn, rgm_fn, rwm_fn, rcsf_fn)
    # ----------------------------------------------------------------------------------------------