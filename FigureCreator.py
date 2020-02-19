#!/usr/bin/python
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from .utils import checkFileExists

#Value for the carpet plot colorbar range : [-PSC_MIN_MAX, +PSC_MIN_MAX]
from .config import PSC_MIN_MAX
# Tissue names
TISSUE_NAMES = ['GM','WM','CSF']
# Parameters for the Figure
TISSUE_COLORS = [COL_GM,COL_WM,COL_CSF] = ['mediumblue','limegreen','gold']
# Colors of the horizontal lines separating GM/WM/CSF
LC_TISSUE_SEP = 'y'
LW_TISSUE_SEP = 1.5
# Transparent color to generate color map [transparent -> color]
COL_TRANSP = '#66000000'
# Linewidth of the left-side tissue bar on the carpet image plot
LW_TISSUE_BAR = 5
# Transparency value of the left-side tissue bar on the carpet image plot
ALPHA_TISSUE_BAR = .7
# VERTICAL SEPARATIONS BETWEEN RUNS/SESSIONS
# Color of the vertical line separating several Runs/sessions
LC_RUN_SEP = '#52CC00' #line color
LS_RUN_SEP = ':' #linestyle
LW_RUN_SEP = 1.5 #linewidth

class CarpetFigureCreator:
    """
    Class to perform an informative carpet plot
    It is created onto a 4 columns x 6rows GridSpecification.
    The first column contains snap views of the functional file overlaid with either GM, WM or CSF segmentation map.
    The first row contains the Framewise Displacement (FD) graph
    The rest is the carpet plot (using 3 columns and 5 rows)
    """
    def __init__(self):
        """
        Sets default values for some plotting parameters
        """
        self.showRuns = False #
        self.title = '' #Super title for the figure
        self.highlightVolumes = []

    def setCarpetMatrix(self,cp_img):
        """
        Set 2D array representing the 4D functional in a specific way (ThePlot)
        :param cp_img: numpy 2D-array
        """
        self.carpet_img = cp_img

    def setFramewiseDisplacementVector(self,fd_vec):
        """
        Set vector containing estimated framewise displacement per volume
        :param fd_vec: numpy 1D-array
        """
        self.fd_vec = fd_vec

    def setNVoxelsPerTissue(self,n_GM,n_WM,n_CSF):
        """
        Set number of voxels in each tissue (in the functional data space)
        :param n_GM: int, number of GM voxels
        :param n_WM: int, number of WM voxels
        :param n_CSF: int, number of CSF voxels
        """
        (self.n_GM,self.n_WM,self.n_CSF) = (n_GM,n_WM,n_CSF)
        self.nVoxSeg = self.n_GM + self.n_WM + self.n_CSF

    def setBinarySegmentations(self,gm_bin,wm_bin,csf_bin):
        """
        Set binary segmentation maps (in the functional data space)
        :param gm_bin: numpy 3D-array, GM segmentation map
        :param wm_bin: numpy 3D-array, WM segmentation map
        :param csf_bin: numpy 3D-array, CSF segmentation map
        :return:
        """
        self.gm_bin = gm_bin
        self.wm_bin = wm_bin
        self.csf_bin = csf_bin

    def setSliceIndex(self,sl_index):
        """
        Sets the slice index for the Z axis. Used to display functional images in the Carpet Figure
        :param sl_index:
        :return:
        """
        self.sl_index = int(sl_index)

    def setFunctionalFileToDisplay(self,fn_file):
        """
        Sets the functional nifti filename (.nii or.nii.gz) used to display on the figure
        :param fn_file: str, path to the 4D functional file
        """
        self.functional_fn = nib.load(checkFileExists(fn_file, 'Functional run (functional_filename.nii)'))
        # Sets the raw data directly in class attribute
        self.f_img = self.functional_fn.get_data()
        self.Nt = self.f_img.shape[-1]
        # Take as a reference TR the value contained in the header of the nifti file used to display functional data
        # TR is only used to compute the amount of time displayed onto the carpet plot
        # self.TR = self.functional_fn.header.get_zooms()[-1]

    def addRunIndices(self,nVolsPerRun):
        """
        This option allows to indicate a new run/session on the plots or images using vertical green-dashed bars.
        This option shall be used when the user wants to generate a plot of the same subject that was scanned across
        multiple runs
        :param nVolsPerRun: numpy 1D-array, containing the number of volumes in each run
        """
        self.showRuns = True
        self.nVolsPerRun = nVolsPerRun

    def setTitle(self,title):
        """

        :param title:
        :return:
        """
        self.title = title

    def setVolumesToHighlight(self,listVolumeTuples):
        """

        :param listVolumeTuples: list, of tuples [(i0,i1),(i2,i3)] where each tuple defines a starting volume index and
        an ending volume index to higlight on the graphs
        :return:
        """
        self.highlightVolumes = listVolumeTuples

    def plot(self,output_file,psc_min_max=PSC_MIN_MAX):
        """
        It is created onto a 4 columns x 6rows GridSpecification.
        The first column contains snap views of the functional file overlaid with either GM, WM or CSF segmentation map.
        The first row contains the Framewise Displacement (FD) graph
        The rest is the carpet plot (using 3 columns and 5 rows)
        :param output_file: string, path to save the output graph. If empty the graph is shown to the user.
        """

        # Create a Figure
        fig = plt.figure(constrained_layout=True, figsize=(13, 8))

        # Definition of a grid of 4 rows ans 6 columns to create a balanced figure.
        gs = GridSpec(4, 6, figure=fig)
        # 1) Axes for the snap views
        ax_all = fig.add_subplot(gs[0, 0])
        ax_gm = fig.add_subplot(gs[1, 0])
        ax_wm = fig.add_subplot(gs[2, 0])
        ax_csf = fig.add_subplot(gs[3, 0])
        # 2) Axes for the FD plot
        ax_fd = fig.add_subplot(gs[0, 1:])
        # 3) Axes for the carpet plot
        ax_cp = fig.add_subplot(gs[1:, 1:])

        # 1) Snap Views Perform the functional image plotting overlaid with each tissue segmentation map
        img = self.f_img[:, :, self.sl_index, 0].T
        ax_all.imshow(img, cmap='gray', interpolation='none')
        ax_all.set_xticks([]); ax_all.set_yticks([])
        ax_all.set_title('3 tissues' + ' (~' + str(int(self.nVoxSeg / 1000)) + 'K)')
        tissueAxes = [ax_gm,ax_wm,ax_csf]
        bin_maps = [self.gm_bin,self.wm_bin,self.csf_bin]
        nVoxelsPerTissue = [self.n_GM,self.n_WM,self.n_CSF]
        for ax,tissueName, bin_map, nVox, tcol in zip(tissueAxes, TISSUE_NAMES, bin_maps, nVoxelsPerTissue, TISSUE_COLORS):
            # Display image, then overlay binary segmentation, remove x/y ticks and add tissue-specific title
            # make the colormaps
            #Colormap for the segmentation
            cmap1 = colors.LinearSegmentedColormap.from_list('my_cmap', [COL_TRANSP, tcol], 256)
            #cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)
            mask = bin_map[:, :, self.sl_index].T
            ax.imshow(img , cmap='gray', interpolation='none')
            ax.imshow(mask, cmap=cmap1 , interpolation='none', alpha=.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(tissueName + ' (~' + str(int(nVox / 1000)) + 'K)')
            ax_all.imshow(mask, cmap=cmap1 , interpolation='none', alpha=.7)


        n_timepoints = self.fd_vec.shape[0]

        # 2) Framewise-displacement plotting
        ax_fd.plot(self.fd_vec,color='r')
        ax_fd.set_xlim(0, n_timepoints)
        ax_fd.set_ylim(0,2) #max to 2 mm
        ax_fd.set_title('Framewise displacement')
        ax_fd.set_ylabel('Head motion (mm)')
        ax_fd.set_xlabel('fMRI volumes')
        ax_fd.axhline(y=0.5, color='b', linestyle='--')
        #Add vertical lines to show different runs ont the same plot
        if self.showRuns:
            for r in range(len(self.nVolsPerRun)):
                if r>0:
                    ax_fd.axvline(x=float(np.sum(self.nVolsPerRun[:r])), color=LC_RUN_SEP, linestyle=LS_RUN_SEP, lw=LW_RUN_SEP)
        #Check if the user asked to highlight specific volumes
        if len(self.highlightVolumes)>0:
            for (t0,tN) in self.highlightVolumes:
                p = plt.Rectangle((t0,0),width=tN-t0,height=2,color='r',alpha=.3,zorder=-1)
                ax_fd.add_patch(p)

        # 3) Carpet plot
        im = ax_cp.imshow(self.carpet_img, 'gray', vmin=-psc_min_max, vmax=psc_min_max)
        ax_cp.axhline(y=self.n_GM, color=LC_TISSUE_SEP, linestyle='--', lw=LW_TISSUE_SEP)
        ax_cp.axhline(y=self.n_GM + self.n_WM, color=LC_TISSUE_SEP, linestyle='--', lw=LW_TISSUE_SEP)
        ax_cp.set_title('Carpet plot')

        # total_time = (n_timepoints*self.TR)/60 #Compute total time
        ax_cp.set_xlabel('fMRI volumes')
        ax_cp.set_ylabel('Voxels')
        #Instead of counting number of voxels in the y axis, just show GM,WM and CSF in the middle of each horizontal part
        ax_cp.set_yticks([self.n_GM / 2, (self.n_GM + self.n_WM / 2), (self.n_GM + self.n_WM + self.n_CSF / 2)])
        ax_cp.set_yticklabels(TISSUE_NAMES)
        cbar = fig.colorbar(im,ticks=[-psc_min_max,0,psc_min_max],shrink=0.6)
        cbar.ax.set_yticklabels(['-{0}%'.format(psc_min_max),'0','+{0}%'.format(psc_min_max)])# vertically oriented colorbar
        cbar.set_label('Signal')
        # Add rectangle patches to the axes
        p_GM = self.n_GM/float(self.nVoxSeg)
        p_WM = self.n_WM/float(self.nVoxSeg)
        for x in [0,n_timepoints-1]:
            ax_cp.axvline(x=x, ymin=1 - p_GM,      ymax=1,           alpha=ALPHA_TISSUE_BAR, color=COL_GM,  linewidth=LW_TISSUE_BAR)
            ax_cp.axvline(x=x, ymin=1 - p_GM-p_WM, ymax=1-p_GM,      alpha=ALPHA_TISSUE_BAR, color=COL_WM,  linewidth=LW_TISSUE_BAR)
            ax_cp.axvline(x=x, ymin=0,             ymax=1-p_GM-p_WM, alpha=ALPHA_TISSUE_BAR, color=COL_CSF, linewidth=LW_TISSUE_BAR)

        # Add vertical lines to show different runs ont the same plot
        if self.showRuns:
            for r in range(len(self.nVolsPerRun)):
                if r > 0:
                    ax_cp.axvline(x=float(np.sum(self.nVolsPerRun[:r])), color=LC_RUN_SEP, linestyle=LS_RUN_SEP, lw=LW_RUN_SEP)
        plt.gca().set_aspect('auto')

        #Add title if it was defined by the user
        if self.title:
            plt.suptitle(self.title)

        # Save figure or show
        if not output_file:
            plt.show()
        else:
            plt.savefig(output_file, dpi=100)
        plt.close()
