#Import libraries
import os
import sys
import re
import pandas as pd
import numpy as np
import tifffile as tiff
import datetime
import mahotas
from mahotas import convolve
from mahotas.features import pftas
# from stardist.models import StarDist2D 
# import h5py
# import tkinter as tk
# from tkinter import filedialog
# import matplotlib.pyplot as plt
# import warnings

#import modules=======================
#=====================================

#======================================
# General Functions
#======================================
# def initialize_directories():
#     """
#     Initializes the directories for feature extraction. 
#     First it asks for the location containing the raw data.
#     Next it asks for the location to save the output folder. 
#     The extracted features will be dumped into the output folder created inside the given output directory.
    
#     returns:
    
#     raw_image_folder: the location of the raw data
    
#     output_folder: the location to the directory where the output files are saved
#     """
#     root = tk.Tk()
#     root.lift()
#     root.withdraw()

#     print('Please select the folder containing the raw images')
#     raw_image_folder = filedialog.askdirectory(title='Please select the folder containing the raw images')
#     raw_image_folder = os.path.abspath(raw_image_folder)

#     print('Please select path for data to be stored in')
#     output_folder = filedialog.askdirectory(title='Please select path for data to be stored in')
#     output_folder = os.path.abspath(output_folder)

#     today = pd.to_datetime('today')
#     output_folder = os.path.join(output_folder, "Feature Extraction Output {}-Hour{}".format(today.date(), today.hour))
#     try:
#         os.makedirs(output_folder)
#     except:
#         warnings.warn('Unable to create output folders. Check if they already exist')
#     return raw_image_folder, output_folder

# def find_all_files (path: str = os.getcwd(), 
#                     search_str: str = 'RAWDATA'):
#     '''
#     This function takes a path to a directory and returns a list of filepaths to all files in the directory matching a 
#     search string. The default path is the current working directory
#     '''
#     file_paths = []
#     for dirpath, dirnames, files in os.walk(path):
#         print('Searching in {} for keyword {}'.format(dirpath, search_str))
#         for file in files:
#             if search_str in file:
#                 print("found {} in {}".format(file, dirpath))
#                 file_paths.append(os.path.join(dirpath, file))

#     return file_paths

# def _check_channel_orders(channel_image_dict):
#     ch1 = list(channel_image_dict.keys())[0]
#     length = len(channel_image_dict[ch1])
#     for i in range(length):
#         wellindex = extract_wellindex_from_filename(channel_image_dict[ch1][i])
#         fov = extract_FOV_from_filename(channel_image_dict[ch1][i])
#         for ch in list(channel_image_dict.keys())[1:]:
#             if wellindex != extract_wellindex_from_filename(channel_image_dict[ch][i]):
#                 raise ValueError("_check_channel_orders: Wellindex for input images does not match. Check file naming scheme.\
#                                     error caused by {}, which is incompatible with {}".format(channel_image_dict[ch][i], 
#                                                                                               channel_image_dict[ch1][i])
#                                     )
#             if fov != extract_FOV_from_filename(channel_image_dict[ch][i]):
#                 raise ValueError("_check_channel_orders: FOV for input images does not match. Check file naming scheme.\
#                                     error caused by {}, which is incompatible with {}".format(channel_image_dict[ch][i], 
#                                                                                               channel_image_dict[ch1][i]))

# def load_file_paths (raw_image_folder):
#     """
#     Description load_file_paths: 
#     loads files into memory, saves their paths to a dict
#     INPUTS #=====================================
#     raw_image_folder: str = master folder containing raw images to be analyzed. This folder may contain subfolders, 
#     they will all be searched.
#     OUTPUTS #=====================================    
#     dict = saved as channel_image_dict.npy, a dictionary containing imaged keyed by channel names 
#     in numerical order (ch1-chN).
#     list = channelinfo_ordered, the keys of channel_image_dict.
#     #================================================
#     Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
#     """   
    
#     file_paths = find_all_files(raw_image_folder, search_str='.tif')
#     image_file_list = pd.Series(file_paths)

#     print("Found ", len(image_file_list), '.tif files in the given directory')

#     ## We will start by generating new names for the files
#     channel_column_row_FOV = r"([a-zA-Z0-9]+_[0-9]*_+[A-Z]_[0-9]{3}_[a-zA-Z]_[0-9]{4})"
#     match = image_file_list.str.extract(channel_column_row_FOV, expand = False)
#     print(match)
#     ch = match.str.split('_').map(lambda split: split[0])

#     #check if any of the channels have more/less images
#     if len(ch.value_counts().unique()) != 1:
#         print('ERROR: MISSING OR EXTRA IMAGES. \
#               Number of images given for one channel does not match number of images given for another channel.')

#     #create a list of all channels detected
#     channel_info = list(ch.unique())

#     #check if channels result in a non-integer division (ie, missing images)
#     if len(image_file_list)/len(channel_info) != len(image_file_list)//len(channel_info):
#         raise ValueError('ERROR: NUMBER OF IMAGES PROVIDED DOES NOT AGREE WITH NUMBER OF CHANNELS DETECTED. \
#             PLEASE CHECK INPUT IMAGES')

#     #reorder all channels using userinput
#     channel_orders = {}
#     for i in range(len(channel_info)):
#         check = False
#         while check == False:
#             user_in = input("please select "
#                             +channel_info[i]
#                             +' channel number (enter a value between 1 and '
#                             +str(len(channel_info))
#                             +'):')
#             if user_in == '':
#                 if input('Are you sure you want to escape? please enter y/n') == 'y':
#                     raise ValueError('load_file_paths: Channel Order Was Not Selected')
#             try: 
#                 int(user_in)
#                 if abs(int(user_in)) <= len(channel_info):
#                     if user_in not in channel_orders.keys():
#                         check = True
#                     else:
#                         print('Invalid entry: this channel number has already been given:\
#                             {} is in {}'.format(user_in, 
#                                                 channel_orders.keys()))
#                 else:
#                     print('Invalid entry: this channel number is too large: {}'.format(user_in))
#             except:
#                 print('Invalid entry: please enter a valid number')
            
#         channel_orders[int(user_in)-1] = channel_info[i]

#     def rearrange(dictionary):
#                 return [dictionary[i] for i in range(len(dictionary))]
#     channelinfo_ordered = rearrange(channel_orders)

#     #create a dictionary of the image files (sorted) keyed by the channel info (sorted)
#     channel_image_dict = {
#         ch: image_file_list[image_file_list.str.contains(ch)]
#         .sort_values()
#         .reset_index(drop = True) for ch in channelinfo_ordered
#         }
#     _check_channel_orders(channel_image_dict)
#     np.save(os.path.join(raw_image_folder, 'channel_image_dict.npy'), channel_image_dict)
#     return channel_image_dict, channelinfo_ordered

# def select_filepath_tk ():
#     """opens a window to select a file, returns the filepath"""
#     root = tk.Tk()
#     root.withdraw()

#     file = filedialog.askopenfilename(title = 'Please select a file')
#     file = os.path.abspath(file)
#     return file

# def select_directory_tk ():
#     """opens a window to select a file, returns the filepath"""
#     root = tk.Tk()
#     root.withdraw()

#     file = filedialog.askdirectory(title = 'Please select a directory')
#     file = os.path.abspath(file)
#     return file

# def create_directory ():
#     location = select_directory_tk()
#     user_working = True
#     while user_working:
#         directory_name = input("Please input the name of the folder you would like to create")
#         userin = input('Are you sure you want to make the following directory? (type y/n)... {}'
#                        .format(os.path.join(location, directory_name)))
#         if userin == 'y': user_working = False
#     try:
#         os.mkdir(os.path.join(location, directory_name))
#         return os.path.join(location, directory_name)
#     except:
#         return ValueError()

# def save_numpy_to_h5_file(filename: str, data: dict):
#     with h5py.File(filename, 'a') as hf:
#         for dataset in data:
#             hf.create_dataset(dataset, data = data[dataset], dtype = data[dataset].dtype)
#         hf.close()

# def read_numpy_from_h5_file(filename: str, *, dataset_name: str = None, leave_open: bool = False):
#         if leave_open:
#             return h5py.File(filename, 'r')
#         else:
#             with h5py.File(filename, 'r') as hf:
#                 data = hf[dataset_name][:]
#                 hf.close()
#             return data

# def save_dataframe_to_h5_file(filename: str, dataframe: pd.DataFrame, name = None):
#     with h5py.File(filename, 'a') as hf:
#         data = dataframe.to_numpy()
#         index = np.array(list(dataframe.index))
#         columns = dataframe.columns.values.astype('S')
#         hf.create_dataset('data', data = data, dtype = data.dtype)
#         hf.create_dataset('index', data = index, dtype = index.dtype)
#         hf.create_dataset('columns', data = columns, dtype = columns.dtype)
#         if name is not None:
#             hf.create_dataset('name', data = np.array([name]), dtype = '<U{}'.format(len(name)))
#         hf.close()

# def read_dataframe_from_h5_file(filename):
#     with h5py.File(filename, 'r') as hf:
#         data = hf['data'][:]
#         index = hf['index'][:]
#         columns = hf['columns'][:].astype(str)
#         df = pd.DataFrame(data, index = index, columns = columns)
#         if 'name' in hf.keys():
#             df.name = str(hf['name'][:][0])
#         hf.close()
#     df.loc[:,'Path'] = filename
#     return df

# def extract_well_metadata(platemap, wellindex):
#     """Extracts metadata from a platemap dataframe given a wellindex"""
#     return platemap[platemap.WellIndex == wellindex]

# def read_all_h5_outputs (df_name: str, file_folder_loc: str, search_str: str = 'RAWDATA'):
#     '''
#         Reads in all data containing search_str from all files contained in file_folder_loc and its subfolders.
#         Inputs:
#             raw_data_folder: the folder containing the raw data
#             df_name: str to name the dataframe
#             search_str: str to find the files. Default = 'RAWDATA'
#         Outputs:
#             raw_data: a pandas dataframe containing the raw data
#     '''
#     files = find_all_files(file_folder_loc, search_str)
#     data = []
#     for file in files:
#         data.append(read_dataframe_from_h5_file(file))

#     raw_data = pd.concat(data, ignore_index=True)
#     check_shape = raw_data.shape[0]
#     platemap_path = find_all_files(file_folder_loc, 'platemap.txt')
#     platemap = pd.read_csv(platemap_path[0], sep = '\t')
#     raw_data = platemap.merge(raw_data, how = 'inner', left_on='WellIndex', right_on='WellIndex')    
#     if raw_data.shape[0] != check_shape:
#         raise ValueError("ERROR: SOME CELLS WERE LOST IN MERGE. CHECK PLATEMAP AND RAW DATA INPUTS FOR MISSING ENTRIES")
#     raw_data.name = df_name
#     return raw_data


#================================================
# Image Segmentation and Preprocessing Functions
#================================================
# GLOBAL VARS:

# load the pretrained model for cell nuclei detection
# creates a pretrained model
# MODEL_2D_FLUORESCENT_CELL_SEGMENTATION = StarDist2D.from_pretrained('2D_versatile_fluo')


# # Description glaylconvert
# # img ==> numpy array
# # Kenta Ninomiya @ Kyushu University: 2021/3/19
# def glaylconvert(img, orgLow, orgHigh, qLow, qHigh):
#     #Quantization of the grayscale levels in the ROI
#     img=np.where(img>orgHigh, orgHigh, img)
#     img=np.where(img<orgLow, orgLow, img)
#     cImg=((img-orgLow)/(orgHigh-orgLow))*(qHigh-qLow)+qLow
#     return(cImg) 

# def segment_images_generate_mask_and_details_filenames(segmentation_output_folder):
#     """Create filenames given a directory"""
#     mask_filename = os.path.join(segmentation_output_folder, "mask_files.h5")
#     details_filename = os.path.join(segmentation_output_folder, "detail_files.h5")
#     return mask_filename, details_filename

# def _segment_images_initialize_run (channel_image_dict, raw_image_folder):
#     """Initialize the channel to segment on and create output directories"""
#     #get user input for segmentation channel
#     ask = True
#     while ask == True:
#         segmentation_channel = input('Please type the name of the channel used for nuclei segmentation (one of {})'
#                                      .format(list(channel_image_dict.keys())))
#         if segmentation_channel in channel_image_dict.keys():
#             print("Selected Channel '{}'...".format(segmentation_channel))
#             ask = False
#         elif segmentation_channel in ['','stop', 'exit', 'done']:
#             raise ValueError("segment_images: no segmentation channel was selected")
#         else:
#             print('{} is an invalid entry. Must be one of {}'
#                   .format(segmentation_channel, list(channel_image_dict.keys())))
#     #create a subfolder with todays date for saving files to
#     today = pd.to_datetime('today')
#     segmentation_output_folder = os.path.join(raw_image_folder, "{}_segmentation_files-{}-StartHour{}"
#                                               .format(segmentation_channel,  today.date(), today.hour))
#     try:
#         if os.path.isdir(segmentation_output_folder):
#             print('WARNING: "segment_images" found the output folder below already exists.')
#             print(segmentation_output_folder)
#         else:
#             os.mkdir(segmentation_output_folder)
#     except:
#         raise ValueError("segment_images: Unable to create output folders. \
#             Check if the following input for raw_image_folder is valid... {}".format(raw_image_folder))
#     #create filenames for masks and details
#     mask_filename, details_filename = segment_images_generate_mask_and_details_filenames(segmentation_output_folder)
#     return segmentation_channel, segmentation_output_folder, mask_filename, details_filename

# def segment_images(channel_image_dict, raw_image_folder, chunking = 20):
#     """
#     Description segment_images: 
#     segments fluorescent images using stardist.
#     INPUTS #=====================================
    
#     channel_image_dict: dict = dictionary keyed by channel (in order), containing image file paths
#     raw_image_folder: str = path to the folder containing the raw images. Used for saving segmentation masks.
#     chunking: int = number of images to analyze before saving the data to a .h5 file. 
#         EX: chunking = 20 means that the script will hold 20 image masks in local memory before 
#         saving them to the .h5 file
#     OUTPUTS #=====================================
#     dict = saved as mask_file_dict.npy, a dictionary keyed by image filename which contains the identifier 
#     for the mask when reading the .h5 mask data
#     dict = saved as details_file_dict.npy, a dictionary keyed by image filename which contains the identifier 
#     for the details when reading the .h5 mask data
#     str = saved as segmentation_channel.npy, the name of the channel used for segmenting in this run
#     str = segmentation_output_folder, the folder containing the data generated
#     #================================================
#     Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
#     ADAPTED FROM: 
#     @Kenta Ninomiya 2021, s1_o2_segmentation_size_thresholding.py
#     """
#     #create empty dictionaries for mask and detail files
#     mask_file_dict = {}
#     details_file_dict = {}
#     #initialize run information
#     segmentation_channel, segmentation_output_folder, mask_filename, details_filename = _segment_images_initialize_run (channel_image_dict, raw_image_folder)
#     #iterate through every file in the given file dict/segmentation channel pair to segment each set of images
#     chunk_i = 0
#     masks = {}
#     detailsdict = {}
#     for imgfile in channel_image_dict[segmentation_channel]:
#         # obtain the filename of the image which will be segmented
#         segmentation_image_name = imgfile.split("\\")[-1]
#         print("segmenting {}...".format(segmentation_image_name))
#         # load the image in the segmentation channel, normalize, generate dataset names
#         tmpImg = tiff.imread(imgfile)
#         normImg = glaylconvert(tmpImg, np.percentile(tmpImg, 1), np.percentile(tmpImg, 99), 0, 1)
#         mask_set_name = segmentation_image_name.replace('.tif','_seg')
#         mask_file_dict[imgfile] = mask_set_name
#         details_file_dict[imgfile] = [segmentation_image_name.replace('.tif',
#                                                                       '_seg_details_{}'.format(i))
#                                       for i in ['points', 'prob']]
#         # segment the image, save masks/details to the dictionary
#         masks[mask_set_name], details = MODEL_2D_FLUORESCENT_CELL_SEGMENTATION.predict_instances(normImg)
#         detailsdict[details_file_dict[imgfile][0]] = details['points'] 
#         detailsdict[details_file_dict[imgfile][1]] = details['prob']
#         chunk_i+=1
#         # for every chunking step, save the masks and details to a numpy file and clear the local dictionaries
#         if chunk_i == chunking:
#             save_numpy_to_h5_file(mask_filename, masks)
#             save_numpy_to_h5_file(details_filename, detailsdict)
#             chunk_i = 0
#             masks = {}
#             detailsdict = {}
#     #finished, save the dictionaries which give easy access to the .h5 files created
#     print('Segmentation Done!')
#     print('Files saved to {}'.format(segmentation_output_folder))
#     np.save(os.path.join(segmentation_output_folder, 'segmentation_channel.npy'), 
#             segmentation_channel, 
#             allow_pickle = True)
#     np.save(os.path.join(segmentation_output_folder, 'mask_file_dict.npy'), 
#             mask_file_dict, 
#             allow_pickle = True)
#     np.save(os.path.join(segmentation_output_folder, 'details_file_dict.npy'), 
#             details_file_dict, 
#             allow_pickle = True)
#     return mask_file_dict, details_file_dict, segmentation_channel, segmentation_output_folder

#======================================
# Feature Extraction Functions
#======================================
# Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
# ADAPTED FROM: 1) Mahotas package v: mahotas\features\tas.py 
#               2) https://github.com/DWALab/Schormann-et-al/blob/master/MBF_texture_suite_b2.proc.txt
#======================================
# GLOBAL VARS:

# for 2D
# _M2 = a 3x3 matrix with all 1s and 10 in the middle
# _bins2 = np.array([0,1,2,3,4,5,6,7,8,9,10])
_M2 = np.ones((3, 3))
_M2[1, 1] = 10
_bins2 = np.arange(10)

# Modified Bin edges for similarity to original TAS paper
# Note that the bin edges are of unit length. 
# This allows for density=True to give the proper output.
_bins2_mod = np.arange(9.5,19.5)

# for 3D
# _M3 = 3 3x3 matrices with all 1s and 28 in the middle
# _bins3 = np.array([0,1,2,3,4,5,6,7,8,9,10])
_M3 = np.ones((3, 3, 3))
_M3[1,1,1] = _M3.sum() + 1
_bins3 = np.arange(28)

_masknumber_dict = {0:'mean-plus-{x}percent -- mean-minus-{x}percent', 
                        1:"max -- mean-minus-{x}percent", 
                        2:"max -- mean-plus-{x}percent", 
                        3:"max -- mean"}



def count_pixels_mahotas_tas(img: np.array):
    """
    count_pixels_mahotas_tas: takes an image and performs the pixel counting step 
    in TAS feature extraction via convolution.
    INPUTS #=====================================
    img: np.array = the image whose pixel values are to be analyzed
    OUTPUTS #=====================================
    np.array = the TAS feature values for the given image
    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    ADAPTED FROM Mahotas package v: mahotas\features\tas.py 
    """

    # if IMG == 2D
    if len(img.shape) == 2:
        M = _M2
        bins = _bins2
        saved = 9
    # if IMG == 3D
    elif len(img.shape) == 3:
        M = _M3
        bins = _bins3
        saved = 27
    else:
        raise ValueError('mahotas.tas: Cannot compute TAS for image of %s dimensions' % len(img.shape))

    def _ctas(img):
        # Convolve the image with the TAS Kernel M
        V = convolve(img.astype(np.uint8), M)
        # Count the values for each bin (0-9, ie number of neighbors)
        values,_ = np.histogram(V, bins=bins)
        # Slice out only the values up to 9
        values = values[:saved] ########## using this would count the number of pixels with value=0 whose
                                ########## neighbors have value=1, I beleive this is a mistake. See original TAS paper
                                ########## see _ctas_mod below for a new implementation
                                ########## - Martin Alvarez-Kuglen, Sanford Burnham Prebys Medical Discovery Institute
        # Sum the values
        s = values.sum()
        if s > 0:
            # return the normalized number
            return values/float(s)
        # else return 0s
        return values
    
    def _ctas_mod(bwimg,M,bins):
        # Convolve the image with the TAS Kernel _M2
        V = convolve(bwimg.astype(np.uint8), 
                     M)
        # Count the values for each bin (10-19, ie white pixels with number of white neighbors)
        # Density=True calcualtes the PDF, which in the case of unit bin edges is the 
        # equivalent of normalizing the values to their sum.
        values,_ = np.histogram(V, 
                                bins=bins,
                                range=(9.5, 18.5),
                                density=True
                                )
        # np.histogram gives NaN values when division by 0 occurs,
        # so check if nan in the array, if so fill nan with 0 values
        if True in np.isnan(values):
            np.nan_to_num(values,
                          nan=0.0,
                          copy=False)
        # use the check below if uncertain about density=True statement
        # if int(values.sum())!=1:
        #     if int(values.sum().astype(np.float16))!=1:
        #         raise RuntimeError("_ctas_mod: sum of the TAS features does not add to 1, density=True not working")
        return values

    return _ctas_mod(img > 0,M,bins)

def MIELv023_tas_name_features(channel: str or int, mask_number: int, percent: float, original_names: str = True, dim: int = 2):
    """
    Description MIELv023_tas_name_features: 
    Generates a list of 9 strings (one for each TAS statistic, ie, N-neighbors counted)
    INPUTS #=====================================
    channel: str = channel name. EX: Dapi
    mask_number: int in range (0,3) = kind of threshold used. See _masknumber_dict for naming. 
    percent: float = percentage used when thresholding for mask (see x above). Use None if mask_number = 3
    OUTPUTS #=====================================
    list = 9 strings (one for each TAS statistic, ie, N-neighbors counted)
    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    ADAPTED FROM Mahotas package v: mahotas\features\tas.py 
    """
    # if original_names:
    #     if mask_number == 3:
    #         names = ["_".join((str(channel), "TXT", "TAS","{}")).format(i+28) for i in range(9)]
        
    #     else:
    #         names = ["_".join((str(channel), "TXT", "TAS","{}", str(int(percent*100))))
    #                  .format(i+9*mask_number+1) for i in range(9)]

    # else:
    #     if percent == None:
    #         names = ["_".join((str(channel), "TXT", "TAS","{}neighbors", _masknumber_dict[mask_number]))
    #                  .format(i) for i in range(9)]
        
    #     else:
    #         names = ["_".join((str(channel), "TXT", "TAS","{}neighbors", _masknumber_dict[mask_number]
    #                            .format_map({'x':int(percent*100)}))).format(i) for i in range(9)]
    if percent == None:
        names = ["_".join((str(channel), "TXT", "TAS","{}neighbors", _masknumber_dict[mask_number]))
                    .format(i) for i in range(3**dim)]
    
    else:
        names = ["_".join((str(channel), "TXT", "TAS","{}neighbors", _masknumber_dict[mask_number]
                            .format_map({'x':int(percent*100)}))).format(i) for i in range(3**dim)]
    return names

def MIELv023_tas_masking(image: np.array, mu: float or int, mask_number: int, percentage_number: float):
    """
    Description MIELv023_tas_masking: 
    Generates a masked image according to one of four different thresholding categories:
    0 = (mean+{x}%, mean-{x}%), 1 = (max, mean-{x}%), 2 = (max, mean+{x}%), 3 = (max, mean)
    INPUTS #=====================================
    image: np.array = raw image to be masked
    mu: average pixel intensity for pixels with value > 0 in image
    mask_number: int (0-3) = which mask option to use (see above for options)
    percentage_number: float = percentage used when thresholding for mask
    OUTPUTS #=====================================
    np.array = new image masked by the threshold values given above.
    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    ADAPTED FROM Mahotas package v: mahotas\features\tas.py 
    """

    if mask_number == 0:
        maximum, minimum = (1+percentage_number, 1-percentage_number)
        mask1 = np.where(image < minimum*mu, 0, 1)
        mask2 = np.where(image < maximum*mu, 0, 1)
    if mask_number == 1:
        minimum = 1-percentage_number
        mask1 = np.where(image < minimum*mu, 0, 1)
        mask2 = np.zeros_like(image)
    if mask_number == 2:
        minimum = 1+percentage_number
        mask1 = np.where(image < minimum*mu, 0, 1)
        mask2 = np.zeros_like(image)
    if mask_number == 3:
        mask1 = np.where(image < mu, 0, 1)
        mask2 = np.zeros_like(image)

    newImg = np.subtract(mask1, mask2)
    return newImg

def extract_MIELv023_tas_features (segCellImg: np.array, ch: str or int, average_intensity: float, 
                                   percentages: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Description extract_MIELv023_tas_features
    Generates TAS data as described in ELIFE PAPER
    INPUTS #=====================================
    segCellImg: np.array = raw image to be analyzed
    ch: str = image channel
    percentages: list of floats = percentages used for thresholding. KEEP AS DEFAULT IF REPLICATING ACAPELLA
    OUTPUTS #=====================================
    pd.Series = TAS features as described in acaeplla v2.4
    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    ADAPTED FROM:
    https://github.com/DWALab/Schormann-et-al/blob/master/MBF_texture_suite_b2.proc.txt
    """
    if average_intensity == None:
        average_intensity = np.mean(segCellImg, where = segCellImg>0)

    # initialize series to store TAS data for each mode (0-3)
    tas_data0 = []
    tas_data1 = []
    tas_data2 = []
    #Extract TAS Features
    for percent in percentages:
        tas_data0.append(
            pd.Series(
                data = count_pixels_mahotas_tas(
                    MIELv023_tas_masking(segCellImg, 
                                         average_intensity, 
                                         0, 
                                         percent)
                    ), 
                index = MIELv023_tas_name_features(channel = ch, 
                                                   mask_number = 0, 
                                                   percent = percent,
                                                   dim=len(segCellImg.shape)), 
                dtype='float64'
                )
            )
        tas_data1.append(
            pd.Series(
                data = count_pixels_mahotas_tas(
                    MIELv023_tas_masking(segCellImg, 
                                         average_intensity, 
                                         1, 
                                         percent)), 
                index = MIELv023_tas_name_features(channel = ch, 
                                                   mask_number = 1, 
                                                   percent = percent,
                                                   dim=len(segCellImg.shape)), 
                dtype='float64'
                )
            )
        tas_data2.append(
            pd.Series(
                data = count_pixels_mahotas_tas(
                    MIELv023_tas_masking(segCellImg, 
                                         average_intensity, 
                                         2, 
                                         percent)), 
                index = MIELv023_tas_name_features(channel = ch, 
                                                   mask_number = 2, 
                                                   percent = percent,
                                                   dim=len(segCellImg.shape)), 
                dtype='float64'
                )
            )
    tas_data3 = pd.Series(data = count_pixels_mahotas_tas(MIELv023_tas_masking(segCellImg, 
                                                                               average_intensity,
                                                                               3,
                                                                               percent)),
                          index = MIELv023_tas_name_features(channel = ch, 
                                                             mask_number = 3, 
                                                             percent = None,
                                                             dim=len(segCellImg.shape)),
                          dtype = 'float64'
                          )
    return pd.concat([pd.concat(tas_data0), pd.concat(tas_data1), pd.concat(tas_data2), tas_data3])

# Description cellselecter
# separate cells from the image
# Kenta Ninomiya @ Kyushu University: 2021/07/29
def cellselecter(img, label, margin, cellIdx):
    # get the binary image of the "celIdxl"-th cell
    objCellLabel=np.where(label==cellIdx, 1, 0) #set teh value one to the "celIdxl"-th cell, zero for the others 
    
    # get the size of the image
    [rowNum, colNum]=objCellLabel.shape
    
    # get a maximum and minimum row coordinate
    coordinateRow=np.arange(0, rowNum)
    idxRow=np.any(a=objCellLabel==1, axis=int(1))
    
    # get a maximum and minimum column coordinate
    coordinateCol=np.arange(0, colNum)
    idxCol=np.any(a=objCellLabel==1, axis=int(0))
    
    rowMin=coordinateRow[idxRow][0]
    rowMax=coordinateRow[idxRow][-1]
    colMin=coordinateCol[idxCol][0]
    colMax=coordinateCol[idxCol][-1]
    
    # slicing the matrix
    objImg=img[rowMin:rowMax+1,colMin:colMax+1]
    objImg=np.pad(objImg, [margin,margin], 'constant')
    
    objCellLabel=objCellLabel[rowMin:rowMax+1,colMin:colMax+1]
    objCellLabel=np.pad(objCellLabel, [margin,margin], 'constant')
    
    return objImg, objCellLabel

def extract_wellindex_from_filename(filename, 
                                    rowlet_colnum = re.compile('__[a-zA-Z]_[0-9]+'), 
                                    let = re.compile('[a-zA-Z]'), 
                                    num = re.compile('[0-9]+')):
    """Extracts a wellindex from a filename given the filename is in the IC200 output file format"""
    mat = rowlet_colnum.search(filename)[0]
    rowlet = let.search(mat)[0]
    colnum = num.search(mat)[0]

    column_lettertonum_dict = {
                                "A":"1",
                                "B":"2",
                                "C":"3",
                                "D":"4",
                                "E":"5",
                                "F":"6",
                                "G":"7",
                                "H":"8",
                                "I":"9",
                                "J":"10",
                                "K":"11",
                                "L":"12",
                                "M":"13",
                                "N":"14",
                                "O":"15",
                                "P":"16",
                                }

    return int(column_lettertonum_dict[rowlet]+colnum)

def extract_FOV_from_filename(filename, row = re.compile('_r_[0-9]+_'), col = re.compile('_c_[0-9]+_'), 
                              num = re.compile('[0-9]+')):
    """Extracts a field of view from a filename given the filename is in the IC200 output file format"""

    rowmat = row.search(filename)[0]
    rownum = num.search(rowmat)[0]

    colmat = col.search(filename)[0]
    colnum = num.search(colmat)[0]

    # FOV = 'row'+rownum[-2:]+'col'+colnum[-2:]
    FOV = int('0'+rownum[-2:]+colnum[-2:])
    return FOV

def threshold_object_size(size_series, size_thresh_low, size_thresh_high):
    """excludes objects of a particular value given high and low thresholds"""
    bool_mask = ((size_series.values > size_thresh_low) & (size_series.values < size_thresh_high))
    in_objects = size_series[bool_mask].index.values
    #out_objects = size_series[~bool_mask].index.values
    return in_objects

def calc_elliptical_eccentricity (bwImg):
    """Calculates the elliptical eccentricity from the semimajor and semiminor\
        axes of an object given by a black/white image"""
    semimajor, semiminor = mahotas.features.ellipse_axes(bwImg)
    #check if math breaks
    if semimajor == 0:
        return np.nan
    if semiminor/semimajor > 1:
        return np.nan
    return np.sqrt(1 - (np.square(semiminor) / np.square(semimajor)))

def _initialize_run (segmentation_mask_file_location, output_directory, instance):
    #load in the directory dictionaries
    segmentation_details_file_dict = np.load(os.path.join(segmentation_mask_file_location, 
                                                          'details_file_dict.npy'), 
                                             allow_pickle = True).item()
    segmentation_mask_file_dict = np.load(os.path.join(segmentation_mask_file_location, 
                                                       'mask_file_dict.npy'), 
                                          allow_pickle = True).item()
    segmentation_channel = np.load(os.path.join(segmentation_mask_file_location, 
                                                'segmentation_channel.npy'), 
                                   allow_pickle = True).item()

    #create output directory
    instance_output_directory = os.path.join(output_directory, 'output{}'.format(instance))
    try:
        if os.path.isdir(instance_output_directory):
            print('WARNING: "load_images_extract_features" Unable to create output folders because they already exist')
        else:
            os.makedirs(instance_output_directory)
    except:
        raise ValueError("_initialize_run: instance_output_directory is not a directory or cannot be created.\
                            Please check your input for output_directory: {}".format(output_directory))
    return segmentation_details_file_dict, segmentation_mask_file_dict, segmentation_channel, instance_output_directory

def run_feature_extraction (channel_image_dict: dict, 
                            segmentation_mask_file_location: str,
                            output_directory: str,
                            instance: int, 
                            size_thresh_low: float, 
                            size_thresh_high: float, 
                            *, 
                            label_channels: list = [],
                            only_count: bool = False,
                            add_morphology_features = True):

    """
    Description load_images_extract_features: 
    extracts TAS features from images given segmentation masks.
    INPUTS #=====================================
    channel_image_dict: dict = dictionary keyed by channel (in order), containing image file paths
    segmentation_mask_file_location: str = path to the folder containing the segmentation files.
    output_directory: str = path to the folder where the output directory will be made (ie where the data will be saved).
    Instance: int = integer number of the instance number. Used for making the output directory.
    size_thresh_low: int = minimum pixel area (inclusive). Objects size <= size_thresh_low will be removed.
    size_thresh_high: int = maximum pixel area (inclusive). Objects size >= size_thresh_high will be removed.
    OUTPUTS #=====================================
    ImgDATA: pd.DataFrame = dataframe containing TAS features for all images in all, saved as np.array in a .h5 file.
    #================================================
    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    """
    # initialize run information
    (segmentation_details_file_dict, 
     segmentation_mask_file_dict, 
     segmentation_channel, 
     instance_output_directory) = _initialize_run (segmentation_mask_file_location, 
                                                   output_directory, 
                                                   instance)
    # create report file
    extraction_report_data = {}
    #create empty dict for saving well/FOV files for concatenation later on
    wellindex_fov_output_dict = {}
    #check if the channels are in order
    _check_channel_orders(channel_image_dict)
    # open the mask, points, probability, and report files
    with read_numpy_from_h5_file(
        os.path.join(segmentation_mask_file_location, 'mask_files.h5'), 
        leave_open=True) as masks_openfile, \
            read_numpy_from_h5_file(
                os.path.join(segmentation_mask_file_location, 'detail_files.h5'), 
                leave_open=True) as details_openfile, \
                            create_report_txt_file('extraction_report', 
                                                   output_directory,
                                                   leave_open=True) as extraction_report:
        #iterate through all indices in the file lists (should be the same for all channels)
        for i in channel_image_dict[segmentation_channel].index:
            #extract wellindex and FOV, check if it matches segmentation images. Raise error if not
            wellindex = extract_wellindex_from_filename(channel_image_dict[segmentation_channel][i])
            fov = extract_FOV_from_filename(channel_image_dict[segmentation_channel][i])
            #add wellindex to the output dict for storage
            if wellindex not in wellindex_fov_output_dict.keys():
                wellindex_fov_output_dict[wellindex] = {}
            #create filename for data to be saved under
            rawdata_filename = str("RAWDATA"
                +channel_image_dict[segmentation_channel][i]
                .split("\\")[-1].replace(segmentation_channel, '').replace('.tif', '.h5'))
            #check if the file already exists, continue if it does
            if os.path.exists(os.path.join(instance_output_directory, rawdata_filename)):
                write_to_report_txt_file(
                    "##########################################################",
                    "\nREPORT FOR WELL {} FOV {}".format(wellindex, fov),
                    "\n{}".format(os.path.join(instance_output_directory, rawdata_filename)),
                    "\nABOVE FILE ALREADY EXISTS IN THE OUTPUT{} FOLDER".format(instance),
                    report_filename=extraction_report,
                    leave_open=True
                    )                
                continue
            # create a dictionary of all image files in the set
            images_by_channel = {ch: tiff.imread(channel_image_dict[ch][i]) for ch in channel_image_dict.keys()}
            #print('Found image data type', images_by_channel[segmentation_channel].dtype)        
            #read in masks
            masks = masks_openfile[segmentation_mask_file_dict[channel_image_dict[segmentation_channel][i]]][:]
            points = details_openfile[segmentation_details_file_dict[channel_image_dict[segmentation_channel][i]][0]][:]
            prob = details_openfile[segmentation_details_file_dict[channel_image_dict[segmentation_channel][i]][1]][:]
            # FOR OLD TECHNIQUE -- ERASE AFTER V1 STABLE
            # masks = read_numpy_from_h5_file(os.path.join(segmentation_mask_file_location, 'mask_files.h5'), 
            #                                 segmentation_mask_file_dict[channel_image_dict[segmentation_channel][i]])
            # points = read_numpy_from_h5_file(os.path.join(segmentation_mask_file_location, 'detail_files.h5'), 
            #                                  segmentation_details_file_dict[channel_image_dict[segmentation_channel][i]][0])
            # prob = read_numpy_from_h5_file(os.path.join(segmentation_mask_file_location, 'detail_files.h5'), 
            #                                segmentation_details_file_dict[channel_image_dict[segmentation_channel][i]][1])

            #get the unique number of the objects --> object identifier
            colors, counts = np.unique(masks.reshape(-1, 1),
                                return_counts = True,
                                axis = 0)
            colors = list(np.delete(colors,0))
            if len(colors) == 0:
                print("WARNING: Well {} FOV {} had no segmented objects".format(wellindex,
                                                                                fov))
                write_to_report_txt_file(
                    "##########################################################",
                    "\nWARNING: Well {} FOV {} had no segmented objects".format(wellindex,
                                                                                fov),
                    "MOVING TO NEXT WELL",
                report_filename=extraction_report,
                leave_open=True                    
                )
                continue
            #create empty dataframe to append data to, indexed by cell number
            #add in the relevant data for wellindex, FOV, x&y coordinates, object area, morphology
            ImgDATA = pd.DataFrame(index = colors)
            ImgDATA['WellIndex'] = wellindex
            ImgDATA['Field Of View'] = fov
            ImgDATA['XCoord'] = ImgDATA.index.map(lambda i: points[:,0][i-1])
            ImgDATA['YCoord'] = ImgDATA.index.map(lambda i: points[:,1][i-1]) 
            ImgDATA['prob'] = ImgDATA.index.map(lambda i: prob[i-1])
            ImgDATA['object pixel area'] = ImgDATA.index.map(lambda i: np.count_nonzero(np.where(masks == i, 1, 0)))
            ImgDATA['object size (micrometer squared)'] = ImgDATA['object pixel area'].map(lambda i: i*0.105625)
            if add_morphology_features:
                ImgDATA['{}_MOR_object_roundness'.format(segmentation_channel)] = 0
                ImgDATA['{}_MOR_graph_eccentricity'.format(segmentation_channel)] = 0
                ImgDATA['{}_MOR_elliptical_eccentricity'.format(segmentation_channel)] = 0
            print('Analyzing well {}, FOV {}....'.format(wellindex, fov))
            colors_size_thresholded = threshold_object_size(ImgDATA['object pixel area'], 
                                                            size_thresh_low, 
                                                            size_thresh_high)
            
            if 'total_object_count' not in extraction_report_data.keys():
                extraction_report_data['total_object_count']=ImgDATA.shape[0]
                extraction_report_data['thresholded_object_count']=colors_size_thresholded.shape[0]
            else:
                extraction_report_data['total_object_count']+=ImgDATA.shape[0]
                extraction_report_data['thresholded_object_count']+=colors_size_thresholded.shape[0]

         #   extraction_report_data[wellindex][fov] = 
            
            write_to_report_txt_file(
                "##########################################################",
                "\nREPORT FOR WELL {} FOV {}".format(wellindex, fov),
                "\nSize thresholding at min={}, max={} eliminated {} cells".format(size_thresh_low, 
                                                                                   size_thresh_high,
                                                                                   ImgDATA.shape[0] 
                                                                                   - colors_size_thresholded.shape[0]),
                "\n",
                "\nTotal Object Count: {}".format(ImgDATA.shape[0]),
                "\nObjects After Thresholding: {} ({}%)".format(colors_size_thresholded.shape[0],
                                                                np.round(colors_size_thresholded.shape[0]
                                                                        /ImgDATA.shape[0],
                                                                        3)
                                                                *100),
                report_filename=extraction_report,
                leave_open=True
                )
            # iterate through all channels
            for ch in images_by_channel:
                if only_count == True:
                    continue
                #Create column for adding channel intensity, set values = 0
                ImgDATA['{}_object_average_intensity'.format(ch)] = 0
                #get the image for the channel
                temp_img = images_by_channel[ch]
                #create a dict to create a df for the data
                channel_TASfeatures_to_df_dict = {}
                # iterate through all segmented objects
                for objectIdx in colors_size_thresholded:
                    # get the image with 0 pixel margin
                    object_img, object_label = cellselecter(temp_img, masks, 1, objectIdx)
                    if add_morphology_features:
                        if ch == segmentation_channel:
                            roundness=mahotas.features.roundness(object_label)
                            graph_eccentricity=mahotas.features.eccentricity(object_label)
                            # COMMENTED CODE FOR TESTING -- 3-17-22
                            # elliptical_eccentricity=calc_elliptical_eccentricity(object_label)
                            # print("roundness\n",
                            #       roundness,
                            #       "graph_eccentricity\n",
                            #       graph_eccentricity,
                            #       "elliptical_eccentricity\n",
                            #       elliptical_eccentricity)
                            ImgDATA.loc[objectIdx,
                                       str(segmentation_channel)
                                       +'_MOR_object_roundness']=roundness
                            ImgDATA.loc[objectIdx,
                                       str(segmentation_channel)
                                       +'_MOR_object_eccentricity']=graph_eccentricity
                            # ImgDATA.loc[objectIdx,
                            #            str(segmentation_channel)
                            #            +'_MOR_elliptical_eccentricity']=elliptical_eccentricity
                            # print(ImgDATA.loc[objectIdx, segmentation_channel+'_MOR_object_roundness'])
                            # print(ImgDATA.loc[objectIdx, segmentation_channel+'_MOR_graph_eccentricity'])
                            # print(ImgDATA.loc[objectIdx, segmentation_channel+'_MOR_elliptical_eccentricity'])
                    # set pixels outside of mask = 0
                    seg_object_img = np.where(object_label == 1, object_img, 0)
                    # average intensity of object (ie, pixels inside the mask)
                    object_avg_int = np.mean(seg_object_img, where = (object_label==1))
                    # add intensity calculation as a feature
                    ImgDATA.at[objectIdx, '{}_object_average_intensity'.format(ch)] = object_avg_int
                    #continue without TAS extraction for label-only channels
                    if ch in label_channels:
                        continue                
                    # extract TAS features
                    channel_TASfeatures_to_df_dict[objectIdx] = extract_MIELv023_tas_features(seg_object_img, 
                                                                                              ch, 
                                                                                              object_avg_int)
                # if the channel is not a label channel, create and merge TAS df
                if ch not in label_channels:
                    df = pd.DataFrame.from_dict(channel_TASfeatures_to_df_dict, orient = 'index')
                    ImgDATA = ImgDATA.merge(df, left_index=True, right_index=True)

                print(ImgDATA.shape)
            save_dataframe_to_h5_file(os.path.join(instance_output_directory, rawdata_filename), ImgDATA)
            wellindex_fov_output_dict[wellindex][fov] = rawdata_filename
        np.save(os.path.join(instance_output_directory, 'wellindex_fov_output_dict.npy'), 
                wellindex_fov_output_dict, 
                allow_pickle = True)
        if 'total_object_count' in extraction_report_data.keys():
            write_to_report_txt_file(
                "##########################################################",
                "\nCONCLUSION",
                "\nSize thresholding at min={}, max={} eliminated {} cells".format(size_thresh_low, 
                                                                                    size_thresh_high,
                                                                                    extraction_report_data['total_object_count']
                                                                                    -extraction_report_data['thresholded_object_count']),
                "\n",
                "\nTotal Object Count: {}".format(extraction_report_data['total_object_count']),
                "\nObjects After Thresholding: {} ({}%)".format(extraction_report_data['thresholded_object_count'],
                                                                np.round(extraction_report_data['thresholded_object_count']
                                                                        /extraction_report_data['total_object_count'],
                                                                        3)
                                                                *100),
                report_filename=extraction_report,
                leave_open='close'
                )

    return wellindex_fov_output_dict#, ImgDATA

def write_to_report_txt_file(*args, report_filename: str, leave_open: str or bool=False, **kwargs):
    if leave_open not in [True, False, 'close']:
        raise ValueError("write_to_report_txt_file:\
                         \nleave_open must be in {} but {} was given".format([True, False, 'close'],
                                                                                                      leave_open))

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    if leave_open == True:
        print(*args, file=report_filename)
    elif leave_open == 'close':
        print(*args, file=report_filename)
        report_filename.close()
    elif leave_open == False:
        with open(report_filename, 'a') as report:
            print(*args, file=report)
        report.close()
    pd.set_option("display.max_rows", 30, "display.max_columns", 30)

def create_report_txt_file(name: str, output_folder: str, leave_open: bool = False):
    filename = os.path.join(output_folder, re.sub('\W+','_', name+'_{}'.format(datetime.datetime.now()))+'.txt')
    if leave_open == True:
        report = open(filename, 'w')
        report.write(name+"..... File Created: {}".format(datetime.datetime.now()))
        print("__________________________________________________________________________", file = report)
        return report
    else:
        with open(filename, 'w') as report:
            report.write(name+"..... File Created: {}".format(datetime.datetime.now()))
            print("__________________________________________________________________________", file = report)
        report.close()
        return filename