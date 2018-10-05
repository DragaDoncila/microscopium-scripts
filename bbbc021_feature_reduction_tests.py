import pandas as pd
import os
import re

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from microscopium import io
from microscopium.preprocess import montage_stream
from microscopium.preprocess import correct_multiimage_illumination
from microscopium.preprocess import find_background_illumination
from microscopium.features import default_feature_map


#FILE_NAME_PREFIX = "Week1_150607_"
IMAGE_FILE_PATH = "/data/BBBC021/data.broadinstitute.org/bbbc/BBBC021/"
OUTPUT_FILE_PATH = "/data/bbbc_out/"

FEATURES_FILE = "./all_features.csv"
DATA_FILE = OUTPUT_FILE_PATH + "Data.csv"
DATA_TSNE = OUTPUT_FILE_PATH + "Data_TSNE.csv"
DATA_UMAP = OUTPUT_FILE_PATH + "Data_UMAP.csv"

def main():

    ## get valid filenames and build output filenames
    filenames, names_illum = get_valid_file_names(IMAGE_FILE_PATH) 

    ## run illumination correction on images
   # for i, directory_files in enumerate(filenames):
   #     run_illum(directory_files, names_illum[i])
   #     print("Directory processed: ", directory_files[0])
   
    all_names_montage = []
    ## montage illumed images
    for i, filename_dir in enumerate(names_illum): 
         sorted_filenames = sorted(filename_dir)
         names_montage = make_montage_names(sorted_filenames)
         all_names_montage.extend(names_montage)
   #      run_montage(sorted_filenames, names_montage)
   #      print("Directory {} out of {} processed".format(i, len(names_illum)))


     ## run features on images
    ims = map(io.imread, all_names_montage)
    output_features(ims, all_names_montage, FEATURES_FILE)
    
     ## get x y coordinates
    coords_pca = pca_transform(FEATURES_FILE)
    print("Coords PCA retrieved")
     ## generate CSV of coordinates
    generate_bokeh_csv(coords_pca, all_names_montage, DATA_FILE)

    coords_tsne = tsne_transform(FEATURES_FILE)
    print("Coords tsne retrieved")
    generate_bokeh_csv(coords_tsne, all_names_montage, DATA_TSNE)

    coords_umap = umap_transform(FEATURES_FILE)
    print("Coords umap retrieved")
    generate_bokeh_csv(coords_umap, all_names_montage, DATA_UMAP)

def make_montage_names(filenames):
    """
    Construct filenames for the montaged images

    :param filenames: names of the illumed images with separate quadrants and channels
    :return names_montage: names for each file which will result from montaging
    """
    filename_reg = r'(.*Week._.*_)(...)(_s._w..*)(_illum)(\.tif)$'
    names_montage = []
    for filename in filenames:
        match = re.search(filename_reg, filename)
        if match:
            potential_name = match.group(1) + match.group(2) + "_montaged" + match.group(5)
            if potential_name not in names_montage:
                names_montage.append(potential_name)
    return names_montage

def get_valid_file_names(filepath):
    """
    Get full filenames relative to top level directory for each file in the BBBC trial, and
    construct filenames with paths for saving output

    :param filepath: path to the directory containing folders of images
    :return (valid_filenames, illum_filenames): tuple with lists of filenames for reading and saving
                                                                    images
    """
    filename_reg = r'(^Week._.*)(_..._s._w.)(.*)(\.tif)$'

    valid_filenames = []
    illum_filenames = []

    for root, directories, filenames in os.walk(IMAGE_FILE_PATH):
        current_subdir = root
        new_subdir = []
        new_subdir_illum = []
        for filename in os.listdir(current_subdir):
            match = re.search(filename_reg, filename)
            if match:
               new_subdir.append(os.path.join(root, match.group(1) + match.group(2) + match.group(3) + match.group(4)))
               new_subdir_illum.append("".join([OUTPUT_FILE_PATH, root[51:]]) + "_" +  match.group(1) + match.group(2) + match.group(3)  + "_illum" + match.group(4))
        if len(new_subdir) != 0 and len(new_subdir_illum) != 0:
               valid_filenames.append(new_subdir)
               illum_filenames.append(new_subdir_illum)

    return (valid_filenames, illum_filenames)


def run_illum(filenames, names_out):
    """
    Find background illumination and correct all images corresponding to elements in filenames.

    Save corrected images using names_out which includes a relative path from the top level directory.

    :param filenames: list of valid filenames with relative paths from top level directory
    :param names_out: list of valid filenames for saving output with relative paths from top level directory
    """
    illum = find_background_illumination(filenames)
    corrected_images = correct_multiimage_illumination(filenames, illum=illum)
    for (image, name) in zip(corrected_images, names_out):
        io.imsave(name, image)


def run_montage(filenames, names_out):
    """
    Read images from filenames and stitch and stack their quadrants and channels before saving to new files using
    names_out

    :param filenames: list of filenames with relative paths to top level sorted by well, quadrant and channel e.g.
                        filenames = ['B02_s1_w1_illum.tif', 'B02_s1_w2_illum.tif', 'B02_s1_w4_illum.tif',
                                    'B02_s2_w1_illum.tif', 'B02_s2_w2_illum.tif', 'B02_s2_w4_illum.tif',
                                    'B02_s3_w1_illum.tif', 'B02_s3_w2_illum.tif', 'B02_s3_w4_illum.tif',
                                    'B02_s4_w1_illum.tif', 'B02_s4_w2_illum.tif', 'B02_s4_w4_illum.tif']
                        will result in one image (B02) with quadrants [[s1, s2], [s3, s4]] where each quadrant
                        is stacked in the order [w4, w2, w1]. This example assumes files at the top level directory
    :param names_out: list of filenames with relative paths to top level for output
    """
    illumed_ims = map(io.imread, filenames)
    montaged_ims = montage_stream(illumed_ims, montage_order=[[0, 1], [2, 3]], channel_order=[2, 1, 0])
    for (image, name) in zip(montaged_ims, names_out):
        io.imsave(name, image)


def output_features(ims, filenames, out_file):
    """
    Build a default feature map for each image in ims and output a dataframe of
    [filenames, features] to out_file as csv for reading in

    :param ims: opened nparray images
    :param filenames: filenames corresponding to each image in ims with relative path to top level directory
    :param out_file: name of CSV file to save dataframe, with relative path to top level directory
    """
    # generate filenames column to exist as first column of feature DF
    filenames_col = ["Filenames"]
    filenames_col.extend(filenames)
    filenames_col = pd.DataFrame(filenames_col)

    all_image_features = pd.DataFrame()
    # set up flag to only add header row once
    flag = True
    i = 1
    for im, im_name in zip(ims, filenames):
        image_features, feature_names = default_feature_map(im, sample_size=100)
        # make sure header row is added to dataframe in first iteration
        if flag:
            all_image_features = all_image_features.append(pd.DataFrame(feature_names).transpose())
            flag = False
        image_features = pd.DataFrame(image_features).transpose()
        all_image_features = all_image_features.append(image_features, ignore_index=True)
        print("Image processed: {} out of {}".format(i, len(filenames)))
        i += 1
    # concatenate filenames column to the features and save to CSV.
    all_image_features = pd.concat([filenames_col, all_image_features], axis=1)
    all_image_features.to_csv(out_file)


def pca_transform(features_filename):
    """
    Read a file of image features into dataframe and perform a 2 component PCA, returning the 2 component values
    of each image

    :param features_filename: filename of CSV containing image features
    :return coords: np array of 2 components for each image
    """
    all_image_features = pd.read_csv(features_filename)
    pca = PCA(2)
    coords = pca.fit_transform(all_image_features.iloc[1:, 2:])

    return coords

def tsne_transform(features_filename):
    """
    Read a file of image features into dataframe and perform TSNE transformation, returning the 2 component values
    of each image

    :param features_filename: filename of CSV containing image features
    :return coords: np array of 2 components for each image
    """
    all_image_features = pd.read_csv(features_filename)

    return TSNE().fit_transform(all_image_features.iloc[1:, 2:])


def umap_transform(features_filename):
    """
    Read a file of image features into dataframe and perform umap transformation, returning the 2 component values
    of each image

    :param features_filename: filename of CSV containing image features
    :return coords: np array of 2 components for each image
    """
    all_image_features = pd.read_csv(features_filename)

    return umap.UMAP().fit_transform(all_image_features.iloc[1:, 2:])

def generate_bokeh_csv(coords, names, filename):
    """
    Generate a CSV of columns
        index,info,url,x,y
    to work with Bokeh app.

    :param coords: the x,y components of each data point
    :param names: the names of the images you wish to load into bokeh, relative to the top level directory
    """
    coords_df = pd.DataFrame(coords)
    filename_reg = r'(/.*/.*/)(Week.*_...)(.*_montaged\.tif)$'
    indices = []
    info = []
    urls = []
    
    for name in names:
        match = re.search(filename_reg, name)
        if match:
            indices.append(match.group(2))
            info.append(match.group(2) + "_info")
            urls.append(match.group(2) + match.group(3))

    indices = pd.DataFrame(indices)
    info = pd.DataFrame(info)
    urls = pd.DataFrame(urls)

    coord_csv = pd.concat([indices, info, urls, coords_df], axis=1)
    coord_csv.columns = ["index", "info", "url", "x", "y"]
    coord_csv.to_csv(filename)

main()
