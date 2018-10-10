import pandas as pd
import os
import re

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

from microscopium import io
from microscopium.preprocess import montage_stream
from microscopium.preprocess import correct_multiimage_illumination
from microscopium.preprocess import find_background_illumination
from microscopium.features import default_feature_map

IMAGE_FILE_PATH = "/data/BBBC021/data.broadinstitute.org/bbbc/BBBC021/"
OUTPUT_FILE_PATH = "/data/bbbc_out/"

FEATURES_FILE = "./all_features.csv"
DATA_FILE = OUTPUT_FILE_PATH + "Data_scaled.csv"
DATA_TSNE = OUTPUT_FILE_PATH + "Data_TSNE_scaled.csv"
DATA_UMAP = OUTPUT_FILE_PATH + "Data_UMAP_scaled.csv"


def main():
    ## get valid filenames and build output filenames
    filenames, names_illum = get_valid_file_names(IMAGE_FILE_PATH)

    ## run illumination correction on images
    for i, directory_files in enumerate(filenames):
        run_illum(directory_files, names_illum[i])
        print("Directory processed: ", directory_files[0])

    all_names_montage = []
    ## montage illumed images
    for i, filename_dir in enumerate(names_illum):
        sorted_filenames = sorted(filename_dir)
        names_montage = make_montage_names(sorted_filenames)
        all_names_montage.extend(names_montage)
        run_montage(sorted_filenames, names_montage)

    ## run features on images
    # ims = map(io.imread, all_names_montage)
    # output_features(ims, all_names_montage, FEATURES_FILE)


    ## scale features
    feature_df = pd.read_csv(FEATURES_FILE)
    tidy_df = feature_df[list(feature_df)[2:]]
    pd.set_option("display.max_columns", None)
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(tidy_df.values))
    scaled_df.columns = list(feature_df)[2:]
    scaled_df["Filenames"] = feature_df["Filenames"]

    col_names = list(scaled_df)[:-1]
    scaled_vals = scaled_df[col_names]

    ## get x y coordinates
    coords_pca = PCA(2).fit_transform(scaled_vals)
    generate_bokeh_csv(coords_pca, all_names_montage, DATA_FILE)

    coords_tsne = TSNE().fit_transform(scaled_vals)
    generate_bokeh_csv(coords_tsne, all_names_montage, DATA_TSNE)

    coords_umap = umap.UMAP().fit_transform(scaled_vals)
    generate_bokeh_csv(coords_umap, all_names_montage, DATA_UMAP)


def make_montage_names(filenames):
    """Construct filenames for the montaged images

    Parameters
    ----------
    filenames: python string list
        List of names with file path for each illumed quadrant and channel

    Returns
    -------
    names_montage: python string list
        List of names for each image once montaged - 12 filenames make 1 montaged name
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
    """Get full filenames relative to top level directory for each file in the BBBC trial, and
    construct filenames with paths for saving output

    Parameters
    ----------
    filepath: string
        path to the directory containing folders of images

    Returns
    -------
    valid_filenames: python string list
        All the valid images names in the directory and subdirectories found at filepath
    illum_filenames: python string list
        One name for each valid filename with output path for saving illumed images
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
                new_subdir_illum.append(
                    "".join([OUTPUT_FILE_PATH, root[51:]]) + "_" + match.group(1) + match.group(2) + match.group(
                        3) + "_illum" + match.group(4))
        if len(new_subdir) != 0 and len(new_subdir_illum) != 0:
            valid_filenames.append(new_subdir)
            illum_filenames.append(new_subdir_illum)

    return (valid_filenames, illum_filenames)


def run_illum(filenames, names_out):
    """Find background illumination and correct all images corresponding to elements in filenames.
    Save corrected images using names_out which includes a relative output path - OUTPUT_FILE_PATH.

    Parameters
    ----------
    filenames: python string list
        All valid filenames of images to be read and illuminated, with path relative to script
    names_out: python string list
        Names under which to save illumed images, with output file path prepended

    Returns
    -------
    None
    """
    illum = find_background_illumination(filenames)
    corrected_images = correct_multiimage_illumination(filenames, illum=illum)
    for (image, name) in zip(corrected_images, names_out):
        io.imsave(name, image)


def run_montage(filenames, names_out):
    """Read images from filenames and stitch and stack their quadrants and channels before saving to new files using
    names_out

    Parameters
    ----------
    filenames: python string list
        list of filenames with relative paths to top level sorted by well, quadrant and channel e.g.
            filenames = ['B02_s1_w1_illum.tif', 'B02_s1_w2_illum.tif', 'B02_s1_w4_illum.tif',
                        'B02_s2_w1_illum.tif', 'B02_s2_w2_illum.tif', 'B02_s2_w4_illum.tif',
                        'B02_s3_w1_illum.tif', 'B02_s3_w2_illum.tif', 'B02_s3_w4_illum.tif',
                        'B02_s4_w1_illum.tif', 'B02_s4_w2_illum.tif', 'B02_s4_w4_illum.tif']
            will result in one image (B02) with quadrants [[s1, s2], [s3, s4]] where each quadrant
            is stacked in the order [w4, w2, w1]. This example assumes files at top level directory
    names_out: python string list
        list of names to use for saving montaged images, prepended with directory path

    Returns
    -------
    None
    """
    illumed_ims = map(io.imread, filenames)
    montaged_ims = montage_stream(illumed_ims, montage_order=[[0, 1], [2, 3]], channel_order=[2, 1, 0])
    for (image, name) in zip(montaged_ims, names_out):
        io.imsave(name, image)


def output_features(ims, filenames, out_file):
    """Build a default feature map for each image in ims and output a dataframe of
    [filenames, features] to out_file as csv for reading in


    Parameters
    ----------
    ims: 3D np.ndarray of float or uint8.
        the input images
    filenames: python string list
        filenames corresponding to each image in ims with relative path to top level directory
    out_file: string
        name of CSV file to save dataframe, with relative path to top level directory

    Returns
    -------
    None
    """
    # generate filenames column to exist as first column of feature DF
    filenames_col = pd.DataFrame(filenames)
    filenames_col.columns = ['Filenames']

    all_image_features = []
    feature_names = []
    for im, im_name in zip(ims, filenames):
        image_features, feature_names = default_feature_map(im, sample_size=100)
        all_image_features.append(image_features)

    # convert to dataframe and assign column names
    all_image_features = pd.DataFrame(all_image_features)
    all_image_features.columns = feature_names

    # concatenate filenames column to the features and save to CSV.
    all_image_features = pd.concat([filenames_col, all_image_features], axis=1)
    all_image_features.to_csv(out_file)


def generate_bokeh_csv(coords, names, filename):
    """Generate a CSV of columns
        index,info,url,x,y
    to work with Bokeh app.

    Parameters
    ----------
    coords: arraylike shape (num files, 2)
        (x, y) coordinates retrieved from dimension reduction
    names: python string lists
        Filenames corresponding to the images which created the coordinates
    filename: string
        Output filename prepended with filepath - should be same directory as images

    Returns
    -------
    None
    """
    coords_df = pd.DataFrame(coords)
    compounds_df = pd.read_csv("BBBC021_v1_image.csv")
    moa_df = pd.read_csv("BBBC021_v1_moa.csv")

    filename_reg = r'(/.*/.*/)(Week.*)(_Week.*_)(...)(.*_montaged\.tif)$'
    indices = []
    info = []
    urls = []

    for name in names:
        match = re.search(filename_reg, name)
        if match:
            indices.append(match.group(2) + "_" + match.group(4))
            image_info = get_info(match, compounds_df, moa_df)
            info.append(image_info)
            urls.append(match.group(2) + match.group(3) + match.group(4) + match.group(5))

    indices = pd.DataFrame(indices)
    info = pd.DataFrame(info)
    urls = pd.DataFrame(urls)

    coord_csv = pd.concat([indices, info, urls, coords_df], axis=1)
    coord_csv.columns = ["index", "info", "url", "x", "y"]
    coord_csv.to_csv(filename)


def get_info(reg_match, compounds_df, moa_df):
    """Retrieve correct compound, concentration and moa info from the dataframes based on the information in the
    regular expression match

    Parameters
    ----------
    reg_match: re match
        The match object retrieved from reg_exing the filename of the image
    compounds_df: pd dataframe
        The crossreference table matching filenames to compounds and concentrations
    moa_df: pd dataframe
        The crossreference table matching compounds and concentrations to moa

    Returns
    -------
    info: string
        Compound, concentration and moa (if available) information about the given matched string.
        In the form
        "compound_concentration_moa"
        OR
        "compound_concentration_NONE"

    """
    valid_row = compounds_df.loc[(compounds_df['Image_Metadata_Plate_DAPI'] == reg_match.group(2)) & (
        compounds_df['Image_Metadata_Well_DAPI'] == reg_match.group(4))]
    valid_compound = valid_row.iloc[0]['Image_Metadata_Compound']
    valid_concentration = valid_row.iloc[0]['Image_Metadata_Concentration']

    valid_moa = moa_df.loc[(valid_compound == moa_df['compound']) & (valid_concentration == moa_df['concentration'])]
    info = valid_compound + "_" + str(valid_concentration) + "_"
    if len(valid_moa) != 0:
        info += valid_moa.iloc[0]['moa']
    else:
        info += "NONE"
    return info


main()
