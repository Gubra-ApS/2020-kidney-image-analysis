###############################################################################
###############################################################################
### IMPORT LIBRARIES
import os 
import glob
import numpy as np; print("os version: {}".format(np.__version__))
import pandas as pd

import cv2
import scipy.ndimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.segmentation


def load_tif_stack(folder_input):
    file_search = sorted(glob.glob(os.path.join(folder_input, '*.tif')))    
    dummy = skimage.io.imread(file_search[0], plugin='tifffile')    
    img = np.zeros((len(file_search),) + dummy.shape, dtype=dummy.dtype)
    for z, file in enumerate(file_search):
        img[z] = skimage.io.imread(file, plugin='tifffile')

    return img


def save_tif_stack(img, folder_output):
    for z in range(img.shape[0]):
        file_save = os.path.join(folder_output, str(z).zfill(3) + '.tif')
        skimage.io.imsave(file_save, img[z], plugin='tifffile', check_contrast=False)


def create_folder(folder_path):
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    return folder_path


###############################################################################
###############################################################################
### STEP 0: LOAD DATA, CREATE OUTPUT FOLDERS, AND SET ANALYSIS PARAMETERS
print('STEP 0: LOADING DATA')

# select input folder in current directory
folder_input = 'image_kidney'

# create output folders in current directory
folder_mask_kidney = create_folder('mask_kidney')
folder_mask_glom = create_folder('mask_glom')
folder_mask_cortex = create_folder('mask_cortex')
folder_mask_medulla = create_folder('mask_medulla')

# load image stack
img = load_tif_stack(folder_input)

# image size
n_slices, n_rows, n_cols = img.shape

# voxel size along each axis of image volume (z,y,x)
voxel_size = np.array([10.0, 4.79, 4.79])

# glomeruli size parameters
rmin = 20 # glomeruli minimum radius [microns]
rmax = 50 # glomeruli maximum radius [microns]

# glom area threshold [voxels]
glom_area_min = np.pi*(rmin**2) / np.prod(voxel_size[1:])
glom_area_max = np.pi*(rmax**2) / np.prod(voxel_size[1:])

# glom volume threshold [voxels]
glom_volume_min = (4/3)*np.pi*(rmin**3) / np.prod(voxel_size)
glom_volume_max = (4/3)*np.pi*(rmax**3) / np.prod(voxel_size)


###############################################################################
###############################################################################
### STEP 1: PRE-PROCESSING
print('STEP 1: PRE-PROCESSING')

# slice-by-slice median filtering
for i in range(img.shape[0]):
    img[i] = cv2.medianBlur(img[i], 3)    

# convert to 'uint8'
scale_limit = np.percentile(img, (99.999))
img = skimage.exposure.rescale_intensity(img, in_range=(0, scale_limit), out_range='uint8').astype('uint8')


###############################################################################
###############################################################################
### STEP 2: KIDNEY SEGMENTATION
print('STEP 2: KIDNEY SEGMENTATION')

# initial kidney mask
otsu_threshold = skimage.filters.threshold_otsu(img)
mask = img > otsu_threshold

# select largest connected component (assumed to be the kidney)
label_image, n_labels = scipy.ndimage.label(mask)
labels = np.arange(1, n_labels+1)
h = scipy.ndimage.labeled_comprehension(mask, label_image, labels, np.sum, int, 0)
mask = label_image == labels[np.argwhere(h == h.max())]

# crude downsampling of kidney mask (in yx-plane) for faster morphological closing of holes in the segmentation mask
mask_down = mask[:, 0::10, 0::10]

# slice-by-slice morphological closing to repair segmentation holes
disk_radius = 50
border_width = disk_radius + 5

mask_down_close = np.zeros_like(mask_down)
for i in range(mask_down.shape[0]):
    
    # add temporary border to mask as the closing structure element is large
    mask_border = np.zeros( np.array(mask_down_close[i].shape)+2*border_width )
    mask_border[border_width:-border_width, border_width:-border_width] = mask_down[i]
    
    # do morphology in mask with added borders
    mask_border = cv2.morphologyEx(mask_border.astype('uint8'),
                                   cv2.MORPH_OPEN,
                                   skimage.morphology.disk(3))
    
    # do morphology in mask with added borders
    mask_border = cv2.morphologyEx(mask_border.astype('uint8'),
                                   cv2.MORPH_CLOSE,
                                   skimage.morphology.disk(disk_radius))
    
    # extract original size image data
    mask_down_close[i] = mask_border[border_width:-border_width, border_width:-border_width]

# isolate filled segmentation holes by XOR operation
mask_down_fill = mask_down ^ mask_down_close

# resize filled segmentation holes to full and combine with otsu segmentation
# to obtain final kidney mask
mask_fill = np.zeros(img.shape, dtype='uint8')
mask_kidn = np.zeros(img.shape, dtype='uint8')
mask_glom = np.zeros(img.shape, dtype='bool')
marker_image = np.zeros( img.shape, dtype='uint16' )

for i in range(mask_down_fill.shape[0]):    
    mask_fill[i] = cv2.resize( mask_down_fill[i].astype('uint8'), (n_cols, n_rows) ).astype('bool')    
    mask_kidn[i] = mask[i] | mask_fill[i]    
    mask_kidn[i] = cv2.morphologyEx(mask_kidn[i].astype('uint8'), cv2.MORPH_CLOSE, skimage.morphology.disk(9))

# save kidney mask as uint8 to corresponding output folder
save_tif_stack(255*(mask_kidn).astype('uint8'), folder_mask_kidney)


###############################################################################
###############################################################################
### STEP 3: GLOMERULI SEGMENTATION
print('STEP 3: GLOMERULI SEGMENTAION')

# calculate gradient image (anisotropic smoothing due to anisotropic voxel size)
img_grad = scipy.ndimage.gaussian_gradient_magnitude(img.astype('float32'), sigma=(0.5, 1.0, 1.0))

# set blob detection parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.filterByConvexity = False
params.filterByCircularity = False
params.filterByArea = True
params.minArea = glom_area_min
params.maxArea = glom_area_max
params.filterByInertia = True
params.minInertiaRatio = 0.3
params.maxInertiaRatio = 1.0
params.minThreshold = 50
params.maxThreshold = 255

# apply blob detection slice-by-slice..
detector = cv2.SimpleBlobDetector_create(params)
coords = []
count = 1
for i in range(img.shape[0]):
    keypoints = detector.detect(img[i])

    # extract blob centroids from keypoint structure and store in labelled
    # marker image for seeded watershed segmentation
    for kp in keypoints:
        pts = kp.pt
        if img[i, int(round(pts[1])), int(round(pts[0])) ] > params.minThreshold:
            coords.append(kp.pt)
            marker_image[i, int(round(pts[1])), int(round(pts[0])) ] = count
            count += 1

# perform seeded watershed
ws = skimage.segmentation.watershed(img_grad, marker_image)

# label the marker image
regions = skimage.measure.regionprops(label_image=ws)

# filter away segmentation regions that are too large and coordinates
glom_coords = []
for prop in regions:
    if (prop.area < glom_volume_max):
        count += 1
        glom_coords.append(prop.coords)

# use coords to get final segmentation
for coords in glom_coords:
    for c in coords:
        mask_glom[ c[0], c[1], c[2] ] = True 

# save glom mask as uint8 to corresponding output folder
save_tif_stack(255*(mask_glom).astype('uint8'), folder_mask_glom)


###############################################################################
###############################################################################
### STEP 4: CORTEX/MEDULLA SEGMENTATION
print('STEP 4: CORTEX/MEDULLA SEGMENTATION')

# create distance-from-kidney-surface distance map
mask_kidn_dist = scipy.ndimage.distance_transform_edt(mask_kidn, voxel_size)

# extract region properties for each segmented glom, using
# distance-to-surface map as intensity image
label_image, _ = scipy.ndimage.label(mask_glom)
rp_dist = skimage.measure.regionprops(label_image, intensity_image=mask_kidn_dist)

# extract individual glom-sizes amd glom-to-kidney-surface distances
glom_size = []
glom_dist = []
for rp_dist_ in rp_dist:
    glom_size.append(rp_dist_.area * np.prod(voxel_size))
    glom_dist.append(rp_dist_.mean_intensity)

# sort according to distance from kidney surface
glom_dist_sorted = np.sort(glom_dist)

# index to 95th glom dist percentile
idx_95pct = int( np.round( 0.95*len(glom_dist) ) )

# extract 95th percentile distance
dist_95pct = glom_dist_sorted[idx_95pct]

# create initial cortex/medulla masks
mask_cortex  = mask_kidn & (mask_kidn_dist < dist_95pct) & (mask_kidn_dist > 50)
mask_medulla = mask_kidn & (mask_kidn_dist > dist_95pct + 100)

# erode cortex/medulla masks to add margin
mask_glom_dilated = scipy.ndimage.binary_dilation(mask_glom, structure=skimage.morphology.ball(4))

# remove dilated gloms from both cortex and medulla mask
mask_cortex[ mask_glom_dilated == True ] = False
mask_medulla[ mask_glom_dilated == True ] = False

# save cortex/medulla masks as uint8 to corresponding output folders
save_tif_stack(255*(mask_cortex).astype('uint8'), folder_mask_cortex)
save_tif_stack(255*(mask_medulla).astype('uint8'), folder_mask_medulla)


###############################################################################
###############################################################################
### STEP 5: SAVE GLOM STATISTICS
print('STEP 5: SAVE GLOM STATISTICS')

# crate data frame and save as .csv in current directory
df = pd.DataFrame()
df['Glomeruli size [um3]'] = glom_size
df['Glomeruli distance to kidney surface [um]'] = glom_dist
df.to_csv('glom_data.csv', index=None)