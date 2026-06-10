# requires four python packages to read directly from the zip file (inspired in neurofinder)
#
# - numpy
# - scipy
# - matplotlib


from zipfile import ZipFile
from PIL import Image  # $ pip install pillow
from scipy.misc import imread
import numpy as np
import caiman as cm
import json
import pylab as plt
# %% extract array from set of zipped images
def get_movie_from_zip(zip_filename, start, end):
    with ZipFile(zip_filename) as archive:
        for idx, entry in enumerate(archive.infolist()):
            if start <= idx < end:
                with archive.open(entry) as file:
                    if idx == start:
                        img = imread(file)
                        mov = np.zeros([end-start, *img.shape], dtype=np.float32)
                        mov[idx-start] = img
                    else:
                        mov[idx-start] = np.array(Image.open(file))

                    if idx % 100 == 0:
                        print(idx)
    return mov
# %% create datasets in tiff format that are at most 3000 frames long (example for N.02.00
zip_filename = './N.02.00/images/images.zip' # for other files change path accordingly
max_size_tiff = 3000
base_names = zip_filename[:-4]
with ZipFile(zip_filename) as archive:
    num_frames = len(archive.infolist())

for selec in np.arange(0, num_frames, max_size_tiff):
    m = get_movie_from_zip(zip_filename, selec, np.minimum(num_frames, selec + max_size_tiff))
    print('SAVING .... ' + base_names + '_' + str(selec) + '.tif')
    cm.movie(m).save(base_names + '_' + str(selec) + '.tif')

dims = m.shape[1:]
#%%
# load the regions (training data only)
regions_filename = './N.02.00/regions/consensus_regions.json' # for other files change path accordingly
with open(regions_filename) as f:
    regions = json.load(f)

def tomask(coords):
    mask = np.zeros(dims)
    for coor in coords:
        mask[coor[0], coor[1]] = 1
    return mask

masks = np.array([tomask(s['coordinates']) for s in regions])

# show the outputs
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(m.sum(axis=0), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(masks.sum(axis=0), cmap='gray')
plt.show()


