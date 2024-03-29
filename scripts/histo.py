import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os.path import join
from os import makedirs, walk
from torch.utils.data import DataLoader, Dataset


START, END = 260, 670
THRESH = 300000
VTHRESH = 25000
PAD = 0

EXPECTED_HEIGHT = 37
MERGED_ROW_TOLERANCE = 10

def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

    
def split_erroneously_merged_rows(z, p):
    n_z, n_p = [], []
    for _z, _p in zip(z, p):
        #print(_p, _z)
        #if _z > (EXPECTED_HEIGHT * 2) - MERGED_ROW_TOLERANCE and (_z % EXPECTED_HEIGHT < MERGED_ROW_TOLERANCE or _z % EXPECTED_HEIGHT > EXPECTED_HEIGHT - MERGED_ROW_TOLERANCE):
        if _z > EXPECTED_HEIGHT + MERGED_ROW_TOLERANCE:
            #if _z % EXPECTED_HEIGHT < MERGED_ROW_TOLERANCE:
            factor = _z // EXPECTED_HEIGHT
            if _z % EXPECTED_HEIGHT > EXPECTED_HEIGHT - MERGED_ROW_TOLERANCE:
                #factor = (_z // EXPECTED_HEIGHT) + 1
                factor += 1
            #print("\tfound row to be split", _p, _z, 'factor:', factor)
            #print(factor)
            for i in range(factor):
                height = _z // factor
                start = _p + (i * height)
                #print(_p, start, _z, height)
                n_z.append(1)
                n_z.append(height)
                n_z.append(1)
                n_p.append(start - 1)
                n_p.append(start)
                n_p.append(start + height)
        else:
            n_z.append(_z)
            n_p.append(_p)
    return n_z, n_p


def process(img_path, out_dir, save_thresh = None):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img_gray.shape)
    img_bin = img_gray
    img_bin[img_bin < 150] = 0
    img_bin[img_bin >= 150] = 255


    #plt.hist(img_gray.flatten(), bins = 'auto')
    #plt.show()

    # NOTE: names should be somewhere in here
    img_bin_piece = img_bin[:, START:END]
    #cv2.imwrite('ex_binarized.jpg', binarized)

    #vdist = np.sum(binarized, axis = 1)
    hdist = np.sum(img_bin_piece, axis = 0)

    boundaries = hdist > THRESH
    z, p, v = rle(boundaries)
    b_z, b_p = None, None
    dist = float('inf')
    for i in zip(p, z, v):
        #print(dist)
        if i[2]:
            if abs(i[1] - 400) < dist:
                dist = abs(i[1] - 400)
                b_z = i[1]
                b_p = i[0]
        #print(i)

    #assert b_z > 360, 'ERROR: name column too small at ' + str(b_z)
    if b_z is None or b_z < 360:
        return
    name = img[:, START + b_p - PAD : START + b_p + b_z + PAD]
    name_bin = img_bin[:, START + b_p - PAD : START + b_p + b_z + PAD]

    vdist = np.sum(name_bin, axis = 1)
    #vdist = vdist[:1000] / 100000


    #x_data = np.array([i for i in range(len(vdist))])
    #params, params_covariance = optimize.curve_fit(test_func, x_data, vdist, p0 = [2, 2])
    #print(params)
    prefix = img_path
    prefix = prefix[prefix.rfind('/') + 1:prefix.rfind('.')]

    boundaries = vdist > VTHRESH
    z, p, v = rle(boundaries)
    z, p = split_erroneously_merged_rows(z, p)
    count = 0
    for i, (_p, _z) in enumerate(zip(p, z)):
        #print(_p, _z)
        if _p > 200 and _z >= 25:
            #print(i, _p, _z, p[i - 1], p[i + 1])
            start = p[i - 1] - 2
            try:
                end = p[i + 1] + z[i + 1] + 20
            except:
                end = len(name)

            count += 1
            
    if count == 50:
        try:
            makedirs(join(out_dir, prefix))
        except:
            #print('out dir already exists!')
            pass

        count = 0
        for i, (_p, _z) in enumerate(zip(p, z)):
            #print(_p, _z)
            if _p > 200 and _z >= 25:
                #print(i, _p, _z, p[i - 1], p[i + 1])
                start = p[i - 1] - 2
                try:
                    end = p[i + 1] + z[i + 1] + 20
                except:
                    end = len(name)

                snippet = name[start:end, :]
                out_path = join(out_dir, prefix, prefix + '_' + str(count) + '.jpg')
                #print('out path', out_path)
                if len(snippet) > 70:
                    print('WARNING: dims look weird', out_path, snippet.shape, start, end, _p, p[i - 1], p[i + 1], _z, z[i + 1])
                #print('outputting to', out_path)
                cv2.imwrite(out_path, snippet)
                count += 1
    else:
        count = None

    return count
    
