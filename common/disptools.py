#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def pred_visualization(fname, arrays, picks, img_shape, tile_spacing=(0,0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """Used for visualization of predictions
    Args:
        fname: filename for saving the image
        arrays: list of arrays containing the frames, first array is assumed to be
            ground truth (all of shape Nxnframesxframesize**2)
        picks: list containing indices of cases that should be used
        img_shape: shape of a frame
        tile_spacing: spacing between the tiles
        scale_rows_to_unit_interval: see tile_raster_images
        output_pixel_vals: see tile_raster_images
    """
    ncases = len(picks)
    narrays = len(arrays)
    if narrays > 1:
        horizon = arrays[1].shape[1]
        horizon_gt = arrays[0].shape[1]
        n_presteps = horizon_gt - horizon
        if n_presteps > 0:
            visdata = np.ones((ncases, horizon_gt * narrays, np.prod(img_shape)))
            visdata[:,:horizon_gt] = arrays[0][picks]
            for i in range(1, narrays):
                visdata[:, i*horizon_gt:(i+1)*horizon_gt] = \
                    np.hstack((
                        (np.ones((ncases, n_presteps, np.prod(img_shape)))),
                         arrays[i][picks]))

        else:
            visdata = np.hstack([arrays[i][picks] for i in range(narrays)])
    else:
        horizon = arrays[0].shape[1]
        horizon_gt = horizon
        visdata = np.hstack([arrays[i][picks] for i in range(narrays)])

    visdata = visdata.reshape(ncases*narrays*horizon_gt,-1)

    im = tile_raster_images(visdata, img_shape, (ncases*narrays, horizon_gt),
                            tile_spacing,
                            scale_rows_to_unit_interval, output_pixel_vals)
    for i in range(len(picks)*len(arrays)):
        #insert white patches for n_presteps
        for j in range(horizon_gt-horizon):
            if i % len(arrays) != 0:
                im[i*img_shape[0] + i*tile_spacing[0]:(i+1)*img_shape[0] + i*tile_spacing[0],
                   j*img_shape[1] + j*tile_spacing[1]:(j+1)*img_shape[1] + j*tile_spacing[1]] = 255


    #np.insert(im, [i * len(arrays) * img_shape[0] + i * (len(arrays)-1) * tile_spacing[0] for i in range(len(picks))], 0)


    h,w = im.shape

    fig = plt.figure(frameon=False)
    #fig.set_size_inches(1,h/np.float(w))
    fig.set_size_inches(w/24.,h/24.)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()

    fig.add_axes(ax)
    ax.imshow(im, aspect='normal', interpolation='nearest')
    fig.savefig(fname, dpi=24)
    return im

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing = (0, 0),
              scale_rows_to_unit_interval = True, output_pixel_vals = True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0, 1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0, 0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            X[tile_row * tile_shape[1] +
                              tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] +
                                     tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * c
        return out_array

def dispims_white(invwhitening, M, height, width, border=0, bordercolor=0.0,
                  layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    M = np.dot(invwhitening, M)
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 = int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * np.ones(((height+border)*n0+border,
                                (width+border)*n1+border), dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] =\
                        np.vstack((
                            np.hstack((
                                np.reshape(M[:, i*n1+j],
                                           (height, width)),
                                bordercolor*np.ones((height, border),
                                                    dtype=float))),
                            bordercolor*np.ones((border, width+border),
                                                dtype=float)))
    plt.imshow(im, cmap=matplotlib.cm.gray, interpolation='nearest', **kwargs)

def CreateMovie(filename, plotter, numberOfFrames, fps):
    for i in range(numberOfFrames):
        plotter(i)
        fname = '_tmp%05d.png' % i
        plt.savefig(fname)
        plt.clf()
    #os.system("rm %s.mp4" % filename)
    #os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png "+filename+".mp4")
    os.system("convert -delay 20 -loop 0 _tmp*.png " +filename+".gif")
    os.system("rm _tmp*.png")


def dispimsmovie_patchwise(filename, M, inv, patchsize, fps=5, *args,
                           **kwargs):
    numframes = M.shape[0] / inv.shape[1]
    n = M.shape[0]/numframes

    def plotter(i):
        M_ = M[i*n:n*(i+1)]
        M_ = np.dot(inv,M_)
        width = int(np.ceil(np.sqrt(M.shape[1])))
        image = tile_raster_images(
            M_.T, img_shape=(patchsize,patchsize),
            tile_shape=(10,10), tile_spacing = (1,1),
            scale_rows_to_unit_interval = True, output_pixel_vals = True)
        plt.imshow(image,cmap=matplotlib.cm.gray,interpolation='nearest')
        plt.axis('off')

    CreateMovie(filename, plotter, numframes, fps)


def dispimsmovie(filename, W, filters, nframes, fps=5):
    patchsize = np.uint8(np.sqrt(W.shape[0]))
    def plotter(i):
        dispims_white(W, filters[i*W.shape[1]:(i+1)*W.shape[1], :], patchsize,
                      patchsize, 1, bordercolor=filters.mean(),
                      vmin=filters.min(), vmax=filters.max()*0.8)
        plt.axis('off')
    CreateMovie(filename, plotter, nframes, fps)
    
    
def visualizefacenet(fname, imgs, patches_left, patches_right,
                     true_label, predicted_label):
    """Builds a plot of facenet with attention per RNN step and
    classification result
    """
    nsamples = imgs.shape[0]
    nsteps = patches_left.shape[1]
    is_correct = true_label == predicted_label
    w = nsteps + 2 + (nsteps % 2)
    h = nsamples * 2
    plt.clf()
    plt.gray()
    for i in range(nsamples):
        plt.subplot(nsamples, w//2, i*w//2 + 1)
        plt.imshow(imgs[i])
        msg = ('Prediction: ' + predicted_label[i] + ' TrueLabel: ' +
               true_label[i])
        if is_correct[i]:
            plt.title(msg,color='green')
        else:
            plt.title(msg,color='red')
        plt.axis('off')
        for j in range(nsteps):
            plt.subplot(h, w, i*2*w + 2 + 1 + j)
            plt.imshow(patches_left[i, j])
            plt.axis('off')
            plt.subplot(h, w, i*2*w + 2 + 1 + j + w)
            plt.imshow(patches_right[i, j])
            plt.axis('off')
    plt.show()
    plt.savefig(fname)

if __name__ == '__main__':
    
    from scipy.misc import lena
    imgs = lena()[None, ...].repeat(3, axis=0)
    patches_left = lena()[None, None, :256].repeat(3, axis=0).repeat(5, axis=1)
    patches_right = lena()[None, None, 256:].repeat(3, axis=0).repeat(5, axis=1)
    true_label = np.array(['angry', 'angry', 'sad'])
    predicted_label = np.array(['sad'] * 3)
    visualizefacenet('lena.pdf', imgs, patches_left, patches_right,
                     true_label, predicted_label)
    


# vim: set ts=4 sw=4 sts=4 expandtab:
