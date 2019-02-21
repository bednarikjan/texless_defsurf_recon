# Python std.
import re
import os
import shutil
import string
import yaml
import datetime
import random

# 3rd party
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt


def ls(path, exts=None, ignore_dot_underscore=True):
    """ Lists the directory and returns it sorted. Only the files with
    extensions in `ext` are kept. The output should match the output of Linux
    command "ls". It wrapps os.listdir() which is not guaranteed to produce
    alphanumerically sorted items.

    Args:
        path (str): Absolute or relative path to list.
        exts (str or list of str or None): Extension(s). If None, files with
            any extension are listed. Each e within `exts` can (but does
            not have to) start with a '.' character. E.g. both
            '.tiff' and 'tiff' are allowed.
        ignore_dot_underscore (bool): Whether to ignore files starting with
            '._' (usually spurious files appearing after manipulating the
            linux file system using sshfs)

    Returns:
        list of str: Alphanumerically sorted list of files contained in
        directory `path` and having extension `ext`.
    """
    if isinstance(exts, str):
        exts = [exts]

    files = [f for f in sorted(os.listdir(path))]

    if exts is not None:
        # Include patterns.
        extsstr = ''
        for e in exts:
            extsstr += ('.', '')[e.startswith('.')] + '{}|'.format(e)
        patt_ext = r'.*({})$'.format(extsstr[:-1])
        re_ext = re.compile(patt_ext)

        # Exclude pattern.
        patt_excl = '^/'
        if ignore_dot_underscore:
            patt_excl = '^\._'
        re_excl = re.compile(patt_excl)

        files = [f for f in files if re_ext.match(f) and not re_excl.match(f)]

    return files


def jn(*parts):
    """ Returns the file system path composed of `parts`.

    Args:
        *parts (str): Path parts.

    Returns:
        str: Full path.
    """
    return os.path.join(*parts)


def load_img(path, dtype='float32', keep_alpha=False):
    """ Loads the image and converts it to one of following given the `dtype`:
    uint8   - pixels in [0, 255]
    float32 - pixels in [0.0, 1.0]

    Args:
        path (str): Absolute path to file.
        dtype (str): Output data type with corresponding value range.
            One of {'uint8', 'float32'}
        keep_alpha (bool): Whether to keep alpha channel. Only applies for
            RGB images and alpha is assumed to be 4th channel.

    Returns:
        np.array: RGB or GS image, possibly with 4th alpha channel (if
        the input image has it and `keep_alpha is True`). Data type and pixel
        values range is given by `dtype`.
    """
    img = plt.imread(path)

    # Check dtype of the image.
    if img.dtype not in (np.uint8, np.float32):
        raise Exception('Loaded image {p} has unsupported dtype {dt}. '
                        'load_cvt() only supports one of (uint8, float32).'.
                        format(p=path, dt=img.dtype))

    # Check dtype argument.
    if dtype not in ('uint8', 'float32'):
        raise Exception('Supported values for "dtype" are ("uint8, float32"), '
                        'got {dt}'.format(dt=dtype))

    # Keep or remove alpha channel.
    if img.ndim == 3 and img.shape[2] == 4 and not keep_alpha:
        img = img[..., :3]

    # Convert data type.
    if img.dtype == np.uint8 and dtype == 'float32':
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.float32 and dtype == 'uint8':
        img = np.round(img * 255.0).astype(np.uint8)

    return img


def load_save_conf(path, fn, fn_args={}):
    """ Loads and returns the configuration file (.yaml) and saves this
    file into the output directory, which is created using an item
     'path_train_run' within config file and `fn_trrun_name` function.

    Args:
        path (str): Absolute path to the configuration file.
        fn (callable): Function which takes a config dict as input
            and produces a name for the train run subdirectory.
        fn_args (dict): Named arguments to pass to `fn`.

    Returns:
        conf (dict): Loaded config file.
        out_path (str):
    """

    # Load conf.
    conf = load_conf(path)

    # Get train run subdir path.
    trrun_subdir = fn(conf, **fn_args)
    out_path = jn(conf['path_train_run'], trrun_subdir)

    # Create train run dir.
    if os.path.exists(out_path):
        out_path_old = out_path
        out_path = unique_dir_name(out_path)
        print('[WARNING]: The output path {} exists, creating new dir {}'.
              format(out_path_old, out_path))
    make_dir(out_path)

    # Save config.
    shutil.copy(path, jn(out_path, os.path.basename(path)))

    return conf, out_path


def load_conf(path):
    """ Returns the loaded .cfg config file.

    Args:
        path (str): Aboslute path to .cfg file.

    Returns:
    dict: Loaded config file.
    """

    with open(path, 'r') as f:
        conf = yaml.load(f)
    return conf


def make_dir(path):
    """ Creates directory `path`. If already exists, does nothing.

    Args:
        path (str): Path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def unique_dir_name(d):
    """ Checks if the `dir` already exists and if so, generates a new name
     by adding current system time as its suffix. If it is still duplicate,
     it adds a random string as a suffix and makes sure it is unique. If
     `dir` is unique already, it is not changed.

    Args:
        d (str): Absolute path to `dir`.

    Returns:
        str: Unique directory name.
    """
    unique_dir = d

    if os.path.exists(d):
        # Add time suffix.
        dir_name = add_time_suffix(d, keep_extension=False)

        # Add random string suffix until the file is unique in the folder.
        unique_dir = dir_name
        while os.path.exists(unique_dir):
            unique_dir += add_random_suffix(unique_dir, keep_extension=False)

    return unique_dir


def add_time_suffix(name, keep_extension=True):
    """ Adds the current system time suffix to the file name.
    If `keep_extension`, then the suffix is added before the extension
    (including the ".") if there is any.

    Args:
        name (str): File name.
        keep_extension (bool): Add the suffix before the extension?

    Returns:
        str: New file name.
    """

    # Get time suffix.
    time_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Generate new name.
    if keep_extension:
        n, e = split_name_ext(name)
        new_name = n + '_' + time_suffix + ('', '.{}'.format(e))[len(e) > 0]
    else:
        new_name = name + '_' + time_suffix

    return new_name


def add_random_suffix(name, length=5, keep_extension=True):
    """ Adds the random string suffix of the form '_****' to a file name,
    where * stands for an upper case ASCII letter or a digit.
    If `keep_extension`, then the suffix is added before the extension
    (including the ".") if there is any.

    Args:
        name (str): File name.
        length (int32): Length of the suffix: 1 letter for underscore,
            the rest for alphanumeric characters.
        keep_extension (bool): Add the suffix before the extension?

    Returns:
        str: New name.
    """
    # Check suffix length.
    if length < 2:
        print('[WARNING] Suffix must be at least of length 2, '
              'using "length = 2"')

    # Get random string suffix.
    s = ''.join(random.choice(string.ascii_uppercase + string.digits)
                for _ in range(length - 1))

    # Generate new name.
    if keep_extension:
        n, e = split_name_ext(name)
        new_name = n + '_' + s + ('', '.{}'.format(e))[len(e) > 0]
    else:
        new_name = name + '_' + s

    return new_name


def split_name_ext(fname):
    """ Splits the file name to its name and extension.

    Args:
        fname (str): File name without suffix (and without '.').

    Returns:
        str: Name without the extension.
        str: Extension.
    """
    parts = fname.rsplit('.', 1)
    name = parts[0]

    if len(parts) > 1:
        ext = parts[1]
    else:
        ext = ''

    return name, ext


def next_batch(dl, use_nmap, use_dmap, use_pc):
    """ Gets next batch from data loader.

    Args:
        dl (DataLoader): Instance of DataLoader class.
        use_nmap (bool): Whether normal maps are loaded.
        use_dmap (bool): Whether depth maps are loaded.
        use_pc (bool): Whether point clouds are loaded.

    Returns:
        b_img (np.array): Batch of images.
        gt (dict): GT values for namps, dmaps, pclouds.
    """
    batch = iter(next(dl))
    b_img = next(batch)
    gt = {}
    for stream, o in zip([use_nmap, use_dmap, use_pc],
                         ['out_normals', 'out_depth_maps', 'out_verts']):
        if stream:
            gt[o] = next(batch)

    return b_img, gt


def cerate_trrun_name(conf):
    """ Creates the training run name based on the configuration params.

    Args:
        conf (dict): Configuration.

    Returns:
        str: Name of the train run.
    """
    un = conf['normals_stream']
    ud = conf['depth_stream']
    up = conf['mesh_stream']

    return ('', 'N')[un] + ('', 'D')[ud] + ('', 'M')[up] + \
           ('', '_wn{:.1f}'.format(conf.get('w_normals', -1)))[un] + \
           ('', '_wd{:.1f}'.format(conf.get('w_depth', -1)))[ud] + \
           ('', '_wm{:.1f}'.format(conf.get('w_coords', -1)))[up] + \
           ('', '_k{:.1f}'.format(conf.get('kappa', -1)))[un]


def extract_epoch(path_params):
    """ Returns the epoch number extracted from the params file name.
    It is expected that the file name ends with 'epNUM.EXT', where NUM
    is arbitrary numbr of digits and EXT is one of ('pkl', 'h5').

    Args:
        path_params (str): Path to params file.

    Returns:
        int: Epoch number.
    """
    regexp = re.compile('ep(\d+)\.(pkl|h5)$')
    res = regexp.search(path_params)

    if res is None:
        raise Exception('Epoch number not found in "{}"'.
                        format(path_params))
    return int(res.group(1))


def get_conf_model_optim(path_trrun):
    """ Returns the paths to config, model params and optimizer params files
    in `path_trrun`. Files are found based on the file extension
    (config - .yaml, model - .h5, optim - .pkl).

    Args:
        path_trrun (str): Path to training run.

    Returns:
        conf (str): Path to configuration file (.yaml).
        model_params (str): Path to model params file (.h5).
        optim_params (str): Path to optimizer params file (.pkl).
    """
    conf = ls(path_trrun, exts='yaml')
    model_params = ls(path_trrun, exts='h5')
    optim_params = ls(path_trrun, exts='pkl')
    assert(len(conf) == 1 and len(model_params) == 1 and len(optim_params) == 1)
    return jn(path_trrun, conf[0]), \
           jn(path_trrun, model_params[0]), \
           jn(path_trrun, optim_params[0])


def get_mask(img, fgrd=True):
    """ Returns the binary mask where foregournd corresponds to non-zero
    pixels (at least one channel has to be non-zero). `fgrd` flag controls
    whether foreground pixels are True or False.

    Args:
        img (np.array of float): (H, W, 3)-tensor, (H, W)-image of 3D normals.
        fgrd (bool): Whether to return foreground mask.

    Returns:
        np.array of bool: (H, W)-matrix, True - foreground, False - background.
    """

    mask = np.sum(np.abs(img), axis=2).astype(np.bool)

    if not fgrd:
        mask = (1 - mask).astype(np.bool)

    return mask


def normals2depth(nmap, s=1.0, t=0.0):
    """ Computes depth map from normal map using least squares and finite
    differences in depth. Orthographic camera model is assumed, however,
    this normally works pretty well even if the data come from perspective
    (pinhole) camera model. The actual normals do not need to be rectangular,
    but the background is required to be [0, 0, 0].

    Expected coordinate frame as in OpenCV:

      _ z
      /|
     /
    +---> x (= u)
    |
    |
    v y (= v)

    Args:
        nmap (np.array): (H, W, 3)-tensor, (H, W)-image of unit-length 3D
            normals.
        s (float): Scale mapping real size to pixel size. s = dx/du = dy/dv,
            i.e. assuming orthographic projection, s tells how big one pixel is
            in real unit (e.g. meters). Since the reconstruction is up to scale
            s, if set properly, the reconstruction would best relate to real
            object.
        t (float): Translation. The reconstruction is up to translation.

    Returns:
        dmap (np.array): (H, W)-array of pixel-wise depths (depth corresponds
            to Z axis).
    """

    def extend_mask_rd(mask, iters=1):
        """ Extends the mask (foreground) one step to right and down. Example:

        0 0 0 0 0      0 0 0 0 0
        0 1 1 0 0      0 1 1 1 0
        0 1 0 0 0  ->  0 1 1 0 0
        0 1 1 1 1      0 1 1 1 1
        0 0 1 0 0      0 0 1 1 0

        Args:
            mask (np.array): (H, W)-matrix, binary array with 1 = foreground,
                0 = background
            iters (int): Number of iterations to extend the mask.

        Returns:
            np.array: (H, W)-matrix of the same type as `mask`.
        """

        kernel = np.array([[1, 1], [1, 0]], dtype=np.uint8)

        # Convolution, thus kernel needs to be flipped vertic. and horizont.
        kernel = cv2.flip(cv2.flip(kernel, 0), 1)

        m = mask.astype(np.uint8)
        m = cv2.dilate(m, kernel, iterations=iters)

        return m.astype(mask.dtype)

    # Extending the normals map by 2 pixels along right and bottom edge.
    nme = np.zeros((nmap.shape[0] + 2, nmap.shape[1] + 2, 3), dtype=nmap.dtype)
    nme[:-2, :-2] = nmap

    # Copy the normals 1 step right and down - prevent ill-posed LS problem.
    m1 = get_mask(nme, fgrd=False)
    nme[:, 1:] += nme[:, :-1] * m1[:, 1:][..., None]
    m2 = get_mask(nme, fgrd=False)
    nme[1:, :] += nme[:-1, :] * m2[1:, :][..., None]

    # Mask of non-zero normals.
    mask_n = get_mask(nme)
    # Mask of non-zero depth values (to be reconstructed).
    mask_z = extend_mask_rd(mask_n)

    # Get dimensions of the matrix A for the LS system Ax = b
    Arows = np.sum(mask_n) * 2
    Acols = np.sum(mask_z)

    ### Create sparse matrix A.
    inds_z = np.copy(mask_z).astype(np.int32)
    inds_z[inds_z == 0] = -1
    inds_z[inds_z != -1] = np.arange(Acols)

    # minus ones
    inds_z_tl = inds_z[mask_n].flatten()

    i_m_ones = np.arange(Arows)
    j_m_ones = np.stack([inds_z_tl] * 2, axis=1).flatten()
    data_m_ones = -np.ones((Arows,))

    # ones from right
    mask_ones_r = np.zeros_like(mask_n, dtype=np.bool)
    mask_ones_r[:, 1:] = mask_n[:, :-1]
    inds_z_r = inds_z[mask_ones_r].flatten()

    i_ones_r = np.arange(Arows // 2) * 2
    j_ones_r = inds_z_r
    data_ones_r = np.ones((Arows // 2,))

    # ones from down
    mask_ones_d = np.zeros_like(mask_n, dtype=np.bool)
    mask_ones_d[1:, :] = mask_n[:-1, :]
    inds_z_d = inds_z[mask_ones_d].flatten()

    i_ones_d = i_ones_r + 1
    j_ones_d = inds_z_d
    data_ones_d = np.ones((Arows // 2,))

    # Concat to get indices and data for creating sparse matrix.
    data = np.concatenate((data_m_ones, data_ones_r, data_ones_d), axis=0)
    i_inds = np.concatenate((i_m_ones, i_ones_r, i_ones_d), axis=0)
    j_inds = np.concatenate((j_m_ones, j_ones_r, j_ones_d), axis=0)

    As = sparse.coo_matrix((data, (i_inds, j_inds)), (Arows, Acols))

    ### Build dense vector b of the LS system Ax = b.
    b = -(nme[mask_n][:, :2] / nme[mask_n][:, 2][:, None]).flatten()

    # Solve for unknown x: Ax = b.
    z = spla.lsmr(As, b)[0]

    # Scale and translate z.
    z = z * s + t

    # Create depth map.
    dmap = np.zeros_like(mask_z, dtype=np.float64)
    dmap[mask_z] = z

    # Delete artificially constructed depth values on the very right and bottom.
    dmap[get_mask(nme, fgrd=False)] = 0.0

    # Get the depth map of the original normal map size.
    dmap = dmap[:-2, :-2]

    return dmap


def procrustes(x_to, x_from, scaling=False, reflection=False, gentle=True):
    """ Finds Procrustes tf of `x_form` to best match `x_to`.

    Args:
        x_to (np.array): Pcloud to which `x_from` will be aligned. Shape
            (V, 3), V is # vertices.
        x_from (np.array): Pcloud to be aligned. Shape (V, 3), V is # vertices.
        scaling (bool): Whether to use scaling.
        reflection (str): Whether to use reflection.
        gentle (bool): Whether to raise Exception when SVD fails
            (`gentle == False`) or rather to print warning and
            continue with unchanged data (`gentle` == True).

    Returns:
        np.array: Aligned pcloud, shape (V, 3).
    """

    n, m = x_to.shape
    ny, my = x_from.shape

    mu_x = x_to.mean(0)
    mu_y = x_from.mean(0)

    x0 = x_to - mu_x
    y0 = x_from - mu_y

    ss_x = (x0 ** 2.).sum()
    ss_y = (y0 ** 2.).sum()

    # Centred Frobenius norm.
    norm_x = np.sqrt(ss_x)
    norm_y = np.sqrt(ss_y)

    # Scale to equal (unit) norm.
    x0 /= norm_x
    y0 /= norm_y

    if my < m:
        y0 = np.concatenate((y0, np.zeros(n, m - my)), 0)

    # Optimum rotation matrix of Y.
    A = np.dot(x0.T, y0)

    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except:
        if gentle:
            print('WARNING: SVD failed, returning non-changed data.')
            return x_from
        else:
            raise

    V = Vt.T
    T = np.dot(V, U.T)

    # Undo unintended reflection.
    if not reflection and np.linalg.det(T) < 0:
        V[:, -1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)

    trace_TA = s.sum()

    if scaling:
        Z = norm_x * trace_TA * np.dot(y0, T) + mu_x
    else:
        Z = norm_y * np.dot(y0, T) + mu_x

    return Z


def dmap2pcloud_persp(dmap, K):
    """ Generates the point cloud from given depth map `dm_gt` using intrinsic
    camera matrix `K`.

    Args:
        dmap (np.array): Depth map, shape (H, W).
        K (np.array): Camera intrinsic matrix, shape (3, 3).

    Returns:
        np.array: Point cloud, shape (P, 3), P is # non-zero depth values.
    """

    Kinv = np.linalg.inv(K)

    y, x = np.where(dmap != 0.0)
    N = y.shape[0]
    z = dmap[y, x]

    pts_proj = np.vstack((x[None, :], y[None, :], np.ones((1, N))) * z[None, :])
    pcloud = (Kinv @ pts_proj).T

    return pcloud.astype(np.float32)


def dmap2pcloud_ortho(dmap, s):
    """ Converts the dmap to point cloud using orthographic reprojection. Only
    considers non-zero depth values.

    Args:
        dmap (np.array): Depth map, shape (H, W).
        s (float): dx/du, real world spatial size corresponding to one pixel.

    Returns:
        np.float: Pcloud, shape (P, 3), P is # non-zero depth values.
    """
    H, W = dmap.shape
    mask = get_mask(dmap)
    grid = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=2) - \
           np.array([W // 2, H // 2])[None, None, :]
    return np.concatenate([grid * s, dmap[..., None]], axis=2)[mask]


def estim_st_dmap_ortho(dmap_to, dmap_from, K):
    """ Estimates the best scale and translation of the depth map
    `dmap_from` w.r.t. `dmap_to`. It is assumed that the point cloud
    can be obtained from `dmap_to` by perspective reprojection, whereas
    the pointcloud from `dmap_from` can be obtained by orthographic
    reprojection.

    Args:
        dmap_to (np.array): Depth map to which the `dmap_from` should be
            aligned, shape (H, W).
        dmap_from (np.array): Depth map to be aligned, shape (H, W).
        K (np.array): Camera intrinsic matrix, shape (3, 3).

    Returns:
        s (float): Optimal scale.
        t (float): Optimal translation.
    """
    mask = get_mask(dmap_to)
    dmap_from *= mask
    num_d = np.sum(mask)

    pc_persp = dmap2pcloud_persp(dmap_to, K)
    pc_ortho = dmap2pcloud_ortho(dmap_from, 1.)

    A = np.stack([pc_ortho.flatten(),
                  np.tile(np.array([0., 0., 1.]), num_d)], axis=1)
    b = pc_persp.flatten()
    s, t = np.linalg.solve(A.T @ A, A.T @ b)
    return s, t


def estim_st_dmap_persp(dmap_to, dmap_from):
    """ Estimates the best scale and translation of the depth map
    `dmap_from` w.r.t. `dmap_to`. It is assumed that pointclouds
    from both depth maps can be obtained by perspective reprojection.

    Args:
        dmap_to (np.array): Depth map to which the `dmap_from` should be
            aligned, shape (H, W).
        dmap_from (np.array): Depth map to be aligned, shape (H, W).

    Returns:
        s (float): Optimal scale.
        t (float): Optimal translation.
    """
    mask = get_mask(dmap_to)
    num_d = np.sum(mask)
    A = np.stack([dmap_from[mask], np.ones(num_d, )], axis=1)
    b = dmap_to[mask]
    s, t = np.linalg.solve(A.T @ A, A.T @ b)
    return s, t
