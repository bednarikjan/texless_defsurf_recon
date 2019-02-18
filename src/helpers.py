# Python std.
import re
import os
import shutil
import string
import yaml
import datetime
import random

# 3rd party
import numpy as np
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
