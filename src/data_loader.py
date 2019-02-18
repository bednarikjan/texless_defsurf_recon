# Python std.
import abc
from abc import abstractmethod
import threading
import multiprocessing
import time
try:
    import queue
except ImportError:
    import Queue as queue

# Project files.
import helpers as hlp

# 3rd party.
import numpy as np


class TfReshape:
    """ Reshapes the data.

    Args:
        shape (tuple): Output shape.
    """
    def __init__(self, shape):
        self._shape = shape

    def __call__(self, sample):
        return sample.reshape(self._shape)


class Dataset(abc.ABC):
    """An abstract Dataset class. Derived datasets should subclass it and
    override `__len__`, which returns # samples, and `__getitem__`, which
    returns a sample at index idx, where idx in [0, len(self)].
    """
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def get_file_list(path_root, seqs, exts=None):
        """ Gets the list of paths to files contained in dirs listed in `seqs`.
        The paths are relative w.r.t. the `path_root`.

        Args:
            path_root (str): Path to root dir containing sequences (directories)
                of data samples.
            seqs (lsit of str): Names of sequences (dirs) to load.
            exts (list of str): Supported file name extensions.

        Returns:
            list of str: List of paths to files.
        """
        return [hlp.jn(s, f) for s in seqs
                for f in hlp.ls(hlp.jn(path_root, s), exts=exts)]


class DatasetImg(Dataset):
    """ Dataset of images of shape (H, W, C), C is # channels.

    Args:
        path_root_imgs (str): Path to root dir containing seqs of images.
        seqs (lsit of str): Names of sequences to load.
        exts (list of str): Supported file name extensions.
    """

    def __init__(self, path_root_imgs, seqs, exts=('tiff',)):
        self._path_root_imgs = path_root_imgs
        self._files = self.get_file_list(path_root_imgs, seqs, exts=exts)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        return hlp.load_img(hlp.jn(self._path_root_imgs, self._files[idx]))


class DatasetNpz(Dataset):
    """ Dataset of data in .npz files.

    Args:
        path_root (str): Path to root dir of seqs of .npz files.
        seqs (lsit of str): Names of sequences to load.
        key (str): Data field of data in .npz files.
        dtype (str): Output data type.
        transform (callable): Transformation of a sample.
    """

    def __init__(self, path_root, seqs, key, dtype='float32', transform=None):
        self._path_root = path_root
        self._files = self.get_file_list(path_root, seqs, exts=('npz',))
        self._key = key
        self._dtype = dtype
        self._transform = transform

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        d = np.load(hlp.jn(self._path_root, self._files[idx]))[self._key].\
            astype(self._dtype)

        if self._transform is not None:
            d = self._transform(d)

        return d


class DatasetImgNDM(Dataset):
    """ Dataset generating tuples of:
        - img
        - normal map [optional]
        - depth map [optional]
        - mesh [optional]
    Specifying None for `dirs_[normals|dmaps|meshes]` means disregarding that
    dataset.

    Args:
        path_root_imgs (str): Path to root dir containing seqs of imgs.
        seqs (list of str): Names of sequences to load.
        path_root_normals (str): Path to root dir containing seqs of nmaps.
        path_root_dmaps (str): Path to root dir containing seqs of dmaps.
        path_root_meshes (str): Path to root dir containing seqs of meshes.
        tf_normals (callable): Transformation.
        tf_dmaps (callable): Transformation.
        tf_meshes (callable): Transformation.
    """
    def __init__(self, path_root_imgs, seqs, path_root_normals=None,
                 path_root_dmaps=None, path_root_meshes=None,
                 tf_normals=None, tf_dmaps=None, tf_meshes=None):

        self._ds_img = DatasetImg(path_root_imgs, seqs)
        self._ds_ndm = []
        for path_root, key, transf in \
                zip([path_root_normals, path_root_dmaps, path_root_meshes],
                    ['normals', 'depth', 'mesh'],
                    [tf_normals, tf_dmaps, tf_meshes]):
            if path_root:
                self._ds_ndm.append(
                    DatasetNpz(path_root, seqs, key, transform=transf))

        # Check # of samples consistency.
        for ds in self._ds_ndm:
            if len(self._ds_img) != len(ds):
                raise Exception(
                    'Number of samples mismatch in images dataset and {} '
                    'dataset, {} vs {}'.format(ds.__class__, len(self._ds_img),
                                               len(ds)))

    def __len__(self):
        return len(self._ds_img)

    def __getitem__(self, idx):
        return (self._ds_img[idx], ) + tuple(ds[idx] for ds in self._ds_ndm)


class _DataLoaderIter:
    """ Index iterator used by `DataLoader`. It produces batches of indices
    to the arrays of data files. Supports multi-process access.

    Args:
        dataset (Dataset): Instance of Dataset class.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data before each epoch.
    """
    def __init__(self, dataset, batch_size, shuffle):
        self._ds = dataset
        self._lock = threading.Lock()
        self._inds_gen = self._index_generator(len(dataset), batch_size, shuffle)

    @staticmethod
    def _index_generator(N, bs, shuffle):
        """ Generator of the batches of indices.

        Args:
            N (int): Total # of samples in the dataset.
            bs (int): Batch size.
            shuffle (bool): Whether to shuffle the data before each epoch.

        Returns:
            generator: Produces np.array of inds of shape (`bs`, ) or
            (N % `bs`, ).
        """
        batch_idx = 0

        while 1:
            if batch_idx == 0:
                indices = np.arange(N)
                if shuffle:
                    indices = np.random.permutation(N)

            curr_idx = (batch_idx * bs) % N
            reset = (N >= curr_idx + bs)
            curr_bs = (N - curr_idx, bs)[reset]
            batch_idx = (0, batch_idx + 1)[reset]

            yield indices[curr_idx:curr_idx + curr_bs]

    def __next__(self):
        # Get batch of indices.
        with self._lock:
            inds = next(self._inds_gen)

        # Load data samples.
        samples = [self._ds[idx] for idx in inds]
        if not isinstance(samples[0], tuple):
            samples = [(s, ) for s in samples]

        # Form a batch.
        return tuple(np.stack(ds, axis=0) for ds in list(zip(*samples)))


class DataLoader:
    """ Continuous multiprocess loader of the data batches.

    Args:
        dataset (Dataset): Instance of Dataset class.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data after each epoch.
        num_workers (int): # of processes/threads to load data.
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=1):
        data_iter = _DataLoaderIter(dataset, batch_size, shuffle)
        self._que, self._stop, self._gen_threads = \
            self.gen_que(data_iter, max_size=20 * batch_size,
                         num_workers=num_workers)
        self._len = int(np.ceil(len(dataset) / float(batch_size)))

    def __next__(self):
        return self.load_batch(self._que, self._stop)

    def __len__(self):
        """
        Returns:
            int: # of batches within dataset.
        """
        return self._len

    @staticmethod
    def gen_que(generator, max_size=10, wait_t=0.05, num_workers=1,
                pickle_safe=False):
        """ Builds a queue out of a data generator. If pickle_safe, use a
        multiprocessing approach. Else, use threading.

        Args:
            generator (Generator): Produces data sample upon call to next().
            max_size (int): Max que size.
            wait_t (float): Sleep between data loading iters.
            num_workers (int): Number of processes.
            pickle_safe (bool):
        """
        gen_threads = []
        if pickle_safe:
            q = multiprocessing.Queue(maxsize=max_size)
            _stop = multiprocessing.Event()
        else:
            q = queue.Queue()
            _stop = threading.Event()

        try:
            def datagen_task():
                while not _stop.is_set():
                    try:
                        if pickle_safe or q.qsize() < max_size:
                            gen_output = next(generator)
                            q.put(gen_output)
                        else:
                            time.sleep(wait_t)
                    except Exception:
                        _stop.set()
                        raise

            for i in range(num_workers):
                if pickle_safe:
                    # Reset random seed else all children processes share the
                    # same seed.
                    np.random.seed()
                    thread = multiprocessing.Process(target=datagen_task)
                else:
                    thread = threading.Thread(target=datagen_task)
                gen_threads.append(thread)
                thread.daemon = True
                thread.start()
        except:
            _stop.set()
            if pickle_safe:
                # Terminate all daemon processes.
                for p in gen_threads:
                    if p.is_alive():
                        p.terminate()
                q.close()
            raise

        return q, _stop, gen_threads

    @staticmethod
    def load_batch(que, stop):
        """ Gets next data batch.

        Returns:
            tuple: Loaded batch.
        """
        while not stop.is_set():
            if not que.empty():
                gen_output = que.get()
                break
            else:
                time.sleep(0.01)

        if not hasattr(gen_output, '__len__'):
            stop.set()
            raise ValueError('Unexpected output of generator queue, '
                             'found: ' + str(gen_output))
        return gen_output


################################################################################
### Tests
if __name__ == '__main__':
    import jblib.unit_test as jbut
    import matplotlib.pyplot as plt

    ############################################################################
    jbut.next_test('DatasetImg()')

    path_root_imgs = '/cvlabdata1/cvlab/datasets_jan/surfrec/synthetic/real_white_objects/cloth_square/images'
    seqs = ['L0_bottom_edge_a', 'L0_bottom_edge_b', 'L0_bottom_edge_tl_corn']
    idx = 123

    num_files = len([f for s in seqs
                     for f in hlp.ls(hlp.jn(path_root_imgs, s))])

    ds_img = DatasetImg(path_root_imgs, seqs)
    assert(num_files == len(ds_img))

    img = ds_img[idx]
    assert(img.shape == (224, 224, 3))
    assert(img.dtype == np.float32)
    assert(np.min(img) >= 0.0 and np.max(img) <= 1.0)

    img2 = (plt.imread(hlp.jn(
        path_root_imgs, seqs[0],
        hlp.ls(hlp.jn(path_root_imgs, seqs[0]))[idx]))).\
               astype(np.float32) / 255.

    assert(np.allclose(img, img2))

    ############################################################################
    jbut.next_test('DatasetNpz()')

    path_root_normals = '/cvlabdata1/cvlab/datasets_jan/surfrec/synthetic/real_white_objects/cloth_square/normals/normals_224x224_ocv'
    path_root_dmaps = '/cvlabdata1/cvlab/datasets_jan/surfrec/synthetic/real_white_objects/cloth_square/depth_maps/depth_224x224_ocv'
    path_root_meshes = '/cvlabdata1/cvlab/datasets_jan/surfrec/synthetic/real_white_objects/cloth_square/coords'
    dtype = np.float32
    idx = 123
    new_shape = (961, 3)

    ds_n = DatasetNpz(path_root_normals, seqs, key='normals', dtype=dtype)
    ds_d = DatasetNpz(path_root_dmaps, seqs, key='depth', dtype=dtype)
    ds_m = DatasetNpz(path_root_meshes, seqs, key='mesh', dtype=dtype,
                      transform=TfReshape(new_shape))

    assert(len(ds_n) == len(ds_d) and len(ds_d) == len(ds_m))

    nmap = ds_n[idx]
    assert(nmap.shape == (224, 224, 3))
    assert(nmap.dtype == np.float32)

    dmap = ds_d[idx]
    assert (dmap.shape == (224, 224))
    assert (dmap.dtype == np.float32)

    mesh = ds_m[idx]
    assert (mesh.shape == (961, 3))
    assert (mesh.dtype == np.float32)

    def load_sample_manually(path_root, seq, idx, key):
        return np.load(hlp.jn(
            path_root, seq, hlp.ls(hlp.jn(path_root,seq))[idx]))[key]

    nmap2 = load_sample_manually(path_root_normals, seqs[0], idx, 'normals')
    assert(np.allclose(nmap, nmap2))
    dmap2 = load_sample_manually(path_root_dmaps, seqs[0], idx, 'depth')
    assert (np.allclose(dmap, dmap2))
    mesh2 = load_sample_manually(path_root_meshes, seqs[0], idx, 'mesh').\
        reshape(new_shape)
    assert (np.allclose(mesh, mesh2))

    ############################################################################
    jbut.next_test('DatasetImgNDM()')

    ds_im_nm = DatasetImgNDM(path_root_imgs, seqs,
                             path_root_normals=path_root_normals,
                             path_root_meshes=path_root_meshes,
                             tf_meshes=TfReshape(new_shape))

    img, nmap, mesh = ds_im_nm[idx]
    assert(np.allclose(img, img2))
    assert(np.allclose(nmap, nmap2))
    assert(np.allclose(mesh, mesh2))

    ############################################################################
    jbut.next_test('DataLoader()')
    from timeit import default_timer as timer

    bs = 32
    shuffle = True
    num_workers = 10

    # Try with 1 worker and small ds.
    ds_im_ndm = DatasetImgNDM(path_root_imgs, seqs[0:1],
                              path_root_normals=path_root_normals,
                              path_root_dmaps=path_root_dmaps,
                              path_root_meshes=path_root_meshes,
                              tf_meshes=TfReshape(new_shape))
    dl_im_ndm = DataLoader(ds_im_ndm, bs, shuffle=shuffle, num_workers=1)

    num_samples = len(ds_im_ndm)
    num_iters = len(dl_im_ndm)
    assert(num_iters == int(np.ceil(num_samples / bs)))

    print('# samples in ds: {}'.format(num_samples))
    print('# iters = ceil(# samples / bs) = ceil({} / {}) = {}'.
          format(num_samples, bs, num_iters))

    tstart = timer()
    samples_seen = 0
    for i in range(num_iters):
        print('\rProcessing batch {}/{}'.format(i + 1, num_iters), end='')
        b_img, b_nmap, b_dmap, b_mesh = next(dl_im_ndm)

        curr_bs = b_mesh.shape[0]
        assert(curr_bs == bs or curr_bs == num_samples % bs)
        
        assert(b_img.shape == (curr_bs, 224, 224, 3))
        assert(b_nmap.shape == (curr_bs, 224, 224, 3))
        assert(b_dmap.shape == (curr_bs, 224, 224))
        assert(b_mesh.shape == (curr_bs, ) + new_shape)

        samples_seen += curr_bs
    print()
    assert(samples_seen == len(ds_im_ndm))

    tdur = timer() - tstart
    print('# workers: 1, processing speed: {:.1f} batches/s'.
          format(num_iters / tdur))

    # Try with multiple workers and bigger ds.
    ds_im_ndm = DatasetImgNDM(path_root_imgs, seqs,
                              path_root_normals=path_root_normals,
                              path_root_dmaps=path_root_dmaps,
                              path_root_meshes=path_root_meshes,
                              tf_meshes=TfReshape(new_shape))
    dl_im_ndm = DataLoader(ds_im_ndm, bs, shuffle=shuffle,
                           num_workers=num_workers)

    num_samples = len(ds_im_ndm)
    num_iters = len(dl_im_ndm)
    assert (num_iters == int(np.ceil(num_samples / bs)))

    tstart = timer()
    for i in range(num_iters):
        print('\rProcessing batch {}/{}'.format(i + 1, num_iters), end='')
        b_img, b_nmap, b_dmap, b_mesh = next(dl_im_ndm)

        curr_bs = b_mesh.shape[0]
        assert (curr_bs == bs or curr_bs == num_samples % bs)

        assert (b_img.shape == (curr_bs, 224, 224, 3))
        assert (b_nmap.shape == (curr_bs, 224, 224, 3))
        assert (b_dmap.shape == (curr_bs, 224, 224))
        assert (b_mesh.shape == (curr_bs,) + new_shape)
    print()

    tdur = timer() - tstart
    print('# workers: {}, processing speed: {:.1f} batches/s'.
          format(num_workers, num_iters / tdur))

