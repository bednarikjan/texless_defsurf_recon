# Project files.
import helpers as hlp

# Python std.
import pickle
import os


class TrainStateSaver:
    """ Callback to save the train state:
        - 'model_params.h5': Model params.
        - 'optim_params.pkl': Optimizer config and params.

    Args:
        path_dir (str): Path to dir where the training state will be saved.
        model (keras.Model): Model. If None, model params will not be saved.
        optimizer (keras.Optimizer): Optimizer. If None, optimizer state will
            not be saved.
        verbose (bool): Whether to print info messages.
    """
    def __init__(self, path_dir, model=None, optimizer=None, verbose=True):
        self._path_dir = path_dir
        self._model = model
        self._opt = optimizer
        self._verbose = verbose

        # Current file names.
        self._ms_path = None
        self._os_path = None

    def __call__(self, epoch):
        """ Saves the training state.

        Args:
            epoch (int): Epoch number.
        """
        if self._model:
            ms_path_new = \
                hlp.jn(self._path_dir, 'model_params_ep{}.h5'.format(epoch))
            self.save_model_params(self._model, ms_path_new)

            if self._verbose:
                print('Saved model params in {}'.format(ms_path_new))

            if self._ms_path and self._ms_path != ms_path_new:
                os.remove(self._ms_path)
            self._ms_path = ms_path_new

        if self._opt:
            os_path_new = \
                hlp.jn(self._path_dir, 'optim_params_ep{}.pkl'.format(epoch))
            self.save_optim_params(self._opt, os_path_new)

            if self._verbose:
                print('Saved optimizer params in {}'.format(os_path_new))

            if self._os_path:
                os.remove(self._os_path)
            self._os_path = os_path_new

    @staticmethod
    def save_model_params(model, path_params):
        model.save_weights(path_params)

    @staticmethod
    def save_optim_params(optim, path_params):
        opt_state = {'config': optim.get_config(),
                     'params': optim.get_weights()}
        with open(path_params, 'wb') as f:
            pickle.dump(opt_state, f)
