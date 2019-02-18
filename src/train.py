# Python std.
import argparse
import pickle
from timeit import default_timer as timer

# Project files.
import helpers as hlp
import helpers
import callbacks
from model import SegNetMultiTask as SNMT
from data_loader import DatasetImgNDM, DataLoader, TfReshape

# 3rd party
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Parse arguments.
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    '--config', help='Path to configuration file.')
group.add_argument(
    '--cont', help='Path to train run directory in which the training will be '
                   'continued.')
parser.add_argument(
    '--model_state', help='Path to model weights to load.')
parser.add_argument(
    '--optim_state', help='Path to optimizer config and params to load.')
args = parser.parse_args()

# Load the config file, prepare paths.
if args.cont:
    path_conf, path_mparams, path_oparams = \
        helpers.get_conf_model_optim(args.cont)
    conf = hlp.load_conf(path_conf)
    path_trrun = args.cont
    args.model_state = path_mparams
    args.optim_state = path_oparams
else:
    conf, path_trrun = \
        hlp.load_save_conf(args.config, fn=helpers.cerate_trrun_name)

use_nmap = conf['normals_stream']
use_dmap = conf['depth_stream']
use_pc = conf['mesh_stream']

# Set TF session.
if conf['flag_use_gpu_fraction']:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=conf['gpu_fraction'])
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

# Create model.
snmt = SNMT(conf['input_shape'], normals=use_nmap, depth=use_dmap, verts=use_pc,
            mesh_verts=conf['mesh_verts'], name='sn_multi')

# Prepare optimizer, possibly load train state for model and optimizer.
ls = {}
lsw = {}
if use_nmap:
    ls['out_normals'] = SNMT.loss_nmap(conf['kappa'])
    lsw['out_normals'] = conf['w_normals']
if use_dmap:
    ls['out_depth_maps'] = SNMT.loss_dmap()
    lsw['out_depth_maps'] = conf['w_depth']
if use_pc:
    ls['out_verts'] = SNMT.loss_pcloud()
    lsw['out_verts'] = conf['w_coords']

if args.model_state:
    snmt.model.load_weights(args.model_state, by_name=True)

if args.optim_state:
    with open(args.optim_state, 'rb') as f:
        opt_state = pickle.load(f)
    optimizer = Adam.from_config(opt_state['config'])
    snmt.model.compile(optimizer, loss=ls, loss_weights=lsw)
    snmt.model._make_train_function()
    optimizer.set_weights(opt_state['params'])
    ep_start = helpers.extract_epoch(args.model_state)
else:
    optimizer = Adam(lr=conf['lr'])
    snmt.model.compile(optimizer, loss=ls, loss_weights=lsw)
    ep_start = 1

# LR scheduler, early stopping.
redlr = None
earlstop = None
if conf['red_lr_plateau']:
    redlr = ReduceLROnPlateau(
        monitor='loss_va', factor=conf['red_lr_factor'],
        patience=conf['red_lr_patience'], verbose=True, mode='min',
        min_delta=conf['red_lr_eps'], min_lr=conf['red_lr_min_lr'])
    redlr.model = snmt.model
if conf['early_stopping']:
    earlstop = EarlyStopping(
        monitor='loss_va', min_delta=conf['early_stopping_eps'],
        patience=conf['early_stopping_patience'], verbose=True, mode='min')
    earlstop.model = snmt.model
    earlstop.on_train_begin()

# Prepare savers.
saver = callbacks.TrainStateSaver(path_trrun, model=snmt.model,
                                  optimizer=optimizer, verbose=True)

# Create data loaders.
prn = (None, conf['path_normals'])[conf['normals_stream']]
prd = (None, conf['path_dmaps'])[conf['depth_stream']]
prm = (None, conf['path_meshes'])[conf['mesh_stream']]
tf_dm = TfReshape(tuple(conf['input_shape'][:2]) + (1, ))

ds_tr = DatasetImgNDM(conf['path_imgs'], conf['seqs_tr'], path_root_normals=prn,
                      path_root_dmaps=prd, path_root_meshes=prm, tf_dmaps=tf_dm)
ds_va = DatasetImgNDM(conf['path_imgs'], conf['seqs_va'], path_root_normals=prn,
                      path_root_dmaps=prd, path_root_meshes=prm, tf_dmaps=tf_dm)
dl_tr = DataLoader(ds_tr, batch_size=conf['batch_size'], shuffle=True,
                   num_workers=30)
dl_va = DataLoader(ds_va, batch_size=conf['batch_size'], shuffle=False,
                   num_workers=5)

# Training loop.
ep_start = helpers.extract_epoch(args.model_state) if args.cont else 1
for ep in range(ep_start, conf['epochs'] + 1):
    tStart = timer()
    # Training
    for it in range(len(dl_tr)):
        # Get next batch.
        b_img, gt = helpers.next_batch(dl_tr, use_nmap, use_dmap, use_pc)

        # Train on one batch.
        outp = snmt.model.train_on_batch(b_img, gt)
        outp = outp if isinstance(outp, list) else [outp]
        losses = iter([float(v) for v in outp])
        loss_tr = next(losses)
        print('\rEp {}, it {}/{}: loss_tr: {:.4f}, t: {:0.2f} s'.
              format(ep, it + 1, len(dl_tr), loss_tr, timer() - tStart), end='')

    # Validation
    loss_va_run = 0.
    for it in range(len(dl_va)):
        # Get next batch.
        b_img, gt = helpers.next_batch(dl_va, use_nmap, use_dmap, use_pc)

        outp = snmt.model.evaluate(b_img, gt, verbose=0)
        outp = outp if isinstance(outp, list) else [outp]
        losses = iter([float(v) for v in outp])
        loss_va_run += next(losses) * b_img.shape[0]
    loss_va = loss_va_run / len(ds_va)
    print(' loss_va: {:.4f}'.format(loss_va))

    # Save training state.
    if ep % conf['period_save_weights'] == 0:
        saver(ep)

    # LR scheduler.
    if redlr:
        redlr.on_epoch_end(ep, logs={'loss_va': loss_va})

    # Early stopping.
    if earlstop:
        earlstop.on_epoch_end(ep, logs={'loss_va': loss_va})
        if hasattr(snmt.model, 'stop_training') and snmt.model.stop_training:
            break
saver(ep)
