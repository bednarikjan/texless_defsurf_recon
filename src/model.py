# 3rd party.
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np


class MaxPoolingWithArgmax2D(Layer):
    """ Pooling 2D with storing pooling indices. Adapts the original
    implementation by [1].

    [1] https://github.com/ykamikawa/SegNet

    Args:
        pool_size (tuple): Pooling windows size.
        strides (tuple): Pooling window strides.
        padding (str): Type of paddding, one of {'same', 'valid'}.
    """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same',
                 **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides

        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = K.tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=strides, padding=padding)

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None
                        else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    """ Upooling 2D with pooling indices. Adapts the original
    implementation by [1].

    [1] https://github.com/ykamikawa/SegNet

    Args:
            size (tuple): Unpooling size.
    """
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')

            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0],
                                input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]],
                                        axis=0)
            batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'),
                                    shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]),
                                            [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], \
               mask_shape[2] * self.size[1], mask_shape[3]


class SegNetMultiTask:
    """ Architecture based on SegNet [1] with original encoder and up to three
    decoders producing normal maps (DNM), depth maps (DDM) and vertices (DV).
    The DNM and DDM are symmetric to the encoder since they produce the
    tensors of the same spatial size as the input. The DV produces flattened
    array of vertex coordinates.

    [1] V Badrinarayanan et. al. SegNet: A Deep Convolutional Encoder-Decoder
    Architecture for Image Segmentation. TPAMI 2017.

    Args:
        input_shape (tuple of int): Input image size of shape (H, W, C).
        normals (bool): Whether to include output normals stream.
        depth (bool): Whether to include output depth maps stream.
        verts (bool): Whether to include output vertices stream.
        kernel (int): Conv kernel size.
        pool_size (tuple): Pooling window size.
        blocks (tuple): # of conv-BN-relu blocks at each stage.
        filters (tuple): # of conv filters used at each stage.
        mesh_verts (int): # of mesh vertices.
        name (str): Name of the keras.Model.
    """

    def __init__(self, input_shape=(224, 224, 3),
                 normals=True, depth=True, verts=True,
                 kernel=3, pool_size=(2, 2),
                 blocks=(2, 2, 3, 3, 3), filters=(32, 64, 128, 256, 256),
                 mesh_verts=961, name=''):
        self._input_shape = input_shape
        self._kernel = kernel
        self._pool_size = pool_size
        self._blocks = blocks
        self._filters = filters
        self._mesh_verts = mesh_verts
        self._name = name

        if len(blocks) != len(filters):
            raise Exception('"blocks" and "filters" must have the same number '
                            'of items, found {} != {}'.
                            format(len(blocks), len(filters)))

        self.model = self._build(normals, depth, verts)

    def _build(self, normals, depth, verts):
        """ Builds the model.

        Args:
            normals (bool): Whether to include output normals stream.
            depth (bool): Whether to include output depth maps stream.
            verts (bool): Whether to include output vertices stream.

        Returns:
            keras.Model: Model.
        """
        x = Input(shape=self._input_shape)
        latent, masks = self._encoder(x)
        outputs = []

        if normals:
            out_n = self._decoder_fc(latent, masks, 3, name_conv='', name_bn='',
                                     name_out='out_normals')
            outputs.append(out_n)
        if depth:
            out_d = self._decoder_fc(latent, masks, 1, name_conv='conv_{}_d',
                                     name_bn='bn_{}_d',
                                     name_out='out_depth_maps')
            outputs.append(out_d)
        if verts:
            out_v = self._decoder_dense(latent, self._mesh_verts * 3,
                                        name_suff='_v', name_out='out_verts')
            outputs.append(out_v)

        return Model(inputs=[x], outputs=outputs, name=self._name)

    @staticmethod
    def conv_bn_act(x, filts, k, padding='same', act='relu',
                     name_conv='', name_bn=''):
        """ A block involving conv2D, batch normalization and activation
        function.

        Args:
            x (tf.Tensor): Input tensor.
            filts (int): # of filters.
            k (int): Conv kernel size.
            padding (str): Padding.
            act (str): Activation funtion.
            name_conv (str): Name of conv layer.
            name_bn (str): Name of BN layer.

        Returns:
            tf.Tensor: Resulting tensor.
        """
        x = Convolution2D(filts, (k, k), padding=padding, name=name_conv)(x)
        x = BatchNormalization(name=name_bn)(x)
        x = Activation(act)(x)
        return x

    def _encoder(self, x):
        """ Builds the encoder.

        Args:
            x (tf.Tensor): Input.

        Returns:
            tf.Tensor: Latent representation.
        """
        masks = []
        for num_blocks, num_filts in zip(self._blocks, self._filters):
            for bi in range(num_blocks):
                x = self.conv_bn_act(x, num_filts, self._kernel)
            x, mask = MaxPoolingWithArgmax2D(self._pool_size)(x)
            masks.append(mask)

        return x, masks

    def _decoder_fc(self, x, masks, out_channels, name_conv='', name_bn='',
                    name_out=''):
        """ Decoder using fully convolutional layers (symmetric to `_encoder`).

        Args:
            x (tf.Tensor): Input, latent representation.
            masks (tf.Tensor): Pooling indices.
            out_channels (int): # of channels of output tensor.
            name_conv (str): Conv layer name template with one parameter, which
                will be filled with an iteration number. e.g. 'conv_{}'.
            name_bn (str): BN layer name template with one parameter, which
                will be filled with an iteration number, e.g. 'bn_{}'
            name_out (str): Name of the output layer.

        Returns:
            tf.Tensor
        """
        blocks = self._blocks[::-1]
        filters = self._filters[::-1] + (out_channels, )
        masks = masks[::-1]
        stages = len(blocks)

        iters = 1
        for si in range(stages):
            x = MaxUnpooling2D(self._pool_size)([x, masks[si]])

            for bi in range(blocks[si]):
                num_filts = filters[(si, si + 1)[bi == blocks[si] - 1]]
                last_layer = (si == stages - 1 and bi == blocks[si] - 1)
                k = (self._kernel, 1)[last_layer]
                act = ('relu', 'linear')[last_layer]
                x = self.conv_bn_act(
                    x, num_filts, k, act=act, name_conv=name_conv.format(iters),
                    name_bn=name_bn.format(iters))
                iters += 1
        return Activation('linear', name=name_out)(x)

    def _decoder_dense(self, x, num_out, name_suff='', name_out=''):
        """ Decoder with one conv2d layer, avg pooling and a fully connected
        layer.

        Args:
            x (tf.Tensor): Input, latent representation.
            num_out (int): # of output values.
            name_suff (str): Name suffix of layers.
            name_out (str): Name of the output layer.

        Returns:
            tf.Tensor, shape (B, `num_out`).
        """
        # Produces (N, 3, 3, 64)
        x = Convolution2D(64, (1, 1), padding='same',
                          name='conv_1_{}'.format(name_suff))(x)

        x = AveragePooling2D((3, 3), padding='same')(x)
        x = Flatten()(x)
        x = Dense(num_out, name='dense_1_{}'.format(name_suff))(x)
        return Activation('linear', name=name_out)(x)

    @staticmethod
    def loss_nmap(kappa):
        """ Returns the normal maps loss function. """

        def loss(nmap_gt, nmap_pred):
            """ Returns the TF computation graph for masked per-pixel angular
            error loss function defined as loss = kappa * loss_ang + loss_len,
            where

            loss_ang = 1/N * \sum_{i}{\acos(ngt_i * np_i /
                       || ngt_i ||*|| np_i ||) / \pi},
            loss_len = 1/N * \sum_{i}{(|| np_i ||^2 - 1)^2},

            where ngt_i, np_i is i-th GT and predicted normal vector
            respectively and kappa sets the relative influcne of the terms.

            Args:
                nmap_gt (tf.Tensor): GT normal maps, shape (B, H, W, 3).
                nmap_pred (tr.Tensor): Pred. normal maps, shape (B, H, W, 3).
                kappa (float): Mixing constant.

            Returns:
                tf.Tensor: Loss, scalar.
            """
            # Constants
            EPS = tf.constant(1e-6, tf.float32)
            PIREC = tf.constant(1.0 / np.pi, tf.float32)

            # Get masks.
            masks_fgrd = tf.reduce_mean(tf.abs(nmap_gt), axis=3)
            masks_fgrd = tf.cast(masks_fgrd, tf.bool)

            # Select data within foreground mask (normals).
            nm_gt_fgrd = tf.boolean_mask(nmap_gt, masks_fgrd)
            mp_pred_fgrd = tf.boolean_mask(nmap_pred, masks_fgrd)

            # Get normals' lengths.
            norms_len_gt = tf.norm(nm_gt_fgrd, axis=1)
            norms_len_pred = tf.norm(mp_pred_fgrd, axis=1)

            # Compute loss for predicted normals lengths (how they differ from 1).
            with tf.name_scope('L_norms_len'):
                llen = tf.reduce_mean(
                    tf.square(tf.constant(1.0, tf.float32) - norms_len_pred),
                    name='loss_norms_len')

            # Compute angular distance.
            with tf.name_scope('L_norms_and_dist'):
                lang = tf.reduce_mean(
                    tf.acos(tf.reduce_sum(nm_gt_fgrd * mp_pred_fgrd, axis=1) /
                            (norms_len_gt * norms_len_pred + EPS)) * PIREC,
                    name='loss_norms_ang_dist')

            # Final loss
            loss = kappa * lang + llen
            loss = tf.reshape(loss, [1])
            loss.set_shape(tf.TensorShape([1]))

            return loss
        return loss

    @staticmethod
    def loss_dmap():
        """ Returns the depth maps loss function. """

        def loss(dmap_gt, dmap_pred):
            """ Returns the TF computation graph for masked per-pixel absolute depth
            error defined as

            loss = 1/N * \sum_{i}{|dgt_i - dp_i|}

            Args:
                dmap_gt (tf.Tensor): GT depth maps, shape (B, H, W, 1).
                dmap_pred (tf.Tensor): Predicted depth maps, shape (B, H, W, 1).

            Returns:
                tf.Tensor: Loss, scalar.
            """
            # Get masks.
            masks_fgrd = tf.cast(dmap_gt, tf.bool)

            # Select data within foreground mask.
            dm_gt_fgrd = tf.boolean_mask(dmap_gt, masks_fgrd)
            dm_pred_fgrd = tf.boolean_mask(dmap_pred, masks_fgrd)

            # Compute abs L1 distance.
            loss = tf.reduce_mean(tf.abs(dm_gt_fgrd - dm_pred_fgrd))
            loss = tf.reshape(loss, [1])
            loss.set_shape(tf.TensorShape([1]))

            return loss
        return loss

    @staticmethod
    def loss_pcloud():
        """ Returns the point cloud loss function. """

        def loss(pc_gt, pc_pred):
            """ Returns the TF computation graph for mean per-point L2 error.

            Args:
                pc_gt (tf.Tensor): GT pcloud, shape (B, V * 3).
                pc_pred (tf.Tensor): Pred pcloud, shape (B, V * 3).

            Returns:
                tf.Tensor: Loss, scalar.
            """
            sh = tf.shape(pc_gt)
            N = sh[0]
            V = sh[1] // 3

            pc_gt = tf.reshape(pc_gt, [N, V, 3])
            pc_pred = tf.reshape(pc_pred, [N, V, 3])

            loss = tf.reduce_mean(tf.norm(pc_gt - pc_pred, axis=2))
            loss = tf.reshape(loss, [1])
            loss.set_shape(tf.TensorShape([1]))

            return loss
        return loss
