# Modified by:
# -----------------------------------------------------------------------------
# Copyright (c) 2016 Ryutaro Yamauchi, Tsuguo Mogami
# Copyright (c) 2016 Albert, Inc.
# -----------------------------------------------------------------------------

# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

import math
import numpy as np
import six

from chainer import cuda
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


class SpatialConvolution2D(link.Link):

    """
    Spatial Convolution Layer.
    """

    def __init__(self, in_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(SpatialConvolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)

        # For backward compatibility
        self.initialW = initialW
        self.wscale = wscale

        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, scale=math.sqrt(wscale))

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initilizer = initializers._get_initializer(initial_bias)
            self.add_param('b', in_channels, initializer=bias_initilizer)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (in_channels, kh, kw)
        self.add_param('W', W_shape, initializer=self._W_initializer)

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])
        return spatial_convolution_2d(
            x, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class SpatialConvolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, cover_all=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 3,
            x_type.shape[1] == w_type.shape[0],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def conv_cpu(self, W, img, kh, kw,
                 sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1):
        n, c, h, w = img.shape
        dkh, dkw = kh + (kh - 1) * (dy - 1), kw + (kw - 1) * (dx - 1)
        out_h = conv.get_conv_outsize(h, kh, sy, ph, cover_all, dy)
        assert out_h > 0, 'Height in the output should be positive.'
        out_w = conv.get_conv_outsize(w, kw, sx, pw, cover_all, dx)
        assert out_w > 0, 'Width in the output should be positive.'

        img = np.pad(img,
                     ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                     mode='constant', constant_values=(pval,))

        y = np.zeros((n, c, out_h, out_w), dtype=img.dtype)
        for j in six.moves.range(0, dkh, dy):
            j_lim = j + sy * out_h
            for i in six.moves.range(0, dkw, dx):
                i_lim = i + sx * out_w
                y += W[np.newaxis, :, j // dy, i // dx][:, :, None, None]\
                    * img[:, :, j:j_lim:sy, i:i_lim:sx]
        return y

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw = W.shape[1:3]

        y = self.conv_cpu(W, x,
                          kh, kw, self.sy, self.sx, self.ph, self.pw,
                          cover_all=self.cover_all)

        if b is not None:
            y += b[:, None, None]
        return y,

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        out_c, kh, kw = W.shape
        n, c, h, w = x.shape

        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
                                      cover_all=self.cover_all)
        assert out_h > 0, 'Height in the output should be positive.'
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
                                      cover_all=self.cover_all)
        assert out_w > 0, 'Width in the output should be positive.'

        y = cuda.cupy.zeros((n, out_c, out_h, out_w), dtype=x.dtype)
        cuda.elementwise(
            'raw T x, raw T W, int32 h, int32 w, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
            'int32 ch',
            'T y',
            '''
                int c0 = i / (out_h * out_w);
                int out_y = i / out_w % out_h;
                int out_x = i % out_w;
                for (int j = 0; j < kh; j++) {
                    for (int k = 0; k < kw; k++) {
                        int in_y = out_y * sy - ph + j;
                        int in_x = out_x * sx - pw + k;
                        if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
                            y += x[in_x + w * (in_y + h * c0)] * \
                            W[(k + kw * (j + kh * c0)) % (kw * kh * ch)];
                        }
                    }
                }
            ''',
            'conv')(x.reduced_view(), W.reduced_view(),
                    h, w, out_h, out_w, kh, kw, self.sy, self.sx,
                    self.ph, self.pw, out_c, y)

        if b is not None:
            y += b[:, None, None]

        return y,

    def deconv_cpu(self, W, img, gy, sy, sx, ph, pw, h, w, dy=1, dx=1):
        n, c, out_h, out_w = gy.shape
        _, kh, kw = W.shape
        pval = 0
        dkh, dkw = kh + (kh - 1) * (dy - 1), kw + (kw - 1) * (dx - 1)

        gx = np.zeros_like(img)
        img = np.pad(img,
                     ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                     mode='constant', constant_values=(pval,))
        gx = np.pad(gx,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
        gW = np.zeros_like(W)

        for j in six.moves.range(0, dkh, dy):
            j_lim = j + sy * out_h
            for i in six.moves.range(0, dkw, dx):
                i_lim = i + sx * out_w
                gx[:, :, j:j_lim:sy, i:i_lim:sx] += \
                    W[np.newaxis, :, j // dy, i // dx][:, :, None, None]\
                    * gy[:, :, ::dy, ::dx]
                gW[np.newaxis, :, j // dy, i // dx][:, :, None, None] += \
                    np.sum(gy[:, :, ::dy, ::dx] *
                           img[:, :, j:j_lim:sy, i:i_lim:sx], axis=(0, 2, 3),
                           keepdims=True)

        return gx[:, :, ph:ph+h, pw:pw+w], gW

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]
        _b, _n, _h, _w = gy.shape

        gx, gW = self.deconv_cpu(W, x, gy,
                                 self.sy, self.sx, self.ph, self.pw, h, w)
        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        n, c, h, w = x.shape
        kh, kw = W.shape[1:3]
        _b, _n, _h, _w = gy.shape
        gy = gy.reshape(_b, _n, 1, 1, _h, _w)

        col = conv.im2col_gpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)

        gW = cuda.cupy.sum(gy * col, (0, 4, 5), keepdims=True)

        gcol = gy * W[np.newaxis, :, :, :, None, None]
        gx = conv.col2im_gpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        if b is not None:
            gb = gy.sum(axis=(0, 2, 3, 4, 5))

        if b is None:
            return gx, gW[0, :, :, :, 0, 0]
        else:
            return gx, gW[0, :, :, :, 0, 0], gb


def spatial_convolution_2d(x, W, b=None, stride=1, pad=0,
                   cover_all=False):
    func = SpatialConvolution2DFunction(stride, pad, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
