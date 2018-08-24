import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from general_conv import special_conv
import time

class FLKConv(gluon.nn.HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1,1), dilation=(1,1),
                 padding=(0,0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(FLKConv, self).__init__(prefix=prefix, params=params)
        self._channels = channels
        self._in_channels = in_channels
        self._kwargs = {
            'kernel': kernel_size, 'kernel_max': kernel_max, 'stride': strides,
            'dilate': dilation, 'pad': padding, 'num_filter': channels, 'num_group':groups,
            'no_bias': not use_bias, 'layout': layout
        }
        self.kernel_mask = kernel_mask
        wshape = (channels, in_channels, kernel_max)
        self.weight = self.params.get('weight', shape=wshape, init=weight_initializer,
                                      allow_deferred_init=True)
        if use_bias:
            self.bias = self.params.get('bias', shape=(channels,),
                                        init=bias_initializer, allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.contrib.FixLengthKernelConvolution(x,self.kernel_mask,weight,name='fwd', **self._kwargs)
        else:
            out = F.contrib.FixLengthKernelConvolution(x,self.kernel_mask,weight,bias,name='fwd', **self._kwargs)
        return out
    
    def set_mask(self, kernel_mask):
        self.kernel_mask = kernel_mask.copy()


class MyConv(gluon.nn.HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1, 1), dilation=(1, 1),
                 padding=(0, 0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None,
                 bias_initializer='zeros',
                 prefix=None, params=None):
        super(MyConv, self).__init__(prefix=prefix, params=params)
        self._channels = channels
        self._in_channels = in_channels
        self._kwargs = {
            'kernel': kernel_size, 'kernel_max': kernel_max, 'stride': strides,
            'dilate': dilation, 'pad': padding, 'num_filter': channels, 'num_group': groups,
            'no_bias': not use_bias, 'layout': layout
        }
        self.kernel_mask = kernel_mask
        wshape = (channels, in_channels, kernel_max)
        self.weight = self.params.get('weight', shape=wshape, init=weight_initializer,
                                      allow_deferred_init=True)
        if use_bias:
            self.bias = self.params.get('bias', shape=(channels,),
                                        init=bias_initializer, allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.contrib.MyConvolution(x, self.kernel_mask, weight, name='fwd', **self._kwargs)
        else:
            out = F.contrib.MyConvolution(x, self.kernel_mask, weight, bias, name='fwd', **self._kwargs)
        return out

    def set_mask(self, kernel_mask):
        self.kernel_mask = kernel_mask.copy()

class Conv(gluon.nn.HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, strides=(1, 1), dilation=(1, 1),
                 padding=(0, 0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None,
                 bias_initializer='zeros',
                 prefix=None, params=None):
        super(Conv, self).__init__(prefix=prefix, params=params)
        self._channels = channels
        self._in_channels = in_channels
        self._kwargs = {
            'kernel': kernel_size, 'stride': strides,
            'dilate': dilation, 'pad': padding, 'num_filter': channels, 'num_group': groups,
            'no_bias': not use_bias, 'layout': layout
        }
        wshape = (channels, in_channels, kernel_size[0], kernel_size[1])
        self.weight = self.params.get('weight', shape=wshape, init=weight_initializer,
                                      allow_deferred_init=True)
        if use_bias:
            self.bias = self.params.get('bias', shape=(channels,),
                                        init=bias_initializer, allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.Convolution(x, weight, name='fwd', **self._kwargs)
        else:
            out = F.Convolution(x, weight, bias, name='fwd', **self._kwargs)
        return out

def get_random_mask(Cin, Cout, kernel, kernel_max):
    '''
    :param Cin: int number of input channel
    :param Cout: int number of output channel
    :param kernel: tuple, kernel size, (kh, kw)
    :param kernel_max: int number of kept kernel size
    :return: NDArray, mask array in shape of [Cout, Cin, Kernel_max]
    '''
    return nd.random.uniform(0,kernel[0]*kernel[1],(Cin,Cout,kernel_max)).astype('int').astype('float32')

def FLK_im2col(inpt_im, # (Cin,H,W)
  kernel_mask, # (Cout,Cin,k_max)
  kernel_size=(3,3),pad=(0,0), stride=(1,1),dilate=(1,1), ctx=mx.gpu()):
    dshape = inpt_im.shape
    kshape = kernel_mask.shape
    height_col = dshape[1]+2*pad[0]-(dilate[0] * (kernel_size[0] - 1) + 1) / stride[0] + 1
    width_col = dshape[2]+2*pad[1]-(dilate[1] * (kernel_size[1] - 1) + 1) / stride[1] + 1
    height = dshape[1]
    width  = dshape[2]
    channel_in = kshape[1]
    kernel_max = kshape[2]
    col_data = nd.zeros((kshape[0]*kshape[1]*kshape[2]*height_col*width_col))
    kernel_mask = kernel_mask.reshape((-1,))
    data_im = inpt_im.reshape((-1,))
    for index in range(kshape[0]*kshape[1]*height_col*width_col):
        w_col = index % width_col
        h_col = (index / width_col) % height_col
        c_im = ((index / width_col) / height_col) % channel_in
        c_col_out = ((index / width_col) / height_col) / channel_in
        h_offset = h_col * stride[0] - pad[0]
        w_offset = w_col * stride[1] - pad[1]
        data_col_index = ((c_col_out * channel_in + c_im) * kernel_max * height_col + h_col) * width_col + w_col
        data_im_index = (c_im * height + h_offset) * width + w_offset
        kernel_mask_index = (c_col_out * channel_in + c_im)*kernel_max

        for k in range(kernel_max):
            k_index = int(kernel_mask[kernel_mask_index+k].asscalar())
            i = k_index // kernel_size[0]
            j = k_index % kernel_size[1]
            h_im = h_offset + i * dilate[0]
            w_im = w_offset + j * dilate[1]
            col_data[data_col_index] = data_im[data_im_index+(i*dilate[0]*width+j*dilate[1])] if \
                (h_im >= 0 and w_im >= 0 and h_im < height and w_im < width) else 0
            col_buffer = col_data.reshape((kshape[0],kshape[1]*kshape[2],height_col,width_col))
            data_col_index += height_col * width_col
    return col_data.reshape((kshape[0]*kshape[1]*kshape[2],height_col*width_col))


if __name__ == '__main__':
    ctx = mx.gpu()
    mx.random.seed(128)
    Cout = 16
    kernel_max = 3
    N, Cin, Height, Width = (1,3,10,10)
    # mask = get_random_mask(Cout,Cin,(3,3),kernel_max).as_in_context(ctx)
    mask = nd.array([[[0,1,2]]*Cin]*Cout, dtype='float32',ctx=ctx)
    weight = nd.random.normal(0,1e-2,(Cout,Cin,kernel_max))
    net = FLKConv(Cout,Cin,(kernel_max,kernel_max),kernel_max,mask,weight_initializer=mx.init.Constant(weight))
    inpt = nd.random.uniform(0,1,(N,Cin,Height,Width),ctx=ctx)

    # col_buffer = FLK_im2col(inpt[0],mask)
    # weight_2d = weight.reshape((-1,1))
    # out_buffer = weight_2d * col_buffer
    # out_test = nd.sum(out_buffer.reshape(Cout, kernel_max*Cin, 8*8),1).reshape((N,Cout,8,8))

    net.initialize(ctx=ctx)
    tic = time.time()
    amount = 10
    for _ in range(amount):
        out1 = net(inpt)
        nd.waitall()
    # out1 = net(inpt)
    timeuse = (time.time() - tic)/amount


    net2 = MyConv(Cout,Cin,(1,kernel_max),kernel_max,mask,
                  weight_initializer=mx.init.Constant(weight),padding=(0,1))
    net2.initialize(ctx=ctx)
    tic = time.time()
    for _ in range(amount):
        out2 = net2(inpt)
        nd.waitall()
    # out2 = net2(inpt)
    timeuse2 = (time.time() - tic)/amount
    out2 = out2[:,:,:-2,1:-1]
    # net3 = Conv(Cout,Cin, (kernel_max,1),weight_initializer=mx.init.Constant(weight.expand_dims(3)))
    # net3.initialize(ctx=ctx)
    # tic = time.time()
    # for _ in range(100):
    #     out3 = net3(inpt)
    # # nd.waitall()
    # # out3 = net3(inpt)
    # timeuse3 =  time.time() - tic
    # print timeuse/1000, timeuse3/1000

    net4 = special_conv(Cout,(1,kernel_max),in_channels=Cin,
                        weight_initializer=mx.init.Constant(weight.reshape(Cout,Cin,1,kernel_max)),
                        padding=1)
    net4.initialize(ctx=ctx)
    net4.set_mask(mask.asnumpy())
    out4 = net4(inpt)[:,:,1:-1,1:-1]

    diff = out1-out2
    print(out1.shape)

