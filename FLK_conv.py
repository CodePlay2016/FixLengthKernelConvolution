import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from general_conv import special_conv
import time

class FLKConv(gluon.nn.HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1,1), dilation=(1,1),
                 padding=(0,0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None,**kwargs):
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
        self._kwargs.update(kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.contrib.FixLengthKernelConvolution(x,self.kernel_mask,weight,name='fwd', **self._kwargs)
        else:
            out = F.contrib.FixLengthKernelConvolution(x,self.kernel_mask,weight,bias,name='fwd', **self._kwargs)
        return out
    
    def set_mask(self, kernel_mask):
        self.kernel_mask = kernel_mask.copy()

class FLKConv_v2(gluon.nn.HybridBlock):
    '''
    combine the im2col with broadcast multiplication
    '''
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1, 1), dilation=(1, 1),
                 padding=(0, 0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None, **kwargs):
        super(FLKConv_v2, self).__init__(prefix=prefix, params=params)
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
        self._kwargs.update(kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.contrib.FixLengthKernelConvolutionV2(x, self.kernel_mask, weight, name='fwd', **self._kwargs)
        else:
            out = F.contrib.FixLengthKernelConvolutionV2(x, self.kernel_mask, weight, bias, name='fwd', **self._kwargs)
        return out

    def set_mask(self, kernel_mask):
        self.kernel_mask = kernel_mask.copy()

class FLKConv_v3(gluon.nn.HybridBlock):
    '''
    combine the im2col with broadcast multiplication
    '''
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1, 1), dilation=(1, 1),
                 padding=(0, 0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None, **kwargs):
        super(FLKConv_v3, self).__init__(prefix=prefix, params=params)
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
        self._kwargs.update(kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.contrib.FixLengthKernelConvolutionV3(x, self.kernel_mask, weight, name='fwd', **self._kwargs)
        else:
            out = F.contrib.FixLengthKernelConvolutionV3(x, self.kernel_mask, weight, bias, name='fwd', **self._kwargs)
        return out

    def set_mask(self, kernel_mask):
        self.kernel_mask = kernel_mask.copy()

class FLKConv_v4(gluon.nn.HybridBlock):
    '''
    combine the im2col with broadcast multiplication
    '''
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1, 1), dilation=(1, 1),
                 padding=(0, 0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None, **kwargs):
        super(FLKConv_v4, self).__init__(prefix=prefix, params=params)
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
        self._kwargs.update(kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            out = F.contrib.FixLengthKernelConvolutionV4(x, self.kernel_mask, weight, name='fwd', **self._kwargs)
        else:
            out = F.contrib.FixLengthKernelConvolutionV4(x, self.kernel_mask, weight, bias, name='fwd', **self._kwargs)
        return out

    def set_mask(self, kernel_mask):
        self.kernel_mask = kernel_mask.copy()

class MyConv(gluon.nn.HybridBlock):
    def __init__(self, channels, in_channels, kernel_size, kernel_max, kernel_mask, strides=(1, 1), dilation=(1, 1),
                 padding=(0, 0), groups=1, use_bias=True, layout='NCHW', weight_initializer=None,
                 bias_initializer='zeros',prefix=None, params=None,**kwargs):
        super(MyConv, self).__init__(prefix=prefix, params=params)
        self._channels = channels
        self._in_channels = in_channels
        self._kwargs = {
            'kernel': kernel_size, 'kernel_max': kernel_max, 'stride': strides,
            'dilate': dilation, 'pad': padding, 'num_filter': channels, 'num_group': groups,
            'no_bias': not use_bias, 'layout': layout
        }
        self.kernel_mask = kernel_mask
        wshape = (channels, in_channels, kernel_size[0], kernel_size[1])
        self.weight = self.params.get('weight', shape=wshape, init=weight_initializer,
                                      allow_deferred_init=True)
        if use_bias:
            self.bias = self.params.get('bias', shape=(channels,),
                                        init=bias_initializer, allow_deferred_init=True)
        self._kwargs.update(kwargs)

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

def get_weights(layer_schedule, kernel_size):
    weight_dict = {}
    for name, param in layer_schedule:
        if 'conv' in name:
            weight_dict[name] = nd.random.normal(0,1e-3,shape=(1,1))

class test_block(gluon.nn.HybridBlock):
    def __init__(self, kernel_max, layer_schedule,conv_type='FLK', fix_weight=False, weight_dict=None, ctx=mx.gpu(), **kwargs):
        super(test_block,self).__init__(**kwargs)
        net_dict = {
            'normal': MyConv,
            'FLK': FLKConv,
            'FLKv2': FLKConv_v2,
            'FLKv4': FLKConv_v4
        }
        assert (conv_type in net_dict.keys()),"conv_type not supported!"
        kernel = (kernel_max,)*2
        conv = net_dict[conv_type]
        masks = build_mask(kernel_max, layer_schedule, ctx)
        self.net = gluon.nn.HybridSequential()
        jconv = 0
        for i,(name,param) in enumerate(layer_schedule):
            if 'conv' in name:
                wi = mx.init.Constant(weight_dict[name]) if fix_weight else mx.init.Xavier()
                layer = conv(param[0],param[1],kernel,kernel_max,masks[jconv],weight_initializer=wi)
                jconv += 1
            elif 'relu' in name:
                layer = gluon.nn.Activation('relu')
            elif 'bn' in name:
                layer = gluon.nn.BatchNorm()
            else:
                layer = gluon.nn.MaxPool2D()
            self.net.add(layer)
    def hybrid_forward(self, F, x, *args, **kwargs):
        for layer in self.net:
            tic = time.time()
            x = layer(x)
            nd.waitall()
            time0 = time.time()-tic
            time0 -= 0
        return x

def get_random_mask(Cin, Cout, kernel, kernel_max):
    '''
    :param Cin: int number of input channel
    :param Cout: int number of output channel
    :param kernel: tuple, kernel size, (kh, kw)
    :param kernel_max: int number of kept kernel size
    :return: NDArray, mask array in shape of [Cout, Cin, Kernel_max]
    '''
    return nd.random.uniform(0,kernel[0]*kernel[1],(Cin,Cout,kernel_max)).astype('int').astype('float32')

def get_layer_schedule(base_channels,model_type='VGG'):
    schedule = []

    if model_type == 'VGG':
        schedule.append(['conv1',[base_channels,3]])
        schedule.append(['bn1',[]])
        schedule.append(['relu1',[]])
        schedule.append(['conv2',[base_channels,base_channels]])
        schedule.append(['bn2',[]])
        schedule.append(['relu2',[]])
        schedule.append(['pool1',[]])

        schedule.append(['conv3',[base_channels*2,base_channels]])
        schedule.append(['bn3',[]])
        schedule.append(['relu3',[]])
        schedule.append(['conv4',[base_channels*2,base_channels*2]])
        schedule.append(['bn4',[]])
        schedule.append(['relu4',[]])
        schedule.append(['pool2',[]])

        schedule.append(['conv5',[base_channels*4,base_channels*2]])
        schedule.append(['bn5',[]])
        schedule.append(['relu5',[]])
        schedule.append(['conv6',[base_channels*4,base_channels*4]])
        schedule.append(['bn6',[]])
        schedule.append(['relu6',[]])
        schedule.append(['conv7',[base_channels*4,base_channels*4]])

    return schedule

def build_mask(kernel_max, layer_schedule, ctx):
    return [nd.array([[list(range(kernel_max))]*param[1]]*param[0],ctx=ctx) for _,param in layer_schedule if len(param)]

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

def test1():
    ctx = mx.gpu()
    mx.random.seed(128)
    Cout = 256
    kernel_max = 3
    N, Cin, Height, Width = (128, 256, 112, 112)
    # mask = get_random_mask(Cout,Cin,(3,3),kernel_max).as_in_context(ctx)
    mask = nd.array([[[0, 1, 2]] * Cin] * Cout, dtype='float32', ctx=ctx)
    mask_ = nd.array([[0, 1, 2]] * Cin, dtype='float32', ctx=ctx)

    weight = nd.random.normal(0, 1e-2, (Cout, Cin, kernel_max))
    amount = 10
    inpt = nd.random.uniform(0, 1, (N, Cin, Height, Width), ctx=ctx)

    # custom original convolution
    net0 = MyConv(Cout, Cin, (1, kernel_max), kernel_max, mask,
                  weight_initializer=mx.init.Constant(weight.expand_dims(2)), padding=(0, 1))
    net0.initialize(ctx=ctx)
    tic = time.time()
    for _ in range(amount):
        out0 = net0(inpt)
        nd.waitall()
    # out0 = net0(inpt)
    timeuse0 = (time.time() - tic) / amount
    # out0 = out0[:, :, :-2, 1:-1]

    # FLK v1
    # net = FLKConv(Cout, Cin, (kernel_max, kernel_max), kernel_max, mask, weight_initializer=mx.init.Constant(weight))
    # net.initialize(ctx=ctx)
    # tic = time.time()
    # for _ in range(amount):
    #     out1 = net(inpt)
    #     nd.waitall()
    # timeuse1 = (time.time() - tic) / amount

    # net2 = FLKConv_v2(Cout, Cin, (kernel_max, kernel_max), kernel_max, mask,
    #                   weight_initializer=mx.init.Constant(weight))
    # net2.initialize(ctx=ctx)
    # tic = time.time()
    # for _ in range(amount):
    #     out2 = net2(inpt)
    #     nd.waitall()
    # timeuse2 = (time.time() - tic) / amount
    #
    # net3 = FLKConv_v3(Cout, Cin, (kernel_max, kernel_max), kernel_max, mask,
    #                   weight_initializer=mx.init.Constant(weight))
    # net3.initialize(ctx=ctx)
    # tic = time.time()
    # for _ in range(amount):
    #     out3 = net3(inpt)
    #     nd.waitall()
    # timeuse3 = (time.time() - tic) / amount

    net4 = FLKConv_v4(Cout, Cin, (kernel_max, kernel_max), kernel_max, mask_,
                      weight_initializer=mx.init.Constant(weight))
    net4.initialize(ctx=ctx)
    tic = time.time()
    for _ in range(amount):
        out4 = net4(inpt)
        nd.waitall()
    timeuse4 = (time.time() - tic) / amount


    print('')
    # # original convolution
    # net3 = Conv(Cout,Cin, (kernel_max,1),weight_initializer=mx.init.Constant(weight.expand_dims(3)))
    # net3.initialize(ctx=ctx)
    # tic = time.time()
    # for _ in range(100):
    #     out3 = net3(inpt)
    # # nd.waitall()
    # # out3 = net3(inpt)
    # timeuse3 =  time.time() - tic
    # print timeuse/1000, timeuse3/1000

    # net4 = special_conv(Cout,(1,kernel_max),in_channels=Cin,
    #                     weight_initializer=mx.init.Constant(weight.reshape(Cout,Cin,1,kernel_max)),
    #                     padding=1)
    # net4.initialize(ctx=ctx)
    # net4.set_mask(mask.asnumpy())
    # out4 = net4(inpt)[:,:,1:-1,1:-1]

    # col_buffer = FLK_im2col(inpt[0],mask)
    # weight_2d = weight.reshape((-1,1))
    # out_buffer = weight_2d * col_buffer
    # out_test = nd.sum(out_buffer.reshape(Cout, kernel_max*Cin, 8*8),1).reshape((N,Cout,8,8))

def test_model(model, inpt, amount, wait=True):
    tic = time.time()
    for i in range(amount):
        if amount >1 and i == 1: tic = time.time()
        out = model(inpt)
        if wait: nd.waitall()
    amount = amount - 1 if amount > 1 else amount
    time_use = (time.time() - tic)/amount
    return time_use, out

if __name__ == '__main__':
    # test1()
    ctx = mx.gpu()
    mx.random.seed(128)
    kernel_max = 3
    base_channels=32
    N, Cin, Height, Width = (8 , 3, 112, 112)
    inpt = nd.random.uniform(0, 1, (N, Cin, Height, Width), ctx=ctx)

    model0 = test_block(kernel_max,get_layer_schedule(base_channels=base_channels),conv_type='normal')
    model1 = test_block(kernel_max,get_layer_schedule(base_channels=base_channels),conv_type='FLK')
    model2 = test_block(kernel_max,get_layer_schedule(base_channels=base_channels),conv_type='FLKv2')
    model3 = test_block(kernel_max,get_layer_schedule(base_channels=base_channels),conv_type='FLKv4')

    model3.initialize(init=mx.init.Xavier(),ctx=ctx)
    model0.initialize(init=mx.init.Xavier(),ctx=ctx)
    model1.initialize(init=mx.init.Xavier(),ctx=ctx)
    model2.initialize(init=mx.init.Xavier(),ctx=ctx)

    amount = 10
    wait = True
    time0, out0 = test_model(model0, inpt, amount, wait)
    time1, out1 = test_model(model1, inpt, amount, wait)
    time2, out2 = test_model(model2, inpt, amount, wait)
    time3, out3 = test_model(model3, inpt, amount, wait)

    print('')
