# FixLengthKernelConvolution
>> There are 4 versions of it.
>> The first three version are of non-aligned constraint, which means the prune pattern for each filter in a layer is different. Among all these three, the V3 version is the fastest
>> The V4 version is of output-channel-aligned constraint.
>> Now it is only implemented in GPU.
>> One can add these files to the source code of MXNet, and make it just following the official guide for 'install from source'. Now it does not support large channel size calculation.
