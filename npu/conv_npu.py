import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width      # val:512
    #here, we can set oour X matrix tile with a row as the matrix width, it would work...

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax    # val:128
    c_out_pmax = nl.tile_size.pmax #val:128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    #first, we reshape matrix W and do transpose.
    #cause W is static compared to X, we do transpose for W, also X should be the matrix B
    #because of B is allowed to have bigger width. A position needs to be transposed.

    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    W_origin = nl.ndarray(
        (n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width),
        dtype=W.dtype, buffer=nl.sbuf
        ) 
    #load the value from device mem to sbuf by whole chunk
    for c_out_i in nl.affine_range(n_tiles_c_out):
        W_origin[c_out_i] = nl.load(W[c_out_i]) 
    #now we have a sbuf block with whole W value

    #define transpose matrix and do the transpose work:
    #noticed that for efficiency, we move filter dim to top level dimenstion
    W_t = nl.ndarray(
        (filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax),
        dtype=W.dtype, buffer=nl.sbuf
        )

    #temperal buffer(be operated by transpose op, so need sbuf type)


    for c_out_i in nl.affine_range(n_tiles_c_out):
        for c_in_i in nl.affine_range(n_tiles_c_in):
            for f_h in nl.affine_range(filter_height):
                for f_w in nl.affine_range(filter_width):
                    w_buffer = nl.ndarray(
                        (nl.par_dim(c_out_pmax), c_in_pmax),
                        dtype=W.dtype, buffer=nl.sbuf
                        )    
                    w_buffer= nl.copy(
                        W_origin[c_out_i, :, c_in_i, :, f_h, f_w], dtype=W.dtype
                        )
                    W_t[f_h, f_w, c_out_i, c_in_i] = nisa.nc_transpose(w_buffer)



    #here, let's generate a output matrix for per images output(sbuf register)
    #size overhead test::

    ##problematic
    #Output_origin = nl.zeros((n_tiles_c_out,nl.par_dim(c_out_pmax),out_height, out_width), dtype = X.dtype, buffer=nl.sbuf)


    # Process the images in batches
    for b_i in nl.affine_range(batch_size):  #iterate through batch images
        #raise RuntimeError("Please fill your implementation of computing convolution"
        #                   " of X[b] with the weights W and bias b and store the result in X_out[b]")
        bias_indicator = 0

        #we do the row tile chunk by chunk: 
        out_h_tile_size = 4
        out_h_tile_amount = (out_height+out_h_tile_size-1) // out_h_tile_size

        #here, because conv needs additional lines(the marginal), we need
        #to load additional rows for input to calculate out out_h_tile_size rows of 
        #output matrix:
        input_h_tile_size = out_h_tile_size + filter_height - 1
        for out_h_tile_i in nl.affine_range(out_h_tile_amount): #iterate through row tiles
            input_h_start = out_h_tile_i*out_h_tile_size
            output_h_start = out_h_tile_i*out_h_tile_size
            input_h_end = input_h_start + input_h_tile_size
            output_h_end = output_h_start + out_h_tile_size
            #generate a bsuf tile for X loaded:

            #choose the specific x block here; satisfy necessary chunk
            #X_input_tile = nl.ndarray(
            #    shape=(n_tiles_c_in,nl.par_dim(c_in_pmax), input_h_tile_size, input_width),
            #    dtype=X.dtype,
            #    buffer=nl.sbuf
            #)
            #load X tile value(only for specific input conv tile)
            #for tile_i in nl.affine_range(n_tiles_c_in):
            #    X_input_tile[tile_i] = nl.load(X[b_i, (tile_i)*c_in_pmax:(tile_i+1)*c_in_pmax,
            #                         (input_h_start):(input_h_end),:])
     



                #begin tiled matrix mul: (out as row amount, in as column amount)
            for c_out_tile_i in nl.affine_range(n_tiles_c_out):

                #each time, generate a whole size output matrix(psum register)


                #Output_tiles= nl.zeros((c_out_pmax, out_h_tile_size, 
                #                          out_width), 
                #                        dtype = X.dtype, buffer=nl.sbuf
                #                )


                for out_h_i in nl.affine_range(out_h_tile_size):
                    if output_h_start+out_h_i<out_height:
                        continue
                    Output_row = nl.zeros((c_out_pmax, 
                                          out_width), 
                                        dtype=nl.float32, buffer=nl.psum
                                        )
                    for c_in_tile_i in nl.affine_range(n_tiles_c_in):
                        #smaller x_input_tiles:
                        X_input_tile_s = nl.ndarray(
                        shape=(nl.par_dim(c_in_pmax), filter_height, input_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf
                        )

    
                        input_h_end = input_h_start+out_h_i+filter_height
                        X_input_tile_s= nl.load(X[b_i, (c_in_tile_i)*c_in_pmax:(c_in_tile_i+1)*c_in_pmax,
                                                (input_h_start+out_h_i):(input_h_end),:]
                                                ,mask=(input_h_end <= input_height)
                        )


                        for f_i in nl.affine_range(filter_height):
                            for f_j in nl.affine_range(filter_width):
                              Output_row+= nl.matmul(
                                  W_t[f_i,f_j,c_out_tile_i, c_in_tile_i],
                                  #X_input_tile[c_in_tile_i,:,out_h_i+f_i, f_j:f_j+out_width],
                                  X_input_tile_s[:,f_i, f_j:f_j+out_width],
                                  transpose_x = True
                              )
                    #final, add the bias.
                    bias_vertical_vec = nl.ndarray(
                    (nl.par_dim(c_out_pmax),1), dtype=bias.dtype, buffer=nl.sbuf
                        )
                    bias_vertical_vec = nl.load(bias[c_out_tile_i*c_out_pmax:(c_out_tile_i+1)*c_out_pmax])
                         
                    Output_row = nisa.tensor_scalar(Output_row, np.add, bias_vertical_vec)
                    #Output_tiles[:, out_h_i,:] = nl.copy(Output_row)
                #now Ouput_tiles is calculated completely::
                    c_out_tile_start = c_out_tile_i*c_out_pmax
                    c_out_tile_end = c_out_tile_start+c_out_pmax
                    nl.store(X_out[b_i,c_out_tile_start:c_out_tile_end,output_h_start+out_h_i],
                        Output_row
                        ,mask=(output_h_start+out_h_i<out_height)
                        )
                      
    return X_out

