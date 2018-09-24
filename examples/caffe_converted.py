from __future__ import print_function
import numpy as np
import sys, argparse
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from six import text_type as _text_type


__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    n = caffe.NetSpec()

    n.input           = L.Input(shape=[dict(dim=[1, 1, 32, 32])], ntop=1)
    n.HardNetnSequentialnfeaturesnnConv2dn0n7 = L.Convolution(n.input, kernel_h=7, kernel_w=7, stride=1, num_output=32, pad_h=0, pad_w=0, group=1, bias_term=True, ntop=1)
    n.HardNetnSequentialnfeaturesnnTanhn1n8 = L.TanH(n.HardNetnSequentialnfeaturesnnConv2dn0n7, ntop=1)
    n.HardNetnSequentialnfeaturesnnMaxPool2dn2n9 = L.Pooling(n.HardNetnSequentialnfeaturesnnTanhn1n8, pool=0, kernel_size=2, pad_h=0, pad_w=0, stride=2, ntop=1)
    n.HardNetnSequentialnfeaturesnnConv2dn3n10 = L.Convolution(n.HardNetnSequentialnfeaturesnnMaxPool2dn2n9, kernel_h=6, kernel_w=6, stride=1, num_output=64, pad_h=0, pad_w=0, group=1, bias_term=True, ntop=1)
    n.HardNetnSequentialnfeaturesnnTanhn4n11 = L.TanH(n.HardNetnSequentialnfeaturesnnConv2dn3n10, ntop=1)
    n.HardNetnSequentialnclassifiernnLinearn0n13 = L.InnerProduct(n.HardNetnSequentialnfeaturesnnTanhn4n11, num_output=128, bias_term=True, ntop=1)
    n.HardNetnSequentialnclassifiernnTanhn1n14 = L.TanH(n.HardNetnSequentialnclassifiernnLinearn0n13, ntop=1)

    return n

def make_net(prototxt):
    n = KitModel()
    with open(prototxt, 'w') as fpb:
        print(n.to_proto(), file=fpb)

def gen_weight(weight_file, model, prototxt):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    net = caffe.Net(prototxt, caffe.TRAIN)

    for key in __weights_dict:
        if 'weights' in __weights_dict[key]:
            net.params[key][0].data.flat = __weights_dict[key]['weights']
        elif 'mean' in __weights_dict[key]:
            net.params[key][0].data.flat = __weights_dict[key]['mean']
            net.params[key][1].data.flat = __weights_dict[key]['var']
            if 'scale' in __weights_dict[key]:
                net.params[key][2].data.flat = __weights_dict[key]['scale']
        elif 'scale' in __weights_dict[key]:
            net.params[key][0].data.flat = __weights_dict[key]['scale']
        if 'bias' in __weights_dict[key]:
            net.params[key][1].data.flat = __weights_dict[key]['bias']
        if 'gamma' in __weights_dict[key]: # used for prelu, not sure if other layers use this too
            net.params[key][0].data.flat = __weights_dict[key]['gamma']
    net.save(model)
    return net



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate caffe model and prototxt')
    parser.add_argument('--weight_file', '-w', type=_text_type, default='IR weight file')
    parser.add_argument('--prototxt', '-p', type=_text_type, default='caffe_converted.prototxt')
    parser.add_argument('--model', '-m', type=_text_type, default='caffe_converted.caffemodel')
    args = parser.parse_args()
    # For some reason argparser gives us unicode, so we need to conver to str first
    make_net(str(args.prototxt))
    gen_weight(str(args.weight_file), str(args.model), str(args.prototxt))


