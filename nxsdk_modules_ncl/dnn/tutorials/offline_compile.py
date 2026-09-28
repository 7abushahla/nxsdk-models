"""Compile small real NxTF networks to portable public partition artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from nxsdk_modules_ncl.dnn.src.dnn_layers import (
    NxConv2D, NxDense, NxFlatten, NxInputLayer, NxModel,
)
from nxsdk_modules_ncl.dnn.src.offline import compile_model


def build_example(topology='dense', reset_mode='soft', logdir=None):
    """Construct a quantized model with an exactly representable reset weight."""
    params = dict(resetMode=reset_mode, vThMant=255, biasExp=6,
                  compartmentKwargs={'compartmentCurrentDecay': 4096,
                                     'compartmentVoltageDecay': 0,
                                     'refractoryDelay': 0})
    shape = (3,) if topology == 'dense' else (3, 3, 1)
    source = NxInputLayer(shape, name='input', **params)
    output = source.input
    if topology == 'conv':
        conv = NxConv2D(1, (2, 2), name='conv', **params)
        output = conv(output)
        conv.set_weights([np.array([20, 40, 60, 80], dtype=np.float32).reshape(2, 2, 1, 1),
                          np.zeros(1, dtype=np.float32)])
        output = NxFlatten(name='flatten')(output)
    elif topology != 'dense':
        raise ValueError('topology must be dense or conv')
    dense = NxDense(2, name='dense', **params)
    output = dense(output)
    weights = (np.array([[64, 32], [128, 64], [32, 128]], dtype=np.float32)
               if topology == 'dense' else
               np.array([[64, 32], [128, 64], [32, 128], [128, 32]], dtype=np.float32))
    dense.set_weights([weights, np.zeros(2, dtype=np.float32)])
    return NxModel(source.input, output, logdir=logdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology', choices=['dense', 'conv'], default='dense')
    parser.add_argument('--reset-mode', choices=['hard', 'soft'], default='soft')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logdir', type=Path, default=Path('offline-output'))
    args = parser.parse_args()
    manifest = compile_model(build_example(args.topology, args.reset_mode, str(args.logdir)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(manifest["partitions"])} compiled partitions and '
          f'{len(manifest["edges"])} connections to {args.output}')


if __name__ == '__main__':
    main()
