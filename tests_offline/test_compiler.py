"""Behavioral tests of the real public compiler without an NxSDK install."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class PublicCompilerTests(unittest.TestCase):
    def api(self):
        try:
            layers = importlib.import_module('nxsdk_modules_ncl.dnn.src.dnn_layers')
            offline = importlib.import_module('nxsdk_modules_ncl.dnn.src.offline')
        except ImportError as error:
            self.fail(f'Public compilation must import without NxSDK: {error}')
        return layers, offline

    def dense(self, reset_mode, directory):
        import numpy as np
        layers, offline = self.api()
        parameters = dict(resetMode=reset_mode, vThMant=255,
                          compartmentKwargs={'compartmentCurrentDecay': 4096,
                                             'refractoryDelay': 0})
        source = layers.NxInputLayer((3,), name='input', **parameters)
        dense = layers.NxDense(2, name='dense', **parameters)
        output = dense(source.input)
        dense.set_weights([np.array([[1,-2],[3,4],[-5,6]], dtype='float32'),
                           np.array([7,-8], dtype='float32')])
        model = layers.NxModel(source.input, output, logdir=directory)
        return model, offline

    def test_public_import_does_not_require_nxsdk(self):
        self.api()

    def test_hard_reset_exports_compiled_connectivity(self):
        with tempfile.TemporaryDirectory() as directory:
            model, offline = self.dense('hard', directory)
            result = offline.compile_model(model)
            self.assertEqual(result['schema'], 'nxtf-partitions-v1')
            self.assertFalse(result['provenance']['sdk_mapping'])
            self.assertFalse(result['provenance']['hardware_validation'])
            self.assertIsNone(model.board)
            self.assertEqual(len(result['partitions']), 2)
            actual = {(e['source_cx'], e['target_cx']): e['weight_mantissa']
                      for e in result['edges']}
            self.assertEqual(actual, {(0,0):1,(0,1):-2,(1,0):3,(1,1):4,(2,0):-5,(2,1):6})
            self.assertFalse(any(e['soft_reset'] for e in result['edges']))
            dense = next(p for p in result['partitions'] if p['layer_id']=='dense')
            self.assertEqual([c['bias_mantissa'] for c in dense['compartments']], [7,-8,0,0])
            self.assertTrue(dense['synapse_formats'])
            self.assertTrue(dense['synapse_groups'])
            self.assertTrue(dense['input_axon_groups'])
            self.assertEqual(result, json.loads(json.dumps(result)))

    def test_soft_reset_keeps_two_compartments_and_feedback_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            model, offline = self.dense('soft', directory)
            result = offline.compile_model(model)
            forward = [e for e in result['edges'] if not e['soft_reset']]
            reset = [e for e in result['edges'] if e['soft_reset']]
            self.assertEqual(len(forward), 6)
            self.assertEqual(len(reset), 5)
            self.assertEqual({(e['source_cx'], e['target_cx']):e['weight_mantissa']
                              for e in forward},
                             {(1,0):1,(1,2):-2,(3,0):3,(3,2):4,(5,0):-5,(5,2):6})
            for edge in reset:
                self.assertEqual(edge['source_partition'], edge['target_partition'])
                self.assertEqual(edge['source_cx'], edge['target_cx'] + 1)
                self.assertEqual(edge['weight_mantissa'], -255)
                self.assertEqual(edge['weight_exponent'], 0)
                self.assertEqual(edge['delay'], 0)
            counts = {p['layer_id']:len(p['compartments']) for p in result['partitions']}
            self.assertEqual(counts, {'input':8,'dense':4})
            inp = next(p for p in result['partitions'] if p['layer_id']=='input')
            self.assertEqual([c['role'] for c in inp['compartments']],
                             ['dendrite','soma','dendrite','soma','dendrite','soma','dummy','dummy'])
            self.assertEqual([c['logical_id'] for c in inp['compartments']],
                             [0,0,1,1,2,2,None,None])
            self.assertTrue(all('sign_mode' in e for e in result['edges']))

    def test_fractional_weights_fail_before_export(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            model, offline = self.dense('hard', directory)
            dense = model.layers[-1]
            weights, biases = dense.get_weights()
            weights[0,0] = 0.25
            dense.set_weights([weights,biases])
            with self.assertRaisesRegex(ValueError, 'integer'):
                offline.compile_model(model)

    def test_cross_partition_edges_preserve_global_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            model, offline = self.dense('soft', directory)
            for layer in model.layers:
                layer.maxNumCompartments = 4
            result = offline.compile_model(model)
            lookup = {(p['id'],c['id']):c['logical_id'] for p in result['partitions']
                      for c in p['compartments']}
            forward = [e for e in result['edges'] if not e['soft_reset']]
            self.assertGreater(len(result['partitions']),2)
            actual = {(lookup[(e['source_partition'],e['source_cx'])],
                       lookup[(e['target_partition'],e['target_cx'])]):e['weight_mantissa']
                      for e in forward}
            self.assertEqual(actual, {(0,0):1,(0,1):-2,(1,0):3,(1,1):4,(2,0):-5,(2,1):6})

    def test_conv_flatten_dense_keeps_connections_across_virtual_layer(self):
        import numpy as np
        layers, offline = self.api()
        with tempfile.TemporaryDirectory() as directory:
            source = layers.NxInputLayer((3,3,1),name='input')
            conv = layers.NxConv2D(1,(2,2),name='conv')
            convolved = conv(source.input)
            flattened = layers.NxFlatten(name='flatten')(convolved)
            dense = layers.NxDense(2,name='dense')
            output = dense(flattened)
            conv.set_weights([np.array([1,2,3,4],dtype='float32').reshape(2,2,1,1), np.array([0],dtype='float32')])
            dense.set_weights([np.array([[5,6],[7,8],[9,10],[11,12]],dtype='float32'),np.array([0,0],dtype='float32')])
            result = offline.compile_model(layers.NxModel(source.input,output,logdir=directory))
            self.assertEqual({p['layer_id'] for p in result['partitions']}, {'input','conv','dense'})
            self.assertEqual(len(result['edges']),24)
            dense_edges = [e for e in result['edges'] if e['target_partition']=='dense:0']
            lookup = {(p['id'],c['id']):c['logical_id'] for p in result['partitions']
                      for c in p['compartments']}
            self.assertEqual({(lookup[(e['source_partition'],e['source_cx'])],
                               lookup[(e['target_partition'],e['target_cx'])]):e['weight_mantissa']
                              for e in dense_edges},
                             {(0,0):5,(0,1):6,(1,0):9,(1,1):10,(2,0):7,(2,1):8,(3,0):11,(3,1):12})


if __name__ == '__main__':
    unittest.main()
