"""Export the public NxTF partition compiler's intermediate representation.

This module runs the original partition optimizer, layer compilers and synapse
encoder. It does not execute NxSDK's synapse memory compiler, assign physical
chip/core addresses, or validate hardware behavior.
"""
import hashlib
from pathlib import Path

import numpy as np


SCHEMA = 'nxtf-partitions-v1'


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _format(fmt):
    return {name: _json(getattr(fmt, name)) for name in fmt.__slots__}


def _entry(entry):
    return {'prefixOffset': int(entry.prefixOffset), 'idxs': _json(entry.idxs),
            'weights': _json(entry.weights), 'delays': _json(entry.delays),
            'kernelIds': _json(entry.kernelIds), 'synFmtId': int(entry.synFmtId)}


def _resource_counts(partition):
    cxids = partition.compartmentGroup.cxIds
    configured = int(np.max(cxids)) + 1 if len(cxids) else 0
    return {'compiled_compartments': len(cxids),
            'configured_compartments': configured,
            'num_update_groups': (configured + 3) // 4,
            'update_slots': 4 * ((configured + 3) // 4),
            **{name: int(getattr(partition, name)) for name in
               ('numSyn', 'numSynEntries', 'numSynMemWords', 'numInputAxons',
                'numOutputAxons', 'numOutputAxonCfgEntries')}}


def _compartments(partition, logical_count, counts):
    group = partition.compartmentGroup
    global_ids = dict(zip(map(int, group.cxIds),
                          map(int, group.relToAbsDestCxIdxMap)))
    soft = partition.resetMode == 'soft'
    size = 2 if soft else 1
    compartments = []
    for index in range(counts['update_slots']):
        global_id = global_ids.get(index)
        real = global_id is not None and 0 <= global_id < logical_count * size
        role = ('dendrite' if index % 2 == 0 else 'soma') if soft else 'hard'
        # The mapper configures every index through max(cxIds), then updates
        # complete groups of four. Trailing unconfigured slots retain defaults.
        configured = index < counts['configured_compartments']
        compartments.append({
            'id': index, 'global_id': global_id,
            'logical_id': global_id // size if real else None,
            'role': role if real else 'dummy',
            'bias_mantissa': int(group.biasMant[index]) if configured else 0,
            'bias_exponent': int(group.biasExp[index]) if configured else 0,
            'configured': configured,
        })
    return compartments


def compile_model(model):
    """Run ``NxModel.partition()`` and return a JSON-compatible manifest.

    The model is expected to contain quantized integer-valued weights and
    biases, as required by the original NxTF compiler. No SDK namespace or
    SDK compiler is substituted. Raw IR arrays use their upstream field names.
    All group references are remapped to deterministic artifact identifiers.
    """
    for layer in model.layers:
        for weights in layer.get_weights():
            values = np.asarray(weights)
            if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
                raise ValueError(f'{layer.name} requires finite integer-valued weights and biases')
    model.partition()
    layers = model.partitionOptimizer.getLayers()
    model_layers = {layer.name: layer for layer in model.layers}
    input_groups = {}
    group_ids = {}
    for layer in layers:
        for partition in layer.partitions:
            pid = f'{layer.id}:{partition.id}'
            for index, group in enumerate(partition.inputAxonGroups):
                input_groups[group.id] = (pid, partition, group)
                group_ids[group.id] = f'{pid}:input:{index}'

    manifest = {'schema': SCHEMA,
                'provenance': {
                    'compiler': 'NxTF public partition compiler',
                    'sdk_mapping': False, 'hardware_validation': False,
                    'physical_placement': False,
                    'synapse_memory_compilation': False,
                    'source_sha256': {name: hashlib.sha256(
                        Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('dnn_layers.py', 'optimization.py',
                                     'synapse_compression.py', 'data_structures.py',
                                     'utils.py', 'offline.py')},
                }, 'layers': [], 'partitions': [], 'edges': []}
    for layer in layers:
        frontend = model_layers[layer.id]
        shape = tuple(frontend.output_shape[1:])
        logical_count = int(np.prod(shape))
        if getattr(frontend, 'signed', False):
            logical_count *= 2
        manifest['layers'].append({
            'id': layer.id, 'type': layer.type, 'output_shape': list(shape),
            'reset_mode': frontend.resetMode,
            'input_mode': _json(getattr(frontend, 'inputMode', None)),
            'signed': bool(getattr(frontend, 'signed', False)),
            'coreIdMap': _json(layer.coreIdMap),
            'multiplicityMap': _json(layer.multiplicityMap),
            'coreOccupancy': _json(layer.coreOccupancy),
        })
        for partition in layer.partitions:
            pid = f'{layer.id}:{partition.id}'
            group = partition.compartmentGroup
            counts = _resource_counts(partition)
            record = {
                'id': pid, 'layer_id': layer.id, 'type': layer.type,
                'partition_id': int(partition.id),
                'reset_mode': partition.resetMode,
                'compartment_parameters': _json(layer.compartmentKwargs),
                'connection_parameters': _json(layer.connectionKwargs),
                'sizeInterleaved': int(partition.sizeInterleaved),
                'chipCounter': int(partition.chipCounter),
                'isInhibitory': bool(partition.isInhibitory),
                'compartment_group': {name: _json(getattr(group, name)) for name
                    in ('cxIds', 'biasMant', 'biasExp', 'relToAbsDestCxIdxMap')},
                'compartments': _compartments(partition, logical_count, counts),
                'resource_counts': counts,
                'synapse_formats': [_format(fmt) for fmt in partition.synFmts],
                'synapse_groups': [
                    {'id': int(sg.id), 'synEntries': [
                        [_entry(entry) for entry in entries]
                        for entries in sg.synEntries]}
                    for sg in partition.synapseGroups],
                'input_axon_groups': [
                    {'id': group_ids[ag.id], 'srcNodeIds': _json(ag.srcNodeIds),
                     'multiplicity': _json(ag.multiplicity),
                     'synGroupId': int(ag.synGroup.id), 'cxBase': int(ag.cxBase)}
                    for ag in partition.inputAxonGroups],
                'output_axon_groups': [],
            }
            for index, axon in enumerate(partition.outputAxonGroups):
                target_pid, target, target_group = input_groups[axon.inAxGrpId]
                record['output_axon_groups'].append({
                    'id': f'{pid}:output:{index}', 'cxIds': _json(axon.cxIds),
                    'multiplicity': _json(axon.multiplicity),
                    'relSrcIds': _json(axon.relSrcIds),
                    'inAxGrpId': group_ids[axon.inAxGrpId],
                })
                for source_index, source_cx in enumerate(axon.cxIds):
                    relative = int(axon.relSrcIds[source_index])
                    # Match the public mapper's discrete-axon source indexing.
                    if axon.multiplicity[source_index] != 1 and relative >= target_group.numNodes:
                        relative = source_index
                    entries = target_group.synGroup.synEntries[relative]
                    for entry in entries:
                        fmt = entry.synFmt
                        reset = target.resetMode == 'soft' and fmt.softReset
                        exponent = target.layer.connectionKwargs[
                            'weightExpSR' if reset else 'weightExponent']
                        destinations = target_group.cxBase + entry.getCxIds()
                        for syn_index, destination in enumerate(destinations):
                            manifest['edges'].append({
                                'source_partition': pid,
                                'source_cx': int(source_cx),
                                'target_partition': target_pid,
                                'target_cx': int(destination),
                                'weight_mantissa': int(entry.weights[syn_index]),
                                'weight_exponent': int(exponent),
                                'num_weight_bits': int(fmt.numWgtBits),
                                'sign_mode': int(fmt.signMode),
                                'soft_reset': bool(reset),
                                'delay': 0 if entry.delays is None else int(entry.delays[syn_index]),
                            })
            manifest['partitions'].append(record)
    return manifest
