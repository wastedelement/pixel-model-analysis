"""Passive diagnostics around the unchanged canonical training program.

No new RNG draws, optimizer settings, data order, loss or clipping policy.
Controls and interventions both use this wrapper; the canonical file is
checksum-locked and is never edited. Diagnostics may synchronize CUDA.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   ensure_ascii=False).encode()).hexdigest()


def attach_diagnostics(module, output):
    torch = module.torch
    state = {'optimizer_updates': 0, 'initial_parameter_hashes': {},
             'train_order_by_epoch': [], 'canonical_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
             'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'torch': torch.__version__, 'cuda': torch.version.cuda,
             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    live = {'model': None, 'train_dataset_assigned': False, 'order_count': 0,
            'order_hash': hashlib.sha256(), 'last_gradient': None}
    original_load = module.PIXELForSequenceClassification.from_pretrained

    def load_model(*args, **kwargs):
        model = original_load(*args, **kwargs)
        live['model'] = model
        for name, parameter in model.named_parameters():
            if 'classifier' in name or 'pooler' in name or name.startswith('vit.layernorm.'):
                values = parameter.detach().contiguous().cpu().numpy().tobytes()
                state['initial_parameter_hashes'][name] = hashlib.sha256(values).hexdigest()
        state['initial_heads_sha256'] = digest(state['initial_parameter_hashes'])
        return model

    module.PIXELForSequenceClassification.from_pretrained = staticmethod(load_model)
    original_dataset = module.PixelDataset

    class RecordingDataset(original_dataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.record_order = not live['train_dataset_assigned']
            if self.record_order:
                live['train_dataset_assigned'] = True
                state['train_n'] = len(self.items)
                state['train_subset_sha256'] = digest(self.items)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            if self.record_order:
                live['order_hash'].update(struct.pack('<Q', int(index)))
                live['order_count'] += 1
                if live['order_count'] % len(self.items) == 0:
                    state['train_order_by_epoch'].append({
                        'epoch': live['order_count'] // len(self.items),
                        'example_index_order_sha256': live['order_hash'].hexdigest()})
                    live['order_hash'] = hashlib.sha256()
            return result

    module.PixelDataset = RecordingDataset
    original_clip = torch.nn.utils.clip_grad_norm_

    def sample_step(step):
        return step <= 10 or step % 100 == 0 or step in (25, 50, 188, 376, 752, 1880)

    def clip_grad_norm(parameters, *args, **kwargs):
        parameters = list(parameters)
        step = state['optimizer_updates'] + 1
        groups = None
        if sample_step(step):
            sums = {k: 0.0 for k in ('encoder_0', 'encoder_6', 'encoder_11',
                                     'final_layernorm', 'task_head')}
            counts = dict.fromkeys(sums, 0)
            for name, parameter in live['model'].named_parameters():
                if parameter.grad is None:
                    continue
                group = None
                for layer in (0, 6, 11):
                    if name.startswith(f'vit.encoder.layer.{layer}.'):
                        group = f'encoder_{layer}'
                if name.startswith('vit.layernorm.'):
                    group = 'final_layernorm'
                elif 'pooler' in name or 'classifier' in name:
                    group = 'task_head'
                if group is not None:
                    norm = float(parameter.grad.detach().norm().item())
                    sums[group] += norm * norm
                    counts[group] += 1
            groups = {key: {'norm': math.sqrt(value), 'parameter_tensors': counts[key]}
                      for key, value in sums.items()}
        total = original_clip(parameters, *args, **kwargs)
        live['last_gradient'] = ({'global_preclip_norm': float(total.item()),
                                 'groups_preclip': groups} if groups is not None else None)
        return total

    torch.nn.utils.clip_grad_norm_ = clip_grad_norm
    original_optim = module.AdamW

    class RecordingAdamW(original_optim):
        def step(self, *args, **kwargs):
            step = state['optimizer_updates'] + 1
            used_lr = [float(group['lr']) for group in self.param_groups]
            result = super().step(*args, **kwargs)
            state['optimizer_updates'] = step
            if sample_step(step):
                record = {'optimizer_update': step, 'lr_used': used_lr,
                          'gradient': live['last_gradient']}
                with open(output / 'optimization_diagnostics.jsonl', 'a') as f:
                    f.write(json.dumps(record) + '\n')
            return result

    module.AdamW = RecordingAdamW
    return state
