"""Evaluate an already selected PIXEL checkpoint on the fixed development holdout.

Portable evaluation entry point added for this release; no training or selection.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/mnli_holdout/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an evaluation result')
    from train_pixel_sweep import (torch, PixelDataset, evaluate, PangoCairoTextRenderer,
                                   PIXELForSequenceClassification, PoolingMode,
                                   get_transforms, resize_model_embeddings)
    from torch.utils.data import DataLoader
    from prepare_mnli_holdout import digest
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    manifest = json.loads(args.manifest.read_text())
    processor = PangoCairoTextRenderer.from_pretrained(args.checkpoint, rgb=True)
    processor.max_seq_length = 256
    transform = get_transforms(do_resize=True, size=(processor.pixels_per_patch,
                                                     processor.pixels_per_patch * 256))
    model, loading = PIXELForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=3, pooling_mode=PoolingMode.MEAN,
        add_layer_norm=True, output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys'):
        if loading.get(key):
            raise ValueError('Checkpoint loading mismatch: ' + key)
    resize_model_embeddings(model, 256)
    image_size = [processor.pixels_per_patch, processor.pixels_per_patch * 256]
    model.config.image_size = image_size
    model.image_size = image_size
    model.vit.embeddings.patch_embeddings.image_size = image_size
    model.vit.embeddings.patch_embeddings.num_patches = 256
    model.to('cuda').eval()
    scores = {}
    for name, items in manifest['holdout'].items():
        if len(items) != 2000 or digest(items) != manifest['audit'][name]['holdout_sha256']:
            raise ValueError('Holdout changed')
        loader = DataLoader(PixelDataset(items, processor, transform, 256,
                                        ('premise', 'hypothesis')),
                            batch_size=32, shuffle=False, num_workers=0)
        scores[name] = evaluate(model, loader, 'cuda')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'purpose': 'development holdout, not benchmark test',
                                     'eval': scores}, indent=2) + '\n')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
