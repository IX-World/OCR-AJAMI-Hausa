import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import unicodedata
from datetime import datetime
from types import SimpleNamespace
from PIL import Image

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def read_samples(root, name, corrections):
    folder = (root / 'data' / name).resolve()
    metadata = folder / 'metadata.jsonl'
    if not metadata.is_file():
        raise FileNotFoundError(f'Métadonnées absentes : {metadata}')
    entries = []
    for number, line in enumerate(metadata.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        path = (folder / row['file_name']).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f'Chemin hors dataset : {path}')
        if not path.is_file() and path.name.endswith('png') and (not path.name.endswith('.png')):
            repaired = path.with_name(path.name[:-3] + '.png')
            if repaired.is_file():
                corrections.append({'split': name, 'original': row['file_name'], 'resolved': str(repaired.relative_to(folder))})
                path = repaired
                row['file_name'] = str(path.relative_to(folder))
        text = unicodedata.normalize('NFC', row['transcript'])
        if not text.strip():
            raise ValueError(f'Transcription vide : {metadata}:{number}')
        with Image.open(path) as image:
            gray = image.convert('L')
            width, height = gray.size
            digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        entries.append(dict(source='old_' + name, file_name=row['file_name'], transcript=text, manuscript=str(row.get('source', 'unknown')), sha256_pixels=digest, width=width, height=height))
    if not entries:
        raise ValueError(f'Dataset vide : {metadata}')
    return entries

def prepare(args):
    root = Path(args.dataset).resolve()
    corrections = []
    raw_train = read_samples(root, 'train', corrections)
    raw_test = read_samples(root, 'test', corrections)
    seen = {}
    duplicates = []
    pools = {'train': [], 'test': []}
    for name, rows in [('test', raw_test), ('train', raw_train)]:
        for row in rows:
            key = row['sha256_pixels']
            if key in seen:
                previous = seen[key]
                if previous['transcript'] != row['transcript']:
                    raise ValueError(f"Image identique avec transcriptions contradictoires : {previous['file_name']} / {row['file_name']}")
                duplicates.append({'removed': row['file_name'], 'split': name, 'kept': previous['file_name']})
            else:
                seen[key] = row
                pools[name].append(row)
    if len(pools['train']) < 3:
        raise ValueError('Au moins 3 images train distinctes sont nécessaires.')
    train_count = getattr(args, 'train_count', None)
    candidates = pools['train'] if train_count is None else pools['train'] + pools['test']
    shuffled = sorted(candidates, key=lambda r: (r['sha256_pixels'], r['file_name']))
    random.Random(args.seed).shuffle(shuffled)
    if train_count is None:
        val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * args.val_fraction)))
        split = {'train': shuffled[val_count:], 'validation': shuffled[:val_count], 'test': pools['test']}
        split_policy = 'official test deduplicated; validation from official train'
    else:
        val_count = args.val_count
        if train_count < 1 or val_count < 1 or train_count + val_count >= len(shuffled):
            raise ValueError('Le total doit laisser au moins une ligne pour le test.')
        split = {'train': shuffled[:train_count], 'validation': shuffled[train_count:train_count + val_count], 'test': shuffled[train_count + val_count:]}
        split_policy = 'custom split from combined original train and test; not comparable to original test scores'
    invalid_ctc = []
    for row in split['train']:
        text = row['transcript']
        steps = max(4, round(row['width'] * 96 / row['height'])) // 4
        required = len(text) + sum((a == b for a, b in zip(text, text[1:])))
        if steps < required:
            invalid_ctc.append(row['file_name'])
    if invalid_ctc:
        raise ValueError('Images train trop étroites pour CTC à hauteur 96 : ' + repr(invalid_ctc))
    chars = sorted(set(''.join((row['transcript'] for row in split['train']))))
    vocab = {'<BLANK>': 0, '<UNK>': 1}
    vocab.update({char: index + 2 for index, char in enumerate(chars)})
    oov = {name: sorted(set(''.join((r['transcript'] for r in rows))) - set(chars)) for name, rows in split.items()}
    canonical = json.dumps(split, ensure_ascii=False, sort_keys=True).encode('utf-8')
    fingerprint = hashlib.sha256(canonical).hexdigest()
    output = Path(args.output) / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + fingerprint[:8])
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(dataset=str(root), output=str(output.resolve()), dataset_fingerprint=fingerprint, architecture='V7_CNN_BiLSTM256x2_CTC', height=96, views_per_image=1, dynamic_augmentation=True, augmentation_probability=0.55, rtl_mirror=True, split_policy=split_policy)
    report = {'counts': {name: len(rows) for name, rows in split.items()}, 'original_counts': {'train': len(raw_train), 'test': len(raw_test)}, 'duplicates_removed': duplicates, 'filename_corrections': corrections, 'unseen_characters': oov, 'vocab_size': len(vocab), 'dataset_fingerprint': fingerprint, 'note': 'Validation par lignes, pas par manuscrits. Les images identiques sont dédupliquées ; cela ne détecte pas les recadrages similaires.'}
    write_json(output / 'split.json', split)
    write_json(output / 'vocab.json', vocab)
    write_json(output / 'config.json', config)
    write_json(output / 'dataset_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('Résultats :', output.resolve(), flush=True)
    return (config, split, vocab)

def train_v7(config):
    import os
    import json
    import csv
    import time
    import random
    import unicodedata
    import io
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Sampler
    OLD_TRAIN_DIR = os.path.join(config['dataset'], 'data', 'train')
    OLD_TEST_DIR = os.path.join(config['dataset'], 'data', 'test')
    SPLIT_PATH = os.path.join(config['output'], 'split.json')
    VOCAB_PATH = os.path.join(config['output'], 'vocab.json')
    OUTPUT_DIR = config['output']
    BEST_CER_PATH = os.path.join(OUTPUT_DIR, 'best_cer_v7.pt')
    BEST_WER_PATH = os.path.join(OUTPUT_DIR, 'best_wer_v7.pt')
    LAST_PATH = os.path.join(OUTPUT_DIR, 'last_v7.pt')
    HISTORY_PATH = os.path.join(OUTPUT_DIR, 'history_v7.csv')
    HEIGHT = 96
    BATCH_SIZE = config['batch_size']
    BUCKET_SIZE = BATCH_SIZE * 8
    VIEWS_PER_IMAGE = 1
    AUGMENT_PROBABILITY = 0.55
    MAX_EPOCHS = config['epochs']
    INITIAL_LR = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_WORKERS = 0
    SEED = config['seed']
    VRAM_LIMIT_MB = 5500
    MIN_EPOCHS = 30
    PATIENCE = 15
    MIN_DELTA = 0.0005
    TREND_WINDOW = 10
    SLOPE_THRESHOLD = 0.001
    TRAIN_LOSS_THRESHOLD = 0.20
    WINDOW_GAIN_THRESHOLD = 0.008
    V8_CER = 0.2895
    V8_WER = 0.758
    V9_CER = 0.2808
    V9_WER = 0.742
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    BLANK_ID = vocab['<BLANK>']
    char_to_id = {c: i for c, i in vocab.items() if c != '<BLANK>'}
    id_to_char = {i: c for c, i in char_to_id.items()}
    id_to_char[vocab['<UNK>']] = '�'
    NUM_CLASSES = len(vocab)
    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split = json.load(f)
    train_entries = split['train']
    val_entries = split['validation']
    test_entries = split['test']
    assert len(train_entries) > 0
    assert len(val_entries) > 0
    assert len(test_entries) > 0

    def resolve_path(source, filename):
        if source == 'old_train':
            return os.path.join(OLD_TRAIN_DIR, filename)
        if source == 'old_test':
            return os.path.join(OLD_TEST_DIR, filename)
        raise ValueError('Source inconnue : ' + str(source))

    def augment_image(image, rng):
        if rng.random() < 0.4:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
        if rng.random() < 0.45:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.22))
        if rng.random() < 0.25:
            angle = rng.uniform(-1.25, 1.25)
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        morphology = rng.random()
        if morphology < 0.06:
            image = image.filter(ImageFilter.MinFilter(3))
        elif morphology < 0.12:
            image = image.filter(ImageFilter.MaxFilter(3))
        if rng.random() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.55)))
        return image

    class DynamicTrainDataset(Dataset):

        def __init__(self, entries):
            self.epoch = 0
            self.base_samples = []
            self.expected_widths = []
            for item in entries:
                source = item['source']
                filename = item['file_name']
                text = unicodedata.normalize('NFC', item['transcript'])
                path = resolve_path(source, filename)
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                unknown = [c for c in text if c not in char_to_id]
                if unknown:
                    raise RuntimeError(f'Caractères inconnus {filename}: {unknown}')
                with Image.open(path) as img:
                    w, h = img.size
                expected_width = max(4, round(w * HEIGHT / h))
                self.base_samples.append((path, text, filename, source, expected_width))
            for base_index in range(len(self.base_samples)):
                expected = self.base_samples[base_index][4]
                for _ in range(VIEWS_PER_IMAGE):
                    self.expected_widths.append(expected)

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(self.base_samples) * VIEWS_PER_IMAGE

        def __getitem__(self, index):
            base_index = index // VIEWS_PER_IMAGE
            variant = index % VIEWS_PER_IMAGE
            path, text, filename, source, _ = self.base_samples[base_index]
            image = Image.open(path).convert('L')
            rng = random.Random(SEED + self.epoch * 1000003 + base_index * 9176)
            variant = int(rng.random() < AUGMENT_PROBABILITY)
            if variant:
                image = augment_image(image, rng)
            w, h = image.size
            new_w = max(4, round(w * HEIGHT / h))
            image = image.resize((new_w, HEIGHT), Image.Resampling.LANCZOS)
            image = ImageOps.mirror(image)
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            data = data.reshape(HEIGHT, new_w).float()
            data /= 255.0
            data = (data - 0.5) / 0.5
            data = data.unsqueeze(0)
            target = torch.tensor([char_to_id.get(c, vocab['<UNK>']) for c in text], dtype=torch.long)
            return (data, target, text, new_w, filename, source, variant)

    class CleanDataset(Dataset):

        def __init__(self, entries):
            self.samples = []
            for item in entries:
                source = item['source']
                filename = item['file_name']
                text = unicodedata.normalize('NFC', item['transcript'])
                path = resolve_path(source, filename)
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                self.samples.append((path, text, filename, source))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, text, filename, source = self.samples[index]
            image = Image.open(path).convert('L')
            w, h = image.size
            new_w = max(4, round(w * HEIGHT / h))
            image = image.resize((new_w, HEIGHT), Image.Resampling.LANCZOS)
            image = ImageOps.mirror(image)
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            data = data.reshape(HEIGHT, new_w).float()
            data /= 255.0
            data = (data - 0.5) / 0.5
            data = data.unsqueeze(0)
            target = torch.tensor([char_to_id.get(c, vocab['<UNK>']) for c in text], dtype=torch.long)
            return (data, target, text, new_w, filename, source, 0)

    def collate_fn(batch):
        images = []
        targets = []
        texts = []
        widths = []
        filenames = []
        sources = []
        variants = []
        for image, target, text, width, filename, source, variant in batch:
            images.append(image)
            targets.append(target)
            texts.append(text)
            widths.append(width)
            filenames.append(filename)
            sources.append(source)
            variants.append(variant)
        max_width = max((x.shape[-1] for x in images))
        padded = []
        for image in images:
            pad = max_width - image.shape[-1]
            image = F.pad(image, (0, pad, 0, 0), value=1.0)
            padded.append(image)
        return (torch.stack(padded), torch.cat(targets), torch.tensor([len(t) for t in targets], dtype=torch.long), texts, torch.tensor(widths, dtype=torch.long), filenames, sources, variants)

    class WidthBucketSampler(Sampler):

        def __init__(self, dataset, batch_size, bucket_size, seed):
            self.dataset = dataset
            self.batch_size = batch_size
            self.bucket_size = bucket_size
            self.seed = seed
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

        def __iter__(self):
            rng = random.Random(self.seed + self.epoch)
            indices = list(range(len(self.dataset)))
            rng.shuffle(indices)
            batches = []
            for start in range(0, len(indices), self.bucket_size):
                bucket = indices[start:start + self.bucket_size]
                bucket.sort(key=lambda idx: self.dataset.expected_widths[idx])
                for pos in range(0, len(bucket), self.batch_size):
                    batch = bucket[pos:pos + self.batch_size]
                    if batch:
                        batches.append(batch)
            rng.shuffle(batches)
            for batch in batches:
                yield batch

    class CRNN(nn.Module):

        def __init__(self, num_classes):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, None)))
            self.rnn = nn.LSTM(input_size=384, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
            self.classifier = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.cnn(x)
            x = x.squeeze(2)
            x = x.permute(0, 2, 1)
            x, _ = self.rnn(x)
            return self.classifier(x)

    def output_lengths(widths):
        return widths // 2 // 2

    def decode_ctc(ids):
        chars = []
        previous = None
        for idx in ids:
            idx = int(idx)
            if idx != BLANK_ID and idx != previous:
                chars.append(id_to_char.get(idx, ''))
            previous = idx
        return ''.join(chars)

    def edit_distance(a, b):
        previous = list(range(len(b) + 1))
        for i, aa in enumerate(a, 1):
            current = [i]
            for j, bb in enumerate(b, 1):
                current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (aa != bb)))
            previous = current
        return previous[-1]

    @torch.no_grad()
    def evaluate(model, loader, max_examples=5):
        model.eval()
        char_errors = 0
        char_total = 0
        word_errors = 0
        word_total = 0
        exact = 0
        total_lines = 0
        examples = []
        for batch in loader:
            images, targets, target_lengths, texts, widths, filenames, sources, variants = batch
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=-1).cpu()
            lengths = output_lengths(widths)
            for i, reference in enumerate(texts):
                prediction = decode_ctc(predictions[i, :int(lengths[i])])
                reference = unicodedata.normalize('NFC', reference)
                prediction = unicodedata.normalize('NFC', prediction)
                char_errors += edit_distance(reference, prediction)
                char_total += len(reference)
                rw = reference.split()
                pw = prediction.split()
                word_errors += edit_distance(rw, pw)
                word_total += len(rw)
                if prediction == reference:
                    exact += 1
                total_lines += 1
                if len(examples) < max_examples:
                    examples.append((filenames[i], reference, prediction))
        return (char_errors / char_total, word_errors / word_total, exact / total_lines, examples)

    def linear_slope(values):
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = 0.0
        denominator = 0.0
        for i, value in enumerate(values):
            dx = i - x_mean
            numerator += dx * (value - y_mean)
            denominator += dx * dx
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def save_checkpoint(path, epoch, model, optimizer, val_cer, val_wer, val_exact):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_cer': val_cer, 'val_wer': val_wer, 'val_exact_rate': val_exact, 'version': 'V7_DYNAMIC_AUG', 'height': HEIGHT, 'batch_size': BATCH_SIZE, 'base_train_images': len(train_entries), 'views_per_image': VIEWS_PER_IMAGE, 'virtual_train_size': len(train_entries) * VIEWS_PER_IMAGE, 'dynamic_augmentation': True, 'augmentation_probability': AUGMENT_PROBABILITY, 'rtl_mirror': True, 'from_scratch': True, 'split': config['dataset_fingerprint'], 'vocab': vocab}, path)
    print('=' * 72)
    print('CREATION DATASET VIRTUEL V7')
    print('=' * 72)
    train_dataset = DynamicTrainDataset(train_entries)
    val_dataset = CleanDataset(val_entries)
    test_dataset = CleanDataset(test_entries)
    train_sampler = WidthBucketSampler(train_dataset, BATCH_SIZE, BUCKET_SIZE, SEED)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=BLANK_ID, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-06)
    print()
    print('=' * 72)
    print('HAUSA CTC V7 - DYNAMIC AUGMENTATION')
    print('=' * 72)
    print('Images reelles TRAIN :', len(train_entries))
    print('Vues / image         :', VIEWS_PER_IMAGE)
    print('TRAIN virtuel        :', len(train_dataset))
    print('Validation           :', len(val_dataset))
    print('Test final           :', len(test_dataset))
    print('Height               :', HEIGHT)
    print('Batch                :', BATCH_SIZE)
    print('Aug dynamique        : 55 % des lignes TRAIN')
    print('Initialisation       : FROM SCRATCH')
    print()
    print('Ancien V9 TEST100 (autre jeu de test), CER :', V9_CER)
    print('Ancien V9 TEST100 (autre jeu de test), WER :', V9_WER)
    print('=' * 72)
    with open(HISTORY_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_cer', 'val_wer', 'val_exact', 'slope', 'window_gain', 'lr', 'seconds', 'vram_mb'])
    best_cer = float('inf')
    best_wer = float('inf')
    best_cer_epoch = 0
    best_wer_epoch = 0
    best_for_patience = float('inf')
    without_improvement = 0
    cer_history = []
    start_total = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f'\nDEBUT EPOCH {epoch}/{MAX_EPOCHS} - {len(train_loader)} batches', flush=True)
        train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        batches = 0
        start_epoch = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            images, targets, target_lengths, texts, widths, filenames, sources, variants = batch
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)
            input_lengths = output_lengths(widths).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item()
            batches += 1
            if batches == 1 or batches % 20 == 0 or batches == len(train_loader):
                elapsed = time.time() - start_epoch
                remaining = elapsed / batches * (len(train_loader) - batches)
                print(f'Epoch {epoch}/{MAX_EPOCHS} | batch {batches}/{len(train_loader)} | loss {loss.item():.4f} | temps {elapsed:.0f}s | restant estime {remaining / 60:.1f} min', flush=True)
        train_loss = running_loss / batches
        print('TRAIN termine - validation en cours...', flush=True)
        val_cer, val_wer, val_exact, examples = evaluate(model, val_loader, 5)
        cer_history.append(val_cer)
        scheduler.step(val_cer)
        lr = optimizer.param_groups[0]['lr']
        if len(cer_history) >= TREND_WINDOW:
            recent = cer_history[-TREND_WINDOW:]
            slope = linear_slope(recent)
            window_gain = recent[0] - min(recent)
        else:
            slope = 0.0
            window_gain = float('inf')
        if val_cer < best_cer:
            best_cer = val_cer
            best_cer_epoch = epoch
            save_checkpoint(BEST_CER_PATH, epoch, model, optimizer, val_cer, val_wer, val_exact)
            print()
            print('>>> NOUVEAU BEST CER')
        if val_wer < best_wer:
            best_wer = val_wer
            best_wer_epoch = epoch
            save_checkpoint(BEST_WER_PATH, epoch, model, optimizer, val_cer, val_wer, val_exact)
            print()
            print('>>> NOUVEAU BEST WER')
        if val_cer < best_for_patience - MIN_DELTA:
            best_for_patience = val_cer
            without_improvement = 0
        else:
            without_improvement += 1
        seconds = time.time() - start_epoch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            vram_mb = 0.0
        print()
        print('=' * 72)
        print(f'EPOCH {epoch:03d}/{MAX_EPOCHS}')
        print('=' * 72)
        print(f'Loss train  : {train_loss:.4f}')
        print(f'VAL CER     : {val_cer:.4f}')
        print(f'VAL WER     : {val_wer:.4f}')
        print(f'VAL exact   : {val_exact:.4f}')
        if len(cer_history) >= TREND_WINDOW:
            print(f'Pente CER   : {slope:+.6f}/epoch')
            print(f'Gain fenetre: {window_gain:.4f}')
        else:
            print('Pente CER   : attente')
        print(f'LR          : {lr:.8f}')
        print(f'Temps       : {seconds:.1f} sec')
        print(f'VRAM alloc  : {vram_mb:.0f} MB')
        print('Vues epoch  :', len(train_dataset))
        print('Equivalent  :', f'{VIEWS_PER_IMAGE} passages/reelle')
        if vram_mb > VRAM_LIMIT_MB:
            print()
            print('!!! ALERTE VRAM !!!')
        for filename, reference, prediction in examples:
            print()
            print('FILE:', filename)
            print('REF :', reference)
            print('PRED:', prediction)
        print()
        print('Best CER    :', round(best_cer, 4), '@', best_cer_epoch)
        print('Best WER    :', round(best_wer, 4), '@', best_wer_epoch)
        print('Sans gain   :', without_improvement, '/', PATIENCE)
        with open(HISTORY_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, train_loss, val_cer, val_wer, val_exact, slope, window_gain, lr, seconds, vram_mb])
        save_checkpoint(LAST_PATH, epoch, model, optimizer, val_cer, val_wer, val_exact)
        saturation = epoch >= MIN_EPOCHS and len(cer_history) >= TREND_WINDOW and (abs(slope) < SLOPE_THRESHOLD) and (window_gain < WINDOW_GAIN_THRESHOLD) and (train_loss < TRAIN_LOSS_THRESHOLD) and (without_improvement >= PATIENCE)
        if saturation:
            print()
            print('=' * 72)
            print('SATURATION DETECTEE')
            print('=' * 72)
            break
    print()
    print('=' * 72)
    print('TEST FINAL TEST NOUVEAU')
    print('=' * 72)
    checkpoint = torch.load(BEST_CER_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_cer, test_wer, test_exact, examples = evaluate(model, test_loader, 10)
    minutes = (time.time() - start_total) / 60
    print()
    print('=' * 72)
    print('RESUME FINAL V7')
    print('=' * 72)
    print('BEST epoch       :', checkpoint['epoch'])
    print('BEST VAL CER     :', round(checkpoint['val_cer'], 4))
    print('BEST VAL WER     :', round(checkpoint['val_wer'], 4))
    print()
    print('TEST NOUVEAU CER      :', round(test_cer, 4))
    print('TEST NOUVEAU WER      :', round(test_wer, 4))
    print('TEST NOUVEAU exact    :', round(test_exact, 4))
    print()
    print('-' * 72)
    print('Nouveau test : aucune comparaison directe avec les anciens CER V8/V9.')
    print('Temps total     :', round(minutes, 1), 'minutes')
    print()
    print('Best CER :', BEST_CER_PATH)
    print('Best WER :', BEST_WER_PATH)
    print('History  :', HISTORY_PATH)
    print()
    print('=' * 72)
    print('EXEMPLES TEST NOUVEAU')
    print('=' * 72)
    for filename, reference, prediction in examples:
        print()
        print('FILE:', filename)
        print('REF :', reference)
        print('PRED:', prediction)
    write_json(Path(OUTPUT_DIR) / 'test_metrics.json', {'cer': test_cer, 'wer': test_wer, 'exact_rate': test_exact, 'selected_epoch': checkpoint['epoch'], 'dataset_fingerprint': config['dataset_fingerprint'], 'selection': 'minimum validation CER'})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='C:\\ai-test\\Hausa_repo_nouveau')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent / 'resultats_v7_nouveau'))
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--train-count', type=int, default=3000)
    parser.add_argument('--val-count', type=int, default=100)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error('Paramètres invalides : vérifier val-fraction, batch-size et epochs.')
    config, split, vocab = prepare(args)
    print('V7 :', len(split['train']), 'lignes par epoch, augmentation dynamique 55 %', flush=True)
    if not args.prepare_only:
        train_v7(config)
if __name__ == '__main__':
    main()
