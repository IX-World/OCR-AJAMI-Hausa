import csv


def cer_trend(records, window=5):
    recent = records[-window:]
    if len(recent) < 2:
        return None, None
    x = [int(row["epoch"]) for row in recent]
    y = [float(row["val_cer"]) for row in recent]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator
    return slope, y[0] - y[-1]


def append_history(path, records, record):
    records.append(record)
    fields = list(record)
    rewrite = path.exists() and list(records[0]) != fields
    with path.open("w" if rewrite else "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if rewrite or handle.tell() == 0:
            writer.writeheader()
        writer.writerows(records if rewrite else [record])


def print_epoch_report(config, record, records, predictions, without_gain, show_examples=5):
    slope, gain = cer_trend(records)
    slope_text = f"{slope:+.6f}/epoch" if slope is not None else "n/a (premiere epoque)"
    gain_text = f"{gain:+.4f}" if gain is not None else "n/a"
    print("\n" + "=" * 64)
    print(f"{config['version'].upper()} | EPOCH {record['epoch']:03d}/{config['max_epochs']}")
    print("=" * 64)
    for label, value in (
        ("Loss train", f"{record['train_loss']:.4f}"),
        ("VAL CER", f"{record['val_cer']:.4f} ({record['val_cer']:.2%})"),
        ("VAL WER", f"{record['val_wer']:.4f} ({record['val_wer']:.2%})"),
        ("VAL UWER", f"{record['val_uwer']:.4f} ({record['val_uwer']:.2%})"),
        ("VAL exact", f"{record['val_exact_rate']:.4f}"),
        ("Pente CER (5)", slope_text),
        ("Gain fenetre (5)", gain_text),
        ("LR epoch / suivant", f"{record['lr_used']:.8f} / {record['lr']:.8f}"),
        ("Temps", f"{record['seconds']:.1f} sec"),
        ("VRAM allouee", f"{record['vram_allocated_mib']:.0f} MiB"),
        ("Pic VRAM epoch", f"{record['epoch_peak_vram_mib']:.0f} MiB"),
        ("Canaux", config['channels']),
        ("CNN conv", config['cnn_convolutions']),
        ("Train reel", config['counts']['train']),
        ("Vues epoch", config['virtual_train_size']),
    ):
        print(f"{label:20s}: {value}")
    for row in predictions[:show_examples]:
        print(f"\nFILE: {row['file_name']}\nREF : {row['reference']}\nPRED: {row['prediction']}")
    for field, label in (("val_cer", "Best CER"), ("val_wer", "Best WER"), ("val_uwer", "Best UWER")):
        measured = [row for row in records if row.get(field) not in (None, "")]
        best = min(measured, key=lambda row: float(row[field]))
        print(f"{label:20s}: {float(best[field]):.4f} @ {best['epoch']}")
    print(f"{'Sans gain CER':20s}: {without_gain} / {config['patience']}", flush=True)
