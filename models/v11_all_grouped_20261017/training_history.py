"""Publish the recorded epoch measurements without rerunning training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


UNITS = {
    "epoch": "index starting at 1",
    "train_loss": "arithmetic mean of batch CTC losses during augmented training",
    "val_cer": "character error ratio, not a percentage",
    "val_wer": "word error ratio, not a percentage",
    "val_uwer": "word error ratio after case and diacritic normalization",
    "val_exact_rate": "exact segment ratio",
    "lr_used": "learning rate used during this epoch",
    "lr": "learning rate after scheduler.step, for the next epoch",
    "seconds": "training and validation wall time, before checkpoint saving",
    "vram_allocated_mib": "CUDA memory allocated at the end of validation, MiB",
    "epoch_peak_vram_mib": "peak CUDA allocated memory during this epoch, MiB",
    "peak_vram_mib": "maximum of recorded epoch peaks so far, MiB",
    "val_cer_slope": "CER ratio per epoch, regression over at most five recent epochs",
    "val_cer_window_gain": "first minus last CER ratio over at most five recent epochs",
}


def read_epoch_history(run):
    run = Path(run)
    selected = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != set(UNITS):
            raise ValueError("Colonnes de l'historique non reconnues ; aucune valeur reconstruite.")
        records = []
        for row in reader:
            record = {"epoch": int(row["epoch"])}
            for name in UNITS:
                if name == "epoch":
                    continue
                optional = name in ("val_cer_slope", "val_cer_window_gain")
                value = None if optional and not row[name] else float(row[name])
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"Valeur non finie dans l'historique : {name}")
                record[name] = value
            records.append(record)
    if not records or [r["epoch"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("L'historique doit contenir une ligne par epoque, sans trou ni doublon.")
    best = min(records, key=lambda r: r["val_cer"])
    if best["epoch"] != selected["selected_epoch"]:
        raise ValueError("Le checkpoint publie ne correspond pas au minimum CER de l'historique.")
    recorded = selected.get("validation_at_selected_epoch")
    if recorded:
        for field in ("cer", "wer", "uwer", "exact_rate"):
            if abs(best[f"val_{field}"] - recorded[field]) > 1e-12:
                raise ValueError(f"Historique et checkpoint selectionne incoherents : {field}")
    return records, selected


def plot_history(records, selected_epoch, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in records]
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
        axes = axes.ravel()
        axes[0].plot(epochs, [r["train_loss"] for r in records], color="#175c96")
        axes[0].set(title="Loss CTC du train (avec augmentation)", ylabel="Loss moyenne par batch", ylim=(0, None))
        for field, label, color in (("val_cer", "CER", "#175c96"), ("val_wer", "WER", "#bc3034"), ("val_uwer", "UWER", "#7951a0")):
            axes[1].plot(epochs, [100 * r[field] for r in records], label=label, color=color)
        axes[1].set(title="Erreurs sur la validation", ylabel="Taux d'erreur (%)", ylim=(0, max(100, 105 * max(r[f] for r in records for f in ("val_cer", "val_wer", "val_uwer")))))
        axes[1].legend(loc="lower left")
        axes[2].plot(epochs, [100 * r["val_exact_rate"] for r in records], color="#22754b")
        axes[2].set(title="Segments exactement reconnus : validation", ylabel="Segments exacts (%)", ylim=(0, 100))
        axes[3].step(epochs, [r["lr_used"] for r in records], where="post", color="#ad641c", label="LR utilisé")
        axes[3].set(title="Learning rate utilisé pendant l'époque", ylabel="LR (échelle logarithmique)", yscale="log")
        axes[4].plot(epochs, [r["seconds"] for r in records], color="#175c96")
        axes[4].set(title="Durée de chaque époque : train + validation", ylabel="Secondes", ylim=(0, None))
        axes[5].plot(epochs, [r["epoch_peak_vram_mib"] for r in records], color="#7951a0", label="Pic de l'époque")
        axes[5].plot(epochs, [r["peak_vram_mib"] for r in records], color="#bc3034", linestyle=":", label="Pic cumulé")
        axes[5].set(title="Mémoire CUDA allouée : pics enregistrés", ylabel="MiB", ylim=(0, None))
        axes[5].legend(loc="lower left")
        for ax in axes:
            ax.axvline(selected_epoch, color="#333333", linestyle="--", linewidth=1)
            ax.set(xlabel="Époque", xlim=(1, max(2, epochs[-1])))
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"Historique enregistré — {len(records)} époques\nTrait pointillé : checkpoint sélectionné sur le CER de validation à l'époque {selected_epoch}", fontsize=14)
        fig.savefig(output / "training_curves.png", dpi=180, metadata={"Software": f"Matplotlib {matplotlib.__version__}"})
        fig.savefig(output / "training_curves.svg", metadata={"Date": None})
        plt.close(fig)


def write_epoch_report(run, output=None, plots=False):
    run = Path(run)
    output = Path(output) if output else run
    output.mkdir(parents=True, exist_ok=True)
    records, selected = read_epoch_history(run)
    if plots:
        plot_history(records, selected["selected_epoch"], output)
    history_hash = hashlib.sha256((run / "history.csv").read_bytes()).hexdigest()
    summary = {
        "source": "history.csv", "history_sha256": history_hash,
        "version": selected.get("version"), "dataset_revision": selected.get("dataset_revision"),
        "report_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "split_sha256": selected["split_sha256"], "selected_epoch": selected["selected_epoch"],
        "completed_epochs": len(records), "units": UNITS, "epochs": records,
    }
    (output / "epochs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = "# Valeurs enregistrées à chaque époque\n\n"
    text += f"**{len(records)} époques exécutées ; checkpoint publié : époque {selected['selected_epoch']}, minimum du CER de validation.**\n\n"
    text += "[CSV brut, précision d'origine](history.csv) · [JSON avec unités et provenance](epochs.json)\n\n"
    text += "La loss du train est la moyenne des pertes CTC des batches pendant l'apprentissage avec augmentation. Les taux CER/WER/UWER et exact ci-dessous concernent uniquement la validation, sur images originales. Les CER/WER/UWER du train et du test ne sont pas enregistrés à chaque époque : ils figurent dans [metrics.csv](metrics.csv), pour le checkpoint sélectionné. Aucun score manquant n'a été reconstruit.\n\n"
    if (output / "training_curves.png").is_file():
        text += "![Courbes des mesures enregistrées par époque](training_curves.png)\n\n[Télécharger les courbes vectorielles SVG](training_curves.svg). Courbes brutes sans lissage ; le trait pointillé indique le checkpoint sélectionné.\n\n"
    text += "## Optima observés sur la validation\n\n| Mesure | Meilleure valeur | Première époque correspondante |\n|---|---:|---:|\n"
    for field, label in (("val_cer", "CER"), ("val_wer", "WER"), ("val_uwer", "UWER"), ("val_exact_rate", "Exact")):
        best = (max if field == "val_exact_rate" else min)(records, key=lambda r: r[field])
        text += f"| {label} | {best[field]:.4%} | {best['epoch']} |\n"
    text += "\nSeul le minimum du CER détermine le checkpoint publié ; les autres optima ne désignent pas d'autres poids conservés. Ces scores de validation interviennent dans la sélection et ne constituent pas des résultats de test indépendants.\n\n"
    text += "## Apprentissage et validation — toutes les époques\n\nLes taux sont exprimés en pourcentage ; les valeurs exactes non arrondies sont dans le CSV et le JSON. LR utilisé = apprentissage de cette époque ; LR suivant = valeur après le scheduler.\n\n"
    text += "| Époque | Loss train | VAL CER % | VAL WER % | VAL UWER % | VAL exact % | LR utilisé | LR suivant | Temps (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in records:
        label = f"**{r['epoch']} retenue**" if r["epoch"] == selected["selected_epoch"] else str(r["epoch"])
        text += f"| {label} | {r['train_loss']:.6f} | {r['val_cer']*100:.4f} | {r['val_wer']*100:.4f} | {r['val_uwer']*100:.4f} | {r['val_exact_rate']*100:.4f} | {r['lr_used']:.8g} | {r['lr']:.8g} | {r['seconds']:.3f} |\n"
    text += "\n## Mémoire et évolution du CER\n\nMémoire en MiB : allocation mesurée en fin de validation, pic de l'époque et pic cumulé. Il s'agit de mémoire allouée par PyTorch, pas de toute la mémoire réservée ou occupée sur la carte. La pente est une régression sur au plus cinq époques récentes ; le gain est le CER du début moins celui de la fin de cette fenêtre. Une pente négative ou un gain positif indique une amélioration sur cette fenêtre.\n\n"
    text += "| Époque | VRAM fin (MiB) | Pic époque (MiB) | Pic cumulé (MiB) | Pente CER (points/époque) | Gain CER (points) |\n|---|---:|---:|---:|---:|---:|\n"
    for r in records:
        slope = "n/a" if r["val_cer_slope"] is None else f"{r['val_cer_slope']*100:+.6f}"
        gain = "n/a" if r["val_cer_window_gain"] is None else f"{r['val_cer_window_gain']*100:+.6f}"
        text += f"| {r['epoch']} | {r['vram_allocated_mib']:.3f} | {r['epoch_peak_vram_mib']:.3f} | {r['peak_vram_mib']:.3f} | {slope} | {gain} |\n"
    text += f"\nSHA-256 du CSV d'origine : `{history_hash}`. Le JSON conserve les valeurs numériques du journal, avec `null` pour les mesures non disponibles. Les temps incluent train et validation, avant la sauvegarde du checkpoint ; ils n'incluent pas le bilan final sur les trois ensembles. Une seule graine est mesurée ; les comparaisons entre versions exigent le même protocole.\n"
    (output / "EPOCHS.md").write_text(text, encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporter les vraies mesures par époque en Markdown/JSON et courbes optionnelles.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots", action="store_true", help="Nécessite Matplotlib, uniquement pour la génération des courbes.")
    args = parser.parse_args()
    summary = write_epoch_report(args.run, args.output, args.plots)
    print(f"Historique exporté : {summary['completed_epochs']} époques, checkpoint {summary['selected_epoch']}.")
