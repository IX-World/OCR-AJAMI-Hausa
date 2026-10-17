"""IX-World: verified context and train-only semantic supervision for V13."""
import hashlib
import json
import math
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

SPLIT_SHA256 = "fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5"
VOCAB_SHA256 = "f92c368520aad95130f909a708fb58864e1f766f10f87d18c96690d1b20ced59"
COUNTS = {"train": 4972, "validation": 200, "test": 401}
TEACHER_REPO = "castorini/afriberta_base"
TEACHER_REVISION = "95b703f498c9cee56be5f5bbc5140022dc86099e"
TEACHER_WEIGHTS_SHA256 = "a1b5507320b2175f5aa01cfbb9cd853fdf92dee491dde2359eae8919eb583a8a"
TEACHER_LANGUAGES = {"om", "am", "rw", "rn", "ha", "ig", "pcm", "so", "sw", "ti", "yo"}
CONTEXT_FIELDS = {"language", "locality", "year_start", "year_end", "evidence"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sample_id(row):
    return row["partition"] + "/" + row["file_name"]


def require_frozen_split(report):
    if report["split_sha256"] != SPLIT_SHA256 or report["vocab_sha256"] != VOCAB_SHA256 or report["counts"] != COUNTS:
        raise ValueError("V13 exige exactement le split commun 4972/200/401 et son vocabulaire archive.")
    if not report.get("all_usable_images_used") or report.get("separation_unit") != "source_and_known_scribe":
        raise ValueError("Le protocole commun par source et copiste identifie est obligatoire.")


def validate_context(value):
    if set(value) - CONTEXT_FIELDS:
        raise ValueError(f"Champs de contexte inconnus : {set(value) - CONTEXT_FIELDS}")
    result = dict(value)
    for field in ("language", "locality"):
        text = result.get(field)
        if text is not None and (not isinstance(text, str) or not text.strip() or len(text) > 160):
            raise ValueError(f"Valeur de contexte incorrecte : {field}")
        result[field] = unicodedata.normalize("NFC", text.strip()) if text is not None else None
    if result["language"] is not None and result["language"] not in TEACHER_LANGUAGES:
        raise ValueError(f"Langue non prise en charge par l'encodeur AfriBERTa : {result['language']}")
    start, end = result.get("year_start"), result.get("year_end")
    if (start is None) != (end is None):
        raise ValueError("Une periode exige year_start et year_end, ou deux valeurs null.")
    if start is not None and (type(start) is not int or type(end) is not int or not 1 <= start <= end <= 2100):
        raise ValueError("Periode invalide : annees civiles de 1 a 2100, debut <= fin.")
    result.update(year_start=start, year_end=end)
    if not isinstance(result.get("evidence"), str) or not result["evidence"].strip():
        raise ValueError("Chaque contexte doit citer sa provenance dans evidence.")
    return result


def load_context(path, sources):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("calendar") != "CE":
        raise ValueError("Contexte attendu : schema_version=1, calendar=CE.")
    defaults = validate_context(raw["defaults"])
    if not isinstance(raw.get("sources"), dict) or set(raw["sources"]) - set(sources):
        raise ValueError("Identifiant de source inconnu dans le fichier de contexte.")
    resolved = {}
    for source in sorted(sources):
        override = raw["sources"].get(source, {})
        if override and not override.get("evidence"):
            raise ValueError(f"Une modification du contexte de source {source} exige sa propre evidence.")
        resolved[source] = validate_context(dict(defaults, **override))
    return raw, resolved


def load_annotations(path, train_rows):
    if path is None:
        return {}
    allowed = {sample_id(row) for row in train_rows}
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) - {"sample_id", "verified", "evidence", "context_before", "context_after"}:
            raise ValueError("Annotation semantique : champ inattendu.")
        key = record.get("sample_id")
        if key not in allowed or key in result:
            raise ValueError("Seules des annotations uniques du TRAIN sont autorisees ; validation et test interdits.")
        if record.get("verified") is not True or not isinstance(record.get("evidence"), str) or not record["evidence"].strip():
            raise ValueError("Le contexte textuel doit etre verifie et sa provenance documentee.")
        for field in ("context_before", "context_after"):
            text = record.get(field, "")
            if not isinstance(text, str) or len(text) > 4000:
                raise ValueError("Contexte textuel invalide.")
            record[field] = unicodedata.normalize("NFC", text)
        result[key] = record
    return result


def fit_context_schema(train_rows, contexts):
    schema = {"schema_version": 1, "languages": [None], "localities": [None]}
    for field, plural in (("language", "languages"), ("locality", "localities")):
        schema[plural] += sorted({contexts[str(row["source"])][field] for row in train_rows} - {None})
    periods = [(contexts[str(row["source"])]["year_start"], contexts[str(row["source"])]["year_end"]) for row in train_rows if contexts[str(row["source"])]["year_start"] is not None]
    schema["time_available"] = bool(periods)
    schema["training_period_range"] = [min(p[0] for p in periods), max(p[1] for p in periods)] if periods else None
    return schema


def context_values(rows, contexts, schema):
    language, locality, time = [], [], []
    for row in rows:
        ctx = contexts[str(row["source"])]
        language.append(schema["languages"].index(ctx["language"]) if ctx["language"] in schema["languages"] else 0)
        locality.append(schema["localities"].index(ctx["locality"]) if ctx["locality"] in schema["localities"] else 0)
        if ctx["year_start"] is None or not schema.get("time_available", False):
            time.append([0.0, 0.0, 0.0])
        else:
            midpoint = (ctx["year_start"] + ctx["year_end"]) / 2
            time.append([1.0, (midpoint - 1800) / 500, math.log1p(ctx["year_end"] - ctx["year_start"]) / math.log(2101)])
    return {"language": torch.tensor(language, dtype=torch.long), "locality": torch.tensor(locality, dtype=torch.long), "time": torch.tensor(time, dtype=torch.float32)}


def coverage(rows, contexts):
    return {"examples": len(rows), "known_language": sum(contexts[str(r["source"])]["language"] is not None for r in rows),
            "known_period": sum(contexts[str(r["source"])]["year_start"] is not None for r in rows),
            "known_locality": sum(contexts[str(r["source"])]["locality"] is not None for r in rows)}


def text_input(row, annotation=None):
    annotation = annotation or {}
    before, after = annotation.get("context_before", ""), annotation.get("context_after", "")
    prefix = before + " " if before else ""
    reference = row["transcript"]
    text = prefix + reference + (" " + after if after else "")
    return text, len(prefix), len(prefix) + len(reference)


def semantic_loss(visual, target, references, languages, temperature=0.1, positive_keys=None):
    """Cosine alignment plus same-language, multi-positive contrastive loss.

    Identical teacher inputs (including augmented views) are positives. The
    same spelling with another verified context is excluded from contrastive
    negatives, and retains its own contextual target for cosine alignment.
    """
    valid = languages > 0
    if not bool(valid.any()):
        zero = visual.sum() * 0
        return zero, zero, zero
    visual, target = F.normalize(visual[valid], dim=-1), F.normalize(target[valid], dim=-1)
    indices = valid.nonzero().flatten().tolist()
    labels = [unicodedata.normalize("NFC", references[i]) for i in indices]
    keys = [positive_keys[i] for i in indices] if positive_keys is not None else labels
    language = languages[valid]
    same_language = language[:, None] == language[None, :]
    same_spelling = torch.tensor([[a == b for b in labels] for a in labels], device=visual.device)
    positive = torch.tensor([[a == b for b in keys] for a in keys], device=visual.device) & same_language
    allowed = same_language & (~same_spelling | positive)
    similarity = visual @ target.T / temperature
    denominator = similarity.masked_fill(~allowed, -torch.inf).logsumexp(1)
    numerator = similarity.masked_fill(~positive, -torch.inf).logsumexp(1)
    contrastive = (denominator - numerator).mean()
    cosine = (1 - (visual * target).sum(-1)).mean()
    return 0.5 * (cosine + contrastive), cosine, contrastive
