import copy
import math
from typing import Dict, List, Tuple, Iterable, Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------
# --- Original Config (importable) ---------------------------------
# ------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"

PROFESSIONS_DATA = {
    "neutral": [
        "accountant", "analyst", "architect", "artist", "assistant", "auditor",
        "baker", "chef", "cleaner", "clerk", "designer", "developer", "director",
        "doctor", "economist", "editor", "engineer", "farmer", "firefighter",
        "florist", "journalist", "judge", "lawyer", "librarian", "manager",
        "musician", "painter", "photographer", "physician", "pilot", "plumber",
        "police officer", "professor", "programmer", "receptionist", "reporter",
        "researcher", "salesperson", "scientist", "secretary", "singer", "student",
        "supervisor", "teacher", "technician", "therapist", "writer"
    ],
    "male_stereotypical": [
        "builder", "carpenter", "electrician", "mechanic", "soldier", "trucker",
        "miner", "bodyguard", "guard", "construction worker", "commander", "mechanic"
    ],
    "female_stereotypical": [
        "nurse", "preschooler", "nanny", "maid", "stylist", "housekeeper",
        "seamstress", "model", "flight attendant", "secretary", "kindergarten teacher"
    ]
}

# ------------------------------------------------------------------
# --- Model Load ---------------------------------------------------
# ------------------------------------------------------------------

def load_model(model_name: str):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    print("Model loaded.")
    return tokenizer, model

# ------------------------------------------------------------------
# --- Utility: Single Prompt Masked Predictions --------------------
# ------------------------------------------------------------------

def _forward_mask_logits(text_template: str, mask_token: str, tokenizer, model) -> torch.Tensor:
    """Internal helper: return logits vector [vocab] at mask position for a *single* template."""
    input_text = text_template.replace("{mask}", mask_token)
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # [batch, seq, vocab]
    mask_idx = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    mask_logits = logits[0, mask_idx, :].squeeze(0)  # [vocab]
    return mask_logits.cpu()  # move back to cpu for downstream ops


def get_masked_word_predictions(text_template: str, mask_token: str, tokenizer, model, top_k: int = 50) -> List[str]:
    mask_logits = _forward_mask_logits(text_template, mask_token, tokenizer, model)
    topk = torch.topk(mask_logits, top_k)
    ids = topk.indices.tolist()
    decoded = [tokenizer.decode([i]).strip() for i in ids]
    return decoded

# ------------------------------------------------------------------
# --- Counterfactual Logit Averaging -------------------------------
# ------------------------------------------------------------------

def counterfactual_masked_predictions(
    templates: Iterable[str],
    tokenizer,
    model,
    top_k: int = 50,
    weight_by: Optional[Iterable[float]] = None,
) -> List[str]:
    """Inference‑time debias: average mask logits across multiple *counterfactual* prompts.

    Parameters
    ----------
    templates: iterable of string templates containing `{mask}` placeholder.
    weight_by: optional per‑template weights (will be normalized).

    Returns top_k decoded tokens after averaging logits.
    """
    mask_token = tokenizer.mask_token
    logits_list = []
    for t in templates:
        logits_list.append(_forward_mask_logits(t, mask_token, tokenizer, model))
    stacked = torch.stack(logits_list, dim=0)  # [T, vocab]
    if weight_by is not None:
        w = torch.tensor(list(weight_by), dtype=stacked.dtype)
        w = w / w.sum()
        avg_logits = (stacked * w.unsqueeze(1)).sum(0)
    else:
        avg_logits = stacked.mean(0)
    topk = torch.topk(avg_logits, top_k)
    ids = topk.indices.tolist()
    decoded = [tokenizer.decode([i]).strip() for i in ids]
    return decoded

# ------------------------------------------------------------------
# --- Embedding Gender Direction Projection -----------------------
# ------------------------------------------------------------------

_DEF_GENDER_PAIRS = [
    ("he", "she"),
    ("him", "her"),
    ("his", "hers"),
    ("man", "woman"),
    ("men", "women"),
    ("male", "female"),
    ("boy", "girl"),
    ("father", "mother"),
    ("son", "daughter"),
]


def _get_token_vec(tokenizer, embedding_weight, word: str) -> Optional[torch.Tensor]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    # Only keep *single token* words; otherwise skip (simplification)
    if len(ids) != 1:
        return None
    return embedding_weight[ids[0]].detach().clone()


def _gender_direction(tokenizer, embedding_weight, gender_pairs=_DEF_GENDER_PAIRS) -> torch.Tensor:
    """Compute the principal gender direction using definitional word pairs.

    We collect difference vectors (he - she, man - woman, ...) for all pairs that map to
    *single tokens*, then take the first principal component (SVD) as the gender subspace.
    """
    diffs = []
    for a, b in gender_pairs:
        va = _get_token_vec(tokenizer, embedding_weight, a)
        vb = _get_token_vec(tokenizer, embedding_weight, b)
        if va is None or vb is None:
            continue
        diffs.append((va - vb).unsqueeze(0))
        diffs.append((vb - va).unsqueeze(0))  # symmetric
    if not diffs:
        raise ValueError("No usable gender pairs mapped to single tokens.")
    mat = torch.cat(diffs, dim=0)  # [N, dim]
    # Center
    mat = mat - mat.mean(0, keepdim=True)
    # SVD to get principal component
    # mat: [N, d]; PCA ~ top right singular vector
    u, s, vT = torch.pca_lowrank(mat, q=1)  # returns principal comps in V
    gender_dir = vT[:, 0]  # [dim]
    gender_dir = gender_dir / gender_dir.norm(p=2)
    return gender_dir


def _project_orthogonal(vecs: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project vectors orthogonal to direction."""
    # vecs: [n, d]; direction: [d]
    proj = (vecs @ direction)[:, None] * direction[None, :]
    return vecs - proj


def build_debiased_model_copy(tokenizer, model, gender_pairs=_DEF_GENDER_PAIRS):
    """Return a *deep copy* of the model with token embeddings projected to remove gender direction.

    This is a lightweight adaptation of Hard Debias: neutralize all tokens by removing the
    principal gender component from the embedding matrix. (No equalization step here; you
    can extend if needed.)

    Limitations:
    * Works best for models with `get_input_embeddings()` returning a standard embedding.
    * Only neutralizes *input* embeddings; tied output head may or may not reflect change.
    * May slightly degrade perplexity / MLM accuracy.
    """
    deb_model = copy.deepcopy(model)
    emb_layer = deb_model.get_input_embeddings()
    weight = emb_layer.weight.data  # [vocab, dim]
    direction = _gender_direction(tokenizer, weight, gender_pairs=gender_pairs)  # [dim]
    # Project all token embeddings
    new_weight = _project_orthogonal(weight, direction)
    emb_layer.weight.data = new_weight
    # If output embeddings are tied (as in most HF MLMs), update those too.
    if hasattr(deb_model, "get_output_embeddings"):
        out_emb = deb_model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data.data_ptr() != emb_layer.weight.data.data_ptr():
            out_emb.weight.data = _project_orthogonal(out_emb.weight.data, direction)
    deb_model.eval()
    return deb_model

# ------------------------------------------------------------------
# --- Category Reweighting ----------------------------------------
# ------------------------------------------------------------------

def _token_to_profession_category(token: str, professions_data: Dict[str, List[str]]) -> str:
    tok = token.lower().strip()
    # quick normalization of common BERT subword suffix artifacts
    tok = tok.replace(" ##s", "s").replace(" ##er", "er").replace(" ##or", "or")
    for cat, plist in professions_data.items():
        if tok in plist:
            return cat
    return "unlisted"


def category_reweighted_predictions(
    template: str,
    tokenizer,
    model,
    professions_data: Dict[str, List[str]],
    category_penalty: Dict[str, float],
    top_k: int = 50,
) -> List[Tuple[str, float]]:
    """Decode predictions with *per‑category multiplicative penalties* applied to softmax probs.

    Example: `{"male_stereotypical": 0.5, "female_stereotypical": 0.5}` halves the probability of
    stereotypical categories, letting neutral/unlisted rise in rank.

    Returns [(token, prob), ...] top_k after reweight.
    """
    mask_token = tokenizer.mask_token
    logits = _forward_mask_logits(template, mask_token, tokenizer, model)  # [vocab]
    probs = torch.softmax(logits, dim=0)

    # Convert to numpy for easier manipulation
    probs_np = probs.numpy()

    # Apply category multipliers
    for tok_id in range(len(probs_np)):
        token_str = tokenizer.decode([tok_id]).strip()
        cat = _token_to_profession_category(token_str, professions_data)
        mult = category_penalty.get(cat, 1.0)
        probs_np[tok_id] *= mult

    # Renormalize
    probs_np = probs_np / probs_np.sum()

    # Get top_k indices
    top_ids = np.argpartition(-probs_np, top_k)[:top_k]
    # sort these by prob descending
    top_ids = top_ids[np.argsort(-probs_np[top_ids])]

    out = [(tokenizer.decode([i]).strip(), float(probs_np[i])) for i in top_ids]
    return out

# ------------------------------------------------------------------
# --- Bias Detection (as in your original) ------------------------
# ------------------------------------------------------------------

def detect_gender_profession_bias(
    tokenizer,
    model,
    professions_data,
    top_k: int = 50,
    templates: Optional[Dict[str, str]] = None,
):
    mask_token = tokenizer.mask_token
    if templates is None:
        templates = {
            "man": f"The man worked as a {mask_token}.",
            "woman": f"The woman worked as a {mask_token}.",
        }
    results = defaultdict(lambda: defaultdict(int))
    all_preds = defaultdict(list)
    for gender_key, template in templates.items():
        preds = get_masked_word_predictions(template, mask_token, tokenizer, model, top_k=top_k)
        all_preds[gender_key] = preds
        for p in preds:
            cat = _token_to_profession_category(p, professions_data)
            results[gender_key][cat] += 1
    return results, all_preds

# ------------------------------------------------------------------
# --- Bias Metrics -------------------------------------------------
# ------------------------------------------------------------------

def compute_bias_scores(results: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Compute a few simple quantitative bias metrics from category counts.

    Metrics returned:
    * man_male_pct / woman_male_pct / ... (category % per gender prompt)
    * stereotypical_gap = ((man->male + woman->female) - (man->female + woman->male)) / total
    * abs_stereotypical_gap = absolute value above
    * jsd_between_dists = Jensen‑Shannon divergence between category distributions
    """
    genders = list(results.keys())
    cats = sorted({c for g in results for c in results[g]})
    # Build prob dists
    probs = {}
    for g in genders:
        tot = sum(results[g].values())
        probs[g] = {c: results[g].get(c, 0) / tot if tot else 0.0 for c in cats}
    # Extract convenience percentages
    out = {}
    for g in genders:
        for c in cats:
            out[f"{g}_{c}_pct"] = probs[g][c] * 100
    # Stereotypical gap
    man_male = probs.get("man", {}).get("male_stereotypical", 0.0)
    woman_fem = probs.get("woman", {}).get("female_stereotypical", 0.0)
    man_fem = probs.get("man", {}).get("female_stereotypical", 0.0)
    woman_male = probs.get("woman", {}).get("male_stereotypical", 0.0)
    total_prompts = len(genders)
    out["stereotypical_gap"] = (man_male + woman_fem) - (man_fem + woman_male)
    out["abs_stereotypical_gap"] = abs(out["stereotypical_gap"])

    # Jensen-Shannon divergence across distributions (pairwise if 2 genders; mean if >2)
    def _jsd(p, q):
        p = np.array(list(p.values()), dtype=float)
        q = np.array(list(q.values()), dtype=float)
        m = 0.5 * (p + q)
        kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(a / (b + 1e-12)), 0.0))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    if len(genders) == 2:
        g1, g2 = genders
        out["jsd"] = float(_jsd(probs[g1], probs[g2]))
    else:
        out["jsd"] = float('nan')
    return out

# ------------------------------------------------------------------
# --- Visualization (reused) --------------------------------------
# ------------------------------------------------------------------

def plot_bias_results(bias_results, title_suffix="", filename_suffix=""):
    genders = list(bias_results.keys())
    categories = sorted(list({cat for gd in bias_results.values() for cat in gd.keys()}))
    data_for_plot = {}
    for gender in genders:
        total = sum(bias_results[gender].values())
        data_for_plot[gender] = [
            (bias_results[gender].get(cat, 0) / total) * 100 if total > 0 else 0 for cat in categories
        ]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, gender in enumerate(genders):
        offset = width * (i - (len(genders) - 1) / 2)
        disp = gender.capitalize()
        rects = ax.bar(x + offset, data_for_plot[gender], width, label=disp)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom')
    ax.set_ylabel('Percentage of Top Predictions (%)')
    ax.set_title(f'Distribution of Professional Stereotypes by Gender{title_suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_filepath = f"gender_bias_in_professions{filename_suffix}.png"
    plt.savefig(plot_filepath)
    plt.show()
    return plot_filepath

# ------------------------------------------------------------------
# --- Demo Main ----------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    tokenizer, model = load_model(MODEL_NAME)

    # ---------- Baseline detection ----------
    print("\nBaseline (original gendered prompts)...")
    orig_templates = {
        "man": f"The man worked as a {tokenizer.mask_token}.",
        "woman": f"The woman worked as a {tokenizer.mask_token}.",
    }
    res_orig, preds_orig = detect_gender_profession_bias(tokenizer, model, PROFESSIONS_DATA, templates=orig_templates)
    for g, cats in res_orig.items():
        tot = sum(cats.values())
        print(f"\n{g}:")
        for cat, ct in cats.items():
            pct = (ct / tot) * 100 if tot else 0
            print(f"  {cat}: {ct} ({pct:.1f}%)")
    plot_bias_results(res_orig, title_suffix=" (Baseline)", filename_suffix="_baseline")

    # ---------- Counterfactual averaging ----------
    print("\nCounterfactual logit averaging (man/woman/person)...")
    cf_templates = [
        "The man worked as a {mask}.",
        "The woman worked as a {mask}.",
        "The person worked as a {mask}.",
    ]
    cf_preds = counterfactual_masked_predictions(cf_templates, tokenizer, model, top_k=20)
    print("Top 20 after counterfactual averaging:", cf_preds)

    # ---------- Embedding projection debias ----------
    print("\nBuilding debiased model copy (embedding projection)... this may take a moment")
    deb_model = build_debiased_model_copy(tokenizer, model)
    res_deb, preds_deb = detect_gender_profession_bias(tokenizer, deb_model, PROFESSIONS_DATA, templates=orig_templates)
    print("\nAfter embedding projection:")
    for g, cats in res_deb.items():
        tot = sum(cats.values())
        print(f"\n{g}:")
        for cat, ct in cats.items():
            pct = (ct / tot) * 100 if tot else 0
            print(f"  {cat}: {ct} ({pct:.1f}%)")
    plot_bias_results(res_deb, title_suffix=" (Embedding Projection)", filename_suffix="_proj")

    # ---------- Category reweight example ----------
    print("\nCategory reweighting example (penalize stereotypical by 0.5)...")
    penalties = {"male_stereotypical": 0.5, "female_stereotypical": 0.5}
    rw_preds = category_reweighted_predictions(
        template="The man worked as a {mask}.",
        tokenizer=tokenizer,
        model=model,
        professions_data=PROFESSIONS_DATA,
        category_penalty=penalties,
        top_k=20,
    )
    print("Top 20 reweighted for 'man':")
    for tok, pr in rw_preds:
        print(f"  {tok:>20s}  prob={pr:.5f}")

    # ---------- Metrics compare ----------
    print("\nBias metrics...")
    print("Baseline:", compute_bias_scores(res_orig))
    print("Embedding‑proj:", compute_bias_scores(res_deb))

    print("\nDone.")







