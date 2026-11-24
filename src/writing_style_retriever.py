"""Optional, offline, deterministic writing-guidance retriever for Mewtype.

IMPORTANT: Does not retrieve clinical evidence, never assigns patient source_ids,
never calls a model/network. Intended as a *candidate* integration module for Codex.
No automatic monkey-patching of the user's current Mewtype pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1] / 'resources' / 'clinical_writing'


def _load(name: str, root: Path = BASE) -> dict[str, Any]:
    with (root / name).open(encoding='utf-8') as f:
        return json.load(f)


def infer_tags(text: str) -> set[str]:
    """Simple deterministic specialty labels for route-only use; not diagnosis."""
    matchers = {
        'cardiology': ('PCI', '支架', '胸痛', '冠脉', '心绞痛', '球囊', '心电图'),
        'neurology': ('TIA', '短暂性脑缺血', '偏瘫', '肢体无力', '言语不清', '神经'),
        'obstetrics': ('妊娠', '孕周', '胆汁酸', '孕晚期', '胎心', 'OGTT'),
        'renal': ('肌酐', '肾功能', '贫血', '血红蛋白', '透析'),
        'geriatrics': ('老年', '多病共存'),
    }
    return {tag for tag, cues in matchers.items() if any(c.lower() in text.lower() for c in cues)}


def select_cards(
    *,
    case_text: str = '',
    tags: set[str] | None = None,
    sections: tuple[str, ...] = ('case_features', 'diagnostic_discussion', 'plan', 'evidence', 'review_items', 'specialty'),
    max_cards: int = 12,
    root: Path = BASE,
) -> list[dict[str, Any]]:
    """Return deterministic rule cards. Explicit tags from the planner are preferred."""
    active = set(tags) if tags is not None else infer_tags(case_text)
    cards = _load('style_cards.json', root)['cards']
    matches = []
    for c in cards:
        if c['section'] not in sections:
            continue
        specialty = (c['section'] == 'specialty')
        if specialty and not active.intersection(c['tags']):
            continue
        if specialty or ('general' in c['tags']) or active.intersection(c['tags']):
            relevance = len(active.intersection(c['tags']))
            score = int(c['priority'])*10 + relevance*3 + int(c['section'] != 'specialty')
            matches.append((score,c['id'],c))
    matches.sort(key=lambda entry: (-entry[0], entry[1]))
    # Guarantee major output sections are represented before filling by relevance.
    sections_in_priority = ('case_features', 'diagnostic_discussion', 'plan', 'evidence', 'review_items', 'specialty')
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in sections_in_priority:
        for _, _, c in matches:
            if c['section'] == section and c['id'] not in seen and len(chosen) < max_cards:
                chosen.append(c)
                seen.add(c['id'])
                break
    for _, _, c in matches:
        if len(chosen) >= max_cards:
            break
        if c['id'] not in seen:
            chosen.append(c)
            seen.add(c['id'])
    return chosen


def select_exemplar_lessons(*, tags: set[str], limit: int = 1, root: Path = BASE) -> list[dict[str, Any]]:
    """Find synthetic example *lessons* only. Does not pass patient narratives to Qwen."""
    examples = _load('synthetic_exemplars.json', root)['examples']
    ranked=[]
    for e in examples:
        overlap=len(tags.intersection(e['tags']))
        if overlap:
            ranked.append((-overlap,e['id'],e))
    return [{'example_id':e['id'],'department':e['department'], 'style_takeaways': e['style_takeaways']}
            for _,_,e in sorted(ranked)[:limit]]


def render_writer_guidance(
    *, case_text: str = '', tags: set[str] | None = None,
    max_cards: int = 12, max_chars: int = 2600, include_exemplar_lessons: bool = True,
    root: Path = BASE,
) -> str:
    """Return bounded guidance to pass as extra Writer context, not as patient facts."""
    active=set(tags) if tags is not None else infer_tags(case_text)
    header=(
        'WRITING_STYLE_REFERENCE (NOT_PATIENT_EVIDENCE)\n'
        '以下为原创写作规则，仅用于组织结构、证据关联和措辞；不得当成患者事实、医学指南、治疗医嘱或source_ids。'
        '当前患者facts与sources具有最高事实优先级。范例中的任何数值、药物、日期、诊断均不得迁移到患者。'
        '不改变规定的JSON Schema。\n'
    )
    if len(header) >= max_chars:
        return header[:max_chars]
    result = header
    for c in select_cards(case_text=case_text, tags=active, max_cards=max_cards, root=root):
        line=f'[{c["id"]}/{c["section"]}] {c["writing_rule"]} 需避免：{c["avoid"]}\n'
        if len(result)+len(line)>max_chars:
            continue
        result+=line
    if include_exemplar_lessons:
        for ex in select_exemplar_lessons(tags=active, root=root):
            line=f'[ORIGINAL_SYNTHETIC_STYLE/{ex["example_id"]}] '+ '；'.join(ex['style_takeaways'])+'\n'
            if len(result)+len(line)<=max_chars:
                result+=line
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Preview bounded non-evidence writing guidance; no model calls')
    p.add_argument('--tags', nargs='*', default=[])
    p.add_argument('--case-text', default='')
    p.add_argument('--max-chars',type=int,default=2600)
    p.add_argument('--no-exemplar-lessons',action='store_true')
    args=p.parse_args()
    print(render_writer_guidance(tags=set(args.tags) if args.tags else None,case_text=args.case_text,
        max_chars=args.max_chars,include_exemplar_lessons=not args.no_exemplar_lessons))
