"""golden set schema 完整性測試（不需 index；grounding 驗證見 test_quality_local）。"""
from golden import load_golden

REQUIRED = {"id", "question", "answerable", "gold", "category"}
CATEGORIES = {"factual", "exact-term", "qualitative", "trap", "refuse"}


def test_golden_loads_enough_items():
    assert len(load_golden()) >= 20


def test_ids_unique():
    ids = [it["id"] for it in load_golden()]
    assert len(ids) == len(set(ids))


def test_schema_and_gold_consistency():
    for it in load_golden():
        assert REQUIRED <= set(it), f"{it.get('id')} 缺欄位"
        assert it["category"] in CATEGORIES, it["id"]
        assert isinstance(it["gold"], list)
        if it["answerable"]:
            assert it["gold"], f"{it['id']} answerable 卻無 gold"
        else:
            assert it["gold"] == [], f"{it['id']} refuse 卻有 gold"
