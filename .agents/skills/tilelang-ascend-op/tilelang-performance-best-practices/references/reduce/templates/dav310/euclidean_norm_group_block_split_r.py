STATUS = "DESIGN_ONLY"


def build(groups: int, rows_per_group: int, cols: int, dtype: str = "float16", num_cores: int = 64):
    raise NotImplementedError("group/block split-R is a design candidate; no verified executable PTO kernel is bundled")
