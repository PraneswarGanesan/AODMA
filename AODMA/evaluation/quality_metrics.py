
from evaluation.fid import FIDEvaluator
from evaluation.kid import KIDEvaluator


class QualityEvaluator:

    def __init__(self, device, kid_subset_size=1000):
        self.fid = FIDEvaluator(device)

        self.kid = KIDEvaluator(
            device,
            subset_size=kid_subset_size,
            subsets=50,
        )

    def update_real(self, images):
        self.fid.update_real(images)
        self.kid.update_real(images)

    def update_fake(self, images):
        self.fid.update_fake(images)
        self.kid.update_fake(images)

    def compute(self):
        fid_score = self.fid.compute()
        kid_score = self.kid.compute()

        # Normalize metric outputs for consistent JSON reporting.
        if isinstance(fid_score, dict):
            fid_score = fid_score.get("fid", fid_score.get("mean"))

        if isinstance(kid_score, dict):
            kid_mean = kid_score.get("mean", kid_score.get("kid_mean"))
            kid_std = kid_score.get("std", kid_score.get("kid_std"))
        else:
            kid_mean = kid_score
            kid_std = None

        if hasattr(fid_score, "item"):
            fid_score = fid_score.item()

        if hasattr(kid_mean, "item"):
            kid_mean = kid_mean.item()

        if hasattr(kid_std, "item"):
            kid_std = kid_std.item()

        return {
            "fid": float(fid_score),
            "kid": {
                "mean": float(kid_mean),
                "std": float(kid_std) if kid_std is not None else None,
            },
        }

    def reset(self):
        self.fid.reset()
        self.kid.reset()
