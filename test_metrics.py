import torch
from IMDLBenCo.evaluation.Accuracy import PixelAccuracy
from IMDLBenCo.evaluation.F1 import PixelF1

# Simulate behavior from wild datasets
# Let's say batch size is 4, image size is 512x512
batch_size, channels, h, w = 4, 1, 512, 512

# Case A: Pure Wild Dataset. Masks are either all 0 (real) or all 1 (AI text2img)
# But wait! For wild dataset "run_wild.bat", ALL testing and validation logic assumes "No Local Anomaly" means "Mask is All Zeros!".
# Wait. If gt is 'text2img', abstract_dataset.py returns an all-255 mask. 
# BUT the model for wild dataset is wild_cls_only, meaning it predicts All 0 (red = 0?).
# Wait! In AniXplore mask generation, anomaly is 1. If prediction is 1 for AI, then F1 should be calcuated.

mask = torch.cat([torch.zeros(2, channels, h, w), torch.ones(2, channels, h, w)], dim=0)

# What if model ALWAYS predicts 0.0 (no mask, because featurePyramid_net is frozen on inpainting)
pred_mask = torch.zeros(batch_size, channels, h, w)

evaluator_acc = PixelAccuracy()
res_acc = evaluator_acc.batch_update(pred_mask, mask, None)
print("ACC always pred zeros on mixed dataset:", res_acc)

# What if it predicts a small random noise around 0?
pred_mask = torch.randn(batch_size, channels, h, w) * 0.1
res_acc = evaluator_acc.batch_update(pred_mask, mask, None)
print("ACC small noise:", res_acc)

# What if it predicts exactly 1 for ALL?
pred_mask = torch.ones(batch_size, channels, h, w)
res_acc = evaluator_acc.batch_update(pred_mask, mask, None)
print("ACC always pred ones:", res_acc)

# Wait. PixelAccuracy in IMDLBenCo averages the 4 values in the batch!
import IMDLBenCo.training_scripts.utils.misc as misc
logger = misc.MetricLogger(delimiter="  ")
logger.update(acc=torch.sum(res_acc), _n=batch_size)
print("Logger average:", logger.meters['acc'].global_avg)
