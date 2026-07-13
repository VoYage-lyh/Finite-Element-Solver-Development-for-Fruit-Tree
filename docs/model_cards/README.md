# Model cards

## EfficientFormer visible-wood segmenter

- Backbone: EfficientFormer-L1
- Best epoch: 70
- Validation Dice: 0.8744
- Held-out tree-group test Dice: 0.8082
- Training split: tree groups, before tile extraction
- Full training history: [wood_seg_training_metrics.json](wood_seg_training_metrics.json)
- Local checkpoint: `workspace/models/wood_seg.pt` (ignored by Git)

The metrics file is versioned for thesis and experiment reproducibility. The
checkpoint is intentionally machine-local because of its size. Fully occluded
wood remains outside the annotation target; skeleton corrections stay explicit
in Harvest Console.
