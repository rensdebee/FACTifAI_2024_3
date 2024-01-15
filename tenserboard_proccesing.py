from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

event_acc = EventAccumulator(
    "base_logs/VOC2007/bcos_standard_attrBCos_loclossEnergy_origNone_resnet50_lr0.0001_sll1.0_layerInput\events.out.tfevents.1704996668.gcn33.local.snellius.surf.nl.634617.0"
)
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
scores = []
steps = []
for event in event_acc.Scalars("fscore"):
    scores.append(event.value)
    steps.append(event.step)

scores = np.asarray(scores)
print(scores.max())
print(scores.argmax())
