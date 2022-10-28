import contextlib
import matplotlib.pyplot as plt
import matplotlib as mpl

with contextlib.suppress(Exception):
    import sys, torch

    if not torch.cuda.is_available():
        print("Change runtime type to include a GPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running pytorch", torch.__version__, "using", device.type)

import torchvision
import torch.hub
from netdissect import nethook, proggan

# n = 'proggan_bedroom-d8a89ff1.pth'
n = "proggan_churchoutdoor-7e701dd5.pth"
# n = 'proggan_conferenceroom-21e85882.pth'
# n = 'proggan_diningroom-3aa0ab80.pth'
# n = 'proggan_kitchen-67f1e16c.pth'
# n = 'proggan_livingroom-5ef336dd.pth'
# n = 'proggan_restaurant-b8578299.pth'


try:
    sd = torch.hub.load_state_dict_from_url(
        f"http://gandissect.csail.mit.edu/models/{n}"
    )

except Exception as e:
    sd = torch.hub.model_zoo.load_url(f"http://gandissect.csail.mit.edu/models/{n}")


model = proggan.from_state_dict(sd).to(device)
# print(model)

from netdissect import zdataset

SAMPLE_SIZE = 50
zds = zdataset.z_dataset_for_model(model, size=SAMPLE_SIZE, seed=5555)
# print(len(zds), zds[0][0].shape)
# print(len(zds[0]))
# print(model(zds[0][0][None, ...].to(device))[0, 0, :10].shape)
# print(zds[0][0][None, ...].shape)

from netdissect import renormalize, show

# synthetic_images = [
#     [renormalize.as_image(model(z[None, ...].to(device))[0])] for [z] in zds
# ]
# synthetic_images[0][0].show()

# Hooking a model with InstrumentedModel
"""
InstrumentedModel adds a few useful functions for inspecting a model, including:
   * `model.retain_layer('layername')` - hooks a layer to hold on to its output after computation
   * `model.retained_layer('layername')` - returns the retained data from the last computation
   * `model.edit_layer('layername', rule=...)` - runs the `rule` function after the given layer
   * `model.remove_edits()` - removes editing rules
"""

from netdissect import nethook

if not isinstance(model, nethook.InstrumentedModel):
    model = nethook.InstrumentedModel(model)
model.retain_layer("layer5")

img = model(zds[0][0][None, ...].to(device))

acts: torch.Tensor = model.retained_layer("layer5")

print(acts.shape)
print(acts[0, 0])
print(type(acts))

from netdissect import imgviz

iv = imgviz.ImageVisualizer(100)
# iv.heatmap(acts[0, 1], mode="nearest")

# show(
#     [
#         [
#             "unit %d" % u,  # u stands for unit
#             [iv.image(img[0])],
#             [iv.masked_image(img[0], acts, (0, u))],
#             [iv.heatmap(acts, (0, u), mode="nearest")],
#         ]
#         for u in range(414, 420)
#     ]
# )

print(acts.shape)
print(acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1]).shape)
