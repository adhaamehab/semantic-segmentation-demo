import numpy as np
import requests

from io import BytesIO

from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image


"""Adobted from github"""


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


"""Adobted from github"""


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def inference(model, url):
    """Inferences DeepLab model and visualizes result."""
    try:
        img_ = requests.get(url).content
        original_im = Image.open(BytesIO(img_))
    except IOError:
        print("Invalid image url" + url)
        return

    print("Predicting ....")
    image, seg_map = model.run(original_im)

    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis("off")
    plt.title("input image")

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis("off")
    plt.title("segmentation map")

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis("off")
    plt.title("segmentation overlay")
    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        CM[unique_labels].astype(np.uint8), interpolation="nearest"
    )  # adobted from tensorflow - model zoo
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid("off")

    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return img


NAMES = np.asarray(
    [
        "airplane, aeroplane, plane",
        "animal, animate being, beast, brute, creature, fauna",
        "apparel, wearing apparel, dress, clothes",
        "arcade machine",
        "armchair",
        "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
        "awning, sunshade, sunblind",
        "background",
        "bag",
        "ball",
        "bannister, banister, balustrade, balusters, handrail",
        "bar",
        "barrel, cask",
        "base, pedestal, stand",
        "basket, handbasket",
        "bathtub, bathing tub, bath, tub",
        "bed ",
        "bench",
        "bicycle, bike, wheel, cycle ",
        "blanket, cover",
        "blind, screen",
        "boat",
        "book",
        "bookcase",
        "booth, cubicle, stall, kiosk",
        "bottle",
        "box",
        "bridge, span",
        "buffet, counter, sideboard",
        "building, edifice",
        "bulletin board, notice board",
        "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
        "cabinet",
        "canopy",
        "car, auto, automobile, machine, motorcar",
        "case, display case, showcase, vitrine",
        "ceiling",
        "chair",
        "chandelier, pendant, pendent",
        "chest of drawers, chest, bureau, dresser",
        "clock",
        "coffee table, cocktail table",
        "column, pillar",
        "computer, computing machine, computing device, data processor, electronic computer, information processing system",
        "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
        "counter",
        "countertop",
        "cradle",
        "crt screen",
        "curtain, drape, drapery, mantle, pall",
        "cushion",
        "desk",
        "dirt track",
        "dishwasher, dish washer, dishwashing machine",
        "door, double door",
        "earth, ground",
        "escalator, moving staircase, moving stairway",
        "fan",
        "fence, fencing",
        "field",
        "fireplace, hearth, open fireplace",
        "flag" "floor, flooring",
        "flower",
        "food, solid food",
        "fountain",
        "glass, drinking glass",
        "grandstand, covered stand",
        "grass",
        "hill",
        "hood, exhaust hood",
        "house",
        "hovel, hut, hutch, shack, shanty",
        "kitchen island",
        "lake",
        "lamp",
        "land, ground, soil",
        "light, light source",
        "microwave, microwave oven",
        "minibike, motorbike",
        "mirror",
        "monitor, monitoring device",
        "mountain, mount",
        "ottoman, pouf, pouffe, puff, hassock",
        "oven",
        "painting, picture",
        "palm, palm tree",
        "path",
        "person, individual, someone, somebody, mortal, soul",
        "pier, wharf, wharfage, dock",
        "pillow",
        "plant, flora, plant life",
        "plate",
        "plaything, toy",
        "pole",
        "pool table, billiard table, snooker table",
        "poster, posting, placard, notice, bill, card",
        "pot, flowerpot",
        "radiator",
        "railing, rail",
        "refrigerator, icebox",
        "river",
        "road, route",
        "rock, stone",
        "rug, carpet, carpeting",
        "runway",
        "sand",
        "sconce",
        "screen door, screen",
        "screen, silver screen, projection screen",
        "sculpture",
        "sea",
        "seat",
        "shelf",
        "ship",
        "shower",
        "sidewalk, pavement",
        "signboard, sign",
        "sink",
        "sky",
        "skyscraper",
        "sofa, couch, lounge",
        "stage",
        "stairs, steps",
        "stairway, staircase",
        "step, stair",
        "stool",
        "stove, kitchen stove, range, kitchen range, cooking stove",
        "streetlight, street lamp",
        "swimming pool, swimming bath, natatorium",
        "swivel chair",
        "table",
        "tank, storage tank",
        "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
        "tent, collapsible shelter",
        "toilet, can, commode, crapper, pot, potty, stool, throne",
        "towel",
        "tower",
        "trade name, brand name, brand, marque",
        "traffic light, traffic signal, stoplight",
        "tray",
        "tree",
        "truck, motortruck",
        "van",
        "vase",
        "wall",
        "wardrobe, closet, press",
        "washer, automatic washer, washing machine",
        "water",
        "waterfall, falls",
        "windowpane, window ",
    ]
)

LM = np.arange(len(NAMES)).reshape(len(NAMES), 1)
CM = label_to_color_image(LM)
