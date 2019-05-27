import numpy as np
import requests

from io import BytesIO

from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image


"""Adobted from tensorflow"""


def create_pascal_label_colormap():

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):

  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

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
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return img


LABEL_NAMES = np.asarray([
  'background','wall','building, edifice','sky','floor, flooring','tree','ceiling','road, route','bed ','windowpane, window ','grass','cabinet','sidewalk, pavement','person, individual, someone, somebody, mortal, soul','earth, ground','door, double door','table','mountain, mount','plant, flora, plant life','curtain, drape, drapery, mantle, pall','chair','car, auto, automobile, machine, motorcar','water','painting, picture','sofa, couch, lounge','shelf','house','sea','mirror','rug, carpet, carpeting','field','armchair','seat','fence, fencing','desk','rock, stone','wardrobe, closet, press','lamp','bathtub, bathing tub, bath, tub','railing, rail','cushion','base, pedestal, stand','box','column, pillar','signboard, sign','chest of drawers, chest, bureau, dresser','counter','sand','sink','skyscraper','fireplace, hearth, open fireplace','refrigerator, icebox','grandstand, covered stand','path','stairs, steps','runway','case, display case, showcase, vitrine','pool table, billiard table, snooker table','pillow','screen door, screen','stairway, staircase','river','bridge, span','bookcase','blind, screen','coffee table, cocktail table','toilet, can, commode, crapper, pot, potty, stool, throne','flower','book','hill','bench','countertop','stove, kitchen stove, range, kitchen range, cooking stove','palm, palm tree','kitchen island','computer, computing machine, computing device, data processor, electronic computer, information processing system','swivel chair','boat','bar','arcade machine','hovel, hut, hutch, shack, shanty','bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle','towel','light, light source','truck, motortruck','tower','chandelier, pendant, pendent','awning, sunshade, sunblind','streetlight, street lamp','booth, cubicle, stall, kiosk','television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box','airplane, aeroplane, plane','dirt track','apparel, wearing apparel, dress, clothes','pole','land, ground, soil','bannister, banister, balustrade, balusters, handrail','escalator, moving staircase, moving stairway','ottoman, pouf, pouffe, puff, hassock','bottle','buffet, counter, sideboard','poster, posting, placard, notice, bill, card','stage','van','ship','fountain','conveyer belt, conveyor belt, conveyer, conveyor, transporter','canopy','washer, automatic washer, washing machine','plaything, toy','swimming pool, swimming bath, natatorium','stool','barrel, cask','basket, handbasket','waterfall, falls','tent, collapsible shelter','bag','minibike, motorbike','cradle','oven','ball','food, solid food','step, stair','tank, storage tank','trade name, brand name, brand, marque','microwave, microwave oven','pot, flowerpot','animal, animate being, beast, brute, creature, fauna','bicycle, bike, wheel, cycle ','lake','dishwasher, dish washer, dishwashing machine','screen, silver screen, projection screen','blanket, cover','sculpture','hood, exhaust hood','sconce','vase','traffic light, traffic signal, stoplight','tray','ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin','fan','pier, wharf, wharfage, dock','crt screen','plate','monitor, monitoring device','bulletin board, notice board','shower','radiator','glass, drinking glass','clock','flag'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
