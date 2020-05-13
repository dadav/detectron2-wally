# based on: https://github.com/mrdbourke/airbnb-amenity-detection/blob/master/app/app.py
import streamlit as st
import numpy as np
import json
import random
import cv2
import torch
from PIL import Image

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

# Set up default variables
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MODEL_FILE = "output/model_final.pth"


@st.cache(allow_output_mutation=True)
def create_predictor(model_config, model_weights, threshold):
    """
    Loads a Detectron2 model based on model_config, model_weights and creates a default
    Detectron2 predictor.

    Returns Detectron2 default predictor and model config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_weights
    #cfg.MODEL.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

from detectron2.engine import DefaultPredictor

def make_inference(image, model_config, model_weights, threshold=0.5, save=False):
  """
  Makes inference on image (single image) using model_config, model_weights and threshold.

  Returns image

  Params:
  -------
  image (str) : file path to target image
  model_config (str) : file path to model config in .yaml format
  model_weights (str) : file path to model weights
  threshold (float) : confidence threshold for model prediction, default 0.5
  save (bool) : if True will save image with predicted instances to file, default False
  """
  # Create predictor and model config
  cfg, predictor = create_predictor(model_config, model_weights, threshold)

  # Convert PIL image to array
  image = np.asarray(image)

  # Create visualizer instance
  visualizer = Visualizer(img_rgb=image,
                          metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                          instance_mode=ColorMode.IMAGE_BW,
                          scale=1.2)
  outputs = predictor(image)

  # Get instance predictions from outputs
  instances = outputs["instances"]

  # Draw on predictions to image
  vis = visualizer.draw_instance_predictions(instances.to("cpu"))

  return vis.get_image(), instances

def main():
    st.title("Wo ist Walter?")
    st.write("Diese App findet Walter auf Suchbildern!")
    st.write("## Wie funktioniert das?")
    st.write("Diese App nutzt eine KI basierend auf [detectron2](https://github.com/facebookresearch/detectron2),")
    st.write("die mit [diesen Bildern](https://github.com/tadejmagajna/HereIsWally/tree/master/images) trainiert wurde.")
    st.image(Image.open("images/1.jpg"),
             caption="Beispielbild.",
             use_column_width=True)
    st.write("## Lade dein eigenes Bild hoch!!")
    uploaded_image = st.file_uploader("Wähle eine jpg oder png Bild aus!",
                                      type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Hochgeladenes Bild.", use_column_width=True)

        # Make sure image is RGB
        image = image.convert("RGB")

        if st.button("Finde Walter!"):
          with st.spinner("Suche Walter..."):
            custom_pred, preds = make_inference(
                image=image,
                model_config=model_zoo.get_config_file(CONFIG_FILE),
                model_weights=MODEL_FILE,
            )
            if len(preds) > 0:
                st.image(custom_pred, caption="Walter wurde {} Mal gefunden!".format(len(preds)), use_column_width=True)
            else:
                st.write("Konnte keinen Walter finden :(")

    st.write("Der Code liegt hier: [GitHub](https://github.com/dadav/detectron2-wally)")

if __name__ == "__main__":
    main()
