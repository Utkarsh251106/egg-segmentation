
from pathlib import Path
import io
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_model, predict, create_overlay


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Egg Segmentation AI",
    page_icon="🥚",
    layout="wide",
)

st.title("🥚 Egg Segmentation AI")
st.write(
    "Upload an egg image to automatically segment the egg "
    "using a trained U-Net model."
)


# --------------------------------------------------
# Load model once and cache it
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an egg image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:

    # Read uploaded image as RGB.
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)

    # Convert RGB to BGR for OpenCV.
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    st.subheader("Original Image")
    st.image(
        image_rgb,
        caption="Uploaded Image",
        use_container_width=True,
    )

    # --------------------------------------------------
    # Run segmentation
    # --------------------------------------------------

    if st.button("Segment Egg", type="primary"):

        with st.spinner("Loading model and segmenting egg..."):

            model = get_model()

            # Generate binary mask.
            mask = predict(model, image_bgr)

            # Generate overlay.
            overlay_bgr = create_overlay(
                image_bgr,
                mask,
            )

            # Convert outputs to RGB for Streamlit.
            mask_rgb = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2RGB,
            )

            overlay_rgb = cv2.cvtColor(
                overlay_bgr,
                cv2.COLOR_BGR2RGB,
            )

        st.success("Segmentation completed!")

        # --------------------------------------------------
        # Segmentation statistics
        # --------------------------------------------------

        egg_pixels = int(np.count_nonzero(mask))

        total_pixels = int(
            mask.shape[0] * mask.shape[1]
        )

        egg_percentage = (
            egg_pixels / total_pixels
        ) * 100

        background_pixels = total_pixels - egg_pixels

        st.subheader("Segmentation Statistics")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                label="Egg Area (pixels)",
                value=f"{egg_pixels:,}",
            )

        with metric2:
            st.metric(
                label="Egg Coverage",
                value=f"{egg_percentage:.2f}%",
            )

        with metric3:
            st.metric(
                label="Background Area (pixels)",
                value=f"{background_pixels:,}",
            )

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        st.subheader("Segmentation Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                mask_rgb,
                caption="Predicted Binary Mask",
                use_container_width=True,
            )

        with col2:
            st.image(
                overlay_rgb,
                caption="Egg Segmentation Overlay",
                use_container_width=True,
            )

        # --------------------------------------------------
        # Download predicted mask
        # --------------------------------------------------

        mask_buffer = io.BytesIO()

        Image.fromarray(mask).save(
            mask_buffer,
            format="PNG",
        )

        mask_buffer.seek(0)

        # --------------------------------------------------
        # Download overlay
        # --------------------------------------------------

        overlay_buffer = io.BytesIO()

        Image.fromarray(overlay_rgb).save(
            overlay_buffer,
            format="PNG",
        )

        overlay_buffer.seek(0)

        # --------------------------------------------------
        # Download buttons
        # --------------------------------------------------

        st.subheader("Download Results")

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label="Download Predicted Mask",
                data=mask_buffer,
                file_name="predicted_egg_mask.png",
                mime="image/png",
            )

        with download_col2:
            st.download_button(
                label="Download Overlay",
                data=overlay_buffer,
                file_name="egg_segmentation_overlay.png",
                mime="image/png",
            )

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        st.caption(
            "Egg coverage represents the percentage of image pixels "
            "classified as egg. This is pixel area, not physical area "
            "in cm²."
        )

else:
    st.info(
        "Please upload an image to begin egg segmentation."
    )
