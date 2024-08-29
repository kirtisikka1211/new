import io
import os
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import layoutparser as lp
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import tempfile

# Initialize session state if not already set
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Define the main processing function
def process_image(image_path):
    # Load OCR model
    ocr = PaddleOCR(lang='en')

    # Load Layout model
    model = lp.PaddleDetectionLayoutModel(
        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
        threshold=0.5,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        enforce_cpu=False,
        enable_mkldnn=True
    )

    # Read image
    image_cv = cv2.imread(image_path)
    image_height, image_width = image_cv.shape[:2]

    # Detect layout
    layout = model.detect(image_cv)
    x_1, y_1, x_2, y_2 = 0, 0, 0, 0
    for l in layout:
        if l.type == 'Table':
            x_1, y_1 = int(l.block.x_1), int(l.block.y_1)
            x_2, y_2 = int(l.block.x_2), int(l.block.y_2)
            break

    # Perform OCR
    output = ocr.ocr(image_path)[0]
    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    # Prepare for non-max suppression
    horiz_boxes, vert_boxes = [], []
    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

    # Apply non-max suppression
    horiz_out = tf.image.non_max_suppression(
        horiz_boxes, probabilities, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')
    )
    vert_out = tf.image.non_max_suppression(
        vert_boxes, probabilities, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')
    )

    horiz_lines = np.sort(np.array(horiz_out))
    vert_lines = np.sort(np.array(vert_out))

    # Extract text and table data
    out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        inter = max(x_2 - x_1, 0) * max(y_2 - y_1, 0)
        if inter == 0:
            return 0

        box_1_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
        box_2_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

        return inter / float(box_1_area + box_2_area - inter)

    unordered_boxes = [vert_boxes[i][0] for i in vert_lines]
    ordered_boxes = np.argsort(unordered_boxes)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if iou(resultant, the_box) > 0.1:
                    out_array[i][j] = texts[b]

    return out_array

# Streamlit app
def main():
    st.title("Omkara Extractor")

    # Sidebar for uploading images and other options
    with st.sidebar:
        st.markdown("---")
        st.markdown("### About the App")
        st.markdown(
            "This app allows you to upload an image or PDF and perform OCR and table extraction using PaddleOCR and LayoutParser."
        )

    uploaded_file = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.write("Upload successful!")

        # Create a temporary directory to store the images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Check if file is a PDF
            if uploaded_file.type == "application/pdf":
                # Save uploaded PDF to a temporary file
                pdf_path = os.path.join(temp_dir, "uploaded_file.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert PDF to images
                images = convert_from_path(pdf_path)
                results = []
                for i, image in enumerate(images):
                    image_path = os.path.join(temp_dir, f"page_{i}.png")
                    image.save(image_path, format="PNG")
                    # Process image and append results
                    with st.spinner(f"Processing page {i + 1}..."):
                        out_array = process_image(image_path)
                        results.append(pd.DataFrame(out_array))

                # Concatenate all results into one DataFrame
                final_df = pd.concat(results, keys=[f"Page {i + 1}" for i in range(len(results))], names=["Page"])
            else:
                # Save uploaded image to temp file
                image_path = os.path.join(temp_dir, "uploaded_image.png")
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display uploaded image
                image = Image.open(st.session_state.uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Display processing indicator
                with st.spinner("Processing the image..."):
                    final_df = pd.DataFrame(process_image(image_path))

            # Display results in a table
            st.write("### Extracted Table Data")
            st.dataframe(final_df.style.set_properties(**{'text-align': 'center'}).highlight_null('yellow'))

            # Convert DataFrame to Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False, sheet_name='Extracted Data')

            # Download Excel
            excel_buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name="output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
