import streamlit as st
import numpy as np
from vedo import Plotter, Volume
import plotly.graph_objects as go
from PIL import Image, ImageEnhance, ImageFilter
import io

# Set the page configuration to full-screen mode with wide layout
st.set_page_config(page_title="3D and 4D Image Visualization", layout="wide")

# Streamlit App Setup
st.title("3D and 4D Image Visualization")
st.write("Upload an image to visualize it in 3D and 4D.")
st.write("### Instructions:")
st.write("1. Upload a grayscale image (JPG, PNG, JPEG).")
st.write("2. View different 3-level preprocesses side-by-side.")
st.write("3. Click the buttons to view various 3D visualizations.")
st.write("4. Use the opacity slider to adjust the visibility of the volume rendering.")
st.write("5. Download processed images or visualizations.")

# Step 1: Load Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Convert the uploaded file to a PIL image and then to grayscale
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image) / 255.0  # Normalize pixel values

    # Preprocessing Levels
    def preprocess_image(image, level):
        if level == 1:
            # Level 1: Mild Gaussian blur
            return image.filter(ImageFilter.GaussianBlur(2))
        elif level == 2:
            # Level 2: Medium Gaussian blur + slight contrast increase
            blurred = image.filter(ImageFilter.GaussianBlur(4))
            enhancer = ImageEnhance.Contrast(blurred)
            return enhancer.enhance(1.2)
        elif level == 3:
            # Level 3: Stronger sharpening and higher contrast
            sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            enhancer = ImageEnhance.Contrast(sharpened)
            return enhancer.enhance(1.5)
        return image

    # Apply preprocessing at different levels
    level_1_image = preprocess_image(image, 1)
    level_2_image = preprocess_image(image, 2)
    level_3_image = preprocess_image(image, 3)

    # Display the original and preprocessed images side-by-side
    col1, col2, col3, col4 = st.columns(4)
    col1.image(image, caption="Original Image", use_column_width=True)
    col2.image(level_1_image, caption="Preprocessing Level 1", use_column_width=True)
    col3.image(level_2_image, caption="Preprocessing Level 2", use_column_width=True)
    col4.image(level_3_image, caption="Preprocessing Level 3", use_column_width=True)

    # Step 2: Create a 3D Surface Using Plotly
    def create_3d_surface(image_np):
        x = np.linspace(0, image_np.shape[1] - 1, image_np.shape[1])
        y = np.linspace(0, image_np.shape[0] - 1, image_np.shape[0])
        x, y = np.meshgrid(x, y)
        z = image_np * 20  # Scale z for height

        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Gray", opacity=0.9)])
        fig.update_layout(
            title="3D Surface Plot",
            scene=dict(zaxis=dict(title="Intensity", range=[0, 20]), xaxis=dict(title="X"), yaxis=dict(title="Y")),
            autosize=True,
            width=1000,
            height=600,
        )
        return fig

    # Create 3D surface plot
    surface_fig = create_3d_surface(image_np)
    st.plotly_chart(surface_fig, use_container_width=True)

    # Interactive 3D Volume Visualization
    if st.button("Show Interactive 3D Volume"):
        # Convert 2D image to 3D by stacking along a new axis
        image_3d = np.stack([image_np] * 10, axis=-1)  # Create a 3D volume with 10 slices
        vol = Volume(image_3d, spacing=(1, 1, 1))
        vol.cmap("bone")

        plotter = Plotter(title="Rotate the volume", interactive=True)
        plotter.show(vol)

    # Additional 3D Visualization Tools
    st.write("### Additional 3D Visualization Tools")

    # Tool 1: 3D Contour Plot (Reversed Image)
    if st.button("Show Reversed 3D Contour Plot"):
        reversed_image_2d = np.flipud(image_np)  # Flip the image vertically
        fig_contour = go.Figure(data=go.Contour(z=reversed_image_2d * 20, colorscale='Viridis'))
        fig_contour.update_layout(
            title='Reversed 3D Contour Plot',
            xaxis_title='X', 
            yaxis_title='Y',
            autosize=True,
            width=1000,
            height=600,
        )
        st.plotly_chart(fig_contour, use_container_width=True)

    # Tool 2: 3D Wireframe Plot
    if st.button("Show 3D Wireframe Plot"):
        x = np.linspace(0, image_np.shape[1] - 1, image_np.shape[1])
        y = np.linspace(0, image_np.shape[0] - 1, image_np.shape[0])
        x, y = np.meshgrid(x, y)
        z = image_np * 20  # Scale z for height

        fig_wireframe = go.Figure(data=[go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), mode='lines', line=dict(color='blue', width=2))])
        fig_wireframe.update_layout(
            title='3D Wireframe Plot',
            scene=dict(zaxis=dict(title='Intensity')),
            autosize=True,
            width=1000,
            height=600,
        )
        st.plotly_chart(fig_wireframe, use_container_width=True)

    # Tool 3: Adjusting Volume Opacity
    opacity = st.slider("Adjust Volume Opacity", 0.0, 1.0, 0.5)
    if st.button("Show Adjusted 3D Volume"):
        image_3d = np.stack([image_np] * 10, axis=-1)
        vol = Volume(image_3d, spacing=(1, 1, 1))
        vol.cmap("bone")
        vol.alpha(opacity)  # Set the user-defined opacity

        plotter = Plotter(title="Adjustable Volume", interactive=True)
        plotter.show(vol)

    # Tool 4: 4D Visualization (Time Series)
    st.write("### 4D Visualization Tool")
    time_series = np.stack([image_np * (i + 1) / 10 for i in range(10)], axis=0)  # Simulate a time series of 10 frames

    if st.button("Show 4D Volume"):
        vol_4d = Volume(time_series, spacing=(1, 1, 1))
        vol_4d.cmap("bone")
        plotter_4d = Plotter(title="4D Volume (Time Series)", interactive=True)
        plotter_4d.show(vol_4d)

    # Download Option
    st.write("### Download Processed Image")
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    st.download_button(
        label="Download Original Image",
        data=buffer,
        file_name="processed_image.png",
        mime="image/png"
    )
