import gradio as gr

from features.basic_operations import (
    to_grayscale, to_negative, adjust_color, flip_image, translate,
    scale_image, rotate, crop, blend, adjust_brightness_contrast,
    color_filter, apply_sepia, apply_cyanotype, add_border, overlay,
)
from ui.handlers import (
    tab_pixelwise_op, tab_transform_filter, tab_enhancement, tab_compression,
    tab_segmentation, tab_binary, tab_restoration, tab_matching,
)

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Processing And Recognition")
        with gr.Tabs():
            # Basic Image Operations
            with gr.TabItem("Basic Image Operations"):
                gr.Markdown("### Basic Image Operations")

                # Tab Greyscale
                with gr.TabItem("Greyscale"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")
                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=to_grayscale,
                        inputs=[input_image_basic1],
                        outputs=[output_image_basic]
                    )

                # Tab Negative
                with gr.TabItem("Negative"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")
                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=to_negative,
                        inputs=[input_image_basic1],
                        outputs=[output_image_basic]
                    )

                # Tab Adjust Color
                with gr.TabItem("Adjust Color"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    r_factor = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Red Factor")
                    g_factor = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Green Factor")
                    b_factor = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Blue Factor")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=adjust_color,
                        inputs=[input_image_basic1, r_factor, g_factor, b_factor],
                        outputs=[output_image_basic]
                    )

                # Tab Flip
                with gr.TabItem("Flip"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    flip_dir = gr.Dropdown(["horizontal", "vertical", "diagonal"], value="horizontal", label="Flip Direction")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=flip_image,
                        inputs=[input_image_basic1, flip_dir],
                        outputs=[output_image_basic]
                    )

                # Tab Translate
                with gr.TabItem("Translate"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    trans_x = gr.Slider(-200, 200, value=0, step=1, label="Translate X")
                    trans_y = gr.Slider(-200, 200, value=0, step=1, label="Translate Y")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=translate,
                        inputs=[input_image_basic1, trans_x, trans_y],
                        outputs=[output_image_basic]
                    )

                # Tab Scale
                with gr.TabItem("Scale"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    scale_w = gr.Number(value=None, label="Scale Width (None=auto)")
                    scale_h = gr.Number(value=None, label="Scale Height (None=auto)")
                    scale_keep_aspect = gr.Checkbox(value=True, label="Keep Aspect Ratio")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=scale_image,
                        inputs=[input_image_basic1, scale_w, scale_h, scale_keep_aspect],
                        outputs=[output_image_basic]
                    )

                # Tab Rotate
                with gr.TabItem("Rotate"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    rot_angle = gr.Slider(-180, 180, value=0, step=1, label="Rotate Angle")
                    rot_clockwise = gr.Checkbox(value=True, label="Rotate Clockwise")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=rotate,
                        inputs=[input_image_basic1, rot_angle, rot_clockwise],
                        outputs=[output_image_basic]
                    )

                # Tab Crop
                with gr.TabItem("Crop"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    crop_x = gr.Slider(0, 300, value=0, step=1, label="Crop X")
                    crop_y = gr.Slider(0, 300, value=0, step=1, label="Crop Y")
                    crop_w = gr.Slider(0, 300, value=100, step=1, label="Crop Width")
                    crop_h = gr.Slider(0, 300, value=100, step=1, label="Crop Height")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=crop,
                        inputs=[input_image_basic1, crop_x, crop_y, crop_w, crop_h],
                        outputs=[output_image_basic]
                    )

                # Tab Blend
                with gr.TabItem("Blend"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        input_image_basic2 = gr.Image(label="Input Image 2 (for blend)")
                        output_image_basic = gr.Image(label="Output Image")

                    blend_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Blend Alpha")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=blend,
                        inputs=[input_image_basic1, input_image_basic2, blend_alpha],
                        outputs=[output_image_basic]
                    )

                # Tab Brightness & Contrast
                with gr.TabItem("Brightness & Contrast"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    bright_val = gr.Slider(-100, 100, value=50, step=1, label="Brightness")
                    contrast_val = gr.Slider(0.1, 3.0, value=1.2, step=0.1, label="Contrast")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=adjust_brightness_contrast,
                        inputs=[input_image_basic1, bright_val, contrast_val],
                        outputs=[output_image_basic]
                    )

                # Tab Color Filter
                with gr.TabItem("Color Filter"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    filter_type = gr.Radio(
                        ["Custom", "Sepia", "Cyanotype"],
                        label="Select Filter Type",
                        value="Custom",
                    )

                    filter_lower_H = gr.Slider(0, 179, value=0, step=1, label="Filter Lower H")
                    filter_lower_S = gr.Slider(0, 255, value=0, step=1, label="Filter Lower S")
                    filter_lower_V = gr.Slider(0, 255, value=0, step=1, label="Filter Lower V")

                    filter_upper_H = gr.Slider(0, 179, value=179, step=1, label="Filter Upper H")
                    filter_upper_S = gr.Slider(0, 255, value=255, step=1, label="Filter Upper S")
                    filter_upper_V = gr.Slider(0, 255, value=255, step=1, label="Filter Upper V")

                    run_basic = gr.Button("Apply")

                    def apply_selected_filter(image, filter_type, lower_H, lower_S, lower_V, upper_H, upper_S, upper_V):
                        if filter_type == "Custom":
                            lowerb = (lower_H, lower_S, lower_V)
                            upperb = (upper_H, upper_S, upper_V)
                            return color_filter(image, lowerb, upperb)
                        elif filter_type == "Sepia":
                            return apply_sepia(image)
                        elif filter_type == "Cyanotype":
                            return apply_cyanotype(image)

                    run_basic.click(
                        fn=apply_selected_filter,
                        inputs=[input_image_basic1, filter_type, filter_lower_H, filter_lower_S, filter_lower_V, filter_upper_H, filter_upper_S, filter_upper_V],
                        outputs=[output_image_basic]
                    )

                # Tab Add Border
                with gr.TabItem("Add Border"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        output_image_basic = gr.Image(label="Output Image")

                    border_top = gr.Slider(0, 100, value=0, step=1, label="Border Top")
                    border_bottom = gr.Slider(0, 100, value=0, step=1, label="Border Bottom")
                    border_left = gr.Slider(0, 100, value=0, step=1, label="Border Left")
                    border_right = gr.Slider(0, 100, value=0, step=1, label="Border Right")
                    border_color_str = gr.Textbox(value="0,0,0", label="Border Color (R,G,B)")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=add_border,
                        inputs=[input_image_basic1, border_top, border_bottom, border_left, border_right, border_color_str],
                        outputs=[output_image_basic]
                    )

                # Tab Overlay
                with gr.TabItem("Overlay"):
                    with gr.Row():
                        input_image_basic1 = gr.Image(label="Input Image 1")
                        input_image_basic2 = gr.Image(label="Input Image 2 (for overlay)")
                        output_image_basic = gr.Image(label="Output Image")

                    overlay_x = gr.Slider(0, 300, value=0, step=1, label="Overlay X")
                    overlay_y = gr.Slider(0, 300, value=0, step=1, label="Overlay Y")
                    overlay_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Overlay Alpha")

                    run_basic = gr.Button("Apply")
                    run_basic.click(
                        fn=overlay,
                        inputs=[input_image_basic1, input_image_basic2, overlay_x, overlay_y, overlay_alpha],
                        outputs=[output_image_basic]
                    )

            # Mathematical Operations
            with gr.TabItem("Mathematical Operations"):
                gr.Markdown("### Mathematical Operations on Images")
                with gr.Row():
                    input_image_math1 = gr.Image(label="Input Image 1")
                    input_image_math2 = gr.Image(label="Input Image 2 (ignored if NOT)")
                    output_image_math = gr.Image(label="Output Image")

                operation_math = gr.Radio(
                    choices=["Add", "Subtract", "Multiply", "Divide",
                             "Bitwise (AND)", "Bitwise (OR)", "Bitwise (XOR)", "Bitwise (NOT)"],
                    value="Add",
                    label="Pixelwise Operation"
                )
                run_math = gr.Button("Apply Operation")
                run_math.click(
                    tab_pixelwise_op,
                    inputs=[input_image_math1, input_image_math2, operation_math],
                    outputs=[output_image_math]
                )

            # Transforms & Filtering
            with gr.TabItem("Transforms & Filtering"):
                gr.Markdown("### Transforms & Filtering")

                with gr.Row():
                    input_image_tf = gr.Image(type="numpy", label="Input Image")  # Ensure 'numpy' type for the input image
                    output_image_tf = gr.Image(type="numpy", label="Output Image")  # Ensure 'numpy' type for the output image

                # Dropdown to choose operation
                operation_tf = gr.Radio(
                    choices=["FFT", "Mean Blur", "Gaussian Blur", "Median Blur",
                            "Sobel Edge", "Canny Edge", "Laplacian Edge"],
                    value="FFT",
                    label="Select Operation"
                )

                # Sliders for kernel size, sigma, and Canny thresholds
                ksize_tf = gr.Slider(1, 15, step=1, value=3, label="Kernel Size (blur/edge)")
                sigma_tf = gr.Slider(0.1, 5.0, step=0.1, value=1.0, label="Sigma (Gaussian)")
                t1_tf = gr.Slider(0, 255, value=50, step=1, label="Canny Threshold1")
                t2_tf = gr.Slider(0, 255, value=150, step=1, label="Canny Threshold2")

                run_tf = gr.Button("Process")

                def _process_tf(img, op, k, s, c1, c2):
                    return tab_transform_filter(img, op, k, s, c1, c2)

                # Set up the button click interaction
                run_tf.click(_process_tf,
                            inputs=[input_image_tf, operation_tf, ksize_tf, sigma_tf, t1_tf, t2_tf],
                            outputs=[output_image_tf])

            # Enhancement
            with gr.TabItem("Enhancement"):
                gr.Markdown("### Image Enhancement")
                with gr.Row():
                    input_image_en = gr.Image(label="Input Image")
                    output_image_en = gr.Image(label="Output Image")

                operation_en = gr.Radio(
                    choices=["Histogram Equalization", "Contrast Stretching", "Gamma Correction"],
                    value="Histogram Equalization",
                    label="Select Enhancement"
                )
                in_low_en = gr.Slider(0, 255, value=0, step=1, label="In Low (Contrast Stretch)")
                in_high_en = gr.Slider(0, 255, value=255, step=1, label="In High (Contrast Stretch)")
                gamma_en = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Gamma")

                run_en = gr.Button("Enhance")

                def _process_en(img, op, low, high, gm):
                    return tab_enhancement(img, op, low, high, gm)

                run_en.click(_process_en,
                             inputs=[input_image_en, operation_en, in_low_en, in_high_en, gamma_en],
                             outputs=[output_image_en])

            # Compression
            with gr.TabItem("Compression"):
                gr.Markdown("### Image Compression")
                with gr.Row():
                    input_image_comp = gr.Image(label="Input Image")
                    output_comp_image = gr.Image(label="Output Image", interactive=False)

                operation_comp = gr.Radio(
                    choices=["RLE", "DCT"],
                    value="RLE",
                    label="Select Compression"
                )

                run_comp = gr.Button("Compress")

                def _process_comp(img, op):
                    result = tab_compression(img, op)
                    return result

                run_comp.click(
                    _process_comp,
                    inputs=[input_image_comp, operation_comp],
                    outputs=[output_comp_image]
                )

            # Tab: Segmentation
            with gr.TabItem("Segmentation"):
                gr.Markdown("### Image Segmentation")
                with gr.Row():
                    input_image_seg = gr.Image(label="Input Image")
                    output_image_seg = gr.Image(label="Output Image")

                operation_seg = gr.Radio(
                    choices=["Global Thresholding", "Adaptive Thresholding", "K-Means"],
                    value="Global Thresholding",
                    label="Select Segmentation"
                )
                thresh_seg = gr.Slider(0, 255, value=128, step=1, label="Threshold (Global)")
                block_seg = gr.Slider(3, 31, step=2, value=11, label="Block Size (Adaptive)")
                c_seg = gr.Slider(0, 10, value=2, step=1, label="C (Adaptive)")
                k_seg = gr.Slider(2, 10, value=3, step=1, label="K (K-Means)")

                run_seg = gr.Button("Segment")

                def _process_seg(img, op, th, bs, c_, k_):
                    return tab_segmentation(img, op, th, bs, c_, k_)

                run_seg.click(_process_seg,
                              inputs=[input_image_seg, operation_seg, thresh_seg, block_seg, c_seg, k_seg],
                              outputs=[output_image_seg])

            # Tab: Binary Processing
            with gr.TabItem("Binary Processing"):
                gr.Markdown("### Binary Image Processing")
                with gr.Row():
                    input_image_bin = gr.Image(label="Input Image")
                    output_image_bin = gr.Image(label="Output Image")

                operation_bin = gr.Radio(
                    choices=["Binarize", "Morphological", "Extract Boundary", "Skeletonize"],
                    value="Binarize",
                    label="Select Operation"
                )
                thresh_bin = gr.Slider(0, 255, value=127, step=1, label="Threshold")
                morph_op_bin = gr.Dropdown(
                    choices=["erosion", "dilation", "opening", "closing"],
                    value="erosion",
                    label="Morphological Operation"
                )
                ksize_bin = gr.Slider(1, 15, value=3, step=1, label="Kernel Size")
                iter_bin = gr.Slider(1, 10, value=1, step=1, label="Iterations")

                run_bin = gr.Button("Process Binary")

                def _process_bin(img, op, th, mop, ks, iters):
                    return tab_binary(img, op, th, mop, ks, iters)

                run_bin.click(_process_bin,
                              inputs=[input_image_bin, operation_bin, thresh_bin, morph_op_bin, ksize_bin, iter_bin],
                              outputs=[output_image_bin])

            # Tab: Restoration
            with gr.TabItem("Restoration"):
                gr.Markdown("### Image Restoration")
                with gr.Row():
                    input_image_rest = gr.Image(label="Input Image")
                    output_image_rest = gr.Image(label="Output Image")

                operation_rest = gr.Radio(
                    choices=["Add Noise + Wiener Filter", "Add Noise + Gaussian Filter", "Inpainting"],
                    value="Add Noise + Wiener Filter",
                    label="Select Restoration"
                )
                ksize_rest = gr.Slider(1, 15, value=5, step=1, label="Kernel Size (Wiener/Gaussian)")
                sigma_rest = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Sigma (Gaussian)")
                noise_var_rest = gr.Slider(0.001, 1.0, value=0.01, step=0.01, label="Noise Var (Wiener)")
                top_left_r = gr.Number(value=0, label="Inpaint Top Row")
                top_left_c = gr.Number(value=0, label="Inpaint Left Col")
                bot_right_r = gr.Number(value=50, label="Inpaint Bottom Row")
                bot_right_c = gr.Number(value=50, label="Inpaint Right Col")
                inpaint_rad = gr.Slider(1, 10, value=3, step=1, label="Inpaint Radius")

                run_rest = gr.Button("Restore")

                def _process_rest(img, op, ks, sg, nv, r1, c1, r2, c2, ipr):
                    return tab_restoration(img, op, ks, sg, nv, int(r1), int(c1), int(r2), int(c2), ipr)

                run_rest.click(_process_rest,
                               inputs=[input_image_rest, operation_rest, ksize_rest, sigma_rest, noise_var_rest,
                                       top_left_r, top_left_c, bot_right_r, bot_right_c, inpaint_rad],
                               outputs=[output_image_rest])

            # Tab: Matching
            with gr.TabItem("Matching"):
                gr.Markdown("### Image Matching")
                with gr.Row():
                    input_image_match1 = gr.Image(label="Main Image / Image1")
                    input_image_match2 = gr.Image(label="Template / Image2")
                output_image_match = gr.Image(label="Output Image")

                operation_match = gr.Radio(
                    choices=["Feature Detection & Matching", "Template Matching"],
                    value="Feature Detection & Matching",
                    label="Select Matching Operation"
                )
                feature_method = gr.Dropdown(choices=["ORB", "SIFT"], value="ORB", label="Feature Method")
                match_method = gr.Dropdown(choices=["BF", "FLANN"], value="BF", label="Match Method")
                templ_thresh = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Template Threshold")

                run_match = gr.Button("Match")

                def _process_match(img1, img2, op, fm, mm, thr):
                    return tab_matching(img1, img2, op, fm, mm, thr)

                run_match.click(_process_match,
                                inputs=[input_image_match1, input_image_match2, operation_match,
                                        feature_method, match_method, templ_thresh],
                                outputs=[output_image_match])
    return demo