#!/usr/bin/env python3

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import sys
import cv2
import numpy as np

def hex_to_rgb(hex_str):
    """Converts a hex color string (e.g., '#FF0000' or 'FFF') to an RGB tuple."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color format: {hex_str}")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def bake_exr_to_png(exr_path, png_path, bg_color_hex):
    try:
        bg_rgb = hex_to_rgb(bg_color_hex)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load EXR with raw floating-point channels intact
    exr_img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if exr_img is None:
        print(f"Error: Could not load EXR file at '{exr_path}'", file=sys.stderr)
        sys.exit(1)
        
    if exr_img.shape[2] != 4:
        print("Error: The provided EXR file does not contain an alpha channel.", file=sys.stderr)
        sys.exit(1)

    # Split channels (OpenCV handles images in BGR/BGRA order)
    b_fg, g_fg, r_fg, a_fg = cv2.split(exr_img)

    # Normalize background color (0-255 to 0.0-1.0 float) to match OpenCV BGR order
    b_bg = bg_rgb[2] / 255.0
    g_bg = bg_rgb[1] / 255.0
    r_bg = bg_rgb[0] / 255.0

    # True Premultiplied Compositing Math: C_final = F_rgb + B_rgb * (1 - Alpha)
    b_final = b_fg + b_bg * (1.0 - a_fg)
    g_final = g_fg + g_bg * (1.0 - a_fg)
    r_final = r_fg + r_bg * (1.0 - a_fg)

    baked_float = cv2.merge((b_final, g_final, r_final))

    # Linear to sRGB Gamma Correction (sRGB = Linear^(1/2.2))
    baked_srgb = np.power(np.clip(baked_float, 0, None), 1.0 / 2.2)

    # Quantization & Clipping (Scale to 8-bit integers)
    baked_8bit = np.clip(baked_srgb * 255.0, 0, 255).astype(np.uint8)

    # Save final flat PNG
    if cv2.imwrite(png_path, baked_8bit):
        print(f"Success: Baked '{exr_path}' onto background '{bg_color_hex}' -> '{png_path}'")
    else:
        print(f"Error: Could not write PNG file to '{png_path}'", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Bakes a 32-bit Premultiplied EXR render over a solid color background into a flat PNG, protecting emissive glows from alpha destruction."
    )
    parser.add_argument("input_exr", help="Path to the input Blender OpenEXR file (.exr)")
    parser.add_argument("output_png", help="Path where the output PNG file should be saved (.png)")
    parser.add_argument(
        "-c", "--color", 
        default="#FFFFFF", 
        help="Hex color code of your target vector canvas (e.g., #FFFFFF, #4a4a4a, or F0F). Default is white."
    )

    args = parser.parse_args()
    bake_exr_to_png(args.input_exr, args.output_png, args.color)

if __name__ == "__main__":
    main()
