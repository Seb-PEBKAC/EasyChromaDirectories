""" Use Apple's Vision Framework via PyObjC to detect text in images 
To use:
python3 -m pip install pyobjc-core pyobjc-framework-Quartz pyobjc-framework-Vision wurlitzer
"""
import sys
import os
import pathlib
import argparse
from PIL import Image
# Add the directory containing the script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from Qwen2_VL import process_image_with_qwen2_vl
except ImportError as e:
    print("Warning: Qwen2_VL module not found. --qwen flag will not work.")
    print(f"Import error details: {e}")
    def process_image_with_qwen2_vl(*args, **kwargs):
        print("Qwen2_VL processing is not available. Please install required dependencies.")
        return None

import Quartz
import Vision
from Cocoa import NSURL
from Foundation import NSDictionary
# needed to capture system-level stderr
from wurlitzer import pipes
from pdf2image import convert_from_path


def image_to_text(img, lang="eng"):
    # Check if img is a PIL Image instance
    if isinstance(img, Image.Image):
        # Save the image to a temporary file
        img_path = "/tmp/temp_image.png"
        img.save(img_path)
    else:
        img_path = img

    input_url = NSURL.fileURLWithPath_(img_path)

    with pipes() as (out, err):
        input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)

    vision_options = NSDictionary.dictionaryWithDictionary_({})
    vision_handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        input_image, vision_options
    )
    results = []
    handler = make_request_handler(results)
    vision_request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    error = vision_handler.performRequests_error_([vision_request], None)

    return results

def make_request_handler(results):
    """ results: list to store results """
    if not isinstance(results, list):
        raise ValueError("results must be a list")

    def handler(request, error):
        if error:
            print(f"Error! {error}")
        else:
            observations = request.results()
            for text_observation in observations:
                recognized_text = text_observation.topCandidates_(1)[0]
                results.append([recognized_text.string(), recognized_text.confidence()])
    return handler

def pdf_to_images(pdf_path):
    # Convert PDF to a list of PIL images
    images = convert_from_path(pdf_path)
    return images

def test_ocr():
    """Test function to verify OCR functionality"""
    from create_test_assets import create_test_image
    
    # Create a test image
    test_image_path = create_test_image()
    
    # Process with OCR
    results = image_to_text(test_image_path)
    
    print("Test Results:")
    print("Input Image:", test_image_path)
    print("OCR Results:", results)
    
    # Test Qwen integration if available
    try:
        ocr_text = " ".join([text for text, _ in results])
        qwen_output = process_image_with_qwen2_vl([test_image_path], ocr_text)
        print("Qwen2-VL Output:", qwen_output)
    except Exception as e:
        print("Qwen2-VL test failed:", str(e))
    
    return results

def main():
    import sys
    import pathlib

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process images with Apple OCR and optionally pass to Qwen2-VL.")
    parser.add_argument("file_path", type=str, help="Path to the image or PDF file.")
    parser.add_argument("--qwen", action="store_true", help="Pass OCR text to Qwen2-VL for further processing.")
    args = parser.parse_args()

    file_path = pathlib.Path(args.file_path)

    if not file_path.is_file():
        sys.exit("Invalid file path")
    file_path = str(file_path.resolve())

    # Determine file type and process accordingly
    if file_path.lower().endswith('.pdf'):
        images = pdf_to_images(file_path)
    else:
        images = [Image.open(file_path)]

    # Process each image with OCR
    ocr_results = []
    for img in images:
        results = image_to_text(img)
        ocr_results.extend(results)

    # Print OCR results
    print("OCR Results:", ocr_results)

    # If --qwen flag is set, pass OCR text to Qwen2-VL
    if args.qwen:
        try:
            print(f"\nProcessing {len(images)} images with Qwen2-VL...")
            ocr_text = " ".join([text for text, _ in ocr_results])  # Combine OCR text
            qwen_output = process_image_with_qwen2_vl(images, ocr_text)
            print("\nQwen2-VL Output:")
            print(qwen_output)
        except Exception as e:
            print(f"\nError in Qwen2-VL processing: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Run test if no arguments provided
        test_ocr()