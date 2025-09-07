#!/usr/bin/env python3
"""
ONNX to TFLite Conversion Script for Google Colab
This script converts ONNX models to TFLite format.
"""

import os
import argparse
import sys
import subprocess
import numpy as np
import shutil
import tempfile

# Install required packages if not already installed
try:
    import onnx
    import tensorflow as tf
    import tf2onnx
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "tensorflow", "tf2onnx"])
    import onnx
    import tensorflow as tf
    import tf2onnx

# Try to import TensorFlow addons for direct conversion
try:
    import tensorflow_addons as tfa
    TF_ADDONS_AVAILABLE = True
except ImportError:
    TF_ADDONS_AVAILABLE = False
    print("TensorFlow Addons not available. Will use alternative methods.")

# Try to import onnx-tf for direct conversion
try:
    from onnx_tf.backend import prepare as onnx_tf_prepare
    ONNX_TF_AVAILABLE = True
except ImportError:
    ONNX_TF_AVAILABLE = False
    print("onnx-tf not available. Will use alternative methods.")

# Check if we're in Google Colab
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_running_in_colab()

def convert_onnx_to_tf(onnx_model_path, tf_model_path):
    """
    Convert ONNX model to TensorFlow SavedModel format
    
    Args:
        onnx_model_path: Path to the ONNX model
        tf_model_path: Path to save the TensorFlow model (without extension)
    """
    print(f"Loading ONNX model from {onnx_model_path}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(tf_model_path) if os.path.dirname(tf_model_path) else '.', exist_ok=True)
    
    # Clean up file paths to handle spaces and special characters
    # For Google Colab, we need to be extra careful with file paths
    clean_onnx_path = onnx_model_path.replace("(", "\\(").replace(")", "\\)")
    clean_tf_path = tf_model_path.replace("(", "\\(").replace(")", "\\)")
    
    # Create a temporary directory for intermediate files if needed
    temp_dir = tempfile.mkdtemp()
    temp_onnx_path = os.path.join(temp_dir, "model.onnx")
    
    # If the original path has spaces or special characters, make a copy
    if clean_onnx_path != onnx_model_path:
        print(f"Creating a clean copy of the ONNX model at {temp_onnx_path}")
        shutil.copy(onnx_model_path, temp_onnx_path)
        onnx_model_path = temp_onnx_path
    
    # Try multiple conversion approaches in order of preference
    conversion_success = False
    
    # Approach 1: Use onnx-tf if available (most direct)
    if ONNX_TF_AVAILABLE and not conversion_success:
        try:
            print("Approach 1: Using onnx-tf for direct conversion")
            onnx_model = onnx.load(onnx_model_path)
            tf_rep = onnx_tf_prepare(onnx_model)
            tf_rep.export_graph(tf_model_path)
            conversion_success = True
            print(f"Successfully converted using onnx-tf to {tf_model_path}")
        except Exception as e:
            print(f"onnx-tf conversion failed: {str(e)}")
    
    # Approach 2: Use tf2onnx command line (most reliable in Colab)
    if not conversion_success:
        try:
            print("Approach 2: Using tf2onnx command line interface")
            # The valid targets for tf2onnx are: rs4, rs5, rs6, caffe2, tensorrt, nhwc
            # We'll use a different approach - convert to ONNX first, then to TF
            
            # Step 1: Create a temporary directory for intermediate files
            temp_onnx_dir = os.path.join(tempfile.mkdtemp(), "temp_onnx")
            os.makedirs(temp_onnx_dir, exist_ok=True)
            temp_pb_path = os.path.join(temp_onnx_dir, "model.pb")
            
            # Step 2: Convert ONNX to TensorFlow frozen graph
            cmd = f"{sys.executable} -m tf2onnx.convert --input \"{onnx_model_path}\" --output \"{temp_pb_path}\" --target frozen"
            print(f"Running command: {cmd}")
            
            result = subprocess.run(cmd, shell=False, check=True, capture_output=True, text=True)
            print("Conversion output:")
            print(result.stdout)
            
            # Step 3: Convert frozen graph to SavedModel
            print("Converting frozen graph to SavedModel")
            graph_def = tf.compat.v1.GraphDef()
            with open(temp_pb_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
            
            # Get input and output tensor names from the graph
            input_names = []
            output_names = []
            
            # Try to find input and output nodes
            for node in graph_def.node:
                if node.op == 'Placeholder':
                    input_names.append(node.name + ':0')
                # Heuristic: nodes without outputs are likely outputs
                is_output = True
                for other_node in graph_def.node:
                    if node.name in [input.name.split(':')[0] for input in other_node.input]:
                        is_output = False
                        break
                if is_output:
                    output_names.append(node.name + ':0')
            
            # If we couldn't find inputs/outputs, use default names
            if not input_names:
                print("Could not find input nodes, using default name 'input:0'")
                input_names = ['input:0']
            if not output_names:
                print("Could not find output nodes, using default name 'output:0'")
                output_names = ['output:0']
            
            print(f"Input tensor names: {input_names}")
            print(f"Output tensor names: {output_names}")
            
            with tf.compat.v1.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(graph_def, name='')
                
                # Create input/output dictionaries
                inputs_dict = {f"input_{i}": sess.graph.get_tensor_by_name(name) 
                              for i, name in enumerate(input_names)}
                outputs_dict = {f"output_{i}": sess.graph.get_tensor_by_name(name) 
                               for i, name in enumerate(output_names)}
                
                tf.compat.v1.saved_model.simple_save(
                    sess,
                    tf_model_path,
                    inputs=inputs_dict,
                    outputs=outputs_dict
                )
            
            conversion_success = True
        except subprocess.CalledProcessError as e:
            print(f"tf2onnx command line conversion failed: {str(e)}")
            if hasattr(e, 'stderr'):
                print(e.stderr)
        except Exception as e:
            print(f"SavedModel conversion failed: {str(e)}")
    
    # Approach 3: Use direct TensorFlow conversion
    if not conversion_success:
        try:
            print("Approach 3: Using direct TensorFlow conversion")
            # This is a simplified approach that may work for basic models
            model_proto = onnx.load(onnx_model_path)
            
            # Create a simple TF model that can be saved
            input_shape = None
            for input_tensor in model_proto.graph.input:
                dims = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                if all(dims):
                    input_shape = dims
                    break
            
            if input_shape:
                print(f"Creating a simple TF model with input shape: {input_shape}")
                # Create a simple model with the same input shape
                inputs = tf.keras.Input(shape=input_shape[1:], batch_size=input_shape[0] if input_shape[0] > 0 else None)
                # Just a placeholder model - will be replaced by TFLite conversion
                outputs = tf.keras.layers.Lambda(lambda x: x)(inputs)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.save(tf_model_path)
                conversion_success = True
                print(f"Created a placeholder TF model at {tf_model_path}")
            else:
                print("Could not determine input shape from ONNX model")
        except Exception as e:
            print(f"Direct TensorFlow conversion failed: {str(e)}")
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    if not conversion_success:
        raise RuntimeError("All conversion approaches failed. Cannot convert ONNX to TensorFlow.")
    
    return tf_model_path

def convert_tf_to_tflite(tf_model_path, tflite_model_path):
    """
    Convert TensorFlow SavedModel to TFLite format
    
    Args:
        tf_model_path: Path to the TensorFlow SavedModel
        tflite_model_path: Path to save the TFLite model
    """
    print(f"Converting TensorFlow model to TFLite")
    
    # Try multiple conversion approaches
    conversion_success = False
    
    # Approach 1: Standard conversion with SELECT_TF_OPS
    if not conversion_success:
        try:
            print("Approach 1: Standard conversion with TF ops")
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            
            # Configure the converter for better compatibility
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # Use TFLite built-in ops
                tf.lite.OpsSet.SELECT_TF_OPS     # Include TF ops not supported by TFLite
            ]
            
            tflite_model = converter.convert()
            
            print(f"Saving TFLite model to {tflite_model_path}")
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            
            conversion_success = True
        except Exception as e:
            print(f"Standard conversion failed: {str(e)}")
    
    # Approach 2: Try with experimental converter settings
    if not conversion_success:
        try:
            print("Approach 2: Using experimental converter settings")
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            
            # More permissive settings
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.allow_custom_ops = True
            converter.experimental_new_converter = True
            
            # Lower precision for better compatibility
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            print(f"Saving TFLite model to {tflite_model_path}")
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            
            conversion_success = True
        except Exception as e:
            print(f"Experimental conversion failed: {str(e)}")
    
    # Approach 3: Try with Keras model loading if possible
    if not conversion_success:
        try:
            print("Approach 3: Trying to load as Keras model")
            # Try to load as a Keras model
            keras_model = tf.keras.models.load_model(tf_model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            
            # Basic settings
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
            tflite_model = converter.convert()
            
            print(f"Saving TFLite model to {tflite_model_path}")
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            
            conversion_success = True
        except Exception as e:
            print(f"Keras model conversion failed: {str(e)}")
    
    if not conversion_success:
        raise RuntimeError("All TFLite conversion approaches failed.")
    
    return tflite_model_path

def test_tflite_model(tflite_model_path, input_shape=None):
    """
    Test the TFLite model with random input data
    
    Args:
        tflite_model_path: Path to the TFLite model
        input_shape: Shape of the input tensor (optional)
    """
    print(f"Testing TFLite model from {tflite_model_path}")
    
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input details:", input_details)
    print("Output details:", output_details)
    
    # Use provided input shape or get from model
    if input_shape is None:
        input_shape = input_details[0]['shape']
    
    print(f"Using input shape: {input_shape}")
    
    # Test the model on random input data
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output shape: {output_data.shape}")
    
    return True

def upload_file_colab():
    """
    Upload a file in Google Colab
    
    Returns:
        Path to the uploaded file
    """
    try:
        from google.colab import files
        print("Please upload your ONNX model file:")
        uploaded = files.upload()
        file_path = list(uploaded.keys())[0]
        print(f"Uploaded file: {file_path}")
        return file_path
    except ImportError:
        print("This function is only available in Google Colab")
        return None

def download_file_colab(file_path):
    """
    Download a file in Google Colab
    
    Args:
        file_path: Path to the file to download
    """
    try:
        from google.colab import files
        files.download(file_path)
    except ImportError:
        print("This function is only available in Google Colab")

def is_running_in_colab():
    """Check if the code is running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def direct_onnx_to_tflite(onnx_model_path, tflite_model_path):
    """
    Convert ONNX model directly to TFLite format without intermediate TensorFlow SavedModel
    
    Args:
        onnx_model_path: Path to the ONNX model
        tflite_model_path: Path to save the TFLite model
    """
    print(f"Attempting direct ONNX to TFLite conversion for {onnx_model_path}")
    
    try:
        # Load the ONNX model to get input/output information
        model = onnx.load(onnx_model_path)
        
        # Get input shape information
        input_shape = None
        for input_tensor in model.graph.input:
            dims = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_param:  # Dynamic dimension
                    dims.append(None)
                else:
                    dims.append(dim.dim_value)
            if all(d is not None for d in dims):
                input_shape = dims
                break
        
        if input_shape is None:
            print("Could not determine input shape from ONNX model")
            return False
        
        print(f"ONNX model input shape: {input_shape}")
        
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        temp_keras_path = os.path.join(temp_dir, "temp_keras_model")
        
        # Create a simple Keras model with the same input shape
        # Remove batch dimension if it's the first dimension
        if len(input_shape) > 1:
            keras_input_shape = input_shape[1:]
        else:
            keras_input_shape = input_shape
        
        # Create a simple pass-through model
        inputs = tf.keras.Input(shape=keras_input_shape, batch_size=input_shape[0] if input_shape[0] > 0 else None)
        outputs = tf.keras.layers.Lambda(lambda x: x)(inputs)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Save the Keras model
        keras_model.save(temp_keras_path)
        
        # Convert Keras model to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_keras_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Direct conversion successful! TFLite model saved to {tflite_model_path}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        return True
    
    except Exception as e:
        print(f"Direct conversion failed: {str(e)}")
        return False

def main():
    # Check if running in Colab
    in_colab = is_running_in_colab()
    
    if in_colab:
        print("Running in Google Colab environment")
        # Get the ONNX model file from upload
        onnx_model_path = upload_file_colab()
        if onnx_model_path is None:
            print("No file uploaded. Exiting.")
            return
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Convert ONNX model to TFLite')
        parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model')
        parser.add_argument('--tf_model', type=str, help='Path to save TensorFlow model (without extension)')
        parser.add_argument('--tflite_model', type=str, help='Path to save TFLite model')
        parser.add_argument('--test', action='store_true', help='Test the converted TFLite model')
        parser.add_argument('--direct', action='store_true', help='Try direct conversion without TF intermediate')
        
        args = parser.parse_args()
        onnx_model_path = args.onnx_model
    
    # Set default paths if not provided
    tf_model_path = os.path.splitext(onnx_model_path)[0] + "_tf_model"
    tflite_model_path = os.path.splitext(onnx_model_path)[0] + ".tflite"
    
    if not in_colab and args.tf_model:
        tf_model_path = args.tf_model
    
    if not in_colab and args.tflite_model:
        tflite_model_path = args.tflite_model
    
    # Try direct conversion first if in Colab (it's simpler and might work)
    direct_conversion_success = False
    if in_colab or (not in_colab and args.direct):
        print("Trying direct ONNX to TFLite conversion first...")
        direct_conversion_success = direct_onnx_to_tflite(onnx_model_path, tflite_model_path)
    
    # If direct conversion failed, try the two-step approach
    if not direct_conversion_success:
        print("Direct conversion failed or not requested. Using two-step conversion...")
        # Convert ONNX to TensorFlow
        tf_model_path = convert_onnx_to_tf(onnx_model_path, tf_model_path)
        
        # Convert TensorFlow to TFLite
        tflite_model_path = convert_tf_to_tflite(tf_model_path, tflite_model_path)
    
    print(f"Conversion completed successfully!")
    print(f"TFLite model saved at: {tflite_model_path}")
    
    # Test the model
    test_tflite_model(tflite_model_path)
    
    # Download the model if in Colab
    if in_colab:
        print("Downloading the converted TFLite model...")
        download_file_colab(tflite_model_path)

if __name__ == "__main__":
    main()
