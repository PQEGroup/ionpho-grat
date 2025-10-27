import hdf5storage
import scipy.io
import os
import sys

def convert_mat_to_v73(input_file, output_file=None):
    """
    Converts an older .mat file to the v7.3 format.

    Args:
        input_file (str): Path to the input .mat file.
        output_file (str, optional): Path to save the converted .mat file (v7.3).
                                    If None, will use input_file with '_v73' suffix.
    """
    try:
        # Generate output filename if not provided
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_v73{ext}"
            
        print(f"Loading file: {input_file}")
        try:
            data = scipy.io.loadmat(input_file)
        except NotImplementedError:
            data = hdf5storage.read(input_file)
            
        print(f"Saving to v7.3 format: {output_file}")
        #print(data)
        hdf5storage.write(data, path='/', filename=output_file, matlab_compatible=True)
        print("Conversion successful!")
        return output_file
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_mat_to_v73.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_mat_to_v73(input_file, output_file) 