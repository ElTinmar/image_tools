import sys
import numpy as np
from qtpy.QtWidgets import QApplication
from image_tools import CloneTool 

image = np.load('/media/martin/DATA/Mecp2/processed/2024_09_25_04_MeCP2_fish1_chunk_001.npy') 

def main():
    app = QApplication(sys.argv) 
    window = CloneTool(image) 
    window.show() 
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()