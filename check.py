from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# Directory to save the image
image_dir = "./"
os.makedirs(image_dir, exist_ok=True)

def fetch_sirius_data():
    try:
        # Query Simbad for the star Sirius
        result = Simbad.query_object("Sirius")
        if result:
            print("Sirius Data Retrieved Successfully!")
            print(result)
            # Save the data to a CSV file for further use
            result.write("sirius_data.csv", format="csv", overwrite=True)
            print("Data saved to 'sirius_data.csv'")
        else:
            print("No data found for Sirius.")
    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_sirius_image():
    try:
        # Query SkyView for an image of Sirius
        images = SkyView.get_images(position='Sirius', survey='DSS', pixels='300,300')
        if images:
            # Convert FITS file to an image
            fits_data = images[0][0]
            img_name = os.path.join(image_dir, "sirius.jpg")
            
            # Open the FITS file and plot it
            data = fits_data.data
            plt.figure()
            plt.imshow(data, cmap='gray', origin='lower')
            plt.axis('off')
            plt.savefig(img_name, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            print(f"Image of Sirius saved as {img_name}")
        else:
            print("No image found for Sirius.")
    except Exception as e:
        print(f"Failed to fetch image for Sirius: {e}")

# Run the script
fetch_sirius_data()
fetch_sirius_image()
