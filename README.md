# STAHZAI - A Really Awesome Project 
**Project Title: Star Seekers: AI Mapping of Stellar Types and Habitable Zones (STAHZAI)**

**Favorite topic Idea #1:** AI Model that can predict Star Composition, Classification, and Distance to Habitable Zone

**Summary of your topic choice: Summarize in a 6-10 sentence paragraph what you are planning on doing as well as how the results of your experiment can be used - what is the purpose? Be as specific as possible & include your background research you collected!!**    

For our project, we are developing an AI model that can analyze star spectroscopy graphs to classify the star type and predict the distance to its habitable zone (if we get far enough, more helpful features). The model will process both the visual and spectral characteristics of stars, so we plan on using Python neural networks for image classification and multilayer perceptrons (MLPs) for the spectral data. This project is grounded in the fact that stars are classified based on their spectral types (O, B, A, F, G, K, M), which correspond to temperature and luminosity, key factors in determining the location of their habitable zones. The habitable zone is the region around a star where liquid water could potentially exist on a planet, making it crucial in the search for exoplanets.

The results of this experiment could be applied in astronomy to identify stars with potentially habitable exoplanets by predicting where their habitable zones are located. This could make the search for Earth-like planets much easier and more efficient, assisting astronomers in focusing their resources on stars most likely to harbor life-supporting planets. Additionally, the AI model can contribute to the growing world of AI and space research by demonstrating how machine learning can improve the efficiency and accuracy of astronomical data analysis. It may even expand to identify other important characteristics of stars that might assist astronomers even better (hopefully NASA is benefited!!)

**“Big Question” you are asking:**
Can an AI model accurately predict star composition, classification (if we get far enough then also habitable zone) by analyzing a spectroscopy diagram of a star?

**Background research:**
Doesn’t currently exist for AI predicting star composition

- Spectrum of star can be caused by chemistry composition and temperature 

- Composition of most stars is like sun

- Spectroscopy data is already available, key to our use

- Different elements are created in different temperatures

- Pressure affects spectrum by number of collisions and ionization

- Photosphere(visible layer of star) can determine which element lines appear

- Some elements have identical wavelengths 

- H and He make around 96-99% of mass

- Must take into account Doppler effect (forward and backwards(shifted) and rotation(thicker) lines)

- Luminosity can create thicker lines

- Atharv already has a python program that will test different AI builds and output the most accurate one, so we can implement this into our STAHZAI (In the Github)

**Links to sources :**
Link to sdss: http://cas.sdss.org/dr18/VisualTools/navi

Link to GAIA Star Data: https://cdn.gea.esac.esa.int/Gaia/gdr3/Astrophysical_parameters/astrophysical_parameters/ 

Data example: http://specdash.idies.jhu.edu/?catalog=sdss&specid=4252649379880785920

Spectra of stars: https://courses.lumenlearning.com/suny-astronomy/chapter/the-spectra-of-stars-and-brown-dwarfs/ (go to the end) 

**Hypothesis (must be testable in the format shown below):** 
(example: "If we make several plane designs, then the NACA 0015 wing design will be able to carry the plane the farthest.)

HYPOTHESIS CAN BE AUTOMATED USING ATHARV’S COMPUTER PROGRAM, WE ARE REALLY JUST CODING AN AI MODEL :D

If we set up an AI with epochs=100 and batch_size=32 , it will produce the most accurate AI model.

(THE BLANKS MUST STILL BE DETERMINED USING MORE RESEARCH)

**Independent Variable (what you are manipulating/the condition you are changing):** 

Structure of Neural Network (how we code the AI)

**Dependent Variable (the response, or the data you are collecting):**

Accuracy of guesses (of composition)

**Control Group (if applicable):** 

Human-made data

**Constant conditions:**
 
Same test data

**Materials:**

Computer 

