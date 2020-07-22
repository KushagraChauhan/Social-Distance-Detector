# Social Distance Detector

---
### In the wake of the global pandemic COVID-19, a new term has originated namely Social Distancing.

This is my effort of detecting the number of violations of social-distancing norm in a video.

### How to setup-
1. Install Conda from here - [Conda Install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

2. Clone the repo to your local machine-
```sh
    git clone https://github.com/KushagraChauhan/Social-Distance-Detector.git
```
3. Download the file-
[Yolov3 Weights](https://pjreddie.com/media/files/yolov3.weights.)
and place it in the folder: 'yolo-coco'

4. Create a conda environment-
```sh
    conda env create -f my-env.yml
```
5. Activate the conda environment-
```sh
    conda activate my-env
```
6. Start the Flask server-
```sh
    python3 app.py
```
7. After that go to the browser and open 
```sh
    localhost:5000
```
The browser will start using your webcam and detect the social distance violations



