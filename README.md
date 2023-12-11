# RAAI Module Vehicle Tracking

## Setup

You need to install all the required packages in the requirements.txt file. It is recommended to use a virtual environment. Replace `<VENV_NAME>` with the name of your virtual environment.

```bash
python.exe -m venv <VENV_NAME>
```

And activate it using:

```bash
call <VENV_NAME>\Scripts\activate.bat
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## Download pre-recorded videos (optional)

You can use the [download_resources.py](resources/download_resources.py) script to download pre-recorded videos from the NG:ITL cloud. The downloaded files are stored in the [resources](resources) folder.

## Starting

### Using the Camera Image Stream

Before you will be able to start the tracker, you will have to start the [raai_module_camera_image_stream](https://github.com/vw-wob-it-edu-ngitl/raai_module_camera_image_stream) from GitHub.

### Using a Video File

You can also use a video file instead of the camera image stream. To do so, you will have to change the `use_camera_image_stream` option in the [vehicle_tracking_config.json](vehicle_tracking_config.json) file to `false` and change the `video_file_path` option to the path of the video file you want to use.

### Region of Interest (optional)

Before starting the tracker you should define a region of interest (henceforth ROI) using the [roi_definer.py](vehicle_tracking/roi_definer.py). You will have to do it every time you move the camera or racetrack, as the ROI is defined in coordinates on the screen.

### Starting the Tracker

Once you have defined the ROI, you can start the tracker using the [main.py](main.py). You can change the launch options in the [vehicle_tracking_config.json](vehicle_tracking_config.json) file.

## Unit Testing

### Configuring the Unit Tests

You can use the [virtual_camera.py](tests/mocks/virtual_camera.py) file to generate frames to ease with testing, as you can choose what distractions to add by adding more objects. You can also use the [path_definer.py](tests/mocks/path_definer.py) to all the edge points of the drawn path copied to your clipboard with a correct format to be pasted into a python file.

### Running the Unit Tests

Currently there is only 1 unit test for the [vehicle_tracking.py](vehicle_tracking/vehicle_tracking.py) module. You can run it using tox which is listed under the requirements.txt.

```bash
tox -e tests
```

## Possible Ideas for Improvement

### Config Request

Currently the program uses a while loop and thread for waiting for the request to send the config. You could maybe update the program to be asynchronous.
