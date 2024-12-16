# RAAI Module Vehicle Tracking

## Introduction
This project enables vehicle tracking in videos or live streams. The setup and configuration of the tracker are modular, covering various use cases.

## Setup

### Create a Virtual Environment
1. Create a virtual environment. Replace `<VENV_NAME>` with the name of your virtual environment.

```bash
python.exe -m venv <VENV_NAME>
call <VENV_NAME>\Scripts\activate.bat
pip install -r requirements.txt
```

## Usage

### 1. Pre-Recorded Videos (Optional)
Use the `download_resources.py` script to download pre-recorded videos from the NG:ITL Cloud. The files will be stored in the [resources](resources) folder.

```bash
python resources/download_resources.py
```

### 2. Live Stream (Camera Stream)
If using a camera stream, first start the module [raai_module_camera_image_stream](https://github.com/vw-wob-it-edu-ngitl/raai_module_camera_image_stream) from GitHub.

### 3. Using a Video File
To use a video file instead of the camera stream:
1. Modify the file [vehicle_tracking_config.json](vehicle_tracking_config.json):
    - Set `use_camera_image_stream` to `false`.
    - Set `video_file_path` to the path of your desired video file.

### 4. Define Region of Interest (ROI)
Define the Region of Interest (ROI) before starting the tracker to optimize tracking. This is especially important if the camera position or track layout changes.

Use the script [roi_definer.py](vehicle_tracking/roi_definer.py):

```bash
python vehicle_tracking/roi_definer.py
```


### 5. Start the Tracker
Start the tracker using the file [main.py](main.py). Adjust launch options in the [vehicle_tracking_config.json](vehicle_tracking_config.json) file.

```bash
python main.py
```


## Configuration

### Manual Configuration
Tracker settings can be manually adjusted in the file [vehicle_tracking_config.json](vehicle_tracking_config.json).

### Configurator Module
Alternatively, use the [Configurator Module](https://github.com/vw-wob-it-edu-ngitl/raai_module_vehicle_tracking_configurator/), which provides a visual interface.
- **Steps:**
  1. Start the tracker.
  2. Start the configurator.

```bash
python configurator_module/main.py
```


## Unit Tests

### 1. Test Preparation
The following tools are available to facilitate testing:
- **Virtual Camera:**
  - Generate frames using [virtual_camera.py](tests/mocks/virtual_camera.py) to simulate test conditions.
- **Path Definer:**
  - Use [path_definer.py](tests/mocks/path_definer.py) to automatically copy edge points of a path.

### 2. Running Unit Tests
Currently, there is one unit test for the module [vehicle_tracking.py](vehicle_tracking/vehicle_tracking.py). All tests can be run using `tox`:

```bash
tox -e tests
```


## Ideas for Improvement

### 1. Asynchronous Configuration Requests
The program currently uses a while loop and threads to wait for configuration requests. This could be optimized by implementing an asynchronous approach.

### 2. Verify Pynng Communication
Ensure that all sent data is actually being utilized.
