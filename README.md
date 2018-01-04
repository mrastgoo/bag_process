# bag_process python package

This package initially was designed to be used in the context of the ANR VIPeR project. 

`bag_process` package is based on [rosbag_pandas](https://pypi.python.org/pypi/rosbag_pandas) and is specificaly designed to sort the images and IMU information from the bag files in the format required by [kalibr](https://github.com/ethz-asl/kalibr) toolbox. 

## include

```
	bag_process
	|__  bag
		|__bag
	|__ utils 
		|__ utils
	|__ extract_data.py
	|__ kalibr_preprocess.py
	|__ kalibr_results_analysis.py

```

## required packages 
* opencv
* rosbag_pandas


