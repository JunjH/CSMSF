
Trained Models
-
Download the trained models for baseline approaches including early, intermediate, and late fusion and our method via the link [link](https://drive.google.com/file/d/1FzyOU8gC_xLr590y-VNu22L0j9ccZ1yD/view?usp=drive_link) <br>


Running
-

  
	| Models | Description |
| --- | --- |
| [test_EF.py] | evaluating performance of the early fusion of vision, LiDAR, and thermal data |
| [test_IF.py] | evaluating performance of the intermediate fusion of vision, LiDAR, and thermal data |
| [test_LF.py] | evaluating performance of the late fusion of vision, LiDAR, and thermal data |
| [test_our.py] | evaluating performance of the prposed method for fusion of vision, LiDAR, and thermal data|
| [test_all_ta.py] | evaluating performance of the prposed method for fusion of vision, LiDAR, and thermal data with the introduced trustworthiness assessment method|

+ ### Train<br>
	| Models | Description |
| --- | --- |
| [train_EF.py] | training the early fusion model of vision, LiDAR, and thermal data |
| [train_IF.py] | training the intermediate fusion model of vision, LiDAR, and thermal data |
| [train_LF.py] | training the late fusion model of vision, LiDAR, and thermal data |
| [train_v.py] | training the model for only vision |
| [train_l.py] | training the model for only lidar |
| [train_t.py] | training the model for only thermal |
| [train_vl.py] | training the model for vision-lidar fusion |
| [train_lt.py] | training the model for lidar-thermal fusion |
| [train_vt.py] | training the model for vision-thermal fusion|
| [train_all_our.py] | training the prposed model for fusion of vision, LiDAR, and thermal data|
