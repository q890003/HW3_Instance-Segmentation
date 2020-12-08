---
Selected-Topics-in-Visual-Recognition-using-Deep-Learning HW3
---
<!-- TOC -->

- [Instance Segmentation](#instance-segmentation)
    - [Reproducing the work](#reproducing-the-work)
        - [Enviroment Installation](#enviroment-installation)
        - [Project installation](#project-installation)
    - [Training](#training)
    - [Inference](#inference)

<!-- /TOC -->
# Instance Segmentation
![](https://i.imgur.com/rNpyJmf.png)

## Reproducing the work
### Enviroment Installation
1. install annoconda
2. create python3.x version 
    ```
    take python3.6 for example
    $ conda create --name (your_env_name) python=3.6
    $ conda activate (your_env_name)
    ```
3. install pytorch 
    - check GPU
        - [Check GPU version](https://www.nvidia.com/Download/index.aspx?lang=cn%20) first and check if CUDA support your GPU.
    - [pytorch](https://pytorch.org/get-started/locally/)
### Project installation
1. clone this repository
    ``` 
    git clone https://github.com/q890003/HW3_Instance-Segmentation.git
    ```
2. Data
    1. Download Official Image: 
        - [Test data](https://drive.google.com/file/d/1VbAitjYKun3Tgc-Tl_wDCEEUO4iqlyhi/view?usp=sharing)
            - [test_image_id](https://drive.google.com/file/d/1hVdPqCiprp888o2yRvV0mYQrumdVTRxA/view?usp=sharing)
        - [Train data](https://drive.google.com/file/d/1XBcZ5-gwtK7SeU1N9f9Oswove09HrUgu/view?usp=sharing)
            - [annotation](https://drive.google.com/file/d/1cVO0aBAXm4XBdxu2PGg7bxwOFV_S5Zu8/view?usp=sharing)

    2. Put (Test/Train) data to folder, **data/**, under the root dir of this project. 
        ```
        |- HW3_Instance-Segmentation
            |- data/
                |- test.json
                |- test_images.zip
                |- pascal_train.json 
                |- train_images.zip
            |- checkpoints/ (Need manual creat)
                |- (step3. parameter_file of model)
            |- results/     (Need manual creat)
            |- datasets/    (For pytorch dataloader)
            |- .README.md
            |- train.py
            |- eval.py
            |- model.py
            |- config.py
            ...
        ```
    3. Decompress the (Test/Train) data
        ```
        At dir HW3_Instance-Segmentation/data/
        $ unzip test_images.zip
        $ unzip train_images.zip
        ```

4. Downoad fine-tuned parameters
    - [Mask-RCNN with backbone:resnest50d](https://drive.google.com/file/d/1threB0RzP_1qh-tFtTBLZM_-e0lTP4HO/view?usp=sharing)
    - put the parameter file to checkpoints folder.
## Training
```
$ python train.py
``` 
## Inference

```
$ python eval.py
```

