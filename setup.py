from setuptools import setup

import os
from glob import glob
from urllib.request import urlretrieve

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
        (os.path.join('models', package_name), glob('./models/*.py')),
        (os.path.join('utils', package_name), glob('./utils/*.py')),
        # (os.path.join('bbox_ex_msgs', package_name), glob('./bbox_ex_msgs/*.py')),
        # (os.path.join('share', package_name), glob('../weights/*.pth'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Ar-Ray-code',
    author_email="ray255ar@gmail.com",
    maintainer='Ar-Ray-code',
    maintainer_email="ray255ar@gmail.com",
    description='YOLOv5 + ROS2 Foxy',
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    tests_require=['pytest'],
    entry_points={
        # 'console_scripts': [
        #     'yolov5_ros = '+package_name+'.main:ros_main',
        # ],
        'console_scripts': [
            'vision = yolov5_ros.vision:ros_main'
        ],
    },
    py_modules=[
        'yolov5_ros.models.common',
        'yolov5_ros.utils.general',
        'yolov5_ros.utils.plots',
        'yolov5_ros.utils.torch_utils',
        'yolov5_ros.utils.datasets',
        'yolov5_ros.utils.augmentations',
        'yolov5_ros.utils.downloads',
        'yolov5_ros.utils.metrics',
        'yolov5_ros.bbox_ex_msgs.msg'
        # 'yolov5_ros.bbox_ex_msgs'
    ]
)