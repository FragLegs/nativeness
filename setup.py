# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
    name='nativeness',
    version='0.0.1',
    description='Model for detecting ELL writing',
    url='https://github.com/FragLegs/nativeness',
    author='Shayne Miel',
    license='Copyright ©2017',
    packages=setuptools.find_packages(exclude=['tests*']),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        'dill==0.2.6',
        'pandas==0.19.2',
        'tensorflow==2.5.3',
        'scipy==0.18.1',
        'matplotlib==2.0.0',
        'scikit-learn==0.18.1'
    ]
)
