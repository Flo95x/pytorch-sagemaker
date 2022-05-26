from setuptools import find_packages, setup

# with open('requirements.txt') as fp:
#    install_requires = fp.read()

INSTALL_REQUIRES = [
    'torch',
    'torchvision',
    'Pillow',
    'matplotlib',
    'sagemaker',
    'protobuf~=3.19.0', # hopefully fixes bug regarding sagemaker import ("TypeError: Descriptors cannot not be created directly.")
]

setup(name='torchmaker',
      version='0.1',
      description='Codebase for Pytorch Sagemaker deployment examples',
      author='Florian Teutsch',
      install_requires=INSTALL_REQUIRES,
      license='TODO',
      packages=find_packages(),
      zip_safe=False)
