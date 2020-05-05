from setuptools import setup, find_packages


setup(
    name='aitoolbox',
    version='1.0',
    author='Marko Vidoni',
    author_email='',
    url='https://github.com/mv1388/AIToolbox',
    description='PyTorch Model Training and Experiment Tracking Framework',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license='MIT',
    keywords=['PyTorch', 'deep learning', 'research', 'train loop'],

    python_requires='>=3.6.0',

    packages=find_packages(exclude=['tests', 'examples', 'deprecated']),

    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',

        'torch',
        'torchvision',
        'torchtext',
        'pytorch-nlp',

        'joblib',
        'tqdm',
        'awscli',
        'boto3',
        'botocore',
        'google-cloud-storage',

        'nltk',
        'allennlp',
        'pyrouge',
        'rouge'
    ],

    test_suite='tests',
    tests_require=['nose'],

    zip_safe=False,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)
