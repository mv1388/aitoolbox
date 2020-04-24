from setuptools import setup, find_packages


setup(name='aitoolbox',
      version='1.0',
      author='Marko Vidoni',
      author_email='',
      url='https://github.com/mv1388/AIToolbox',
      description='Toolbox of useful functions often needed in different AI oriented projects',
      long_description=open('README.md').read(),

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

      # scripts=['bin/AWS/'],

      zip_safe=False)
