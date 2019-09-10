from setuptools import setup, find_packages


setup(name='AIToolbox',
      version='0.2',
      author='Marko Vidoni',
      author_email='',
      url='https://github.com/mv1388/AIToolbox',
      description='Toolbox of useful functions often needed in different AI oriented projects',
      long_description=open('README.md').read(),

      python_requires='>=3.6.0',

      packages=find_packages(exclude=['tests', 'examples', 'deprecated']),

      install_requires=['mysql-connector-python',
                        'pymongo',
                        'numpy',
                        'networkx',
                        'beautifulsoup4',
                        'nltk',
                        'gensim',
                        'pandas',
                        'tweepy',
                        'joblib',
                        'tqdm',
                        'awscli',
                        'boto3',
                        'google-cloud-storage',
                        'botocore',
                        'matplotlib',
                        'seaborn',
                        'scikit-learn',
                        'allennlp',
                        'torch',
                        'torchvision',
                        'pytorch-nlp',
                        'rouge',
                        'pyrouge',
                        ],

      test_suite='tests',
      tests_require=['nose'],

      # scripts=['bin/AWS/'],

      zip_safe=False)
