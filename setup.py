from setuptools import setup, find_packages


setup(name='AIToolbox',
      version='0.1',
      author='Marko Vidoni',
      author_email='marko.viwa@gmail.com',
      url='https://github.com/mv1388/AIToolbox',
      description='Toolbox of useful functions often needed in different AI oriented projects',
      long_description=open('README.md').read(),

      python_requires='>=3.6.0',

      packages=find_packages(exclude=['tests', 'ToolboxUseScripts', 'deprecated']),

      install_requires=['mysql-connector-python',
                        'pymongo',
                        'numpy',
                        'networkx',
                        'beautifulsoup4',
                        'nltk',
                        'gensim',
                        'pandas',
                        # 'rpy2',
                        'tweepy',
                        'joblib',
                        'tqdm',
                        'boto3',
                        'botocore',
                        'scikit-learn'],

      test_suite='tests',
      tests_require=['nose'],

      # scripts=['bin/AWS/'],

      zip_safe=False)
