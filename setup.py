from setuptools import setup, find_packages

setup(name='AIToolbox',
      version='0.1',
      author='Marko Vidoni',
      author_email='marko.viwa@gmail.com',
      url='https://github.com/mv1388/AIToolbox',
      description='Toolbox of useful functions often needed in different AI oriented projects',
      long_description=open('README.md').read(),

      packages=['AIToolbox', 'AIToolbox.DataManipulation', 'AIToolbox.RPort', 'AIToolbox.Social'],
      zip_safe=False)
