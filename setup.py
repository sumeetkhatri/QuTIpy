from setuptools import setup

setup(
    name='QuTIpy',
    version='0.1.0',
    description='A package to perform quantum information calculations',
    url='https://github.com/sumeetkhatri/QuTIpy',
    author='Sumeet Khatri',
    author_email='khatri6000@gmail.com',
    license='Apache-2.0',
    packages=['qutipy'],
    install_requires=['numpy','scipy','sympy','cvxpy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3']

)