import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robustcontrol",
    version="0.1.0",
    author="",
    author_email="",
    description="Code contributed by students doing the course CBT700 - 'Multivariable Control Theory' at the University of Pretoria.",
	long_description="This project is aimed at creating Python code for the various code examples in the textbook. Skogestad, S., I. Postlethwaite; Multivariable Feedback Control: Analysis and Design; John Wiley & Sons, 2005. The code is tested on Python 2.7 and 3.6. We are using Python-Future as our compatability layer. The code is largely contributed by students doing the course CBT700 at the University of Pretoria.",
    long_description_content_type="text/markdown",
	license="",
    url="https://github.com/alchemyst/Skogestad-Python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	install_requires=[
	    "numpy>=1.18",
		"scipy>=1.4",
		"matplotlib>=3.1",
		"sympy>=1.5",
		"harold>=1.0",
		"control>=0.8",
		"jupyterlab>=1.2"
	],
    python_requires='>=3.6',
	test_suite='nose.collector',
    tests_require=['nose']
)