import setuptools

with open('README.md', 'r') as f:
	readme = f.read()

setuptools.setup(
	name="sippy",
	version="0.1.0",
	author="Giuseppe Armenise",
	description="Systems Identification Package for Python",
	long_description=readme,
	long_description_content_type='text/markdown',
	url="https://github.com/CPCLAB-UNIPI/sippy",
	packages=setuptools.find_packages(),
	python_requires="==2.7,>=3.5,<=3.6",
	install_requires=(
		"numpy",
		"scipy",
		"control",
		"slycot",
		"future"),
	classifiers=(
		"Development Status :: 4 - Beta",
		"Intended Audience :: Education",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3.5",
		"Programming Language :: Python :: 3.6",
		"Operating System :: OS Independent",
	),
)
