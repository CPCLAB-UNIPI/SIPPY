import setuptools

setuptools.setup(
	name="SIPPY",
	version="0.1.0",
	author="Giuseppe Armenise",
	description="Systems Identification Package for Python",
	url="https://github.com/CPCLAB-UNIPI/SIPPY",
	packages=setuptools.find_packages(),
	python_requires=">=2.7",
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
		"Programming Language :: Python :: 3.6",
		"Operating System :: OS Independent",
	),
)
