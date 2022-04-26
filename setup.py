import setuptools

with open('README.md', 'r') as f:
	readme = f.read()

setuptools.setup(
	name="sippy",
	version="0.2.0",
	author="Giuseppe Armenise",
	description="Systems Identification Package for Python",
	long_description=readme,
	long_description_content_type='text/markdown',
	url="https://github.com/CPCLAB-UNIPI/sippy",
	packages=setuptools.find_packages(),
	python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",#",<=3.8",
	install_requires=(
		"numpy",
		"scipy",
		"control",
		"slycot",
		"future",
		"casadi"),
	classifiers=(
		"Development Status :: 4 - Beta",
		"Intended Audience :: Education",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3.5",
		"Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Operating System :: OS Independent",
	),
)
