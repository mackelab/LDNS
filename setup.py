from setuptools import setup, find_packages

# read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ldns",
    version="0.1.0",
    description="Latent Diffusion for Neural Spiking Data",
    author="Jaivardhan Kapoor, Auguste Schulz, et al.",
    author_email="jaivardhan.kapoor@uni-tuebingen.de",
    packages=['ldns'],
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    extras_require={
        'dev': [
            'jupyter',
            'pytest',
            'black',
            'flake8',
        ],
    },
) 