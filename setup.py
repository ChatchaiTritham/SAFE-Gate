"""Setup configuration for SAFE-Gate package."""

from setuptools import setup, find_packages
import os


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


setup(
    name='safegate',
    version='1.0.0',
    author='Chatchai Tritham, Chakkrit Snae Namahoot',
    author_email='chakkrits@nu.ac.th',
    description='Safety-Assured Fusion Engine with Gated Expert Triage for clinical decision support',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/ChatchaiTritham/SAFE-Gate',
    project_urls={
        'Bug Reports': 'https://github.com/ChatchaiTritham/SAFE-Gate/issues',
        'Source': 'https://github.com/ChatchaiTritham/SAFE-Gate',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'jupyter>=1.0.0',
            'pyyaml>=6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'safegate=src.safegate:main',
        ],
    },
    keywords='clinical-decision-support triage formal-verification safety emergency-medicine',
    include_package_data=True,
    zip_safe=False,
)
