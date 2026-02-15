from setuptools import setup, find_packages

setup(
    name="cat-dog-classifier",
    version="1.0.0",
    description="基于深度学习的猫狗识别Web应用",
    author="kimcxx",
    packages=find_packages(),
    install_requires=[
        "Flask==2.3.3",
        "flask-cors==4.0.0",
        "Pillow==10.1.0",
        "numpy==1.24.3",
    ],
    python_requires=">=3.8",
)