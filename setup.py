import shutil
import setuptools

setuptools.setup(
        name='iax',
        description="iax",
        license="none",
        url="none",
        version="0.219",
        author="Bartosz Schatton",
        author_email="b.schatton@gmail.com",
        packages=setuptools.find_packages(),
        zip_safe=False,
        install_requires=[]
    )

shutil.rmtree('build')

