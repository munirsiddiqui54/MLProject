from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT="-e ."

def getReq(path:str)->List[str]:
    reqList=[]
    with open(path) as file:
        reqList=file.read().splitlines()
        if HYPHEN_E_DOT in reqList:
            reqList.remove(HYPHEN_E_DOT)
    return reqList

setup(
    name="ML Project",
    version="0.0.1",
    author="Munir Siddiqui",
    author_email="munir230204@gmail.com",
    packages=find_packages(),
    install_requires=getReq("requirements.txt")
)