from setuptools import find_packages,setup
from typing import List


def get_requirement_packaged(file_path:str) -> List[str]:
        
        '''
        Docstring for get_requirement_packages
        
        :param file_path: Description
        :type file_path: str
        :return: Description
        :rtype: List[str]
        '''
        required_packages = []
        with open(file_path) as fileobj:
            lines = fileobj.readlines()
            required_packages = [listobj.replace('\n','') for listobj in lines if listobj != '-e .']
        return required_packages

setup(
    name = 'MLPROJECT',
    version = '1.0.1',
    author = 'Karthik',
    author_email = 'karthikerra67@gmail.com',
    packages = find_packages(),
    install_requires = get_requirement_packaged('requirements.txt')

)