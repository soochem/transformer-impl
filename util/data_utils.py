import tarfile
import urllib.request
import progressbar

import lxml.etree as etree
from os import listdir, path, makedirs
from os.path import isfile, join

import logging

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class ProgressBar:
    """
    Show progress while downloading files
    Reference : https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def extract_file_from_web(url, extract_path='.'):
    """
    Request tar files from web url
    :param url: web url
    :param extract_path: path to save extracted file
    :return:
    """
    filename = url.split("/")[-1]
    logging.info("Request %s", url)
    (tar_path, headers) = urllib.request.urlretrieve(url, reporthook=ProgressBar())

    if filename.endswith("tar.gz") or filename.endswith('.tgz'):
        tar = tarfile.open(tar_path, "r:gz")
        tar.extractall(extract_path)
        tar.close()
    elif filename.endswith("tar"):
        tar = tarfile.open(tar_path, "r:")
        tar.extractall(extract_path)
        tar.close()
    else:
        print("Could not extract a file.")

    return


def extract_file(file_path, extract_path='.'):
    """
    Save extracted files from tar file name
    :param file_path:
    :param extract_path: path to save extracted file
    :return:
    """
    if file_path.endswith("tar.gz") or file_path.endswith('.tgz'):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(extract_path)
        tar.close()
    elif file_path.endswith("tar"):
        tar = tarfile.open(file_path, "r:")
        tar.extractall(extract_path)
        tar.close()
    else:
        print("Could not extract a file.")
    return


def load_data_from_file(dir_path, output_path='.'):
    """
    Parse XML file to generate dataset
    Reference
    * https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/iwslt.html

    :param dir_path: directory name that files exist
    :param output_path: path to save text files
    :return:
    """
    # Get all xml file list under dir_path
    # ex. "../data/fr-en\IWSLT17.TED.dev2010.fr-en.en.xml"
    xml_files = [join(dir_path, f) for f in listdir(dir_path)
                 if isfile(join(dir_path, f)) and f.endswith('xml')]
    # print("XML file list : ", str(xml_files))

    if not path.exists(output_path):
        makedirs(output_path)

    for xml_file in xml_files:
        # print("XML file name : ", xml_file)
        txt_file = '.'.join(xml_file.split('.')[-3:-1]) + '.txt'
        txt_file = join(output_path, txt_file)

        # Load tree using lxml.etree
        tree = etree.parse(xml_file)
        root = tree.getroot()

        # Check xml contents with pretty form
        # parsed_str = etree.tostring(tree, pretty_print=True)
        # print(parsed_str)

        if isfile(txt_file):  # file already loaded
            continue

        with open(txt_file, 'w', encoding='utf-8') as tfile:
            # root.findall('doc')  # Tag name, first level only
            docs = root.findall('.//doc')  # XPath, recursive
            for doc in docs:
                for element in doc.findall('.//seg'):
                    tfile.write(element.text.strip() + '\n')  # strip: 앞뒤 trim

    # Get all train file list under dir_path
    other_files = [f for f in listdir(dir_path)
                   if isfile(join(dir_path, f)) and f.startswith('train.tags')]
    # print("Other file list : ", str(other_files))

    for other_file in other_files:
        # print("Other file name : ", other_file)
        txt_file = other_file.replace('tags.', '') + '.txt'
        txt_file = join(output_path, txt_file)
        other_file = join(dir_path, other_file)

        if isfile(txt_file):  # file already loaded
            continue

        with open(txt_file, 'w', encoding='utf-8') as tfile, \
                open(other_file, mode='r', encoding='utf-8') as ofile:
            for line in ofile:
                line = line.strip()
                if not line.startswith('<'):
                    tfile.write(line + '\n')
    return
