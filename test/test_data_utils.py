from data_utils import *


TAR_PATH = '../data/fr-en.tar'
# TAR_PATH = '../data/fr-en.tgz'
TAR_URL = "https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz"
EXTRACT_PATH = '../data'
DATA_PATH = '../data/fr-en'

# Extract file from file name
extract_file(TAR_PATH, EXTRACT_PATH)

# Extract file from web url
extract_file_from_web(TAR_URL, EXTRACT_PATH)

# Parse dataset from xml files
load_data_from_file(DATA_PATH, EXTRACT_PATH + '/output')
