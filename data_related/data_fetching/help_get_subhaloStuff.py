# this is for getting the SubhaloFlag/SubhaloMassType/SubhaloPos of subhalos
# SubhaloFlag mainly for identifying non-cosmological subhalos for ruling out
# SubhaloMassType mainly for identifying subhalos with non-zero stellar mass (our definition for galaxy)
# SubhaloPos is position of subhalo

import requests
import os

headers = {"api-key":"yourApiKeyHere"}

def get(path1, path2, params=None):
	r = requests.get(path2, params=params, headers=headers)

	r.raise_for_status()

	if r.headers["content-type"] == 'application/json':
		return r.json()
	
	if "content-disposition" in r.headers:
		if path1 == "http://www.tng-project.org/api/Illustris-1/":
			prefix = "Illustris-1/"
		elif path1 == "http://www.tng-project.org/api/TNG100-1/":
			prefix = "TNG100-1/"
		elif path1 == "http://www.tng-project.org/api/TNG300-1/":
			prefix = "TNG300-1/"		
		elif path1 == "http://www.tng-project.org/api/Illustris-1-Dark/":
			prefix = "Illustris-1-Dark/"
		elif path1 == "http://www.tng-project.org/api/TNG100-1-Dark/":
			prefix = "TNG100-1-Dark/"
		elif path1 == "http://www.tng-project.org/api/TNG300-1-Dark/":		
			prefix = "TNG300-1-Dark/"

		if not os.path.exists(prefix):
			os.makedirs(prefix)
		
		filename = prefix + r.headers["content-disposition"].split("filename=")[1]
		with open(filename, "wb") as f:
			f.write(r.content)
		return filename

	return r


base_url = "http://www.tng-project.org/api/TNG300-1/"
params = {"Subhalo":"SubhaloFlag"}  # change to SubhaloFlag/SubhaloMassType/SubhaloPos as needed

print(get(base_url, base_url)["num_files_snapshot"])  # should equal 1 (number of files to be downloaded = 1)

file_url = base_url + "files/groupcat-99"
saved_filename = get(base_url, file_url, params)
print(saved_filename)
