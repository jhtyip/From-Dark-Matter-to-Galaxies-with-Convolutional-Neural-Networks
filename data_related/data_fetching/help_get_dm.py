# this is for getting the Coordinates of the dark matter particles

import requests
import os

headers = {"api-key":"yourApiKeyHere"}  # API key of your account on the IllustrisTNG data access site

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


base_url = "http://www.tng-project.org/api/TNG300-1-Dark/"
params = {"dm":"Coordinates"}

print(get(base_url, base_url)["num_files_snapshot"])  # print total number of files

for i in range(get(base_url, base_url)["num_files_snapshot"]):
    file_url = base_url + "files/snapshot-99." + str(i) + ".hdf5"  # 99 refers to the snapshot at z = 0 (today's universe)
    saved_filename = get(base_url, file_url, params)
    print(saved_filename)  # print name of the downloaded file
