import os
import subprocess
import sys


input_path = "shaders/"
output_path = "shaders/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
debug =  int(sys.argv[1])

files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input_path)) for f in fn]

for f in files:
	sub_path = f.replace(input_path,"")
	file = os.path.basename(sub_path)
	folder_name = sub_path.replace(file, "")
	if not os.path.exists(output_path + folder_name):
		os.makedirs(output_path + folder_name)
	file_name = folder_name + os.path.splitext(file)[0]
	extension = os.path.splitext(file)[1][1:]
	extension_new = ""
	if extension == "frag":
		extension_new = "_fragment.spv"
	elif extension == "vert":
		extension_new = "_vertex.spv"
	elif extension == "comp":
		extension_new = "_compute.spv"
	elif extension == "tese":
		extension_new = "_evaluation.spv"
	elif extension == "tesc":
		extension_new = "_control.spv"
	elif extension == "geom":
		extension_new = "_geometry.spv"
	else:
		continue

	debug_key = ""
	if debug:
		debug_key = "-gVS"
	command_line = f"glslangValidator -V {f} -o {output_path}{file_name}{extension_new} {debug_key}"
	print(command_line)
	subprocess.call(command_line)
