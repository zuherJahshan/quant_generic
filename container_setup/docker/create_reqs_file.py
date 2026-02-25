import re
import shutil
import os


input_file = 'requirements.txt'
output_file = 'clean_requirements.txt'


# Remove current requirements.txt file
if os.path.exists(input_file):
    os.system(f'rm -f {input_file}')


# Run the pip freeze command to generate a new requirements.txt file
# This will include all installed packages and their versions
os.system(f'pip freeze > {input_file}')




with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Use regex to remove everything after the package version
        cleaned_line = re.sub(r'\s*@\s*file://.*', '', line)
        outfile.write(cleaned_line)


# mv clean_requirements.txt to requirements.txt
shutil.move(output_file, input_file)




print(f'Cleaned requirements written to {input_file}')
