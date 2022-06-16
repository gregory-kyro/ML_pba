"""
Script to run a python script in chimera 1.16 from the Mac terminal.
"""

/Applications/Chimera.app/Contents/MacOS/chimera --nogui Path/To/process_chimera.py

# NOTE: In rare cases (roughly 1 in 5,000) chimera may fail to add hydrogens to a pdb file. In this case, remove that pdb from the dataset and
# rerun the script.

for f in *
do
  echo $f
  sed -i '' 's/H\.t3p/H	/' $f
  sed -i '' 's/O\.t3p/O\.3  /' $f
  done
