#!/bin/bash
#SBATCH --job-name=math                
#SBATCH --output=../logs/math_%j.out   
#SBATCH --error=../logs/math_%j.err    

#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p general
#SBATCH -A 
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4

mkdir -p ../data/raw/arqmath

echo "Downloading ARQMath dataset..."
wget https://www.cs.rit.edu/~dprl/ARQMath-backup/Collection/Posts.V1.3.zip -P ../data/raw/arqmath/

echo "Unzipping ARQMath dataset..."
unzip -o ../data/raw/arqmath/Posts.V1.3.zip -d ../data/raw/arqmath/

echo "Parsing to Parquet..."
python ../src/parse_arqmath.py

echo "Inspecting Data..."
python ../src/inspect_data.py

echo "Constructing Triplets..."
python ../src/build_triplets.py

echo "Tokenization Checking..."
python ../src/tokenization_check.py

echo "Creating Diagnostic Set..."
python ../src/build_diagnostic_set.py

echo "Done!"