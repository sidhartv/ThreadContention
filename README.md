# ThreadContention
A lock contention prediction pipeline.

## Build mutrace
cd mutrace
./bootstrap.sh
make
sudo make install
cd ..

## Run pipeline
1. Profile the application using `mutrace`
```
mutrace application_name
```
2. This will produce a series of files as thread.[tid].log
3. Process mutrace trace files.
```
python data_preprocess.py --input-dir [path/to/directory/containing/mutrace/files] --output-dir [path/to/directory/for/processed/traces]
```
4. Generate models using the train\_online.py script. Models are generated per thread, per lock, per event (subject to change).
