# Download and process data

This document how to download, compile and run the LUBM UBA Generator in
order to generate the graph data used in this paper.

### Requirements

The UBA requires a valid Java installation. To check for availability, you
can run: `javac --version`. If not available, please download the
[JDK](https://www.oracle.com/java/technologies/downloads/).

### Step 1

Download the generator from the official [LUBM website](https://swat.cse.lehigh.edu/projects/lubm/).
The results in this paper used the UBA 1.7.

Make a folder in the project root, `uba`, and deflate the zip there.

As an option, we provide a script that downloads and extracts UBA 1.7 into
the correct folder structure in `download_data.sh`. However, we cannot
guarantee that this will work indefinitely.

### Step 2

The next step is to compile the UBA.

`cd` into `/uba/src/edu/lehigh/swat/bench/uba`

The `Generator.java` uses backslashes, \\, when generating the files. If
your OS does not, you have to open this file and replace these with forward
slashes. There are two occurences that you need to swap. Failure to do so
does not really cause any errors right away but may cause great mental pain
later down the line.

After this replacement, you can compile the program with:

`javac -d ../../../../../../ *.java`

This will create the folder `edu` inside `project_root/uba`.

### Step 3

Now it is time to generate the graphs.

`cd` into `project_root/uba`.

The generator can be run using:

`java edu.lehigh.swat.bench.uba.Generator -univ <value> -index <value> -seed <value> -daml -onto <value>`

To reproduce our settings, run:

`java edu.lehigh.swat.bench.uba.Generator -univ 360 -index 0 -seed 0 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl`

This will generate approximately 14k `.owl` graphs in the `uba` folder.

Organize them using:

```
cd project_root/uba/
mkdir ../raw_graphs/
mv *.owl ../raw_graphs/
```

Now you have all the raw graphs available.
