The code is tested under python 2.7.

The following libraries need to be installed.

- numpy (1.16.1)
- scipy (1.2.1)
- pandas (0.24.1)
- pandasql (0.7.3)
- vtk

You can do `pip install` for all the libraries except `vtk`.

**Installing VTK**

The code is tested with VTK 6.3.0. Newer/older versions might not work.
You need to install VTK from source. Download the source for VTK 6.3.0 from https://vtk.org/download/

README file provided with VTK contains detailed installation instructions. A summary is given below.

- Install libraries `cmake` and `cmake-curses-gui`
- Extract VTK source to a folder
- Create another folder named `build` in the same folder with the source folder. (So there is a `VTK-6.3.0` and `build` folder in the same folder.)
- Open up a terminal, `cd` into `build` folder and run `ccmake ../VTK-6.3.0`
    - Configuration interface will start. Press `c` to run initial configuration
    - Set `BUILD_EXAMPLES` (not necessary, but nice for checking if VTK works fine), `BUILD_SHARED_LIBS` and `VTK_WRAP_PYTHON` to `ON`
    - Press `c` as many times as necessary until you get the option to generate and exit
    - Press `g` to generate and exit
- Run `make`. This takes some time.
- Run `sudo make install` to install VTK.
- Update environment variables `LD_LIBRARY_PATH` to include VTK library folder (default is `build/lib`) and `PYTHONPATH` to include library folder and VTK python wrapping folder (default `build/Wrapping/Python`).
	- For example, in Ubuntu you can update `.bashrc` as follows
	
	    ```
		export LD_LIBRARY_PATH=YOUR_VTK_FOLDER/build/lib:$LD_LIBRARY_PATH
		export PYTHONPATH=YOUR_VTK_FOLDER/build/Wrapping/Python:YOUR_VTK_FOLDER/build/lib:$PYTHONPATH
		```
