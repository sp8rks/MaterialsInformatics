
# BestPractices in [Julia](www.julialang.org)

  * Why do this? No particular reason other than I like Julia, it has some advantages (and disadvantages) over python. Plus I find when I code up tutorials or problems on my own I actually see and learn what is happening, so using Julia forces me to do this.
  * Whats different? Julia has similar but different packages that enable easy implementation of data science and machine learning approaches/algorithms. For example rather than `pandas` to handle data Julia has [Database.jl]().


All the packages and compatible versions are provided by the manifest.toml and project.toml, so once this repo is cloned, the following commands will set the Julia environment up:
  ```julia
  julia> cd("pluto_notebooks")
  julia> using Pkg
  julia> Pkg.activate(".")
  julia> Pkg.instantiate()
  ```
  This activates the environment and downloads all dependencies and correct versions based on the project.toml. After this setup, a Pluto.jl notebook server can be launched using:

  ```bash
  cd pluto_notebooks
  julia --project="@." --eval "using Pluto; Pluto.run()"
  ```

  This should activate the correct Julia environment for the Pluto notebook so all the packages needed packages are available. However, pluto is still a bit rough around and doesn't always find the proper project.toml so the `Pkg.activate(".")` command is also included in the pluto notebooks. One of the nice things with a Pluto notebook is it is just  pure julia code with commented mark-down syntax, so it runs like any julia file. It is also reactive which is good for poking around in a real-time feedback way. 

  A [runtest.jl](runtest.jl) is provided just for sanity check to ensure, where possible, the julia implementation match with that produced by the original python approach.

### Note on CBFV
At the moment the julia version interfaces with the CBFV library via [PyCall.jl](). This means it is straightforward to use, however, the path to the feature databases in CBFV is determined using `getcwd()` from pythons `os` which is well defined when using PyCall.jl. For this reason I had to copy CBFV to this folder and slightly modify it so that now you can provide the file path for the feature database.
