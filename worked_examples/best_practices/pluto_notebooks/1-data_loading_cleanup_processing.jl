### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ a466cc94-2572-11eb-205a-f13c1a73e12e
begin
	using Pkg; Pkg.activate(".") #sets the Julia package environment 
	using CSV #For processing csv files
	using DataFrames
	path = pwd();
	data_path = path*"/../data/cp_data_demo.csv"
end;

# ╔═╡ d04f7b76-2572-11eb-1923-e1e7ccbaa00d
md"
## Data loading, cleanup, and preprocessing

We follow the initial steps as documented in the [original python jupyter notebook](https://github.com/anthony-wang/BestPractices/blob/master/notebooks/1-data_loading_cleanup_processing.ipynb) but using the Julia equivalent packages, there the supporting markdown text is curtailed.

In addition we are using Pluto.jl instead of jupyter, which has some benefits and limitations.

"

# ╔═╡ 98c04d7e-2573-11eb-174a-a15605b1488b
md"
Read the csv file and then pipe it into a DataFrame data type
"

# ╔═╡ 0c41c3c8-2573-11eb-0cd2-819829c338a4
begin
	dataframe = CSV.File(data_path) |> DataFrame
	md"Orignal DataFrame shape: $(size(dataframe))"
end

# ╔═╡ 67dd36a8-2574-11eb-3e5d-13ea6cb11cd3
md"
Now showing the first several entries in the database. What we will see is that there are repeating chemical compounds and property entries in the database at different conditions.
"

# ╔═╡ 82c5f464-2574-11eb-3f9d-97fd1a326024
first(dataframe,6)

# ╔═╡ e8b56b78-2575-11eb-242d-a133123f0759
md"
We can get additional information about the dataset by calling the `describe` method. In addition its best to rename the dataframe column names for brevity and clarity.
"

# ╔═╡ fe090ec2-2574-11eb-3d14-bbc4ce082e34
describe(dataframe[!,2:3]);

# ╔═╡ 6235fb12-2575-11eb-1ee7-3d61c4f92b60
begin
	newnames = Dict(zip(names(dataframe),("formula","T","Cₚ")))
	renamed_df = rename(dataframe,newnames)
end

# ╔═╡ b04a23ec-2576-11eb-3180-470ef19e0c73
md"
Now we need to check for any incomplete or NaN entries.	Dataframe shape before dropping missing or NaN values: $(size(renamed_df))
"

# ╔═╡ 06c6bd54-2578-11eb-0169-b1e9b2740f1a
begin
	pruned_df = dropmissing(renamed_df)
	md"Dataframe shape after dropping missing or NaN values: $(size(pruned_df))"
end

# ╔═╡ 58bcfa42-2578-11eb-1fb3-196eef5ee6a5
md"
Now check to see if there are any values for $\text{T}\lt 0$ and $\text{C}_\text{p} \lt 0$ which wouldn't make physical sense.
"

# ╔═╡ 2c35a58e-2579-11eb-0651-b1975379228d
keepcondition = row -> row[:T] ≥ 0 && row[:Cₚ] ≥ 0;

# ╔═╡ 78e495da-257a-11eb-0209-4d2532cc782d
processed_df = filter(keepcondition, pruned_df);

# ╔═╡ 3156aa8e-257b-11eb-2fd0-cf1af1aa22eb
md"
Datafreame now has size $(size(processed_df)) after removing unphysical entries. We can now save the cleaned-up data to a new CSV file.
"

# ╔═╡ 3eb39758-257a-11eb-2eb9-0dd1cabd3c50
begin
	cleaned_data = path*"/../data/cp_data_cleaned_jl.csv"
	processed_df |> CSV.write(cleaned_data);
end;

# ╔═╡ Cell order:
# ╟─d04f7b76-2572-11eb-1923-e1e7ccbaa00d
# ╠═a466cc94-2572-11eb-205a-f13c1a73e12e
# ╟─98c04d7e-2573-11eb-174a-a15605b1488b
# ╠═0c41c3c8-2573-11eb-0cd2-819829c338a4
# ╟─67dd36a8-2574-11eb-3e5d-13ea6cb11cd3
# ╠═82c5f464-2574-11eb-3f9d-97fd1a326024
# ╟─e8b56b78-2575-11eb-242d-a133123f0759
# ╠═fe090ec2-2574-11eb-3d14-bbc4ce082e34
# ╠═6235fb12-2575-11eb-1ee7-3d61c4f92b60
# ╟─b04a23ec-2576-11eb-3180-470ef19e0c73
# ╟─06c6bd54-2578-11eb-0169-b1e9b2740f1a
# ╟─58bcfa42-2578-11eb-1fb3-196eef5ee6a5
# ╠═2c35a58e-2579-11eb-0651-b1975379228d
# ╠═78e495da-257a-11eb-0209-4d2532cc782d
# ╟─3156aa8e-257b-11eb-2fd0-cf1af1aa22eb
# ╠═3eb39758-257a-11eb-2eb9-0dd1cabd3c50
