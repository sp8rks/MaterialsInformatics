### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ fab940e4-260e-11eb-02fb-017189d5dd7e
begin
	using Pkg; Pkg.activate(".") #sets the Julia package environment 
	using CSV #For processing csv files
	using DataFrames
	using StatsBase 
	using Query #Enables SQL type query of DataFrames
	using MLDataUtils #Tools for splitting data
	path = pwd();
	data_path = path*"/../data/cp_data_cleaned_jl.csv"

	#Set random seed for reproducibility
	using Random
	const rng_seed = 42
	Random.seed!(rng_seed)
end;

# ╔═╡ d5f6a828-260c-11eb-360a-7fdf63568dd6
md"
# Splitting data into the train/validation/test dataset

It is important to split your full dataset into train/validation/test datasets, and reliably use the same datasets for your modeling tasks later.

An example of using different splitting for train/validation/test is shown in the figure in the [python notebook](https://github.com/anthony-wang/BestPractices/blob/master/notebooks/2-data_splitting.ipynb)

Lets load the Julia packages we need and the preprocessed data generated from [1-data\_loading\_cleanup\_processing.ipynb]() 
"

# ╔═╡ b038f004-260f-11eb-021d-cd976cca1f98
dataframe = CSV.File(data_path) |> DataFrame

# ╔═╡ e632ec3c-260f-11eb-1432-43dad5c863e0
md"
Now we need to split the data into `inputs` and `output`. Here the input is the chemical composition and temperature and the output is the heat capacity. The standard notation is to describe inputs as `features` or `descriptors` with variable `X` and output as the `target` with variable `y`.

Note that the use of `!` in the dataframe type indicates all rows.

"

# ╔═╡ 32c4846e-2610-11eb-0a57-5f7bdb29198f
begin
	X = dataframe[!,[:formula,:T]]
	y = dataframe[!,[:Cₚ]]
	md"Size of input: $(size(X))
	   Size of output: $(size(y))
	"
end

# ╔═╡ 8f072bd2-2612-11eb-383f-c1688c98898a
md"
The next step is to utilize standard ML tools that enable random shuffling of the dataset and then splitting it into training and testing. Its more common to split this into training, validation, and testing where the validation is used to help assist with model suitability and hyperparameter evaluation, but in materials science the datasets usually are too small that such splittign would limit valuable training data.

As mentioned in the orignal jupyter notebook, a rule of thumb one can go by is:


  --             | Training Split | Validation Split | Testing Split
:--------------  | :------------  | :--------------: | --------:
Portion to keep  | 50-70%         | 20-30%           | 10-20%

Note the random number generator used for MLDataUtils uses the `Random.GLOBAL_RNG` value which is set by the seed.
"

# ╔═╡ f7e7cba6-2611-11eb-0ddd-51add42ce94f
begin
	Xshuffle,yshuffle = shuffleobs((X,y))
	(Xtrain,ytrain),(Xtest,ytest) = splitobs((Xshuffle,yshuffle),at=0.80)
	md"Size of train input: $(size(Xtrain))
	   SIze of test input: $(size(Xtest))
	"
end

# ╔═╡ e92ea6d0-2635-11eb-2649-27d8ce85ea43
md"

One thing we didn't consider is the fact that the input and output features in the test set shouldn't also be in the training set. In other words, the data has many entries for the same chemical formula given that the data was taken at different temperatures.

Therefore we have to do some additional data processing to ensure that we properly manage this.
"

# ╔═╡ 4619c080-2638-11eb-048a-571e53d02fef
begin
	nrows = nrow(Xtrain);
	nunique_formulae = length(unique(Xtrain[:formula]));
end;

# ╔═╡ 4de8b85e-2636-11eb-2679-c9b1ea4fd78e
md"
There are a total of $(nrows) in the feature training set. But the number of unique formulae is $(nunique_formulae). Below is the unique formula and the counts for each.
"

# ╔═╡ 5f7430e4-2636-11eb-061b-79d0c6cb107d
counts_train = sort(combine(groupby(Xtrain,[:formula]),nrow=>:count))

# ╔═╡ 6651a89a-2638-11eb-204e-51f5c3182c50
counts_test = sort(combine(groupby(Xtest,[:formula]),nrow=>:count))

# ╔═╡ 9d0c5eca-2638-11eb-38be-1b343afbdb6e
md"
Compare the two unique count databases for train and test we can will see there is overlap which is not good practice.

What we need to do is get the list of unique chemical formula and then split the training and test datasets from that list.
"

# ╔═╡ 51aabf24-263a-11eb-0edc-a56f41717300
X_unique_formulae = unique(X[:formula]);

# ╔═╡ de263442-263a-11eb-110a-43b9ac45b2b0
md"

$(X_unique_formulae)

\

Now from the unique list we will split into training, validation, and test data. Theen we can use these seperate label arrays to split the actual dataframe. We will use the Query.jl package to allow for macros that can make constructing the different data sets easier.
"

# ╔═╡ b05f5720-2638-11eb-0259-eb0d51fad3c0
begin
	val_size = 0.20;
	test_size = 0.10;
	train_size = 1.00-val_size-test_size;
unique_train,unique_val,unique_test = splitobs(shuffleobs(X_unique_formulae),at=(train_size,val_size));
end;

# ╔═╡ 71020cfe-2646-11eb-0a9a-f1a5e3bed706
md"

The number of unique formulae for each dataset is:

Training: $(length(unique_train)) \
Validation: $(length(unique_val)) \
Testing: $(length(unique_test))


Note that this shuffling and splitting won't be the same each time which is different from the original jupyter implementation.
"

# ╔═╡ a69242c4-2643-11eb-36ca-275312dd2739
begin
	df_train = filter(row-> row[:formula] ∈ unique_train, dataframe)
	# Query.jl approach below
	#dataframe |> @filter(_.formula ∈ unique_test) |> DataFrame
	df_val = filter(row -> row[:formula] ∈ unique_val, dataframe)
	df_test = filter(row -> row[:formula] ∈ unique_test, dataframe)
end;

# ╔═╡ b10eb172-264a-11eb-054b-63a9436f86eb
md"

Below we implement the original manual approach from the jupyter notebook. The steps are:

1. Get number of entries based on fraction split.
2. Randomly sample from the list of unique formulae.
3. Update the list by removing those selected.
4. Slice the data from the training, validation, and test list.

"

# ╔═╡ 627b84b2-26cd-11eb-3289-85202be68813
begin
	Random.seed!(rng_seed)
	
	#Get the size of the split data
	all_formulae = copy(X_unique_formulae);
	num_unique = length(all_formulae);
	num_val_samples = round(Int,val_size*num_unique);
	num_test_samples = round(Int,test_size*num_unique);
	num_train_samples = round(Int,train_size*num_unique);
	
	#Select formula from unique list for each data set
	val_formulae = sample(all_formulae,num_val_samples,replace=false);
	all_formulae = [formula for formula in all_formulae if formula ∉ val_formulae];
	test_formulae = sample(all_formulae,num_test_samples,replace=false);
	all_formulae = [formula for formula in all_formulae if formula ∉ test_formulae];
	train_formulae = copy(all_formulae);
	
end	;

# ╔═╡ c1e61032-26d2-11eb-02b0-39feb496857c
begin
	df_train_org = filter(row-> row[:formula] ∈ train_formulae, dataframe)
	df_val_org = filter(row -> row[:formula] ∈ val_formulae, dataframe)
	df_test_org = filter(row -> row[:formula] ∈ test_formulae, dataframe)
end;

# ╔═╡ ebdccc4e-264a-11eb-2b6c-6de724bb411c
begin
	name_df_pair =  zip(("cp_train","cp_val","cp_test"),
				        (df_train_org,df_val_org,df_test_org));
	for (name,df) in name_df_pair
		datafile = path*"/../data/$(name)_jl.csv"
		df |> CSV.write(datafile);
	end
end;

# ╔═╡ Cell order:
# ╟─d5f6a828-260c-11eb-360a-7fdf63568dd6
# ╠═fab940e4-260e-11eb-02fb-017189d5dd7e
# ╠═b038f004-260f-11eb-021d-cd976cca1f98
# ╟─e632ec3c-260f-11eb-1432-43dad5c863e0
# ╠═32c4846e-2610-11eb-0a57-5f7bdb29198f
# ╟─8f072bd2-2612-11eb-383f-c1688c98898a
# ╠═f7e7cba6-2611-11eb-0ddd-51add42ce94f
# ╟─e92ea6d0-2635-11eb-2649-27d8ce85ea43
# ╠═4619c080-2638-11eb-048a-571e53d02fef
# ╟─4de8b85e-2636-11eb-2679-c9b1ea4fd78e
# ╠═5f7430e4-2636-11eb-061b-79d0c6cb107d
# ╠═6651a89a-2638-11eb-204e-51f5c3182c50
# ╟─9d0c5eca-2638-11eb-38be-1b343afbdb6e
# ╠═51aabf24-263a-11eb-0edc-a56f41717300
# ╟─de263442-263a-11eb-110a-43b9ac45b2b0
# ╠═b05f5720-2638-11eb-0259-eb0d51fad3c0
# ╟─71020cfe-2646-11eb-0a9a-f1a5e3bed706
# ╠═a69242c4-2643-11eb-36ca-275312dd2739
# ╟─b10eb172-264a-11eb-054b-63a9436f86eb
# ╠═627b84b2-26cd-11eb-3289-85202be68813
# ╠═c1e61032-26d2-11eb-02b0-39feb496857c
# ╠═ebdccc4e-264a-11eb-2b6c-6de724bb411c
