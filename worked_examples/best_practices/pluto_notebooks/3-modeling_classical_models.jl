### A Pluto.jl notebook ###
# v0.12.14

using Markdown
using InteractiveUtils

# ╔═╡ d16393e0-26de-11eb-17a4-e74d7e9015ae
begin
	using Pkg; Pkg.activate(".") #sets the Julia package environment 
	using CSV #For processing csv files
	using DataStructures #Expands on standard Dict
	using DataFrames
	using StatsBase 
	using StatPlots
	using MLDataUtils #Tools for splitting data
	using ScikitLearn #A Julia implementation of Python's ScikitLearn
	using ScikitLearn: fit!, predict
	using Plots
	using LsqFit # Simple library for least-squares
	
	path = pwd();
	data_path = path*"/../data/cp_data_cleaned_jl.csv"

	#Set random seed for reproducibility
	using Random
	const rng_seed = 42
	Random.seed!(rng_seed)
end;

# ╔═╡ d1e158da-278c-11eb-05ec-b1787dedc72f
using PyCall

# ╔═╡ c785b672-26df-11eb-229e-c904ef98c0ac
md"
# Data Featurization
The original python jupyter noteboook can be found [here](https://github.com/anthony-wang/BestPractices/blob/master/notebooks/3-modeling_classic_models.ipynb). This version was put together by [Stefan Bringuier](mailto:stefanbringuier@gmail.com)

The task at hand now is to featurize the composition input (e.g. 'BaO3') into a material descriptor/feature vector. This enables representing the string as a vector chemical properties based on the constituent atoms. For more detailed discussion refer to the manuscript. There are many different ways to do featurization in materials informatics and will depend on the type of model or analysis one is trying to achieve [^a]."

# ╔═╡ 8f9e387e-26df-11eb-06ea-313299966ce4
md"
Let us load the data sets corresponding to training, validation, and testing. One may also be interested in sub-sampling from these data sets if the size of the datasets is to large to enable easy prototyping and assesments. However, it is usually the case that materials data is to precious to do so. The steps aren't shown in this notebook but see the [original jupyter notebook](https://github.com/anthony-wang/BestPractices/blob/master/notebooks/3-modeling_classic_models.ipynb) if interested.

"

# ╔═╡ f6285478-26e0-11eb-2878-57a500d3f3b3
md"
## Generating features using the CBFV package

The composition-based feature vector(CBFV) utlizes exisiting material property databases (e.g. JARVIS) to determine what the features/descriptors should be given a specific chemical formula.

At the moment I'll make use of modified CBFV package[^1] in this directory and use pandas via PyCall.jl. Eventually it will be more desirable to have a native Julia CBFV package however this is somewhat duplicative. 


"

# ╔═╡ 63254334-27d7-11eb-0252-b1e321613d49
begin
	py"""
	import sys
	sys.path.insert(0,".")
	"""
	generate_features = pyimport("CBFV.cbfv.composition")["generate_features"]
end;

# ╔═╡ e0be63ee-27cc-11eb-0ed9-3d933555f0b1
begin
	df_train = CSV.File("../data/cp_train.csv") |> DataFrame
	df_val = CSV.File("../data/cp_val.csv") |> DataFrame
	df_test = CSV.File("../data/cp_val.csv") |> DataFrame
end;

# ╔═╡ 2970a1d4-27cc-11eb-2ffc-114c91fe46aa
md"
Now lets read in the datasets that were generated from [2-data_splitting](https://github.com/anthony-wang/BestPractices/blob/master/notebooks/2-data_splitting.ipynb) and use the CBFV `generate_function` call to generate a new input features. The way CBFV works requires the dataframe have a column called `target`, in this cases this is the heat capcity $\text{C}_p$.

Lets first read in the datasets...

* Training dataset has $(size(df_train))
* Validation dataset has $(size(df_val))
* Testing dataset has $(size(df_test))
"

# ╔═╡ 5697ea04-27cd-11eb-28cf-07ccf586bb6b
md"
Now renaming $\text{C}_p$ to `target`.[^2]
"

# ╔═╡ c96b6ab8-27cd-11eb-2aed-c760d176ce4b
begin
	df_train_target = rename(df_train, :Cp => "target");
	df_train_val = rename(df_val, :Cp => "target");
	df_train_test = rename(df_test, :Cp => "target");
end

# ╔═╡ 37077d50-2938-11eb-018c-83e72c7374e5
md"
#### Using Pandas via PyCall.jl instead of DataFrames.jl 
"

# ╔═╡ 68926d84-282b-11eb-3923-85ff978fffdf
pandas = pyimport("pandas");

# ╔═╡ abbf1bde-282b-11eb-3aab-afa2182c6e99
md"
Since this excersise interfaces with the python CBFV version at the momement, the steps above are repeated using PyCall.jl and pandas. This gives a PyObject that references the pandas DataFrame object that can then be passed to the `generate_features` call.
"

# ╔═╡ 75610794-282b-11eb-1464-078644009df6
begin
	pdf_train = pandas.read_csv("../data/cp_train.csv")
	pdf_val = pandas.read_csv("../data/cp_val.csv")
	pdf_test = pandas.read_csv("../data/cp_test.csv")
	
	dict_rename = Dict("Cp" => "target")
	pdf_train_raw= pdf_train.rename(columns=dict_rename)
	pdf_val_raw = pdf_val.rename(columns=dict_rename)
	pdf_test_raw = pdf_test.rename(columns=dict_rename)
end;

# ╔═╡ ab92c33a-38f3-11eb-29ea-393ca4cbb2cb
md"
### Sub-sampling Data (Optional)
If your trying to do some model assesement or prototyping it maybe more practical to train which a smaller dataset. Usually this isn't a luxury we have when working materials or chemistry datasets.
"

# ╔═╡ 12c2ede6-38f4-11eb-107b-f702ab0d48ee
begin
	pdf_train_subsampled = pdf_train_raw.sample(n=2000,random_state=rng_seed)
	pdf_val_subsampled = pdf_val_raw.sample(n=200,random_state=rng_seed)
	pdf_test_subsampled = pdf_val_raw.sample(n=200,random_state=rng_seed)
end;

# ╔═╡ 7c596716-27cf-11eb-25dc-b13f0a3618ba
md"
Now that we are ready to use the `generate_features` function we need to specify some key arguments about what is to be done. There following are:

* `elem_prop`: specifies which CBFV featurization scheme to use. This depends on the chemica/material feature database one wants to use.
* `drop_duplicates`: parameter that determines whether to drop duplicate formulae during featurization. Since there are multiple occurances in the data being used (i.e., multiple tempeatures for a given composition) this needs to be `false`.
* `extend_features`: adds the CBFV features onto existing features. Since we have temperature as a feature this will be `true`.
* `sum_feat`: uses/calculates the sum of constituent features.

The settings we will use are:

* `elem_prop=\"oliynyk\"`
* `drop_duplicates=false`
* `extend_features=true`
* `sum_feat=True`

"

# ╔═╡ 81fe1a6c-2834-11eb-0314-91345a00e98d
begin	
	#df --> refers to dataframe, pdf --> refers to pandas dataframe
	cbfv_file = "CBFV/element_properties/oliynyk.csv"

	Xtrain_pdf,ytrain_pdf,formulae_train_pdf,skipped_train_pdf =
					  generate_features(pdf_train_subsampled,
					  	elem_prop="oliynyk",
					  	drop_duplicates=false,
			          	extend_features=true,
			          	sum_feat=true,
					  	file_provided=cbfv_file)
	Xval_pdf,yval_pdf,formulae_val_pdf,skipped_val_pdf =
					  generate_features(pdf_val_subsampled,
					  	elem_prop="oliynyk",
					  	drop_duplicates=false,
			          	extend_features=true,
			          	sum_feat=true,
					  	file_provided=cbfv_file)
	Xtest_pdf,ytest_pdf,formulae_test_pdf,skipped_test_pdf =
					  generate_features(pdf_test_subsampled,
					  	elem_prop="oliynyk",
					  	drop_duplicates=false,
			          	extend_features=true,
			          	sum_feat=true,
					  	file_provided=cbfv_file)
end;

# ╔═╡ ec4e091a-2837-11eb-083b-4f47556f7da4
md"
#### Optional conversion
Now moving back from pandas to DataFrame.jl to continue working with a Julia native environment. We just need a little wrapper function.

However its probably just best to convert everything to Julia arrays as this is what ScikitLearn.jl will use.

"

# ╔═╡ 894ecc98-2838-11eb-32af-ef0fcbb950cd
begin
	function pdf_to_df(df_pd)
    	df= DataFrame()
    	for col in df_pd.columns
        	df[!, col] = get(df_pd,col).values
    	end
    	df
	end
	function ps_to_df(s_pd)
		df = DataFrame();
		cname = s_pd.name;
		df[!,cname] = s_pd.values
		return df
	end
end


# ╔═╡ 6e25ba5e-2839-11eb-2efc-a512b21749ad
begin
	Xtrain,ytrain= pdf_to_df(Xtrain_pdf),ps_to_df(ytrain_pdf)
	Xval,yval = pdf_to_df(Xval_pdf),ps_to_df(yval_pdf)
	Xtest,ytest = pdf_to_df(Xtest_pdf),ps_to_df(ytest_pdf)
end;

# ╔═╡ c963e8d0-283b-11eb-0dc8-d9ea1da79741
md"

$(first(Xtrain,25))

Note that the entries are different from the orignal python jupyter notebook since no subsampling was done in this notebook.
	
"

# ╔═╡ 92a28dea-2842-11eb-2d3f-4bf55b271864
md"
## Data normalization and scaling

The next step that is usually key to improve model training performance is to scale and normalize the input features.

The scaling helps to address the order-of-magnitude spread that can occur between input features, this is a very common occurance in chemical and material data.

Normalization is the rescaling of values within an input feature to improve model training/performance order-of-magnitude spread exist within the feature, this is also common for many chemical and material properties.
"

# ╔═╡ 48baf3fe-3fec-11eb-38d3-7bac96226b3d
function normalize_scale(data_train::DataFrame,data_to_scale::DataFrame)
	# The scaling and normalize should be based on the training statistics fit.
	data_train_arry = convert(Array,data_train);
	centerscale = fit(ZScoreTransform,data_train_arry,dims=1)
	data_arry = convert(Array,data_to_scale);
	scaled_data = StatsBase.transform(centerscale,data_arry)
	return scaled_data
end;

# ╔═╡ a0d7c2fa-2843-11eb-2a33-5baa0ccd294e
begin
	Xtrain_f,Xval_f,Xtest_f = [normalize_scale(Xtrain,d) for d in (Xtrain,Xval,Xtest)] 
	ytrain_f,yval_f, ytest_f = map(x->convert(Array,x),[ytrain,yval,ytest])
end;

# ╔═╡ 644a69ba-293a-11eb-15ef-87dc7794c14b
md"
## ScikitLearn.jl
This is an implementation of scikit learn but in Julia. It offers much of the same model functionlity as in the python implementation. We first use the macro to import linear, ensemble, and support vector machine algorithms/models.

The goal now is to assess different machine learning models performance for the data at hand. You can think of the models of different types of parameterization of the data, this can using arguments that statistical, categorical, or clustering, etc. by nature.

* Ridge regression
* Support vector machine
* Linear support vector machine
* Random forest
* Extra trees
* Adaptive boosting
* Gradient boosting
* k-nearest neighbors
* Dummy (if you can't beat this, something is wrong.)

Unhide below to see the macro for imported models.
"

# ╔═╡ 41831116-293a-11eb-1f82-c9629ca42334
begin
	@sk_import dummy: DummyRegressor
	@sk_import linear_model: Ridge
	@sk_import ensemble: AdaBoostRegressor
	@sk_import ensemble: GradientBoostingRegressor
	@sk_import ensemble: ExtraTreesRegressor
	@sk_import ensemble: RandomForestRegressor
	@sk_import neighbors: KNeighborsRegressor
	@sk_import svm: SVR
	@sk_import svm: LinearSVR
end;

# ╔═╡ fa45ba9e-2b6c-11eb-273a-c9e60f3c940d
md"""
Now creating a function call that instantiates the scikit model and then fits it to the provided training data
"""

# ╔═╡ 814d33f4-2966-11eb-3f4f-09450d8f3d32
function fit_model!(scikit_model,Xtrain::Array,ytrain::Array)
	tᵢ = time();
	model = scikit_model();
	fitmodel = fit!(model,Xtrain,ytrain);
	Δt = time() - tᵢ
	return fitmodel, Δt
end;

# ╔═╡ 299d79be-2b6d-11eb-372f-9b6d14ee8477
md"""
Next is a function to evaluate the performance of the fit. The metrics we will us are the coefficient of determination regression score r$^2$, mean absolute error, and mean-squared error.

To do this we will import the functions from ScikitLearn.jl
"""

# ╔═╡ 9fe0f246-2b71-11eb-3aa2-ffb5f71894d6
begin
	@sk_import metrics: r2_score
	@sk_import metrics: mean_absolute_error
	@sk_import metrics: mean_squared_error
end;

# ╔═╡ ce3a928c-2b6f-11eb-2f60-7bd5e039006b
function evaluate_model(fitmodel,X::Array,y::Array)
	ypredict = predict(fitmodel,X)
	r² = r2_score(y,ypredict)
	mae = mean_absolute_error(y,ypredict)
	rmse = mean_squared_error(y,ypredict,squared=false)
	return (r²,mae,rmse)
end

# ╔═╡ ae2c0eee-2b74-11eb-2ec4-335f23618d58
md"

The next step is to write a function that returns a model ScikitLearn.jl PyObject and the name of it.

Then writing a function to train and evaluate the training set and validation set and then sore the results in a dictionary that will be used to in the `model_results` dataframe.
"

# ╔═╡ 0cfaf15a-2b7b-11eb-10d7-51182f193565
begin
	function get_scikit_model(name_abbrv)
		model_names = dict_scikit_models() 
		return model_names[name_abbrv],eval(model_names[name_abbrv])
	end

	function dict_scikit_models()
		Dict(
    	:dumr => :DummyRegressor,
    	:rr => :Ridge,
    	:abr => :AdaBoostRegressor,
    	:gbr => :GradientBoostingRegressor,
    	:rfr => :RandomForestRegressor,
    	:etr => :ExtraTreesRegressor,
    	:svr => :SVR,
    	:lsvr => :LinearSVR,
    	:knr => :KNeighborsRegressor
		);
	end
end

# ╔═╡ 46e78d20-2b75-11eb-24cb-531a9b80a9af
function fit_evaluate_model(scikit_model_abbrv,Xtrain,ytrain,Xval,yval)
	results = Dict() #store results;
	
	#Model type
	results[:model_abbrv] = String(scikit_model_abbrv);
	model_name,scikit_model = get_scikit_model(scikit_model_abbrv);
	results[:model_name] = String(model_name);
	
	#Fit model
	fitmodel,results[:fit_time] = fit_model!(scikit_model,Xtrain,ytrain);
	results[:model_params] = fitmodel.get_params();
	
	#Evaluate model
	begin
		results[:r2_train],
		results[:mae_train],
		results[:rmse_train] = evaluate_model(fitmodel,Xtrain,ytrain);
		
		results[:r2_val],
		results[:mae_val],
		results[:rmse_val] = evaluate_model(fitmodel,Xval,yval);
	end
	return fitmodel,results
end	

# ╔═╡ 3fa9d45a-2b89-11eb-3162-4701219093b0
md"
A convience function to append and soft the model result database
"

# ╔═╡ 4fe5a448-2b89-11eb-2ef9-ff04a24f9451
function append_and_sort!(dataframe::DataFrame,data::Dict,col::Symbol=:r2_val)
	append!(dataframe,data)
	sort!(dataframe,col)
end

# ╔═╡ 6cdb025a-2940-11eb-1866-312d59c524b7
md"
To make the assesment of models easier we will create a julia DataFrame to store the results.
"

# ╔═╡ 959afe70-2b89-11eb-11fa-f3af9f1ef4cc
md"

The final step is to loop through each model, train it, evaluate it, and store the results.
"

# ╔═╡ b7525c34-2b89-11eb-0af6-6515f965d4a9
begin
	model_results = DataFrame(model_abbrv = String[],
						  model_name = String[],
						  model_params = Dict[],
						  fit_time = Float64[],
						  r2_train = Float64[],
						  mae_train = Float64[],
						  rmse_train = Float64[],
						  r2_val = Float64[],
						  mae_val = Float64[],
						  rmse_val = Float64[]
						  );
	
	stored_models = Dict();
	
	model_names = keys(dict_scikit_models())
	
	for m in model_names
		m_fitmodel,m_results = fit_evaluate_model(m,Xtrain_f,ytrain_f,Xval_f,yval_f)
		append_and_sort!(model_results,m_results)
		stored_models[m] = m_fitmodel;
	end
end

# ╔═╡ 28b419ac-2b8b-11eb-1304-af16c78b7e26
md"
Now the peformance of the models can be explored.
"

# ╔═╡ 4c42b16e-2b8e-11eb-1b7b-6158fbae63f0
model_results

# ╔═╡ d506fdfe-2b8e-11eb-355e-4f52e4b5093a
md"

Ultimately we want something a bit more easy to assess the performance of a model on the validation data. The most common way to do this is to create a scatter plot of predicted vs actual target values. A perfect agreement would be a straight line with slope 1.

Below is a function that plots the model prediction results vs. the actual results and also does a simple linear fit. The r$^2$ score is indicated in t
"

# ╔═╡ b027f326-2b8f-11eb-26a0-d1edcbfff4b6
function plot_performance!(model_name,ypredict,yactual;size=(300,300))
	
	#r2 score for how well variance is captured.
	r² = r2_score(yactual,ypredict)
	
	#GR plot default settings
	gr(size = size, 
		linewidth= 1,
		markersize= 3,
		markeralpha= 0.4,
		markercolor=:silver,
		foreground_color_legend=nothing,
		xlabel="Actual Cp [J/mol K]",
		ylabel="Predicted Cp [J/mol K]",
		tickfont = (8, "arial"),
		guidefont = (8, "arial"),
		title="$(model_name), r2: $(round(r²,sigdigits=3))",
		titlefont = (6,"arial"),
		legendfont = (5,"arial"),
		legend=:topleft);
	
	
	maxbound = maximum(max(yactual,ypredict));
	lims = range(0,maxbound,length=100)
	p = scatter(yactual,ypredict,label=nothing)
	p = plot!(lims,lims,label="ideal",linecolor=:black,
		linestyle=:dash)
	
	#Least-squares linear fit linear algebra approach, not working
	# using LsqFit.jl package
	#coeff = (yactual' * yactual) \ (yactual' * ypredict)
	coeffs = [1.0, 0.0]
	lf = curve_fit((x,p)->p[1].*x .+ p[2], yactual, ypredict, coeffs)
	p = plot!(x->x,x->lf.param[1].*x .+ lf.param[2],yactual,
		label="linear fit",linealpha=0.8,linecolor=:orange)
	
	
	#p = annotate!(300,50,text("r2 score\n $(round(r²,sigdigits=3))",8,"arial"))
	
	return p
end
	

# ╔═╡ 0a11a784-2b91-11eb-093a-71d0e1e731f1
begin
mp = Plots.Plot{Plots.GRBackend}[]
for m in keys(stored_models)
		trained_model = stored_models[m]
		ypredict_val = predict(trained_model,Xval_f)
		model_name = String(dict_scikit_models()[m])
		p =plot_performance!(model_name,vcat(ypredict_val...),vcat(yval_f...),size=(700,700))
		push!(mp,p)
end
end

# ╔═╡ 3ab4364a-3212-11eb-3fcb-db53d1d87e03
md"""
Now plotting all the actual vs. prediction for all the trained models.
"""

# ╔═╡ 068af642-2cf1-11eb-2520-b95506a8cff4
plot(mp...)

# ╔═╡ 459f4870-2d48-11eb-031a-1b2093391a55
md"
Upon inspection of the plots we get a sense for which models perform well, that is, the model predictions which show a general behavior scattered around the ideal line. The linear least-squares fit shows how the model prediction deviates (in an average sense) and the R$^2$ score indicates how the variance or data spread is captured by the model prediction.

## Re-training the best performing model
The next step is to select the best performing machine learning model and use the validation data set to train the hyperparameters.

"

# ╔═╡ eaa5d594-2868-11eb-0a45-3bb4e8ed7921
best_model = last(model_results)

# ╔═╡ 5f379a00-3214-11eb-3716-652b4f197e3a
begin
	best_model_name = best_model[:model_abbrv]
	best_model_params = best_model[:model_params]
end;

# ╔═╡ ad1a1b00-3262-11eb-28b1-719264e3d7dd
trained_model = stored_models[Symbol(best_model_name)]

# ╔═╡ 8b8b6df8-3263-11eb-2faa-a55675ae8c1f
md"
The next step is to combine the training and validation data sets and then retrain the `GradientBoostingRegressor`
"

# ╔═╡ a887ec3a-3263-11eb-14c0-fdb6366ee22c
begin
	X_train_val = vcat(Xtrain_f,Xval_f);
	y_train_val = vcat(ytrain_f,yval_f);
end;

# ╔═╡ 771bc308-335b-11eb-3497-41115ad8b1f1
md"

The optimal parametes obtained from the initial training should be used as the starting point. Here the `stored_models` dictionary object retains the best parameters for the trained model, but you could set them using a function call `set_params!(model;params...)`

Then the model is retrained using the combined traning and validation datasets.
"

# ╔═╡ 22e193b4-3264-11eb-06d7-71a9ea777b2d
begin
	#The stored model retains the optimal parameters, but they can be assigned via:
	#set_params!(trained_model;best_model_params...)
	trained_model.fit(X_train_val,y_train_val)
end


# ╔═╡ 05d481f2-335c-11eb-2850-b543ac5ddb4b
md"
The final step is to see how the trained model now performs on the test data, which was never seen by the model.
"

# ╔═╡ 207d9b40-335c-11eb-229f-5916852a6800
begin
	y_pred_test = trained_model.predict(Xtest_f)
	r2,mae,rmse = evaluate_model(trained_model,Xtest_f,ytest_f)
	md"""
	Fit quality metrics for test data
	
	r$^2$: $(r2)
	
	mae: $(mae)

	mse: $(rmse)
	
	The model performs remarkable well, with metrics indicating that it captures the data variance and average error to a good degree.
	"""
end

# ╔═╡ 14359dc2-4178-11eb-0435-0ffb7c4a5814
# Write R2 value to a file for test, see runtest.jl
begin
	fitoutput = open("../data/R2_GradientBoostingRegressor.fit","w")
	write(fitoutput,"$(r2)");
	close(fitoutput);
end

# ╔═╡ 9d7f2590-335d-11eb-2ff8-97f1302243f9
plot_performance!(best_model_name,y_pred_test,vcat(ytest_f...),size=(500,500))

# ╔═╡ a5200be6-3371-11eb-30af-6dea1964901f
md"""
# Impact on splitting of dataset

This section will show how creating different splits of the datasets impacts the model training and predictive quality.
"""

# ╔═╡ b1d67f78-38f5-11eb-1a52-218b9349201e
begin	
	#df --> refers to dataframe, pdf --> refers to pandas dataframe
	Xtrain_unscaled_pdf,ytrain_unscaled_pdf,_,_ =
					  generate_features(pdf_train_subsampled,
					  	elem_prop="oliynyk",
					  	drop_duplicates=false,
			          	extend_features=true,
			          	sum_feat=true,
					  	file_provided=cbfv_file)
	Xval_unscaled_pdf,yval_unscaled_pdf,_,_=
					  generate_features(pdf_val_subsampled,
					  	elem_prop="oliynyk",
					  	drop_duplicates=false,
			          	extend_features=true,
			          	sum_feat=true,
					  	file_provided=cbfv_file)
	Xtest_unscaled_pdf,ytest_unscaled_pdf,_,_ =
					  generate_features(pdf_test_subsampled,
					  	elem_prop="oliynyk",
					  	drop_duplicates=false,
			          	extend_features=true,
			          	sum_feat=true,
					  	file_provided=cbfv_file)
end;

# ╔═╡ f2388d08-3f36-11eb-01af-2b4453c7e98c
md"
now converting the pandas database to DataFrames.jl for convience
"

# ╔═╡ 0181745a-3f37-11eb-3135-015de061d70c
begin
	Xtrain_unscaled,ytrain_unscaled= pdf_to_df(Xtrain_pdf),ps_to_df(ytrain_pdf)
	Xval_unscaled,yval_unscaled = pdf_to_df(Xval_pdf),ps_to_df(yval_pdf)
	Xtest_unscaled,ytest_unscaled = pdf_to_df(Xtest_pdf),ps_to_df(ytest_pdf)
end;

# ╔═╡ 5b05ac3a-3f37-11eb-1c4b-c9c0015760be
md"
making initial copies of the data so that we can sample and then reassign during the splitting. A total of 10 different dataset splits will be performed (randomly) to show how it impacts training and performance outcomes.
"

# ╔═╡ eda4edbc-3f37-11eb-24e8-c95498dc9629
begin
	Xtrain_org = copy(Xtrain_unscaled);
	ytrain_org = copy(ytrain_unscaled);
	Xval_org = copy(Xval_unscaled);
	Xtest_org = copy(Xtest_unscaled);
end;

# ╔═╡ 5e3c7162-3f38-11eb-27d1-b5ee205371e3
nsplits = 1:10;

# ╔═╡ 3fd375b8-3f38-11eb-195b-e36da010b17c
split_results = DataFrame(split=Int[],
						  r2_train = Float64[],
						  mae_train = Float64[],
						  rmse_train = Float64[],
						  r2_val = Float64[],
						  mae_val = Float64[],
						  rmse_val = Float64[]);

# ╔═╡ 9719d72a-4179-11eb-3a53-cdb597966d83
md"Now set the fraction to 70% for data to split

# ╔═╡ c239a238-3fd7-11eb-0b3f-a9c0e1638772
frac = Int(0.7*nrow(Xtrain_org));

# ╔═╡ 043ad70c-3f39-11eb-2fd5-436ca790fcdb
let
	model_abbrv = :abr #AdaBoostRegressor
	for split in nsplits
		md"Fitting and evaluating random split $(split)"
		xindices = StatsBase.sample(1:nrow(Xtrain_org),frac,replace=false);
    	Xtrain = normalize_scale(Xtrain_org[xindices,:],Xtrain_org[xindices,:]);
    	ytrain = convert(Array,ytrain_org[xindices,:]);
		
	  	Xval = normalize_scale(Xtrain_org[xindices,:],Xval_org);
		yval = convert(Array,yval_unscaled);
		Xtest = normalize_scale(Xtrain_org[xindices,:],Xtest_org);
		ytest = convert(Array,ytest_unscaled);
		
		model_name,scikit_model = get_scikit_model(model_abbrv);
		fitmodel,time = fit_model!(scikit_model,Xtrain,ytrain);
		ypred_val = predict(fitmodel,Xval);
		
		r2_train, mae_train, rmse_train = evaluate_model(fitmodel, Xtrain, ytrain)
    	r2_val, mae_val, rmse_val = evaluate_model(fitmodel, Xval, yval)
		
		split_result = Dict(:split => split,
							:r2_train => r2_train,
							:mae_train => mae_train,
						    :rmse_train => rmse_train,
							:r2_val => r2_val,
							:mae_val => mae_val,
						    :rmse_val => rmse_val);
		
		append_and_sort!(split_results,split_result)
	end
end;


# ╔═╡ fc21a804-417a-11eb-09f2-1976176f5b78
split_results

# ╔═╡ 13641844-417b-11eb-013e-399c2f1a4fb9
md"
To visual inspect the r$^2$ values for each split, lets plot them. To faciliate statistical orientied plots via the StatPlots.jl package. This provies we can use functions and macros that interface with Plots.jl. 

At first glance the variation in r$^2$ values seems more prevalent in the validation data set.

This can be further seen by looking at the mean absolute error


"

# ╔═╡ 2fb6b226-417c-11eb-1829-f31ba635c495
begin
	gr(size = (600,400), 
		linewidth= 2,
		title = "Split fitting of AdaBoostRegressor",
		tickfont = (10, "arial"),
		guidefont = (10, "arial"),
		titlefont = (10,"arial"),
		legendfont = (10,"arial"),
		legend=:topright);

		@df split_results groupedbar(:split,	[:r2_train :r2_val],xlabel="Split",ylabel="r2",ylim=(0.0,1.15))
end

# ╔═╡ 74e9d29c-417c-11eb-27a1-dfeaa2573d18
@df split_results groupedbar(:split,[:mae_train :mae_val],xlabel="Split",ylabel="r2",ylim=(0.0,30.0))

# ╔═╡ 06073c08-4180-11eb-27ea-9154bb3ecbae
md"

Finally, it is common to report the statistical averages of the scores across the splits

Average validation r$^2$: $(mean(split_results[!,:r2_val]))

Average validation MAE: $(mean(split_results[!,:mae_val]))
"

# ╔═╡ 1476bd9c-27cf-11eb-0f1e-7d42c5a8df3e
md"

#### Footnotes

[^a]: Pluto currently doesn't display standard IO, so the progress bar associated with CBFV package won't be shown in the notebook but rather shown in the Julia session that launched the Pluto server. Similar behavior occurs for other IO (e.x. timing).

[^1]: The modification to the CBFV package in pluto_notebooks adds an additional kwarg to `generate_features\(...,file_provided=\"database_file_path\"\)` this was done because the existing composition.py uses the current working directory to find the database file, however, I believe PyCall exist within Julia's OS environment and therefore the two can't be changed indepedently.

[^2]: Note that we don't really need to create a new variable assignment, I could have used the `rename!()` call, but this wasn't choosen because of the way Pluto notebooks are reactive.
"

# ╔═╡ Cell order:
# ╟─c785b672-26df-11eb-229e-c904ef98c0ac
# ╠═d16393e0-26de-11eb-17a4-e74d7e9015ae
# ╟─8f9e387e-26df-11eb-06ea-313299966ce4
# ╟─f6285478-26e0-11eb-2878-57a500d3f3b3
# ╠═d1e158da-278c-11eb-05ec-b1787dedc72f
# ╠═63254334-27d7-11eb-0252-b1e321613d49
# ╟─2970a1d4-27cc-11eb-2ffc-114c91fe46aa
# ╠═e0be63ee-27cc-11eb-0ed9-3d933555f0b1
# ╟─5697ea04-27cd-11eb-28cf-07ccf586bb6b
# ╠═c96b6ab8-27cd-11eb-2aed-c760d176ce4b
# ╟─37077d50-2938-11eb-018c-83e72c7374e5
# ╠═68926d84-282b-11eb-3923-85ff978fffdf
# ╟─abbf1bde-282b-11eb-3aab-afa2182c6e99
# ╠═75610794-282b-11eb-1464-078644009df6
# ╟─ab92c33a-38f3-11eb-29ea-393ca4cbb2cb
# ╠═12c2ede6-38f4-11eb-107b-f702ab0d48ee
# ╟─7c596716-27cf-11eb-25dc-b13f0a3618ba
# ╠═81fe1a6c-2834-11eb-0314-91345a00e98d
# ╟─ec4e091a-2837-11eb-083b-4f47556f7da4
# ╠═894ecc98-2838-11eb-32af-ef0fcbb950cd
# ╠═6e25ba5e-2839-11eb-2efc-a512b21749ad
# ╟─c963e8d0-283b-11eb-0dc8-d9ea1da79741
# ╟─92a28dea-2842-11eb-2d3f-4bf55b271864
# ╠═48baf3fe-3fec-11eb-38d3-7bac96226b3d
# ╠═a0d7c2fa-2843-11eb-2a33-5baa0ccd294e
# ╟─644a69ba-293a-11eb-15ef-87dc7794c14b
# ╠═41831116-293a-11eb-1f82-c9629ca42334
# ╟─fa45ba9e-2b6c-11eb-273a-c9e60f3c940d
# ╠═814d33f4-2966-11eb-3f4f-09450d8f3d32
# ╟─299d79be-2b6d-11eb-372f-9b6d14ee8477
# ╠═9fe0f246-2b71-11eb-3aa2-ffb5f71894d6
# ╠═ce3a928c-2b6f-11eb-2f60-7bd5e039006b
# ╟─ae2c0eee-2b74-11eb-2ec4-335f23618d58
# ╠═0cfaf15a-2b7b-11eb-10d7-51182f193565
# ╠═46e78d20-2b75-11eb-24cb-531a9b80a9af
# ╟─3fa9d45a-2b89-11eb-3162-4701219093b0
# ╠═4fe5a448-2b89-11eb-2ef9-ff04a24f9451
# ╟─6cdb025a-2940-11eb-1866-312d59c524b7
# ╟─959afe70-2b89-11eb-11fa-f3af9f1ef4cc
# ╠═b7525c34-2b89-11eb-0af6-6515f965d4a9
# ╟─28b419ac-2b8b-11eb-1304-af16c78b7e26
# ╠═4c42b16e-2b8e-11eb-1b7b-6158fbae63f0
# ╟─d506fdfe-2b8e-11eb-355e-4f52e4b5093a
# ╠═b027f326-2b8f-11eb-26a0-d1edcbfff4b6
# ╠═0a11a784-2b91-11eb-093a-71d0e1e731f1
# ╟─3ab4364a-3212-11eb-3fcb-db53d1d87e03
# ╠═068af642-2cf1-11eb-2520-b95506a8cff4
# ╟─459f4870-2d48-11eb-031a-1b2093391a55
# ╠═eaa5d594-2868-11eb-0a45-3bb4e8ed7921
# ╠═5f379a00-3214-11eb-3716-652b4f197e3a
# ╠═ad1a1b00-3262-11eb-28b1-719264e3d7dd
# ╟─8b8b6df8-3263-11eb-2faa-a55675ae8c1f
# ╠═a887ec3a-3263-11eb-14c0-fdb6366ee22c
# ╟─771bc308-335b-11eb-3497-41115ad8b1f1
# ╠═22e193b4-3264-11eb-06d7-71a9ea777b2d
# ╟─05d481f2-335c-11eb-2850-b543ac5ddb4b
# ╟─207d9b40-335c-11eb-229f-5916852a6800
# ╟─14359dc2-4178-11eb-0435-0ffb7c4a5814
# ╠═9d7f2590-335d-11eb-2ff8-97f1302243f9
# ╟─a5200be6-3371-11eb-30af-6dea1964901f
# ╠═b1d67f78-38f5-11eb-1a52-218b9349201e
# ╟─f2388d08-3f36-11eb-01af-2b4453c7e98c
# ╠═0181745a-3f37-11eb-3135-015de061d70c
# ╟─5b05ac3a-3f37-11eb-1c4b-c9c0015760be
# ╠═eda4edbc-3f37-11eb-24e8-c95498dc9629
# ╠═5e3c7162-3f38-11eb-27d1-b5ee205371e3
# ╠═3fd375b8-3f38-11eb-195b-e36da010b17c
# ╠═9719d72a-4179-11eb-3a53-cdb597966d83
# ╠═c239a238-3fd7-11eb-0b3f-a9c0e1638772
# ╠═043ad70c-3f39-11eb-2fd5-436ca790fcdb
# ╠═fc21a804-417a-11eb-09f2-1976176f5b78
# ╟─13641844-417b-11eb-013e-399c2f1a4fb9
# ╠═2fb6b226-417c-11eb-1829-f31ba635c495
# ╠═74e9d29c-417c-11eb-27a1-dfeaa2573d18
# ╟─06073c08-4180-11eb-27ea-9154bb3ecbae
# ╟─1476bd9c-27cf-11eb-0f1e-7d42c5a8df3e
