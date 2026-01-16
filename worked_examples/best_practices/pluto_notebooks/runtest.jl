using Pkg
Pkg.activate(".")
using Test


@testset "1-data_loading_cleanup_processing" begin
	using CSV
	using DataFrames
	path = pwd();
	pythonfile = CSV.File(path*"/../data/cp_data_cleaned.csv") |> DataFrame
	juliafile = CSV.File(path*"/../data/cp_data_cleaned_jl.csv") |> DataFrame;
	@test  isequal(pythonfile[!,1],juliafile[!,1])
	@test all(isapprox.(pythonfile[!,2],juliafile[!,2],atol=1.0e-3))
	@test all(isapprox.(pythonfile[!,3],juliafile[!,3],atol=1.0e-3)) 
end

@testset "2-data_splitting" begin
	 using CSV
	 using DataFrames
	 path = pwd();
	 dataset = ("train","val","test")
	 @testset "Comparing number of formulae in $(d) set" for d in dataset
	 	pythonfile = CSV.File(path*"/../data/cp_$(d).csv") |> DataFrame
		juliafile = CSV.File(path*"/../data/cp_$(d)_jl.csv") |> DataFrame;
	 	@test length(unique(pythonfile[!,:formula])) == length(unique(juliafile[!,:formula]))
	end
end

@testset "3-modeling_classical_models" begin
	 path= pwd();
	 juliavalue = parse(Float64,readline(open(path * "/../data/R2_GradientBoostingRegressor.fit")))
	 @test (juliavalue > 0.8442) && (juliavalue < 0.9900)	 
end