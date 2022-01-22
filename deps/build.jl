using PyCall

pydeps = Dict(["torch" => ["--pre", "torch", "-f" ,"https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"], "jax"=> "jax", "functorch"=>"git+https://github.com/pytorch/functorch.git"])
for (dep, pip_suffix) in pydeps
    try
        pyimport(dep)
    catch e
        try
            run(`$(PyCall.pyprogramname) -m pip install $(pip_suffix)`)
        catch ee
            if !(typeof(ee) <: PyCall.PyError)
                rethrow(ee)
            end
            @warn("""
        Python dependencies not installed.
        Either
        - Rebuild `PyCall` to use Conda by running the following in Julia REPL
            - `ENV[PYTHON]=""; using Pkg; Pkg.build("PyCall"); Pkg.build("PyCallChainRules")
        - Or install the dependencies by running `pip`
            - `pip install $(pip_suffix)`
            """
                )              
        end
    end
end