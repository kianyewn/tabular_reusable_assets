# parameter space.py: 
- Defines parameter spaces for each model
- Dataclass is created for each parameter space, because `optuna.Trial` is required as an argument
- `.get_params()` returns a dictionary of parameters for the model. For optuna, the values are sampled from the distribution specified in the parameter space, which requires a `Trial` object as an argument.
- To include non-optuna parameters, we can use `update()` to add them to the dictionary, after the `.get_params()` is called.
- This way, we have a parameter space that is compatible with optuna, and we can use the `update()` method to add non-optuna parameters to the dictionary.