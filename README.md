# PythonBayesianPrediction
Here we are going to evaluate the posterior using MCMC and Variational Inference techniques and use it to make Bayesian Prediction

## Output for MCMC
```

        num_success = 0  # number of success (avoidances) before trial j
        num_failure = 0  # number of previous failures (shocks)

Number of accepted samples: 104 
Number of rejected samples: 9896 
Mean of alpha values: -0.318211
Mean of beta values: -0.171720
Prediction
----------
Number of instances where the dog jumps off: 17
Number of instances where the dog gets shock: 8
Prediction: 
[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Probability values:
[1.0, 0.86795058523318425, 0.78127926839604478, 0.71426770067266721, 0.65721764765182422, 0.60634412993337361, 0.56005544851737443, 0.51758306765186457, 0.47847615370074353, 0.37515473155435314, 0.29433765377905963, 0.23107875521919269, 0.18153027009341799, 0.14269546675356271, 0.11223833952836978, 0.088336537519593555, 0.069567420954823156, 0.054819645595252812, 0.043224480741617916, 0.034102410059129097, 0.026921583051344594, 0.021265464512261252, 0.016807628781346541, 0.013292107968103361, 0.010518060487927621]


        num_success = 1  # number of success (avoidances) before trial j
        num_failure = 0  # number of previous failures (shocks)

Number of accepted samples: 116 
Number of rejected samples: 9884 
Mean of alpha values: -0.310133
Mean of beta values: -0.173461
Prediction
----------
Number of instances where the dog jumps off: 21
Number of instances where the dog gets shock: 5
Prediction: 
[False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Probability values:
[0.74965593818074294, 0.67050867028429451, 0.61054709179688293, 0.56010486740942855, 0.51542554549999808, 0.4749434366000797, 0.37604816510451722, 0.29800106185358033, 0.2363360624545674, 0.18757031535563295, 0.14897498069160126, 0.11840645761177361, 0.094178197320429688, 0.07496177451798007, 0.059709914777183563, 0.047596348714646596, 0.037968701529030618, 0.030311537082137876, 0.02421733635284766, 0.019363686754154747, 0.015495339178184893, 0.01241008373610001, 0.0099476236899835192, 0.0079808051424708395, 0.0064086990344512703]

	num_success = 0  # number of success (avoidances) before trial j
        num_failure = 1  # number of previous failures (shocks)

Number of accepted samples: 106 
Number of rejected samples: 9894 
Mean of alpha values: -0.317536
Mean of beta values: -0.170974
Prediction
----------
Number of instances where the dog jumps off: 18
Number of instances where the dog gets shock: 8
Prediction: 
[False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Probability values:
[0.86862827704226508, 0.78251105218599537, 0.71594697301562504, 0.65925264172928866, 0.60865605605460604, 0.56257687418854085, 0.52025649665986218, 0.48125277516799003, 0.37762519971688946, 0.29647593103290304, 0.23289179375161617, 0.18304300791100908, 0.14394129566086325, 0.11325323572462101, 0.089155612609335341, 0.070223046052653129, 0.055340575187518698, 0.043635592241367496, 0.034424805055966075, 0.027172885056548024, 0.021460208139493109, 0.016957675625759221, 0.01340705113798856, 0.010605597119036473, 0.0083940646518548753]

```

