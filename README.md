# Biomass.jl

Autonomous Biomass Pyrolysis Kinetic Modeling using Chemical Reaction Neural Network

## Abstract

Modeling the burning processes of biomass such as wood, grass, and crops is crucial for the modeling and prediction of wildland and urban fire behavior.  Despite its importance, the burning of solid fuels remains poorly understood, which can be partly attributed to the unknown chemical kinetics of most solid fuels.  Most available kinetic models were built upon expert knowledge, which requires chemical insights and years of experience. This work presents a framework for autonomously discovering biomass pyrolysis kinetic models from thermogravimetric analyzer (TGA) experimental data using the recently developed chemical reaction neural networks (CRNN). The approach incorporated the CRNN model into the framework of neural ordinary differential equations to predict the residual mass in TGA data. In addition to the flexibility of neural-network-based models, the learned CRNN model is interpretable, by incorporating the fundamental physics laws, such as the law of mass action and Arrhenius law, into the neural network structure. The learned CRNN model can then be translated into the classical forms of biomass chemical kinetic models, which facilitates the extraction of chemical insights and the integration of the kinetic model into large-scale fire simulations. We demonstrated the effectiveness of the framework in predicting the pyrolysis and oxidation of cellulose. This successful demonstration opens the possibility of rapid and autonomous chemical kinetic modeling of solid fuels, such as wildfire fuels and industrial polymers.

## Related publication

Ji, Weiqi, Franz Richter, Michael J. Gollner, and Sili Deng. "Autonomous kinetic modeling of biomass pyrolysis using chemical reaction neural networks." Combustion and Flame 240 (2022): 111992.
