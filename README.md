# TCM-Model

## Goal

The purpose of this work is to study the Deep Brain Stimulation (DBS) in the Thalamo-Cortical Microcircuit (TCM).

This will accomplished with the creation of a completely new model using Python.

This model is based on [Farokhniaee and Lowery, 2021](https://iopscience.iop.org/article/10.1088/1741-2552/abee50/data) and [Tsodyks and Markram, 1997](https://www.pnas.org/doi/10.1073/pnas.94.2.719)

## Steps

### STEP 1 - CREATE A MODEL OF THE CORTEX

*Neurons are modeled by Izhikevich, with synaptic connection modeled by Tsodkys and Markram*

- [ ] Model_C 1.0:
    - [ ] 1 neuron in layer D
    - [ ] 1 neuron in layer M
    - [ ] 1 neuron in layer S
    - [ ] 1 neuron in layer CI

- [ ] Model_C 2.0:
    - [ ] 10 neuron in layer D
    - [ ] 10 neuron in layer M
    - [ ] 10 neuron in layer S
    - [ ] 10 neuron in layer CI

- [ ] Model_C 3.0:
    - [ ] 100 neuron in layer D
    - [ ] 100 neuron in layer M
    - [ ] 100 neuron in layer S
    - [ ] 100 neuron in layer CI
    

### STEP 2 - CREATE A MODEL OF THE THALAMUS

*Neurons are modeled by Izhikevich, with synaptic connection modeled by Tsodkys and Markram*

- [ ] Model_T 1.0:
    - [ ] 1 neuron in thalamo-cortical relay nucleus (TC)
    - [ ] 1 neuron in thalamic reticular nucleus (TR)

- [ ] Model_T 2.0:
    - [ ] 10 neuron in thalamo-cortical relay nucleus (TC)
    - [ ] 4 neuron in thalamic reticular nucleus (TR)
    
- [ ] Model_T 3.0:
    - [ ] 100 neuron in thalamo-cortical relay nucleus (TC)
    - [ ] 40 neuron in thalamic reticular nucleus (TR)

### STEP 3 - COUPLING THE CORTEX AND THE THALAMUS

*Connection between the layer D and the Thalamus (TC and TR) -> Pure Facilitating*
*Connection between the TC and layer D -> Pure Depressive*

- [ ] Model_Final:
    - [ ] Modelo_C_1 + Modelo_T_1
    - [ ] Modelo_C_2 + Modelo_T_2
    - [ ] Modelo_C_3 + Modelo_T_3
