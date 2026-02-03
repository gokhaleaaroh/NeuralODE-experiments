# NeuralODE-experiments
A set of experiments for testing the capabilities of Neural ODEs, a deep learning architecture capabale of continuous-time modeling of data.

## Van der Pol Oscillator
Neural ODEs can learn the dynamics of a Van der Pol oscillator by learning through ground truth trajectory samples. The resulting model is a surrogate for the true dynamics and mimics the true dynamics remarkably well. Below are some visualizations comparing the learned dynamics with the ground truth. 

![phase_gt_vs_pred](https://github.com/user-attachments/assets/f9b9eaae-5509-4c9d-a26d-e3ebb37edb07)
<img width="1484" height="595" alt="good_phase_picture" src="https://github.com/user-attachments/assets/938867cb-9bb1-4737-9379-20d040aa2d61" />

## Continuous Normalizing Flows
Neural ODEs also allow learning a continuous version of normalizing flows, which are a way to generate a target distribution by successively applying diffeomorphic transformations to a source distribution. Neural ODEs allow learning this as a continuous-time differential equation. In the following visualizations, we see the result of learning to morph the standard normal distribution into a spiral distribution:

![spiral_morph](https://github.com/user-attachments/assets/7847f893-b69b-4165-ab14-75ff7839386d)
<img width="636" height="566" alt="true-vs-generated" src="https://github.com/user-attachments/assets/ad6afbc3-ccee-4771-bcea-29acf6efe4f8" />
